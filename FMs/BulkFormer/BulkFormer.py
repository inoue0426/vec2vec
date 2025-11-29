import os
import math
import pandas as pd
import numpy as np
from collections import OrderedDict

import torch
from torch.utils.data import TensorDataset, DataLoader
from torch_geometric.typing import SparseTensor

from utils.BulkFormer import BulkFormer
from model.config import model_params
from tqdm import tqdm

import urllib.request  # ← 追加

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CKPT_URL = (
    "https://zenodo.org/records/15559368/files/"
    "Bulkformer_ckpt_epoch_29.pt?download=1"
)

ESM2_URL = (
    "https://zenodo.org/records/15559368/files/"
    "esm2_feature_concat.pt?download=1"
)


def download_file_if_needed(path: str, url: str, label: str = ""):
    """
    指定パスにファイルが無ければ URL からダウンロードする。
    """
    if os.path.exists(path):
        tag = f"[BulkFormer] {label}" if label else "[BulkFormer]"
        print(f"{tag} file found: {path}")
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)
    tag = f"[BulkFormer] {label}" if label else "[BulkFormer]"

    print(f"{tag} file not found. Downloading from:\n  {url}")
    print(f"{tag} Saving to: {path}")

    with urllib.request.urlopen(url) as response, open(path, "wb") as out_f:
        out_f.write(response.read())

    print(f"{tag} Download finished.")


def move_model_to_device(model, device):
    """
    Force-move all tensors in the model to the given device, including those
    not registered as buffers (e.g., inv_freq in Rope).
    """
    # 1. parameters
    for name, param in model.named_parameters(recurse=True):
        if param is not None:
            model._parameters[name] = param.to(device)

    # 2. buffers
    for name, buffer in model.named_buffers(recurse=True):
        if buffer is not None:
            model._buffers[name] = buffer.to(device)

    # 3. attributes directly holding tensors
    for name, value in model.__dict__.items():
        if isinstance(value, torch.Tensor):
            setattr(model, name, value.to(device))

    return model


def load_bulkformer(
    device="cpu",
    graph_path=None,
    weights_path=None,
    gene_emb_path=None,
    ckpt_path=None,
    gene_info_path=None,
    high_var_gene_idx_path=None,
):
    # Default paths
    if graph_path is None:
        graph_path = os.path.join(BASE_DIR, "data/G_gtex.pt")
    if weights_path is None:
        weights_path = os.path.join(BASE_DIR, "data/G_gtex_weight.pt")
    if gene_emb_path is None:
        gene_emb_path = os.path.join(BASE_DIR, "data/esm2_feature_concat.pt")
    if ckpt_path is None:
        ckpt_path = os.path.join(BASE_DIR, "model/Bulkformer_ckpt_epoch_29.pt")
    if gene_info_path is None:
        gene_info_path = os.path.join(BASE_DIR, "data/bulkformer_gene_info.csv")
    if high_var_gene_idx_path is None:
        high_var_gene_idx_path = os.path.join(BASE_DIR, "data/high_var_gene_list.pt")

    # ---- 自動ダウンロード ----
    download_file_if_needed(ckpt_path, CKPT_URL, label="Checkpoint")
    download_file_if_needed(gene_emb_path, ESM2_URL, label="ESM2 embedding")

    """
    Load BulkFormer and related objects, and return an inference-ready bundle.
    """

    # --- graph / weight / gene embedding ---
    graph_raw = torch.load(graph_path, map_location="cpu", weights_only=False)
    weights = torch.load(weights_path, map_location="cpu", weights_only=False)
    graph = SparseTensor(row=graph_raw[1], col=graph_raw[0], value=weights).t().to(device)

    gene_emb = torch.load(gene_emb_path, map_location="cpu", weights_only=False)

    # --- model ---
    model_params_local = dict(model_params)
    model_params_local["graph"] = graph
    model_params_local["gene_emb"] = gene_emb

    model = BulkFormer(**model_params_local).to(device)

    ckpt_model = torch.load(ckpt_path, map_location=torch.device("cpu"), weights_only=False)
    new_state_dict = OrderedDict()
    for key, value in ckpt_model.items():
        new_key = key[7:] if key.startswith("module.") else key
        new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict)
    model.eval()
    model = move_model_to_device(model, device)

    # --- gene list / high var gene index ---
    bulkformer_gene_info = pd.read_csv(gene_info_path)
    bulkformer_gene_list = bulkformer_gene_info["ensg_id"].tolist()

    high_var_gene_idx = torch.load(high_var_gene_idx_path, map_location="cpu", weights_only=False)

    return {
        "model": model,
        "graph": graph,
        "gene_emb": gene_emb,
        "bulkformer_gene_list": bulkformer_gene_list,
        "high_var_gene_idx": high_var_gene_idx,
    }


def align_to_bulkformer_genes(X_df, gene_list):
    """
    log-transformed TPM DataFrame (列が ENSG... ) を、
    BulkFormer が想定している gene_list の順番に揃える。
    足りない遺伝子には -10 を入れる。
    """

    # BulkFormerが期待するgene_listのうち、入力に無い遺伝子
    to_fill_columns = list(set(gene_list) - set(X_df.columns))

    # 入力側に存在する（=カバーできた）遺伝子
    covered_genes = list(set(gene_list) & set(X_df.columns))

    # # ★★★ カバー率をここで Print ★★★
    # print(f"[BulkFormer] Covered genes: {len(covered_genes)} / {len(gene_list)} "
    #       f"({len(covered_genes)/len(gene_list)*100:.2f}%)")
    # print(f"[BulkFormer] Missing genes : {len(to_fill_columns)} (filled with -10)")
    # # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★

    # 欠損遺伝子を -10 で埋める
    padding_df = pd.DataFrame(
        np.full((X_df.shape[0], len(to_fill_columns)), -10, dtype=float),
        columns=to_fill_columns,
        index=X_df.index,
    )

    # 列を結合してBulkFormer順に並べ直す
    X_concat = pd.concat([X_df, padding_df], axis=1)
    X_concat = X_concat[gene_list]

    # mask作成 (0=元の遺伝子, 1=パディングされた遺伝子)
    var = pd.DataFrame(index=X_concat.columns)
    var["mask"] = [1 if g in to_fill_columns else 0 for g in var.index]

    return X_concat, to_fill_columns, var


def bulkformer_embed(
    log_tpm_df: pd.DataFrame,
    bf_obj: dict,
    feature_type: str = "transcriptome_level",
    aggregate_type: str = "max",  # "max" / "mean" / "median" / "all"
    batch_size: int = 32,
    return_expr_value: bool = False,
    device="cpu",
    is_tqdm=False
):
    """
    log-transformed TPM (列: ENSG..., 行: サンプル) から BulkFormer embedding を返す関数。

    Parameters
    ----------
    log_tpm_df : pandas.DataFrame
        - 行: サンプル
        - 列: Ensembl Gene ID (例: ENSG00000000003)
        - 値: log(TPM+1) などの log-transformed expression

    bf_obj : dict
        load_bulkformer() の戻り値 (model, bulkformer_gene_list, high_var_gene_idx など)

    feature_type : {"transcriptome_level", "gene_level"}
        - "transcriptome_level": 1サンプル → 1ベクトル
        - "gene_level": 1サンプル → (遺伝子数 × 次元) のテンソル

    aggregate_type : {"max", "mean", "median", "all"}
        transcriptome_level のときの集約方法

    return_expr_value : bool
        True の場合、embedding ではなくモデルの出力発現値を返す

    Returns
    -------
    torch.Tensor または numpy.ndarray
        - return_expr_value=False の場合: embedding (torch.Tensor)
        - return_expr_value=True の場合: 予測発現値 (numpy.ndarray)
    """
    model = bf_obj["model"]
    gene_emb = bf_obj["gene_emb"]
    bulkformer_gene_list = bf_obj["bulkformer_gene_list"]
    high_var_gene_idx = bf_obj["high_var_gene_idx"]

    # 1) gene list に揃える (欠損遺伝子は -10 埋め)
    input_df, to_fill_columns, var = align_to_bulkformer_genes(
        X_df=log_tpm_df,
        gene_list=bulkformer_gene_list,
    )

    var = var.reset_index()
    valid_gene_idx = list(var[var["mask"] == 0].index)
    
    # 2) DataLoader 作成
    expr_array = input_df.values.astype("float32")
    expr_tensor = torch.tensor(expr_array, dtype=torch.float32, device=device)
    dataset = TensorDataset(expr_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    if is_tqdm:
        loader = tqdm(loader, total=len(loader))

    all_emb_list = []
    all_expr_value_list = []

    model.eval()
    with torch.no_grad():
        if feature_type == "transcriptome_level":
            for (X,) in loader:
                X = X.to(device)
                output, emb = model(X, [2])
                all_expr_value_list.append(output.detach().cpu().numpy())

                emb = emb[2].detach().cpu().numpy()          # [B, N_genes, D]
                emb_valid = emb[:, high_var_gene_idx, :]     # 高変動遺伝子のみ

                if aggregate_type == "max":
                    final_emb = np.max(emb_valid, axis=1)
                elif aggregate_type == "mean":
                    final_emb = np.mean(emb_valid, axis=1)
                elif aggregate_type == "median":
                    final_emb = np.median(emb_valid, axis=1)
                elif aggregate_type == "all":
                    max_emb = np.max(emb_valid, axis=1)
                    mean_emb = np.mean(emb_valid, axis=1)
                    median_emb = np.median(emb_valid, axis=1)
                    final_emb = max_emb + mean_emb + median_emb
                else:
                    raise ValueError(f"Unknown aggregate_type: {aggregate_type}")

                all_emb_list.append(final_emb)

            result_emb = np.vstack(all_emb_list)
            result_emb = torch.tensor(result_emb, device="cpu", dtype=torch.float32)

        elif feature_type == "gene_level":
            for (X,) in loader:
                X = X.to(device)
                output, emb = model(X, [2])
                emb = emb[2].detach().cpu().numpy()      # [B, N_genes, D]
                emb_valid = emb[:, valid_gene_idx, :]    # 欠損ではない遺伝子のみ
                all_emb_list.append(emb_valid)
                all_expr_value_list.append(output.detach().cpu().numpy())

            all_emb = np.vstack(all_emb_list)           # [N_samples, N_valid_genes, D]
            all_emb_tensor = torch.tensor(all_emb, device="cpu", dtype=torch.float32)

            # ESM2 embedding を concat
            esm2_emb_selected = gene_emb[valid_gene_idx]
            esm2_emb_expanded = esm2_emb_selected.unsqueeze(0).expand(all_emb_tensor.shape[0], -1, -1)
            esm2_emb_expanded = esm2_emb_expanded.to("cpu")

            result_emb = torch.cat([all_emb_tensor, esm2_emb_expanded], dim=-1)
        else:
            raise ValueError(f"Unknown feature_type: {feature_type}")

    if return_expr_value:
        return np.vstack(all_expr_value_list)
    else:
        return result_emb
