import numpy as np
import torch
from torch.utils.data import Dataset

def _estimate_bytes(n, gene_dim, drug_dim, bytes_per_elem=4):
    return n * (gene_dim + drug_dim) * bytes_per_elem

class GDSCDataset(Dataset):
    def __init__(
        self,
        pairs_df,
        exp_values,         # np.ndarray または torch.Tensor
        cid_to_row,
        drug_vec,           # dict: SMILES -> np.ndarray
        *,
        dtype: torch.dtype = torch.float32,
        pin_memory: bool = False,      # CUDA の時だけ True に
        copy_arrays: bool = False,
        materialize: str = "auto",
        ram_limit_gb: float = 8.0,
    ):
        self.dtype = dtype
        self.pin_memory = bool(pin_memory and torch.cuda.is_available())

        # --- exp_values を CPU の numpy(C連続) に統一 ---
        if isinstance(exp_values, torch.Tensor):
            exp_values = exp_values.detach().to("cpu").numpy()
        # ここで必ず C連続化（flags.get は使わない）
        exp_values = np.asarray(exp_values, order="C")
        self._exp_values = exp_values  # np.ndarray (CPU)

        # --- indices / arrays ---
        self.df_len = len(pairs_df)
        self._cid_index = np.fromiter(
            (cid_to_row[c] for c in pairs_df["COSMIC_ID"].to_numpy()),
            dtype=np.int64,
            count=self.df_len,
        )

        # SMILES をユニーク化して1回だけスタック
        smiles_all = pairs_df["SMILES"].to_numpy()
        uniq_smiles, inv = np.unique(smiles_all, return_inverse=True)
        xd_np = np.stack([drug_vec[s] for s in uniq_smiles], axis=0)  # (n_uniq, d)
        # ここは xd_np を C連続化（exp_values ではなく！）
        xd_np = np.asarray(xd_np, order="C")
        if copy_arrays:
            xd_np = xd_np.copy()

        # CPU tensor に変換
        self._drug_matrix = torch.from_numpy(xd_np).to(device="cpu", dtype=dtype)
        self._drug_idx = torch.from_numpy(inv.astype(np.int64))  # CPU

        # 目的変数（CPU）
        y_np = pairs_df["Z_score"].to_numpy(dtype=np.float32)
        if copy_arrays:
            y_np = y_np.copy()
        self.y = torch.from_numpy(y_np)  # CPU

        # メモリ見積り & materialize 判定
        gene_dim = self._exp_values.shape[1]
        drug_dim = self._drug_matrix.shape[1]
        if materialize == "auto":
            bytes_est = _estimate_bytes(
                self.df_len,
                gene_dim,
                drug_dim,
                2 if dtype in (torch.float16, torch.bfloat16) else 4,
            )
            want_materialize = "none" if bytes_est > ram_limit_gb * (1024**3) else "all"
        else:
            want_materialize = materialize

        self._materialized = want_materialize == "all"

        if self._materialized:
            # --- 前展開（CPU） ---
            xg_np = np.asarray(self._exp_values[self._cid_index], order="C")  # (N, gene_dim)
            if copy_arrays:
                xg_np = xg_np.copy()
            self.x_gene = torch.from_numpy(xg_np).to(device="cpu", dtype=dtype)
            self.x_drug = self._drug_matrix[self._drug_idx]  # CPU

            if self.pin_memory:  # CUDA の時だけ pin
                self.x_gene = self.x_gene.pin_memory()
                self.x_drug = self.x_drug.pin_memory()
                self.y      = self.y.pin_memory()
        else:
            # 非前展開：行列とインデックス/ラベルだけ pin（CUDA の時だけ）
            if self.pin_memory:
                self._drug_matrix = self._drug_matrix.pin_memory()
                self._drug_idx    = self._drug_idx.pin_memory()
                self.y            = self.y.pin_memory()

        self.gene_dim = gene_dim
        self.drug_dim = drug_dim

    def __len__(self):
        return self.df_len

    def __getitem__(self, i):
        if self._materialized:
            return self.x_gene[i], self.x_drug[i], self.y[i]
        else:
            gi = self._cid_index[i]
            xg = torch.from_numpy(self._exp_values[gi]).to(device="cpu", dtype=self.dtype)
            xd = self._drug_matrix[self._drug_idx[i]]  # CPU
            y  = self.y[i]
            return xg, xd, y
