from __future__ import annotations
from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================
# Base Model (GDSC構造に統一)
# ==========================
class GeneDrugVAEBase(nn.Module):
    def __init__(
        self,
        gene_dim: int,
        drug_dim: int = 768,
        proj_dim: int = 256,
        hidden: int = 512,
        latent: int = 128,
        use_ic50: bool = False,  # True: Z-score回帰ヘッド有効
    ):
        super().__init__()
        self.gene_dim = gene_dim
        self.drug_dim = drug_dim
        self.proj_dim = proj_dim
        self.hidden = hidden
        self.latent = latent
        self.use_ic50 = use_ic50

        # --- Gene Encoder (GDSCと同じ) ---
        self.g1 = nn.Linear(gene_dim, hidden)
        self.g2 = nn.Linear(hidden, hidden // 2)

        # --- Drug Projector (GDSCと同じ: 768→256→256, ReLUのみ) ---
        self.p1 = nn.Linear(drug_dim, proj_dim)
        self.p2 = nn.Linear(proj_dim, proj_dim)

        # --- Latent ---
        comb = hidden // 2 + proj_dim
        self.fc = nn.Linear(comb, comb)
        self.mu = nn.Linear(comb, latent)
        self.lv = nn.Linear(comb, latent)

        # --- Decoder (gene reconstruction) ---
        self.d1 = nn.Linear(latent, hidden // 2)
        self.d2 = nn.Linear(hidden // 2, gene_dim)

        # --- Optional: Z-score regression head ---
        self.head = nn.Linear(latent, 1) if self.use_ic50 else None

    # ===== blocks =====
    def _encode_gene(self, xg: torch.Tensor) -> torch.Tensor:
        hg = F.relu(self.g1(xg))
        hg = F.relu(self.g2(hg))
        return hg

    def _project_drug(self, xd: torch.Tensor) -> torch.Tensor:
        hd = F.relu(self.p1(xd))
        hd = F.relu(self.p2(hd))
        return hd

    def encode(self, xg: torch.Tensor, xd: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hg = self._encode_gene(xg)
        hd = self._project_drug(xd)
        h = torch.cat([hg, hd], dim=-1)
        h = F.relu(self.fc(h))
        mu, lv = self.mu(h), self.lv(h)
        return mu, lv

    def reparam(self, mu: torch.Tensor, lv: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * lv)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.d1(z))
        return self.d2(h)

    def forward(self, xg: torch.Tensor, xd: torch.Tensor):
        mu, lv = self.encode(xg, xd)
        z = self.reparam(mu, lv)
        recon = self.decode(z)
        if self.use_ic50 and self.head is not None:
            z_pred = self.head(z).squeeze(-1)
            return recon, mu, lv, z_pred
        else:
            return recon, mu, lv


# ==========================
# Loss helpers
# ==========================

def kld_loss(mu: torch.Tensor, lv: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp(), dim=-1).mean()


def recon_loss_mse(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(recon, target, reduction="mean")


def zscore_regression_loss(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, y, reduction="mean")


# ==========================
# Datasets (同一モデルで使い回し)
# ==========================
class BasePairsDataset(torch.utils.data.Dataset):
    """(x_gene, x_drug[, y]) を返す。TCGAはy無し。"""
    def __init__(self, pairs_df, exp_values, id_to_row, drug_vec, id_col: str):
        self.df = pairs_df
        self.exp_values = exp_values
        self.id_to_row = id_to_row
        self.drug_vec = drug_vec
        self.id_col = id_col

    def __len__(self):
        return len(self.df)

    def _fetch_gene(self, id_):
        return self.exp_values[self.id_to_row[id_]]

    def _fetch_drug(self, smiles):
        return self.drug_vec[smiles]


class GDSCDataset(BasePairsDataset):
    """Z_score (LN_IC50を薬ごと標準化済み) を y として返す。"""
    def __init__(self, pairs_df, exp_values, cid_to_row, drug_vec):
        super().__init__(pairs_df, exp_values, cid_to_row, drug_vec, id_col="COSMIC_ID")

    def __getitem__(self, i):
        row = self.df.iloc[i]
        cid = row["COSMIC_ID"]
        smi = row["SMILES"]
        z = float(row["Z_score"])  # 既標準化 LN_IC50
        x_gene = torch.from_numpy(self._fetch_gene(cid)).float()
        x_drug = torch.from_numpy(self._fetch_drug(smi)).float()
        y = torch.tensor(z, dtype=torch.float32)
        return x_gene, x_drug, y


class TCGADataset(BasePairsDataset):
    """y無し（再構成のみ）"""
    def __init__(self, pairs_df, exp_values, sid_to_row, drug_vec):
        super().__init__(pairs_df, exp_values, sid_to_row, drug_vec, id_col="SAMPLE_ID")

    def __getitem__(self, i):
        row = self.df.iloc[i]
        sid = row["SAMPLE_ID"]
        smi = row["SMILES"]
        x_gene = torch.from_numpy(self._fetch_gene(sid)).float()
        x_drug = torch.from_numpy(self._fetch_drug(smi)).float()
        return x_gene, x_drug


# ==========================
# Training step (共通)
# ==========================
@torch.no_grad()
def _detach(x):
    return x.detach() if torch.is_tensor(x) else x


def training_step(model: GeneDrugVAEBase, batch, kl_beta: float = 1.0, zscore_lambda: float = 1.0):
    """
    use_ic50=True の場合:  (xg, xd, y) -> recon + KL + z回帰
    use_ic50=False の場合: (xg, xd)     -> recon + KL
    """
    if model.use_ic50:
        if len(batch) != 3:
            raise ValueError("Expected (xg, xd, y) when use_ic50=True")
        xg, xd, y = batch
        recon, mu, lv, z_pred = model(xg, xd)
        loss_recon = recon_loss_mse(recon, xg)
        loss_kld = kld_loss(mu, lv)
        loss_z = zscore_regression_loss(z_pred, y)
        loss = loss_recon + kl_beta * loss_kld + zscore_lambda * loss_z
        return {"loss": loss, "recon": _detach(loss_recon), "kld": _detach(loss_kld), "zscore": _detach(loss_z)}
    else:
        if len(batch) != 2:
            raise ValueError("Expected (xg, xd) when use_ic50=False")
        xg, xd = batch
        recon, mu, lv = model(xg, xd)
        loss_recon = recon_loss_mse(recon, xg)
        loss_kld = kld_loss(mu, lv)
        loss = loss_recon + kl_beta * loss_kld
        return {"loss": loss, "recon": _detach(loss_recon), "kld": _detach(loss_kld)}


# ==========================
# Convenience wrappers
# ==========================
class GDSCModel(GeneDrugVAEBase):
    """GDSC用途: 回帰ヘッド有効(use_ic50=True)。構造はBaseそのもの。"""
    def __init__(self, gene_dim: int, drug_dim: int = 768, proj_dim: int = 256, hidden: int = 512, latent: int = 128):
        super().__init__(gene_dim, drug_dim, proj_dim, hidden, latent, use_ic50=True)


class TCGAModel(GeneDrugVAEBase):
    """TCGA用途: 同一構造で回帰ヘッド無効(use_ic50=False)。"""
    def __init__(self, gene_dim: int, drug_dim: int = 768, proj_dim: int = 256, hidden: int = 512, latent: int = 128):
        super().__init__(gene_dim, drug_dim, proj_dim, hidden, latent, use_ic50=False)


# ==========================
# Example usage
# ==========================
if __name__ == "__main__":
    G = 20000  # gene_dim の例
    model_gdsc = GDSCModel(gene_dim=G)
    model_tcga = TCGAModel(gene_dim=G)
    # これで構造は同一、違いはIC50を使うかどうかだけ。
