import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneDrugVAE(nn.Module):
    def __init__(self, gene_dim, drug_dim=768, proj_dim=256, hidden=512, latent=128):
        super().__init__()
        # Gene encoder
        self.g1 = nn.Linear(gene_dim, hidden)
        self.g2 = nn.Linear(hidden, hidden // 2)

        # Drug projection
        self.p1 = nn.Linear(drug_dim, proj_dim)
        self.p2 = nn.Linear(proj_dim, proj_dim)

        # Latent
        comb = hidden // 2 + proj_dim
        self.fc = nn.Linear(comb, comb)
        self.mu = nn.Linear(comb, latent)
        self.lv = nn.Linear(comb, latent)

        # Decoder (gene reconstruction)
        self.d1 = nn.Linear(latent, hidden // 2)
        self.d2 = nn.Linear(hidden // 2, gene_dim)

        # Z-score regression head (regularizer)
        self.head = nn.Sequential(
            nn.Linear(latent, 128),
            nn.ReLU(),  # ← ReLUでもOK（SiLUでも可）
            nn.Dropout(0.1),
            nn.Linear(128, 1),  # ← 最終は活性化なし
        )

    def encode(self, xg, xd):
        hg = F.relu(self.g1(xg))
        hg = F.relu(self.g2(hg))
        hd = F.relu(self.p1(xd))
        hd = F.relu(self.p2(hd))
        h = torch.cat([hg, hd], dim=-1)
        h = F.relu(self.fc(h))
        return self.mu(h), self.lv(h)

    def reparam(self, mu, lv):
        std = torch.exp(0.5 * lv)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.d1(z))
        return self.d2(h)

    def forward(self, xg, xd):
        mu, lv = self.encode(xg, xd)
        z = self.reparam(mu, lv)
        recon = self.decode(z)  # 生成は z
        z_pred = self.head(mu).squeeze(-1)  # 予測は mu に変更
        return recon, mu, lv, z_pred
