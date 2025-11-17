import torch
from torch import nn
import torch.nn.functional as F
from base_ae import BaseAE
from typing import List, Optional

Tensor = torch.Tensor

class VAE(BaseAE):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 128,
        hidden_dims: Optional[List[int]] = None,
        dop: float = 0.1,
        noise_flag: bool = True,
        **kwargs
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.dop = dop
        self.noise_flag = noise_flag

        hidden_dims = hidden_dims or [512, 256]
        self.hidden_dims = hidden_dims
        self.embedder = self._make_layers(
            [input_dim] + hidden_dims[:-1], hidden_dims, dop
        )
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

        dec_hidden_dims = list(reversed(hidden_dims))
        self.decoder = self._make_layers(
            [latent_dim] + dec_hidden_dims[:-1], dec_hidden_dims, dop
        )
        self.final_layer = nn.Sequential(
            nn.Linear(dec_hidden_dims[-1], dec_hidden_dims[-1]),
            nn.ReLU(),
            nn.Dropout(self.dop),
            nn.Linear(dec_hidden_dims[-1], input_dim)
        )

    @staticmethod
    def _make_layers(in_dims, out_dims, dop):
        layers = []
        for in_dim, out_dim in zip(in_dims, out_dims):
            layers.append(
                nn.Sequential(
                    nn.Linear(in_dim, out_dim, bias=True),
                    nn.ReLU(),
                    nn.Dropout(dop)
                )
            )
        return nn.Sequential(*layers)

    def encode(self, x: Tensor) -> list:
        if self.noise_flag and self.training:
            x = x + torch.randn_like(x) * 0.1
        embed = self.embedder(x)
        mu = self.fc_mu(embed)
        log_var = self.fc_var(embed)
        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        z = self.decoder(z)
        x_recon = self.final_layer(z)
        return x_recon

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: Tensor, **kwargs) -> list:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recons = self.decode(z)
        return [x, recons, mu, log_var]

    def loss_function(self, *args, **kwargs) -> dict:
        x, recons, mu, log_var = args[:4]
        kld_weight = kwargs.get('M_N', 1.0)
        recons_loss = F.mse_loss(x, recons)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1))
        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'recons_loss': recons_loss, 'KLD': kld_loss}

    def sample(self, num_samples: int, current_device: int, **kwargs) -> Tensor:
        z = torch.randn(num_samples, self.latent_dim, device=current_device)
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        return self.forward(x)[1]
