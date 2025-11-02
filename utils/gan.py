import dataclasses
import types

import accelerate
import torch
import torch.nn.functional as F


@dataclasses.dataclass
class VanillaGAN:    
    """
    VanillaGAN implements a simple GAN training loop for embedding-based generators and discriminators.

    This class coordinates the adversarial training between a generator (e.g. translator network)
    and a discriminator that tries to distinguish between "real" (true embeddings) and "fake" 
    (generator-produced) samples.

    The generator is updated using BCE-based adversarial loss, while the discriminator learns
    to classify real vs fake embeddings correctly. Optional R1-style gradient penalty and label
    smoothing are supported.
    """
    
    cfg: types.SimpleNamespace
    generator: torch.nn.Module
    discriminator: torch.nn.Module
    discriminator_opt: torch.optim.Optimizer
    discriminator_scheduler: torch.optim.lr_scheduler._LRScheduler
    accelerator: accelerate.Accelerator

    @property
    def _batch_size(self) -> int:
        return self.cfg.bs
    
    def compute_gradient_penalty(self, d_out: torch.Tensor, d_in: torch.Tensor) -> torch.Tensor:
        gradients = torch.autograd.grad(
            outputs=d_out.sum(),
            inputs=d_in,
            create_graph=True,
            retain_graph=True,
        )[0]
        
        return gradients.pow(2).sum().mean()
    
    def set_discriminator_requires_grad(self, rg: bool) -> None:
        for module in self.discriminator.parameters():
            module.requires_grad = rg

    def _step_discriminator(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float, float]:
        real_data = real_data.detach().requires_grad_(True)
        fake_data = fake_data.detach().requires_grad_(True)
        d_real_logits, d_fake_logits = self.discriminator(real_data), self.discriminator(fake_data)

        device = d_real_logits.device
        batch_size = d_real_logits.size(0)
        real_labels = torch.ones((batch_size, 1), device=device) * (1 - self.cfg.smooth)
        fake_labels = torch.ones((batch_size, 1), device=device) * self.cfg.smooth
        disc_loss_real = F.binary_cross_entropy_with_logits(d_real_logits, real_labels)
        disc_loss_fake = F.binary_cross_entropy_with_logits(d_fake_logits, fake_labels)
        disc_loss = (disc_loss_real + disc_loss_fake) / 2
        disc_acc_real = (d_real_logits.sigmoid() < 0.5).float().mean().item()
        disc_acc_fake = (d_fake_logits.sigmoid() > 0.5).float().mean().item()

        r1_penalty = self.compute_gradient_penalty(d_out=d_real_logits, d_in=real_data)
        r2_penalty = self.compute_gradient_penalty(d_out=d_fake_logits, d_in=fake_data)

        self.generator.train()
        self.discriminator_opt.zero_grad()
        self.accelerator.backward(
            (
                disc_loss + 
                ((r1_penalty + r2_penalty) * self.cfg.loss_coefficient_r1_penalty)
            ) * self.cfg.loss_coefficient_disc
        )
        self.accelerator.clip_grad_norm_(
            self.discriminator.parameters(),
            self.cfg.max_grad_norm
        )
        self.discriminator_opt.step()
        self.discriminator_scheduler.step()
        return (r1_penalty + r2_penalty).detach(), disc_loss.detach(), disc_acc_real, disc_acc_fake

    def _step_generator(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> tuple[torch.Tensor, float]:
        d_fake_logits = self.discriminator(fake_data)
        device = fake_data.device
        batch_size = fake_data.size(0)
        real_labels = torch.zeros((batch_size, 1), device=device)
        gen_loss = F.binary_cross_entropy_with_logits(d_fake_logits, real_labels)
        gen_acc = (d_fake_logits.sigmoid() < 0.5).float().mean().item()
        return gen_loss, gen_acc

    def step_discriminator(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> tuple[torch.Tensor, float, float]:
        if self.cfg.loss_coefficient_disc > 0:
            return self._step_discriminator(real_data, fake_data)
        else:
            return torch.tensor(0.0), 0.0, 0.0

    def step_generator(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> tuple[torch.Tensor, float]:
        if self.cfg.loss_coefficient_gen > 0:
            return self._step_generator(real_data=real_data, fake_data=fake_data)
        else:
            return torch.tensor(0.0), 0.0

    def step(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, float, float, float]:
        self.generator.eval()
        self.discriminator.train()
        self.set_discriminator_requires_grad(True)
        r1_penalty, disc_loss, disc_acc_real, disc_acc_fake = self.step_discriminator(
            real_data=real_data.detach(),
            fake_data=fake_data.detach()
        )
        self.generator.train()
        self.discriminator.eval()
        self.set_discriminator_requires_grad(False)
        gen_loss, gen_acc = self.step_generator(
            real_data=real_data,
            fake_data=fake_data
        )

        return r1_penalty, disc_loss, gen_loss, disc_acc_real, disc_acc_fake, gen_acc


import dataclasses
import types

import accelerate
import torch
import torch.nn.functional as F


# =========================
# MMD / Cauchy–Schwarz / Wasserstein-GP
# =========================

class MMDGAN:
    """
    Use MMD^2(real, fake) as the generator loss. No discriminator is updated.
    Keeps the same .step(...) API as other GAN classes.
    cfg:
      - mmd_sigmas: list[float] = [0.5, 1.0, 2.0, 4.0]
    """
    def __init__(self, cfg, generator, discriminator, discriminator_opt, discriminator_scheduler, accelerator):
        self.cfg = cfg
        self.generator = generator
        self.accelerator = accelerator
        self.sigmas = getattr(cfg, "mmd_sigmas", [0.5, 1.0, 2.0, 4.0])
        # keep for API compatibility
        self.discriminator = discriminator
        self.discriminator_opt = discriminator_opt
        self.discriminator_scheduler = discriminator_scheduler

    @staticmethod
    def _pdist2(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # pairwise squared L2 distances
        x2 = (x**2).sum(dim=1, keepdim=True)           # [B,1]
        y2 = (y**2).sum(dim=1, keepdim=True).transpose(0, 1)  # [1,B]
        return x2 + y2 - 2.0 * (x @ y.transpose(0, 1)) # [B,B]

    def _rbf_mix(self, d2: torch.Tensor) -> torch.Tensor:
        K = 0.0
        for s in self.sigmas:
            gamma = 1.0 / (2.0 * s * s + 1e-12)
            K = K + torch.exp(-gamma * d2)
        return K

    def _mmd2_unbiased(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        d2_xx = self._pdist2(x, x)
        d2_yy = self._pdist2(y, y)
        d2_xy = self._pdist2(x, y)
        Kxx = self._rbf_mix(d2_xx)
        Kyy = self._rbf_mix(d2_yy)
        Kxy = self._rbf_mix(d2_xy)
        B = x.size(0)
        mask = ~torch.eye(B, dtype=torch.bool, device=x.device)
        return (Kxx[mask].mean() + Kyy[mask].mean() - 2.0 * Kxy.mean())

    def step(self, real_data: torch.Tensor, fake_data: torch.Tensor):
        self.generator.train()
        gen_loss = self._mmd2_unbiased(real_data, fake_data)
        # Discriminator-related values are dummies to keep API shape
        zero = torch.tensor(0.0, device=real_data.device)
        return zero, zero, gen_loss, 0.0, 0.0, float('nan')


class CauchySchwarzGAN:
    """
    Use Cauchy–Schwarz divergence between kernel Gram matrices as the generator loss.
    No discriminator is updated.
    cfg:
      - cs_sigma: float = 1.0
    """
    def __init__(self, cfg, generator, discriminator, discriminator_opt, discriminator_scheduler, accelerator):
        self.cfg = cfg
        self.generator = generator
        self.accelerator = accelerator
        self.sigma = float(getattr(cfg, "cs_sigma", 1.0))
        # keep for API compatibility
        self.discriminator = discriminator
        self.discriminator_opt = discriminator_opt
        self.discriminator_scheduler = discriminator_scheduler

    @staticmethod
    def _pdist2(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x2 = (x**2).sum(dim=1, keepdim=True)
        y2 = (y**2).sum(dim=1, keepdim=True).transpose(0, 1)
        return x2 + y2 - 2.0 * (x @ y.transpose(0, 1))

    def _rbf(self, d2: torch.Tensor, sigma: float) -> torch.Tensor:
        gamma = 1.0 / (2.0 * sigma * sigma + 1e-12)
        return torch.exp(-gamma * d2)

    def _cs_divergence(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        Kxx = self._rbf(self._pdist2(x, x), self.sigma)
        Kyy = self._rbf(self._pdist2(y, y), self.sigma)
        Kxy = self._rbf(self._pdist2(x, y), self.sigma)
        # include diagonals for a simple/stable variant
        num = Kxy.sum()
        den = (Kxx.sum() * Kyy.sum()).clamp_min(1e-12).sqrt()
        return -torch.log(num / den)

    def step(self, real_data: torch.Tensor, fake_data: torch.Tensor):
        self.generator.train()
        gen_loss = self._cs_divergence(real_data, fake_data)
        zero = torch.tensor(0.0, device=real_data.device)
        return zero, zero, gen_loss, 0.0, 0.0, float('nan')


class WassersteinGAN_GP(VanillaGAN):
    """
    WGAN with Gradient Penalty (GP). Uses critic scores (no sigmoid).
    cfg:
      - wgan_gp_lambda: float = 10.0
    Notes:
      - Discriminator acts as a critic; do NOT put a sigmoid at the end.
      - Gradient penalty is computed on interpolated points.
    """
    def __init__(self, cfg, generator, discriminator, discriminator_opt, discriminator_scheduler, accelerator):
        super().__init__(cfg, generator, discriminator, discriminator_opt, discriminator_scheduler, accelerator)
        self.gp_lambda = float(getattr(cfg, "wgan_gp_lambda", 10.0))

    def _gradient_penalty(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        B = real.size(0)
        # broadcast eps to input shape
        eps = torch.rand(B, 1, device=real.device)
        eps = eps.expand_as(real)
        x_hat = eps * real + (1.0 - eps) * fake
        x_hat.requires_grad_(True)
        d_hat = self.discriminator(x_hat)
        grads = torch.autograd.grad(
            outputs=d_hat.sum(),
            inputs=x_hat,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        grads = grads.view(B, -1)
        return ((grads.norm(2, dim=1) - 1.0) ** 2).mean()

    def _step_discriminator(self, real_data: torch.Tensor, fake_data: torch.Tensor):
        # detach fake for D update
        real_data = real_data.detach()
        fake_data = fake_data.detach()
        self.discriminator.train()
        self.discriminator_opt.zero_grad(set_to_none=True)

        d_real = self.discriminator(real_data)   # [B,1] or [B]
        d_fake = self.discriminator(fake_data)   # [B,1] or [B]

        # make them scalar tensors
        d_real_mean = d_real.mean()
        d_fake_mean = d_fake.mean()

        gp = self._gradient_penalty(real_data, fake_data) * self.gp_lambda
        loss_d = (d_fake_mean - d_real_mean) + gp

        self.accelerator.backward(loss_d * self.cfg.loss_coefficient_disc)
        self.accelerator.clip_grad_norm_(self.discriminator.parameters(), self.cfg.max_grad_norm)
        self.discriminator_opt.step()
        self.discriminator_scheduler.step()

        # accuracies are not well-defined for WGAN; return NaN
        return gp.detach(), loss_d.detach(), float('nan'), float('nan')

    def _step_generator(self, real_data: torch.Tensor, fake_data: torch.Tensor):
        # Critic eval mode for stabler G loss evaluation
        self.discriminator.eval()
        d_fake = self.discriminator(fake_data)
        gen_loss = -d_fake.mean()
        gen_acc = float('nan')
        self.discriminator.train()
        return gen_loss, gen_acc

    def step(self, real_data: torch.Tensor, fake_data: torch.Tensor):
        # D update
        self.generator.eval()
        self.set_discriminator_requires_grad(True)
        gp, disc_loss, disc_acc_real, disc_acc_fake = self._step_discriminator(real_data, fake_data)

        # G loss (update is performed by outer loop)
        self.generator.train()
        self.set_discriminator_requires_grad(False)
        gen_loss, gen_acc = self._step_generator(real_data, fake_data)

        return gp, disc_loss, gen_loss, disc_acc_real, disc_acc_fake, gen_acc



@dataclasses.dataclass
class VanillaGAN:    
    """
    VanillaGAN implements a simple GAN training loop for embedding-based generators and discriminators.

    This class coordinates the adversarial training between a generator (e.g. translator network)
    and a discriminator that tries to distinguish between "real" (true embeddings) and "fake" 
    (generator-produced) samples.

    The generator is updated using BCE-based adversarial loss, while the discriminator learns
    to classify real vs fake embeddings correctly. Optional R1-style gradient penalty and label
    smoothing are supported.
    """
    
    cfg: types.SimpleNamespace
    generator: torch.nn.Module
    discriminator: torch.nn.Module
    discriminator_opt: torch.optim.Optimizer
    discriminator_scheduler: torch.optim.lr_scheduler._LRScheduler
    accelerator: accelerate.Accelerator

    @property
    def _batch_size(self) -> int:
        return self.cfg.bs
    
    def compute_gradient_penalty(self, d_out: torch.Tensor, d_in: torch.Tensor) -> torch.Tensor:
        gradients = torch.autograd.grad(
            outputs=d_out.sum(),
            inputs=d_in,
            create_graph=True,
            retain_graph=True,
        )[0]
        
        return gradients.pow(2).sum().mean()
    
    def set_discriminator_requires_grad(self, rg: bool) -> None:
        for module in self.discriminator.parameters():
            module.requires_grad = rg

    def _step_discriminator(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float, float]:
        real_data = real_data.detach().requires_grad_(True)
        fake_data = fake_data.detach().requires_grad_(True)
        d_real_logits, d_fake_logits = self.discriminator(real_data), self.discriminator(fake_data)

        device = d_real_logits.device
        batch_size = d_real_logits.size(0)
        real_labels = torch.ones((batch_size, 1), device=device) * (1 - self.cfg.smooth)
        fake_labels = torch.ones((batch_size, 1), device=device) * self.cfg.smooth
        disc_loss_real = F.binary_cross_entropy_with_logits(d_real_logits, real_labels)
        disc_loss_fake = F.binary_cross_entropy_with_logits(d_fake_logits, fake_labels)
        disc_loss = (disc_loss_real + disc_loss_fake) / 2
        disc_acc_real = (d_real_logits.sigmoid() < 0.5).float().mean().item()
        disc_acc_fake = (d_fake_logits.sigmoid() > 0.5).float().mean().item()

        r1_penalty = self.compute_gradient_penalty(d_out=d_real_logits, d_in=real_data)
        r2_penalty = self.compute_gradient_penalty(d_out=d_fake_logits, d_in=fake_data)

        self.generator.train()
        self.discriminator_opt.zero_grad()
        self.accelerator.backward(
            (
                disc_loss + 
                ((r1_penalty + r2_penalty) * self.cfg.loss_coefficient_r1_penalty)
            ) * self.cfg.loss_coefficient_disc
        )
        self.accelerator.clip_grad_norm_(
            self.discriminator.parameters(),
            self.cfg.max_grad_norm
        )
        self.discriminator_opt.step()
        self.discriminator_scheduler.step()
        return (r1_penalty + r2_penalty).detach(), disc_loss.detach(), disc_acc_real, disc_acc_fake

    def _step_generator(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> tuple[torch.Tensor, float]:
        d_fake_logits = self.discriminator(fake_data)
        device = fake_data.device
        batch_size = fake_data.size(0)
        real_labels = torch.zeros((batch_size, 1), device=device)
        gen_loss = F.binary_cross_entropy_with_logits(d_fake_logits, real_labels)
        gen_acc = (d_fake_logits.sigmoid() < 0.5).float().mean().item()
        return gen_loss, gen_acc

    def step_discriminator(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> tuple[torch.Tensor, float, float]:
        if self.cfg.loss_coefficient_disc > 0:
            return self._step_discriminator(real_data, fake_data)
        else:
            return torch.tensor(0.0), 0.0, 0.0

    def step_generator(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> tuple[torch.Tensor, float]:
        if self.cfg.loss_coefficient_gen > 0:
            return self._step_generator(real_data=real_data, fake_data=fake_data)
        else:
            return torch.tensor(0.0), 0.0

    def step(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, float, float, float]:
        self.generator.eval()
        self.discriminator.train()
        self.set_discriminator_requires_grad(True)
        r1_penalty, disc_loss, disc_acc_real, disc_acc_fake = self.step_discriminator(
            real_data=real_data.detach(),
            fake_data=fake_data.detach()
        )
        self.generator.train()
        self.discriminator.eval()
        self.set_discriminator_requires_grad(False)
        gen_loss, gen_acc = self.step_generator(
            real_data=real_data,
            fake_data=fake_data
        )

        return r1_penalty, disc_loss, gen_loss, disc_acc_real, disc_acc_fake, gen_acc



class LeastSquaresGAN(VanillaGAN):
    def _step_discriminator(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float, float]:
        real_data = real_data.detach().requires_grad_(True)
        fake_data = fake_data.detach().requires_grad_(True)
        d_real_logits, d_fake_logits = self.discriminator(real_data), self.discriminator(fake_data)

        device = d_real_logits.device
        batch_size = d_real_logits.size(0)
        real_labels = torch.ones((batch_size, 1), device=device) * (1 - self.cfg.smooth)
        fake_labels = torch.ones((batch_size, 1), device=device) * self.cfg.smooth
        disc_loss_real = (d_real_logits ** 2).mean()
        disc_loss_fake = ((d_fake_logits - 1) ** 2).mean()
        disc_loss = (disc_loss_real + disc_loss_fake) / 2
        disc_acc_real = ((d_real_logits ** 2) < 0.5).float().mean().item()
        disc_acc_fake = ((d_fake_logits ** 2) > 0.5).float().mean().item()

        r1_penalty = self.compute_gradient_penalty(d_out=d_real_logits, d_in=real_data)
        r2_penalty = self.compute_gradient_penalty(d_out=d_fake_logits, d_in=fake_data)
        self.generator.train()
        self.discriminator_opt.zero_grad()
        self.accelerator.backward(
            (disc_loss + ((r1_penalty + r2_penalty) * self.cfg.loss_coefficient_r1_penalty)) * self.cfg.loss_coefficient_disc
        )
        self.accelerator.clip_grad_norm_(
            self.discriminator.parameters(),
            self.cfg.max_grad_norm
        )
        self.discriminator_opt.step()
        self.discriminator_scheduler.step()
        return (r1_penalty + r2_penalty).detach(), disc_loss.detach(), disc_acc_real, disc_acc_fake

    def _step_generator(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> tuple[torch.Tensor, float]:
        d_fake_logits = self.discriminator(fake_data)
        device = fake_data.device
        batch_size = fake_data.size(0)
        gen_loss = ((d_fake_logits) ** 2).mean()
        gen_acc = ((d_fake_logits ** 2) < 0.5).float().mean().item()
        return gen_loss * 0.5, gen_acc


class RelativisticGAN(VanillaGAN):
    def _step_discriminator(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> tuple[torch.Tensor, float, float]:
        self.generator.eval()
        d_real_logits = self.discriminator(real_data)
        d_fake_logits = self.discriminator(fake_data)

        disc_loss = F.binary_cross_entropy_with_logits(d_fake_logits - d_real_logits, torch.ones_like(d_real_logits))
        disc_acc_real = (d_real_logits > d_fake_logits).float().mean().item()
        disc_acc_fake = 1.0 - disc_acc_real

        self.generator.train()
        self.discriminator_opt.zero_grad()
        self.accelerator.backward(disc_loss * self.cfg.loss_coefficient_disc)
        self.accelerator.clip_grad_norm_(
            self.discriminator.parameters(),
            self.cfg.max_grad_norm
        )
        self.discriminator_opt.step()

        return disc_loss, disc_acc_real, disc_acc_fake

    def _step_generator(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> tuple[torch.Tensor, float]:
        self.discriminator.eval()

        d_real_logits = self.discriminator(real_data)
        d_fake_logits = self.discriminator(fake_data)
        gen_loss = F.binary_cross_entropy_with_logits(d_real_logits - d_fake_logits, torch.ones_like(d_real_logits))

        gen_acc = (d_real_logits > d_fake_logits).float().mean().item()
        self.discriminator.train()
        return gen_loss, gen_acc


    

class LeastSquaresGAN(VanillaGAN):
    def _step_discriminator(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float, float]:
        real_data = real_data.detach().requires_grad_(True)
        fake_data = fake_data.detach().requires_grad_(True)
        d_real_logits, d_fake_logits = self.discriminator(real_data), self.discriminator(fake_data)

        device = d_real_logits.device
        batch_size = d_real_logits.size(0)
        real_labels = torch.ones((batch_size, 1), device=device) * (1 - self.cfg.smooth)
        fake_labels = torch.ones((batch_size, 1), device=device) * self.cfg.smooth
        disc_loss_real = (d_real_logits ** 2).mean()
        disc_loss_fake = ((d_fake_logits - 1) ** 2).mean()
        disc_loss = (disc_loss_real + disc_loss_fake) / 2
        disc_acc_real = ((d_real_logits ** 2) < 0.5).float().mean().item()
        disc_acc_fake = ((d_fake_logits ** 2) > 0.5).float().mean().item()

        r1_penalty = self.compute_gradient_penalty(d_out=d_real_logits, d_in=real_data)
        r2_penalty = self.compute_gradient_penalty(d_out=d_fake_logits, d_in=fake_data)
        self.generator.train()
        self.discriminator_opt.zero_grad()
        self.accelerator.backward(
            (disc_loss + ((r1_penalty + r2_penalty) * self.cfg.loss_coefficient_r1_penalty)) * self.cfg.loss_coefficient_disc
        )
        self.accelerator.clip_grad_norm_(
            self.discriminator.parameters(),
            self.cfg.max_grad_norm
        )
        self.discriminator_opt.step()
        self.discriminator_scheduler.step()
        return (r1_penalty + r2_penalty).detach(), disc_loss.detach(), disc_acc_real, disc_acc_fake

    def _step_generator(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> tuple[torch.Tensor, float]:
        d_fake_logits = self.discriminator(fake_data)
        device = fake_data.device
        batch_size = fake_data.size(0)
        gen_loss = ((d_fake_logits) ** 2).mean()
        gen_acc = ((d_fake_logits ** 2) < 0.5).float().mean().item()
        return gen_loss * 0.5, gen_acc


class RelativisticGAN(VanillaGAN):
    def _step_discriminator(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> tuple[torch.Tensor, float, float]:
        self.generator.eval()
        d_real_logits = self.discriminator(real_data)
        d_fake_logits = self.discriminator(fake_data)

        disc_loss = F.binary_cross_entropy_with_logits(d_fake_logits - d_real_logits, torch.ones_like(d_real_logits))
        disc_acc_real = (d_real_logits > d_fake_logits).float().mean().item()
        disc_acc_fake = 1.0 - disc_acc_real

        self.generator.train()
        self.discriminator_opt.zero_grad()
        self.accelerator.backward(disc_loss * self.cfg.loss_coefficient_disc)
        self.accelerator.clip_grad_norm_(
            self.discriminator.parameters(),
            self.cfg.max_grad_norm
        )
        self.discriminator_opt.step()

        return disc_loss, disc_acc_real, disc_acc_fake

    def _step_generator(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> tuple[torch.Tensor, float]:
        self.discriminator.eval()

        d_real_logits = self.discriminator(real_data)
        d_fake_logits = self.discriminator(fake_data)
        gen_loss = F.binary_cross_entropy_with_logits(d_real_logits - d_fake_logits, torch.ones_like(d_real_logits))

        gen_acc = (d_real_logits > d_fake_logits).float().mean().item()
        self.discriminator.train()
        return gen_loss, gen_acc
