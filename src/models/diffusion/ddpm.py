"""
Denoising Diffusion Probabilistic Model (DDPM) Implementation.

Core diffusion model components for radio map generation:
- Forward process: q(x_t | x_0) - gradually adds noise
- Reverse process: p(x_{t-1} | x_t) - learned denoising
- Noise schedules: linear and cosine beta schedules
- Loss computation: simplified MSE and variational lower bound

References:
- Ho et al. "Denoising Diffusion Probabilistic Models" (2020)
- Nichol & Dhariwal "Improved Denoising Diffusion Probabilistic Models" (2021)
"""

import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def linear_beta_schedule(
    num_timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
) -> torch.Tensor:
    """
    Linear noise schedule from DDPM paper.

    Args:
        num_timesteps: Number of diffusion steps (T)
        beta_start: Starting noise level
        beta_end: Ending noise level

    Returns:
        Beta values for each timestep (T,)
    """
    return torch.linspace(beta_start, beta_end, num_timesteps)


def cosine_beta_schedule(
    num_timesteps: int,
    s: float = 0.008,
    max_beta: float = 0.999,
) -> torch.Tensor:
    """
    Cosine noise schedule from Improved DDPM paper.

    Provides smoother noise addition, better for high-resolution images.

    Args:
        num_timesteps: Number of diffusion steps (T)
        s: Small offset to prevent singularity at t=0
        max_beta: Maximum beta value (clipped)

    Returns:
        Beta values for each timestep (T,)
    """
    steps = num_timesteps + 1
    t = torch.linspace(0, num_timesteps, steps)

    # Cosine schedule for alpha_bar
    alpha_bar = torch.cos(((t / num_timesteps) + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]  # Normalize

    # Convert to betas
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    return torch.clip(betas, 0.0001, max_beta)


def sigmoid_beta_schedule(
    num_timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
) -> torch.Tensor:
    """
    Sigmoid noise schedule - smooth transition between start and end.

    Args:
        num_timesteps: Number of diffusion steps (T)
        beta_start: Starting noise level
        beta_end: Ending noise level

    Returns:
        Beta values for each timestep (T,)
    """
    betas = torch.linspace(-6, 6, num_timesteps)
    betas = torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
    return betas


class SinusoidalPositionEmbedding(nn.Module):
    """
    Sinusoidal timestep embedding from Transformer/DDPM.

    Maps integer timestep to continuous embedding vector.
    """

    def __init__(self, dim: int, max_period: int = 10000):
        """
        Args:
            dim: Embedding dimension (must be even)
            max_period: Maximum period for sinusoidal functions
        """
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Embed timesteps.

        Args:
            timesteps: Integer timesteps (B,)

        Returns:
            Embeddings (B, dim)
        """
        half_dim = self.dim // 2
        embeddings = math.log(self.max_period) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=timesteps.device) * -embeddings)
        embeddings = timesteps[:, None].float() * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)

        # Handle odd dimensions
        if self.dim % 2 == 1:
            embeddings = F.pad(embeddings, (0, 1))

        return embeddings


class GaussianDiffusion(nn.Module):
    """
    Gaussian Diffusion Process for DDPM.

    Implements the forward (noising) and reverse (denoising) processes.
    This class handles the diffusion math - the actual denoising network
    is passed in separately.

    Forward process: q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1-alpha_bar_t) * I)
    Reverse process: p(x_{t-1} | x_t) = N(x_{t-1}; mu_theta(x_t, t), sigma_t^2 * I)
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_schedule: str = 'cosine',
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        loss_type: str = 'mse',
        clip_denoised: bool = True,
        prediction_type: str = 'epsilon',  # 'epsilon' or 'x0' or 'v'
    ):
        """
        Initialize diffusion process.

        Args:
            num_timesteps: Number of diffusion steps (T)
            beta_schedule: Type of noise schedule ('linear', 'cosine', 'sigmoid')
            beta_start: Starting beta for linear/sigmoid schedule
            beta_end: Ending beta for linear/sigmoid schedule
            loss_type: Loss function ('mse', 'l1', 'huber')
            clip_denoised: Whether to clip x_0 predictions to [-1, 1]
            prediction_type: What the model predicts:
                - 'epsilon': predicts the noise (standard DDPM)
                - 'x0': predicts the clean image directly
                - 'v': predicts velocity (progressive distillation)
        """
        super().__init__()

        self.num_timesteps = num_timesteps
        self.loss_type = loss_type
        self.clip_denoised = clip_denoised
        self.prediction_type = prediction_type

        # Create noise schedule
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(num_timesteps, beta_start, beta_end)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(num_timesteps)
        elif beta_schedule == 'sigmoid':
            betas = sigmoid_beta_schedule(num_timesteps, beta_start, beta_end)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        # Precompute diffusion parameters
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Register buffers (not parameters, but should move with model)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # Calculations for diffusion q(x_t | x_0)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1))

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        # Clipped log variance for numerical stability
        self.register_buffer(
            'posterior_log_variance_clipped',
            torch.log(torch.cat([posterior_variance[1:2], posterior_variance[1:]]))
        )

        # Posterior mean coefficients
        self.register_buffer(
            'posterior_mean_coef1',
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer(
            'posterior_mean_coef2',
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        )

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: Tuple) -> torch.Tensor:
        """
        Extract values from a 1D tensor for a batch of indices.

        Args:
            a: 1D tensor of values (T,)
            t: Batch of timestep indices (B,)
            x_shape: Shape of x for broadcasting

        Returns:
            Extracted values reshaped for broadcasting (B, 1, 1, 1)
        """
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward diffusion process: sample x_t given x_0.

        q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1-alpha_bar_t) * I)

        Args:
            x_0: Clean data (B, C, H, W)
            t: Timesteps (B,)
            noise: Optional pre-generated noise

        Returns:
            Noisy data x_t (B, C, H, W)
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha_bar = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alpha_bar = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)

        return sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise

    def q_posterior_mean_variance(
        self,
        x_0: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute posterior q(x_{t-1} | x_t, x_0).

        Args:
            x_0: Clean data prediction (B, C, H, W)
            x_t: Noisy data at timestep t (B, C, H, W)
            t: Timesteps (B,)

        Returns:
            Tuple of (posterior_mean, posterior_variance, posterior_log_variance)
        """
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 +
            self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)

        return posterior_mean, posterior_variance, posterior_log_variance

    def predict_x0_from_epsilon(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict x_0 from x_t and predicted noise.

        x_0 = (x_t - sqrt(1-alpha_bar) * epsilon) / sqrt(alpha_bar)

        Args:
            x_t: Noisy data (B, C, H, W)
            t: Timesteps (B,)
            epsilon: Predicted noise (B, C, H, W)

        Returns:
            Predicted clean data x_0 (B, C, H, W)
        """
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * epsilon
        )

    def predict_epsilon_from_x0(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        x_0: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict noise from x_t and x_0.

        epsilon = (x_t - sqrt(alpha_bar) * x_0) / sqrt(1-alpha_bar)

        Args:
            x_t: Noisy data (B, C, H, W)
            t: Timesteps (B,)
            x_0: Clean data (B, C, H, W)

        Returns:
            Noise (B, C, H, W)
        """
        return (
            (x_t - self._extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_0) /
            self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        )

    def p_mean_variance(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[Dict[str, torch.Tensor]] = None,
        clip_denoised: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute mean and variance for reverse process p(x_{t-1} | x_t).

        Args:
            model: Denoising network
            x_t: Noisy data (B, C, H, W)
            t: Timesteps (B,)
            condition: Optional conditioning information
            clip_denoised: Whether to clip x_0 predictions

        Returns:
            Dict with 'mean', 'variance', 'log_variance', 'pred_x0'
        """
        if clip_denoised is None:
            clip_denoised = self.clip_denoised

        # Get model prediction
        if condition is not None:
            model_output = model(x_t, t, **condition)
        else:
            model_output = model(x_t, t)

        # Convert prediction to x_0 estimate
        if self.prediction_type == 'epsilon':
            pred_x0 = self.predict_x0_from_epsilon(x_t, t, model_output)
        elif self.prediction_type == 'x0':
            pred_x0 = model_output
        elif self.prediction_type == 'v':
            # v-prediction: v = sqrt(alpha_bar) * epsilon - sqrt(1-alpha_bar) * x_0
            sqrt_alpha_bar = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
            sqrt_one_minus_alpha_bar = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
            pred_x0 = sqrt_alpha_bar * x_t - sqrt_one_minus_alpha_bar * model_output
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")

        # Optionally clip
        if clip_denoised:
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

        # Get posterior parameters
        mean, variance, log_variance = self.q_posterior_mean_variance(pred_x0, x_t, t)

        return {
            'mean': mean,
            'variance': variance,
            'log_variance': log_variance,
            'pred_x0': pred_x0,
        }

    @torch.no_grad()
    def p_sample(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[Dict[str, torch.Tensor]] = None,
        clip_denoised: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        Sample x_{t-1} from p(x_{t-1} | x_t).

        Single step of the reverse diffusion process.

        Args:
            model: Denoising network
            x_t: Noisy data at timestep t (B, C, H, W)
            t: Timesteps (B,)
            condition: Optional conditioning information
            clip_denoised: Whether to clip x_0 predictions

        Returns:
            Sampled x_{t-1} (B, C, H, W)
        """
        out = self.p_mean_variance(model, x_t, t, condition, clip_denoised)

        # No noise at t=0
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))

        return out['mean'] + nonzero_mask * torch.exp(0.5 * out['log_variance']) * noise

    @torch.no_grad()
    def p_sample_loop(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        condition: Optional[Dict[str, torch.Tensor]] = None,
        return_intermediates: bool = False,
        progress: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, list]]:
        """
        Full reverse diffusion sampling loop.

        Generates samples by iteratively denoising from pure noise.

        Args:
            model: Denoising network
            shape: Shape of samples to generate (B, C, H, W)
            condition: Optional conditioning information
            return_intermediates: Whether to return intermediate samples
            progress: Whether to show progress bar

        Returns:
            Generated samples (B, C, H, W), optionally with intermediates
        """
        device = next(model.parameters()).device
        batch_size = shape[0]

        # Start from pure noise
        x = torch.randn(shape, device=device)

        intermediates = []

        # Reverse diffusion
        timesteps = list(reversed(range(self.num_timesteps)))
        if progress:
            try:
                from tqdm import tqdm
                timesteps = tqdm(timesteps, desc='Sampling')
            except ImportError:
                pass

        for t in timesteps:
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t_batch, condition)

            if return_intermediates:
                intermediates.append(x.clone())

        if return_intermediates:
            return x, intermediates
        return x

    def training_losses(
        self,
        model: nn.Module,
        x_0: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[Dict[str, torch.Tensor]] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training losses for a batch.

        Args:
            model: Denoising network
            x_0: Clean data (B, C, H, W)
            t: Timesteps (B,)
            condition: Optional conditioning information
            noise: Optional pre-generated noise

        Returns:
            Dict with 'loss' and other metrics
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        # Forward diffusion
        x_t = self.q_sample(x_0, t, noise)

        # Get model prediction
        if condition is not None:
            model_output = model(x_t, t, **condition)
        else:
            model_output = model(x_t, t)

        # Compute target based on prediction type
        if self.prediction_type == 'epsilon':
            target = noise
        elif self.prediction_type == 'x0':
            target = x_0
        elif self.prediction_type == 'v':
            # v = sqrt(alpha_bar) * epsilon - sqrt(1-alpha_bar) * x_0
            sqrt_alpha_bar = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
            sqrt_one_minus_alpha_bar = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
            target = sqrt_alpha_bar * noise - sqrt_one_minus_alpha_bar * x_0
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")

        # Compute loss
        if self.loss_type == 'mse':
            loss = F.mse_loss(model_output, target, reduction='none')
        elif self.loss_type == 'l1':
            loss = F.l1_loss(model_output, target, reduction='none')
        elif self.loss_type == 'huber':
            loss = F.smooth_l1_loss(model_output, target, reduction='none')
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Average over all dimensions except batch
        loss = loss.mean(dim=list(range(1, len(loss.shape))))

        return {
            'loss': loss.mean(),
            'loss_per_sample': loss,
        }

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Sample random timesteps for training.

        Args:
            batch_size: Number of timesteps to sample
            device: Device to create tensor on

        Returns:
            Random timesteps (B,)
        """
        return torch.randint(0, self.num_timesteps, (batch_size,), device=device, dtype=torch.long)


class DDIMSampler:
    """
    DDIM (Denoising Diffusion Implicit Models) Sampler.

    Allows faster sampling with fewer steps while maintaining quality.
    Can also be used for deterministic sampling (eta=0).

    Reference: Song et al. "Denoising Diffusion Implicit Models" (2021)
    """

    def __init__(
        self,
        diffusion: GaussianDiffusion,
        ddim_num_steps: int = 50,
        ddim_eta: float = 0.0,
    ):
        """
        Initialize DDIM sampler.

        Args:
            diffusion: GaussianDiffusion instance
            ddim_num_steps: Number of DDIM steps (can be much fewer than T)
            ddim_eta: Stochasticity parameter (0 = deterministic, 1 = DDPM)
        """
        self.diffusion = diffusion
        self.ddim_num_steps = ddim_num_steps
        self.ddim_eta = ddim_eta

        # Compute DDIM timestep sequence
        c = diffusion.num_timesteps // ddim_num_steps
        self.ddim_timesteps = list(range(0, diffusion.num_timesteps, c))

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        condition: Optional[Dict[str, torch.Tensor]] = None,
        progress: bool = True,
    ) -> torch.Tensor:
        """
        Generate samples using DDIM.

        Args:
            model: Denoising network
            shape: Shape of samples to generate (B, C, H, W)
            condition: Optional conditioning information
            progress: Whether to show progress bar

        Returns:
            Generated samples (B, C, H, W)
        """
        device = next(model.parameters()).device
        batch_size = shape[0]

        # Start from pure noise
        x = torch.randn(shape, device=device)

        # DDIM sampling
        timesteps = list(reversed(self.ddim_timesteps))
        timesteps_iter = timesteps  # keep plain list for subscript access
        if progress:
            try:
                from tqdm import tqdm
                timesteps_iter = tqdm(timesteps, desc='DDIM Sampling')
            except ImportError:
                pass

        for i, t in enumerate(timesteps_iter):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

            # Get model prediction
            if condition is not None:
                pred = model(x, t_batch, **condition)
            else:
                pred = model(x, t_batch)

            # Predict x_0
            if self.diffusion.prediction_type == 'epsilon':
                pred_x0 = self.diffusion.predict_x0_from_epsilon(x, t_batch, pred)
            elif self.diffusion.prediction_type == 'x0':
                pred_x0 = pred
            else:
                raise NotImplementedError(f"DDIM not implemented for {self.diffusion.prediction_type}")

            # Clip if needed
            if self.diffusion.clip_denoised:
                pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

            # Get alpha values
            alpha_bar_t = self.diffusion._extract(
                self.diffusion.alphas_cumprod, t_batch, x.shape
            )

            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
                t_prev_batch = torch.full((batch_size,), t_prev, device=device, dtype=torch.long)
                alpha_bar_t_prev = self.diffusion._extract(
                    self.diffusion.alphas_cumprod, t_prev_batch, x.shape
                )
            else:
                alpha_bar_t_prev = torch.ones_like(alpha_bar_t)

            # Compute variance
            sigma_t = self.ddim_eta * torch.sqrt(
                (1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_t_prev)
            )

            # Predict direction
            pred_dir = torch.sqrt(1 - alpha_bar_t_prev - sigma_t ** 2) * (
                (x - torch.sqrt(alpha_bar_t) * pred_x0) / torch.sqrt(1 - alpha_bar_t)
            )

            # Compute x_{t-1}
            x = torch.sqrt(alpha_bar_t_prev) * pred_x0 + pred_dir

            # Add noise (if eta > 0)
            if self.ddim_eta > 0 and i < len(timesteps) - 1:
                noise = torch.randn_like(x)
                x = x + sigma_t * noise

        return x
