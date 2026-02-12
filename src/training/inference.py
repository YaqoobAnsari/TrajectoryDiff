"""
Inference utilities for TrajectoryDiff.

Provides convenient functions for sampling from trained diffusion models,
including batch processing, GPU handling, and output post-processing.
"""

from typing import Dict, Optional, Tuple, Union
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class DiffusionInference:
    """
    Inference wrapper for trained diffusion models.

    Handles model loading, device management, and batch inference
    with proper memory handling.

    Example:
        >>> inference = DiffusionInference.from_checkpoint('model.ckpt')
        >>> samples = inference.sample(condition_batch, use_ddim=True)
    """

    def __init__(
        self,
        module: 'DiffusionModule',
        device: Optional[torch.device] = None,
        use_ema: bool = True,
    ):
        """
        Initialize inference wrapper.

        Args:
            module: Trained DiffusionModule
            device: Device to use for inference (auto-detect if None)
            use_ema: Whether to use EMA model weights for inference
        """
        self.module = module
        self.use_ema = use_ema and module.use_ema and hasattr(module, 'ema_model')

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Move model to device
        self.module = self.module.to(self.device)
        self.module.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        device: Optional[torch.device] = None,
        use_ema: bool = True,
        **module_kwargs,
    ) -> 'DiffusionInference':
        """
        Load inference wrapper from checkpoint.

        Args:
            checkpoint_path: Path to Lightning checkpoint
            device: Device for inference
            use_ema: Whether to use EMA weights
            **module_kwargs: Additional arguments for module loading

        Returns:
            Configured DiffusionInference instance
        """
        from .diffusion_module import DiffusionModule

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load model from checkpoint
        module = DiffusionModule.load_from_checkpoint(
            checkpoint_path,
            map_location='cpu',  # Load to CPU first, then move to device
            **module_kwargs,
        )

        return cls(module, device=device, use_ema=use_ema)

    @torch.no_grad()
    def sample(
        self,
        condition: Dict[str, torch.Tensor],
        use_ddim: bool = True,
        progress: bool = True,
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate samples from the diffusion model.

        Args:
            condition: Dictionary of conditioning tensors
            use_ddim: Whether to use DDIM sampling (faster)
            progress: Whether to show progress bar
            batch_size: If set, process in smaller batches to save memory

        Returns:
            Generated radio map samples [B, 1, H, W]
        """
        # Move condition to device
        condition = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in condition.items()
        }

        # Get batch size from condition
        for v in condition.values():
            if isinstance(v, torch.Tensor):
                total_batch = v.shape[0]
                break

        # Process in batches if needed
        if batch_size is not None and batch_size < total_batch:
            return self._sample_batched(
                condition, use_ddim, progress, batch_size, total_batch
            )

        # Single batch sampling
        return self.module.sample(
            condition=condition,
            use_ddim=use_ddim,
            progress=progress,
        )

    def _sample_batched(
        self,
        condition: Dict[str, torch.Tensor],
        use_ddim: bool,
        progress: bool,
        batch_size: int,
        total_batch: int,
    ) -> torch.Tensor:
        """Process sampling in smaller batches."""
        samples = []

        num_batches = (total_batch + batch_size - 1) // batch_size
        iterator = range(num_batches)
        if progress:
            iterator = tqdm(iterator, desc="Sampling batches")

        for i in iterator:
            start = i * batch_size
            end = min(start + batch_size, total_batch)

            # Slice condition
            batch_condition = {
                k: v[start:end] if isinstance(v, torch.Tensor) else v
                for k, v in condition.items()
            }

            # Sample
            batch_samples = self.module.sample(
                condition=batch_condition,
                use_ddim=use_ddim,
                progress=False,  # Disable inner progress bar
            )

            samples.append(batch_samples.cpu())

        return torch.cat(samples, dim=0)

    @torch.no_grad()
    def sample_from_dataloader(
        self,
        dataloader: DataLoader,
        use_ddim: bool = True,
        ddim_steps: int = 50,
        eta: float = 0.0,
        max_samples: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Generate samples for a full dataloader.

        Args:
            dataloader: DataLoader providing conditioning inputs
            use_ddim: Whether to use DDIM sampling
            ddim_steps: Number of DDIM steps
            eta: DDIM eta parameter
            max_samples: Maximum number of samples to generate

        Returns:
            Tuple of (generated_samples, ground_truth, conditions)
        """
        all_samples = []
        all_targets = []
        all_conditions = {
            'building_map': [],
            'sparse_rss': [],
            'trajectory_mask': [],
            'coverage_density': [],
            'tx_position': [],
        }

        total = 0
        for batch in tqdm(dataloader, desc="Generating samples"):
            # Extract condition
            condition = {
                'building_map': batch['building_map'],
                'sparse_rss': batch['sparse_rss'],
                'trajectory_mask': batch.get('trajectory_mask'),
                'coverage_density': batch.get('coverage_density'),
                'tx_position': batch.get('tx_position'),
            }

            # Remove None values
            condition = {k: v for k, v in condition.items() if v is not None}

            # Generate samples
            # Note: ddim_steps and eta are configured at DDIMSampler init time,
            # not per-call. They are accepted as method params for documentation
            # but not passed through to self.sample().
            samples = self.sample(
                condition=condition,
                use_ddim=use_ddim,
                progress=False,
            )

            # Store results
            all_samples.append(samples.cpu())
            all_targets.append(batch['radio_map'].cpu())

            for key in all_conditions:
                if key in batch:
                    all_conditions[key].append(batch[key].cpu())

            total += samples.shape[0]
            if max_samples is not None and total >= max_samples:
                break

        # Concatenate results
        samples = torch.cat(all_samples, dim=0)
        targets = torch.cat(all_targets, dim=0)
        conditions = {
            k: torch.cat(v, dim=0) if v else None
            for k, v in all_conditions.items()
        }

        # Trim to max_samples
        if max_samples is not None:
            samples = samples[:max_samples]
            targets = targets[:max_samples]
            conditions = {
                k: v[:max_samples] if v is not None else None
                for k, v in conditions.items()
            }

        return samples, targets, conditions


def denormalize_radio_map(
    x: torch.Tensor,
    min_val: float = -186.0,
    max_val: float = -47.0,
) -> torch.Tensor:
    """
    Denormalize radio map from [-1, 1] back to dBm scale.

    The RadioMapSeer dataset encodes pathloss as PNG [0, 255]:
        dBm = (png / 255) * 139 + (-186)
    giving a range of [-186, -47] dBm.

    The dataset normalizes PNG to [-1, 1]:
        norm = png / 255 * 2 - 1

    So the combined inverse is:
        dBm = (norm + 1) / 2 * (max_val - min_val) + min_val

    Args:
        x: Normalized radio map in [-1, 1]
        min_val: Minimum dBm value (default: -186, RadioMapSeer)
        max_val: Maximum dBm value (default: -47, RadioMapSeer)

    Returns:
        Radio map in dBm scale
    """
    # From [-1, 1] to [0, 1]
    x = (x + 1) / 2
    # From [0, 1] to [min_val, max_val]
    x = x * (max_val - min_val) + min_val
    return x


def compute_uncertainty(
    inference: DiffusionInference,
    condition: Dict[str, torch.Tensor],
    num_samples: int = 10,
    use_ddim: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute uncertainty estimates via multiple samples.

    Generates multiple samples from the same condition and computes
    mean and standard deviation as uncertainty estimate.

    Args:
        inference: DiffusionInference instance
        condition: Conditioning tensors
        num_samples: Number of samples to generate
        use_ddim: Whether to use DDIM sampling

    Returns:
        Tuple of (mean_prediction, std_prediction)
    """
    samples = []

    for i in tqdm(range(num_samples), desc="Computing uncertainty"):
        sample = inference.sample(
            condition=condition,
            use_ddim=use_ddim,
            progress=False,
        )
        samples.append(sample.cpu())

    # Stack samples: [num_samples, B, 1, H, W]
    samples = torch.stack(samples, dim=0)

    # Compute statistics
    mean = samples.mean(dim=0)
    std = samples.std(dim=0)

    return mean, std


def sample_interpolation(
    inference: DiffusionInference,
    condition1: Dict[str, torch.Tensor],
    condition2: Dict[str, torch.Tensor],
    num_steps: int = 10,
    use_ddim: bool = True,
) -> torch.Tensor:
    """
    Generate samples along interpolation between two conditions.

    Useful for visualizing how the model transitions between
    different conditioning inputs.

    Args:
        inference: DiffusionInference instance
        condition1: Starting condition
        condition2: Ending condition
        num_steps: Number of interpolation steps
        use_ddim: Whether to use DDIM sampling

    Returns:
        Interpolated samples [num_steps, 1, H, W]
    """
    samples = []

    for i in range(num_steps):
        alpha = i / (num_steps - 1)

        # Interpolate conditions
        condition = {}
        for key in condition1:
            v1 = condition1[key]
            v2 = condition2[key]

            if isinstance(v1, torch.Tensor) and isinstance(v2, torch.Tensor):
                condition[key] = (1 - alpha) * v1 + alpha * v2
            else:
                # Non-interpolatable, use first
                condition[key] = v1

        sample = inference.sample(
            condition=condition,
            use_ddim=use_ddim,
            progress=False,
        )

        samples.append(sample.cpu())

    return torch.cat(samples, dim=0)


__all__ = [
    'DiffusionInference',
    'denormalize_radio_map',
    'compute_uncertainty',
    'sample_interpolation',
]
