"""
Data augmentation transforms for RadioMapSeer.

Physics-consistent augmentations that preserve radio propagation properties:
- Rotations (90°, 180°, 270°)
- Flips (horizontal, vertical)
- Random crops (with coordinate adjustment)

NOT allowed (would break physics):
- Scaling RSS values
- Color jittering
- Elastic deformations
"""

from typing import Dict, Optional, Tuple
import numpy as np
import torch


class RadioMapTransform:
    """
    Physics-consistent augmentation pipeline for radio maps.

    Augmentations are applied consistently to all inputs
    (building map, radio map, sparse samples, masks).
    """

    def __init__(
        self,
        random_rotation: bool = True,
        random_flip: bool = True,
        random_crop: bool = False,
        crop_size: int = 128,
        p_rotation: float = 0.5,
        p_flip: float = 0.5,
        p_crop: float = 0.5,
        seed: Optional[int] = None,
    ):
        """
        Initialize transform pipeline.

        Args:
            random_rotation: Enable 90°/180°/270° rotations
            random_flip: Enable horizontal/vertical flips
            random_crop: Enable random cropping
            crop_size: Size of random crop
            p_rotation: Probability of applying rotation
            p_flip: Probability of applying flip
            p_crop: Probability of applying crop
            seed: Random seed
        """
        self.random_rotation = random_rotation
        self.random_flip = random_flip
        self.random_crop = random_crop
        self.crop_size = crop_size
        self.p_rotation = p_rotation
        self.p_flip = p_flip
        self.p_crop = p_crop
        self.rng = np.random.default_rng(seed)

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply transforms to a sample.

        Args:
            sample: Dict with keys like 'building_map', 'radio_map',
                   'sparse_rss', 'trajectory_mask', 'coverage_density', 'tx_position'

        Returns:
            Transformed sample
        """
        # Get spatial tensors to transform
        spatial_keys = ['building_map', 'radio_map', 'sparse_rss', 'trajectory_mask', 'coverage_density']

        # Track rotation/flip for tx_position adjustment
        rotation_k = 0
        flip_h = False
        flip_v = False
        H, W = 256, 256

        # Random rotation
        if self.random_rotation and self.rng.random() < self.p_rotation:
            rotation_k = self.rng.choice([1, 2, 3])  # 90°, 180°, 270°
            for key in spatial_keys:
                if key in sample:
                    sample[key] = torch.rot90(sample[key], k=rotation_k, dims=[-2, -1])

        # Random horizontal flip
        if self.random_flip and self.rng.random() < self.p_flip:
            flip_h = True
            for key in spatial_keys:
                if key in sample:
                    sample[key] = torch.flip(sample[key], dims=[-1])

        # Random vertical flip
        if self.random_flip and self.rng.random() < self.p_flip:
            flip_v = True
            for key in spatial_keys:
                if key in sample:
                    sample[key] = torch.flip(sample[key], dims=[-2])

        # Adjust tx_position if present
        if 'tx_position' in sample:
            tx = sample['tx_position'].clone()

            # Apply rotation to tx position
            for _ in range(rotation_k):
                x, y = tx[0].item(), tx[1].item()
                tx[0] = H - 1 - y
                tx[1] = x

            # Apply flips
            if flip_h:
                tx[0] = W - 1 - tx[0]
            if flip_v:
                tx[1] = H - 1 - tx[1]

            sample['tx_position'] = tx

        # Random crop (if enabled)
        if self.random_crop and self.rng.random() < self.p_crop:
            sample = self._random_crop(sample, spatial_keys)

        return sample

    def _random_crop(
        self,
        sample: Dict[str, torch.Tensor],
        spatial_keys: list
    ) -> Dict[str, torch.Tensor]:
        """Apply random crop to sample."""
        # Get current size
        for key in spatial_keys:
            if key in sample:
                _, H, W = sample[key].shape
                break
        else:
            return sample

        if H <= self.crop_size or W <= self.crop_size:
            return sample

        # Random crop position
        top = self.rng.integers(0, H - self.crop_size)
        left = self.rng.integers(0, W - self.crop_size)

        # Apply crop
        for key in spatial_keys:
            if key in sample:
                sample[key] = sample[key][
                    :, top:top+self.crop_size, left:left+self.crop_size
                ]

        # Adjust tx_position
        if 'tx_position' in sample:
            tx = sample['tx_position'].clone()
            tx[0] = tx[0] - left
            tx[1] = tx[1] - top
            sample['tx_position'] = tx

        return sample


class Compose:
    """Compose multiple transforms."""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for t in self.transforms:
            sample = t(sample)
        return sample


class ToTensor:
    """Convert numpy arrays to tensors."""

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        result = {}
        for key, value in sample.items():
            if isinstance(value, np.ndarray):
                if value.ndim == 2:
                    value = value[None]  # Add channel dim
                result[key] = torch.from_numpy(value.copy()).float()
            elif isinstance(value, (int, float)):
                result[key] = value
            else:
                result[key] = value
        return result


class Normalize:
    """Normalize radio maps to specified range."""

    def __init__(self, input_range: Tuple[float, float] = (0, 255), output_range: Tuple[float, float] = (0, 1)):
        self.input_range = input_range
        self.output_range = output_range

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        in_min, in_max = self.input_range
        out_min, out_max = self.output_range

        for key in ['radio_map', 'sparse_rss']:
            if key in sample:
                value = sample[key]
                # Normalize
                value = (value - in_min) / (in_max - in_min)
                value = value * (out_max - out_min) + out_min
                sample[key] = value

        return sample


def get_train_transforms(
    random_rotation: bool = True,
    random_flip: bool = True,
    random_crop: bool = False,
    crop_size: int = 128,
) -> RadioMapTransform:
    """Get default training transforms."""
    return RadioMapTransform(
        random_rotation=random_rotation,
        random_flip=random_flip,
        random_crop=random_crop,
        crop_size=crop_size,
        p_rotation=0.5,
        p_flip=0.5,
        p_crop=0.3,
    )


def get_val_transforms() -> None:
    """Get validation transforms (none by default)."""
    return None


def get_test_transforms() -> None:
    """Get test transforms (none by default)."""
    return None
