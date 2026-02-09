"""
PyTorch Lightning DataModule for RadioMapSeer.

Provides train/val/test dataloaders with proper configuration.
"""

from pathlib import Path
from typing import Optional, Union

import lightning as L
from torch.utils.data import DataLoader

from .dataset import RadioMapDataset, UniformSamplingDataset


class RadioMapDataModule(L.LightningDataModule):
    """
    Lightning DataModule for RadioMapSeer dataset.

    Handles train/val/test splits and dataloader creation.
    """

    def __init__(
        self,
        data_dir: Union[str, Path] = 'data/raw',
        variant: str = 'IRT2',
        # Sampling configuration
        sampling_strategy: str = 'trajectory',  # 'trajectory' or 'uniform'
        num_trajectories: int = 3,
        points_per_trajectory: int = 100,
        trajectory_method: str = 'mixed',
        uniform_sampling_rate: float = 0.01,
        # Noise
        position_noise_std: float = 0.5,
        rss_noise_std: float = 2.0,
        coverage_sigma: float = 5.0,
        # Data processing
        normalize: bool = True,
        # Caching
        cache_radio_maps: bool = False,
        trajectory_cache_sets: int = 0,
        # Dataloader
        batch_size: int = 16,
        num_workers: int = 4,
        pin_memory: bool = True,
        # Split
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        split_seed: int = 42,
        # Reproducibility
        seed: Optional[int] = None,
    ):
        """
        Initialize DataModule.

        Args:
            data_dir: Path to data/raw directory
            variant: Radio map variant ('IRT2', 'IRT4', 'DPM')
            sampling_strategy: 'trajectory' or 'uniform'
            num_trajectories: Trajectories per sample (if trajectory)
            points_per_trajectory: Points per trajectory
            trajectory_method: Trajectory type ('mixed', 'shortest_path', etc.)
            uniform_sampling_rate: Sampling rate if uniform
            position_noise_std: Position noise in pixels
            rss_noise_std: RSS noise in PNG units
            coverage_sigma: Coverage density smoothing
            normalize: Normalize maps to [0, 1]
            batch_size: Batch size for dataloaders
            num_workers: Number of dataloader workers
            pin_memory: Pin memory for GPU transfer
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            split_seed: Seed for train/val/test split
            seed: Seed for trajectory generation
        """
        super().__init__()
        self.save_hyperparameters()

        self.data_dir = Path(data_dir)
        self.variant = variant
        self.sampling_strategy = sampling_strategy
        self.num_trajectories = num_trajectories
        self.points_per_trajectory = points_per_trajectory
        self.trajectory_method = trajectory_method
        self.uniform_sampling_rate = uniform_sampling_rate
        self.position_noise_std = position_noise_std
        self.rss_noise_std = rss_noise_std
        self.coverage_sigma = coverage_sigma
        self.normalize = normalize
        self.cache_radio_maps = cache_radio_maps
        self.trajectory_cache_sets = trajectory_cache_sets
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.split_seed = split_seed
        self.seed = seed

        # Will be set in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        """Setup datasets for each stage."""
        # Get split IDs
        splits = RadioMapDataset.get_split_ids(
            total_maps=701,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            seed=self.split_seed
        )

        # Choose dataset class
        DatasetClass = (
            RadioMapDataset if self.sampling_strategy == 'trajectory'
            else UniformSamplingDataset
        )

        # Common kwargs
        common_kwargs = dict(
            data_dir=self.data_dir,
            variant=self.variant,
            position_noise_std=self.position_noise_std,
            rss_noise_std=self.rss_noise_std,
            coverage_sigma=self.coverage_sigma,
            normalize=self.normalize,
            cache_building_maps=True,
            cache_radio_maps=self.cache_radio_maps,
            trajectory_cache_sets=self.trajectory_cache_sets,
        )

        if self.sampling_strategy == 'trajectory':
            common_kwargs.update(
                num_trajectories=self.num_trajectories,
                points_per_trajectory=self.points_per_trajectory,
                trajectory_method=self.trajectory_method,
            )
        else:
            common_kwargs['sampling_rate'] = self.uniform_sampling_rate

        if stage == 'fit' or stage is None:
            self.train_dataset = DatasetClass(
                map_ids=splits['train'],
                seed=self.seed,
                **common_kwargs
            )
            self.val_dataset = DatasetClass(
                map_ids=splits['val'],
                seed=self.seed + 1000 if self.seed else None,
                **common_kwargs
            )

        if stage == 'test' or stage is None:
            self.test_dataset = DatasetClass(
                map_ids=splits['test'],
                seed=self.seed + 2000 if self.seed else None,
                **common_kwargs
            )

    def _dataloader_kwargs(self) -> dict:
        """Common DataLoader kwargs for persistent workers and prefetching."""
        kwargs = {}
        if self.num_workers > 0:
            kwargs['persistent_workers'] = True
            kwargs['prefetch_factor'] = 3
        return kwargs

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            **self._dataloader_kwargs(),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            **self._dataloader_kwargs(),
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            **self._dataloader_kwargs(),
        )

    @property
    def num_train_samples(self) -> int:
        """Number of training samples."""
        if self.train_dataset is None:
            self.setup('fit')
        return len(self.train_dataset)

    @property
    def num_val_samples(self) -> int:
        """Number of validation samples."""
        if self.val_dataset is None:
            self.setup('fit')
        return len(self.val_dataset)

    @property
    def num_test_samples(self) -> int:
        """Number of test samples."""
        if self.test_dataset is None:
            self.setup('test')
        return len(self.test_dataset)
