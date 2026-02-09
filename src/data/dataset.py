"""
PyTorch Dataset for RadioMapSeer with trajectory sampling.

This module provides:
1. RadioMapDataset: Core dataset with trajectory generation
2. Support for train/val/test splits by map ID
3. On-the-fly trajectory generation for data diversity
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from .floor_plan import FloorPlanProcessor, png_to_db
from .trajectory_sampler import MixedTrajectoryGenerator, Trajectory


class RadioMapDataset(Dataset):
    """
    PyTorch Dataset for RadioMapSeer with trajectory sampling.

    Each sample contains:
    - building_map: Binary building footprint (1, H, W)
    - radio_map: Ground truth pathloss map (1, H, W)
    - sparse_rss: Trajectory-sampled RSS values (1, H, W)
    - trajectory_mask: Binary mask of sampled locations (1, H, W)
    - coverage_density: Gaussian-smoothed coverage (1, H, W)
    - tx_position: Transmitter location (2,)
    - metadata: Dict with map_id, tx_id, etc.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        map_ids: List[int],
        variant: str = 'IRT2',
        num_trajectories: int = 3,
        points_per_trajectory: int = 100,
        trajectory_method: str = 'mixed',
        position_noise_std: float = 0.5,
        rss_noise_std: float = 2.0,
        coverage_sigma: float = 5.0,
        normalize: bool = True,
        cache_building_maps: bool = True,
        cache_radio_maps: bool = False,
        trajectory_cache_sets: int = 0,
        seed: Optional[int] = None,
    ):
        """
        Initialize RadioMapDataset.

        Args:
            data_dir: Path to data/raw directory
            map_ids: List of map IDs to include
            variant: Radio map variant ('IRT2', 'IRT4', 'DPM')
            num_trajectories: Number of trajectories per sample
            points_per_trajectory: Points per trajectory
            trajectory_method: 'mixed', 'shortest_path', 'random_walk', 'corridor_biased'
            position_noise_std: Position noise in pixels
            rss_noise_std: RSS measurement noise (in PNG units, ~2 for dB equivalent)
            coverage_sigma: Gaussian smoothing sigma for coverage density
            normalize: If True, normalize signal data to [-1, 1] for diffusion
            cache_building_maps: Cache building maps in memory
            cache_radio_maps: Cache radio maps in memory (~3.5GB for full dataset).
                Eliminates PNG decompression overhead on every __getitem__ call.
            trajectory_cache_sets: Number of trajectory sets to pre-generate per sample.
                0 = generate on-the-fly (default). >0 = pre-generate and cycle through
                cached sets. Reduces per-sample overhead from ~30-50ms to <1ms.
            seed: Random seed for trajectory generation
        """
        self.data_dir = Path(data_dir)
        self.map_ids = sorted(map_ids)
        self.variant = variant
        self.num_trajectories = num_trajectories
        self.points_per_trajectory = points_per_trajectory
        self.coverage_sigma = coverage_sigma
        self.normalize = normalize
        self.cache_building_maps = cache_building_maps
        self.cache_radio_maps = cache_radio_maps
        self.trajectory_cache_sets = trajectory_cache_sets

        # Setup paths
        self.building_dir = self.data_dir / 'png' / 'buildings_complete'
        self.radio_dir = self.data_dir / 'gain' / variant
        self.antenna_dir = self.data_dir / 'antenna'

        # Validate paths
        if not self.building_dir.exists():
            raise FileNotFoundError(f"Building directory not found: {self.building_dir}")
        if not self.radio_dir.exists():
            raise FileNotFoundError(f"Radio map directory not found: {self.radio_dir}")

        # Count transmitters per map (should be 80)
        self.num_tx = 80

        # Create sample index: (map_id, tx_id) pairs
        self.samples = []
        for map_id in self.map_ids:
            for tx_id in range(self.num_tx):
                radio_path = self.radio_dir / f'{map_id}_{tx_id}.png'
                if radio_path.exists():
                    self.samples.append((map_id, tx_id))

        # Floor plan processor
        self.processor = FloorPlanProcessor(erosion_radius=1)

        # Trajectory generator
        self.trajectory_generator = MixedTrajectoryGenerator(
            position_noise_std=position_noise_std,
            rss_noise_std=rss_noise_std,
            seed=seed,
        )

        # Cache
        self._building_cache: Dict[int, np.ndarray] = {}
        self._walkable_cache: Dict[int, np.ndarray] = {}
        self._antenna_cache: Dict[int, np.ndarray] = {}
        self._radio_cache: Dict[Tuple[int, int], np.ndarray] = {}
        # Trajectory cache: maps sample_idx -> list of (sparse_rss, mask) tuples
        self._trajectory_cache: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {}
        self._trajectory_access_count: Dict[int, int] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        map_id, tx_id = self.samples[idx]

        # Load building map (with caching)
        building_map = self._load_building_map(map_id)
        walkable_mask = self._load_walkable_mask(map_id)

        # Load radio map
        radio_map = self._load_radio_map(map_id, tx_id)

        # Load antenna position
        antenna_positions = self._load_antenna_positions(map_id)
        tx_position = antenna_positions[tx_id]

        # Generate trajectories (with optional caching)
        sparse_rss, trajectory_mask = self._get_trajectory_data(
            idx, walkable_mask, radio_map
        )

        # Compute coverage density
        coverage_density = self._compute_coverage_density(trajectory_mask)

        # Normalize if requested
        # Signal data -> [-1, 1] for diffusion model compatibility
        # Masks (trajectory_mask, coverage_density) stay in [0, 1]
        if self.normalize:
            radio_map = radio_map.astype(np.float32) / 255.0 * 2.0 - 1.0
            sparse_rss = sparse_rss / 255.0 * 2.0 - 1.0
            # Zero out sparse_rss where mask is 0 (unobserved = neutral, not -1)
            sparse_rss = sparse_rss * trajectory_mask
            building_map = building_map.astype(np.float32) / 255.0 * 2.0 - 1.0

        # Normalize TX position to [0, 1]
        map_size = radio_map.shape[-1] if radio_map.ndim > 1 else 256
        tx_position_norm = tx_position.astype(np.float32) / float(map_size)

        # Convert to tensors
        return {
            'building_map': torch.from_numpy(building_map[None]).float(),
            'radio_map': torch.from_numpy(radio_map[None]).float(),
            'sparse_rss': torch.from_numpy(sparse_rss[None]).float(),
            'trajectory_mask': torch.from_numpy(trajectory_mask[None]).float(),
            'coverage_density': torch.from_numpy(coverage_density[None]).float(),
            'tx_position': torch.from_numpy(tx_position_norm).float(),
            'map_id': map_id,
            'tx_id': tx_id,
        }

    def _load_building_map(self, map_id: int) -> np.ndarray:
        """Load building map with optional caching."""
        if self.cache_building_maps and map_id in self._building_cache:
            return self._building_cache[map_id]

        path = self.building_dir / f'{map_id}.png'
        img = np.array(Image.open(path))

        if self.cache_building_maps:
            self._building_cache[map_id] = img

        return img

    def _load_walkable_mask(self, map_id: int) -> np.ndarray:
        """Load walkable mask with caching."""
        if map_id in self._walkable_cache:
            return self._walkable_cache[map_id]

        building_map = self._load_building_map(map_id)
        walkable = self.processor.get_walkable_mask(building_map)

        if self.cache_building_maps:
            self._walkable_cache[map_id] = walkable

        return walkable

    def _load_radio_map(self, map_id: int, tx_id: int) -> np.ndarray:
        """Load radio/pathloss map with optional caching."""
        key = (map_id, tx_id)
        if self.cache_radio_maps and key in self._radio_cache:
            return self._radio_cache[key]

        path = self.radio_dir / f'{map_id}_{tx_id}.png'
        img = np.array(Image.open(path))

        if self.cache_radio_maps:
            self._radio_cache[key] = img

        return img

    def _load_antenna_positions(self, map_id: int) -> np.ndarray:
        """Load antenna positions with caching."""
        if map_id in self._antenna_cache:
            return self._antenna_cache[map_id]

        path = self.antenna_dir / f'{map_id}.json'
        with open(path) as f:
            positions = json.load(f)

        positions = np.array(positions)

        if self.cache_building_maps:
            self._antenna_cache[map_id] = positions

        return positions

    def _get_trajectory_data(
        self,
        idx: int,
        walkable_mask: np.ndarray,
        radio_map: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get trajectory sparse map and mask, using cache if enabled."""
        if self.trajectory_cache_sets > 0:
            # Check if we have cached trajectories for this sample
            if idx not in self._trajectory_cache:
                # Pre-generate multiple sets
                self._trajectory_cache[idx] = []
                for _ in range(self.trajectory_cache_sets):
                    trajectories = self.trajectory_generator.generate_multiple(
                        walkable_mask,
                        radio_map,
                        n_trajectories=self.num_trajectories,
                        points_per_trajectory=self.points_per_trajectory,
                    )
                    sparse_rss, mask = self._combine_trajectories(trajectories)
                    self._trajectory_cache[idx].append((sparse_rss.copy(), mask.copy()))
                self._trajectory_access_count[idx] = 0

            # Cycle through cached sets
            count = self._trajectory_access_count[idx]
            result = self._trajectory_cache[idx][count % self.trajectory_cache_sets]
            self._trajectory_access_count[idx] = count + 1
            return result

        # No caching: generate on-the-fly
        trajectories = self.trajectory_generator.generate_multiple(
            walkable_mask,
            radio_map,
            n_trajectories=self.num_trajectories,
            points_per_trajectory=self.points_per_trajectory,
        )
        return self._combine_trajectories(trajectories)

    def _combine_trajectories(
        self,
        trajectories: List[Trajectory]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Combine multiple trajectories into sparse map."""
        sparse_rss = np.zeros((256, 256), dtype=np.float32)
        mask = np.zeros((256, 256), dtype=np.float32)

        for traj in trajectories:
            traj_sparse, traj_mask = traj.to_sparse_map((256, 256))
            # Average overlapping points
            overlap = (mask > 0) & (traj_mask > 0)
            sparse_rss[~overlap] += traj_sparse[~overlap]
            sparse_rss[overlap] = (sparse_rss[overlap] + traj_sparse[overlap]) / 2
            mask = np.maximum(mask, traj_mask)

        return sparse_rss, mask

    def _compute_coverage_density(self, mask: np.ndarray) -> np.ndarray:
        """Compute Gaussian-smoothed coverage density."""
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(mask.astype(np.float32), sigma=self.coverage_sigma)

    @staticmethod
    def get_split_ids(
        total_maps: int = 701,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        seed: int = 42
    ) -> Dict[str, List[int]]:
        """
        Get train/val/test split by map ID.

        Args:
            total_maps: Total number of maps
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            seed: Random seed for reproducibility

        Returns:
            Dict with 'train', 'val', 'test' keys containing map IDs
        """
        rng = np.random.default_rng(seed)
        all_ids = np.arange(total_maps)
        rng.shuffle(all_ids)

        n_train = int(total_maps * train_ratio)
        n_val = int(total_maps * val_ratio)

        return {
            'train': sorted(all_ids[:n_train].tolist()),
            'val': sorted(all_ids[n_train:n_train + n_val].tolist()),
            'test': sorted(all_ids[n_train + n_val:].tolist()),
        }


class UniformSamplingDataset(RadioMapDataset):
    """
    Dataset with uniform random sampling instead of trajectories.

    Used as baseline comparison to show trajectory-awareness matters.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        map_ids: List[int],
        sampling_rate: float = 0.01,
        **kwargs
    ):
        """
        Args:
            sampling_rate: Fraction of pixels to sample (default 1%)
        """
        # Disable trajectory generation in parent
        kwargs['num_trajectories'] = 0
        super().__init__(data_dir, map_ids, **kwargs)
        self.sampling_rate = sampling_rate

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        map_id, tx_id = self.samples[idx]

        # Load data
        building_map = self._load_building_map(map_id)
        walkable_mask = self._load_walkable_mask(map_id)
        radio_map = self._load_radio_map(map_id, tx_id)
        antenna_positions = self._load_antenna_positions(map_id)
        tx_position = antenna_positions[tx_id]

        # Uniform random sampling
        sparse_rss, trajectory_mask = self._uniform_sample(radio_map, walkable_mask)
        coverage_density = self._compute_coverage_density(trajectory_mask)

        # Normalize
        # Signal data -> [-1, 1] for diffusion model compatibility
        # Masks (trajectory_mask, coverage_density) stay in [0, 1]
        if self.normalize:
            radio_map = radio_map.astype(np.float32) / 255.0 * 2.0 - 1.0
            sparse_rss = sparse_rss / 255.0 * 2.0 - 1.0
            # Zero out sparse_rss where mask is 0 (unobserved = neutral, not -1)
            sparse_rss = sparse_rss * trajectory_mask
            building_map = building_map.astype(np.float32) / 255.0 * 2.0 - 1.0

        # Normalize TX position to [0, 1]
        map_size = radio_map.shape[-1] if radio_map.ndim > 1 else 256
        tx_position_norm = tx_position.astype(np.float32) / float(map_size)

        return {
            'building_map': torch.from_numpy(building_map[None]).float(),
            'radio_map': torch.from_numpy(radio_map[None]).float(),
            'sparse_rss': torch.from_numpy(sparse_rss[None]).float(),
            'trajectory_mask': torch.from_numpy(trajectory_mask[None]).float(),
            'coverage_density': torch.from_numpy(coverage_density[None]).float(),
            'tx_position': torch.from_numpy(tx_position_norm).float(),
            'map_id': map_id,
            'tx_id': tx_id,
        }

    def _uniform_sample(
        self,
        radio_map: np.ndarray,
        walkable_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample uniformly at random from walkable area."""
        walkable_coords = np.argwhere(walkable_mask)
        n_samples = int(len(walkable_coords) * self.sampling_rate)
        n_samples = max(n_samples, 10)  # Minimum 10 samples

        rng = np.random.default_rng()
        idx = rng.choice(len(walkable_coords), size=min(n_samples, len(walkable_coords)), replace=False)
        selected = walkable_coords[idx]

        sparse_rss = np.zeros_like(radio_map, dtype=np.float32)
        mask = np.zeros_like(radio_map, dtype=np.float32)

        for y, x in selected:
            sparse_rss[y, x] = radio_map[y, x]
            mask[y, x] = 1.0

        return sparse_rss, mask
