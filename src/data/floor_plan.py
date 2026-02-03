"""
Floor plan processing utilities for RadioMapSeer dataset.

The RadioMapSeer building maps are binary:
- 0 (black): Building/obstacle
- 255 (white): Street/walkable area

This module provides utilities for:
1. Loading and validating floor plans
2. Extracting walkable masks
3. Computing distance transforms (for corridor detection)
4. Finding valid trajectory start/end points
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, binary_erosion


class FloorPlanProcessor:
    """Process RadioMapSeer building maps for trajectory generation."""

    # RadioMapSeer specific constants
    BUILDING_VALUE = 0      # Black = building/obstacle
    WALKABLE_VALUE = 255    # White = street/walkable
    MAP_SIZE = 256          # 256x256 pixels
    RESOLUTION = 1.0        # 1 meter per pixel

    def __init__(self, erosion_radius: int = 1):
        """
        Initialize floor plan processor.

        Args:
            erosion_radius: Pixels to erode from walkable boundary.
                           Prevents trajectories from touching walls.
        """
        self.erosion_radius = erosion_radius

    def load(self, path: Union[str, Path]) -> np.ndarray:
        """
        Load a building map from file.

        Args:
            path: Path to PNG file

        Returns:
            Binary mask (H, W), uint8, values in {0, 255}
        """
        img = Image.open(path)
        arr = np.array(img)

        # Validate
        if arr.shape != (self.MAP_SIZE, self.MAP_SIZE):
            raise ValueError(f"Expected {self.MAP_SIZE}x{self.MAP_SIZE}, got {arr.shape}")

        if len(arr.shape) != 2:
            raise ValueError(f"Expected grayscale, got shape {arr.shape}")

        return arr

    def get_walkable_mask(
        self,
        building_map: np.ndarray,
        erode: bool = True
    ) -> np.ndarray:
        """
        Extract binary walkable mask from building map.

        Args:
            building_map: Raw building map (0=building, 255=walkable)
            erode: If True, erode mask to keep trajectories away from walls

        Returns:
            Binary mask (H, W), bool, True = walkable
        """
        # Convert to binary
        walkable = building_map == self.WALKABLE_VALUE

        # Optionally erode to avoid wall-hugging trajectories
        if erode and self.erosion_radius > 0:
            struct = np.ones((2 * self.erosion_radius + 1, 2 * self.erosion_radius + 1))
            walkable = binary_erosion(walkable, structure=struct)

        return walkable.astype(bool)

    def get_distance_transform(self, walkable_mask: np.ndarray) -> np.ndarray:
        """
        Compute distance from each walkable pixel to nearest obstacle.

        Higher values = further from walls = more "corridor-like".

        Args:
            walkable_mask: Binary walkable mask

        Returns:
            Distance transform (H, W), float, in pixels (meters)
        """
        return distance_transform_edt(walkable_mask)

    def get_corridor_mask(
        self,
        walkable_mask: np.ndarray,
        min_distance: float = 3.0
    ) -> np.ndarray:
        """
        Identify "corridor" regions (areas far from any wall).

        Useful for corridor-biased trajectory sampling.

        Args:
            walkable_mask: Binary walkable mask
            min_distance: Minimum distance from walls to be considered corridor

        Returns:
            Binary corridor mask
        """
        dist = self.get_distance_transform(walkable_mask)
        return dist >= min_distance

    def get_random_walkable_points(
        self,
        walkable_mask: np.ndarray,
        n_points: int,
        min_separation: float = 10.0,
        rng: Optional[np.random.Generator] = None
    ) -> np.ndarray:
        """
        Sample random walkable points with minimum separation.

        Args:
            walkable_mask: Binary walkable mask
            n_points: Number of points to sample
            min_separation: Minimum distance between points (pixels/meters)
            rng: Random number generator

        Returns:
            Points array (n_points, 2) with [y, x] coordinates
        """
        if rng is None:
            rng = np.random.default_rng()

        # Get all walkable coordinates
        walkable_coords = np.argwhere(walkable_mask)
        if len(walkable_coords) < n_points:
            raise ValueError(f"Not enough walkable pixels ({len(walkable_coords)}) for {n_points} points")

        # Sample with separation constraint
        selected = []
        available_mask = np.ones(len(walkable_coords), dtype=bool)

        for _ in range(n_points):
            # Get available indices
            available_idx = np.where(available_mask)[0]
            if len(available_idx) == 0:
                break

            # Random selection
            idx = rng.choice(available_idx)
            point = walkable_coords[idx]
            selected.append(point)

            # Mark nearby points as unavailable
            if min_separation > 0:
                distances = np.linalg.norm(walkable_coords - point, axis=1)
                available_mask &= distances >= min_separation

        return np.array(selected)

    def validate_path(
        self,
        walkable_mask: np.ndarray,
        path: np.ndarray
    ) -> bool:
        """
        Check if a path only traverses walkable pixels.

        Args:
            walkable_mask: Binary walkable mask
            path: Array of (y, x) coordinates

        Returns:
            True if entire path is walkable
        """
        for y, x in path:
            y_int, x_int = int(round(y)), int(round(x))
            if not (0 <= y_int < walkable_mask.shape[0] and
                    0 <= x_int < walkable_mask.shape[1]):
                return False
            if not walkable_mask[y_int, x_int]:
                return False
        return True

    def get_coverage_stats(self, walkable_mask: np.ndarray) -> dict:
        """
        Compute coverage statistics for a walkable mask.

        Args:
            walkable_mask: Binary walkable mask

        Returns:
            Dictionary with coverage statistics
        """
        total_pixels = walkable_mask.size
        walkable_pixels = np.sum(walkable_mask)
        dist = self.get_distance_transform(walkable_mask)

        return {
            'total_pixels': total_pixels,
            'walkable_pixels': int(walkable_pixels),
            'walkable_ratio': float(walkable_pixels / total_pixels),
            'mean_distance_to_wall': float(dist[walkable_mask].mean()) if walkable_pixels > 0 else 0,
            'max_distance_to_wall': float(dist.max()),
        }


def load_building_map(
    data_dir: Union[str, Path],
    map_id: int
) -> np.ndarray:
    """
    Convenience function to load a building map by ID.

    Args:
        data_dir: Path to data/raw directory
        map_id: Map identifier (0-700)

    Returns:
        Building map array
    """
    data_dir = Path(data_dir)
    path = data_dir / 'png' / 'buildings_complete' / f'{map_id}.png'
    processor = FloorPlanProcessor()
    return processor.load(path)


def load_radio_map(
    data_dir: Union[str, Path],
    map_id: int,
    tx_id: int,
    variant: str = 'IRT2'
) -> np.ndarray:
    """
    Load a radio/pathloss map.

    Args:
        data_dir: Path to data/raw directory
        map_id: Map identifier (0-700)
        tx_id: Transmitter identifier (0-79)
        variant: Simulation variant ('IRT2', 'IRT4', 'DPM', etc.)

    Returns:
        Radio map array (256, 256), uint8
    """
    data_dir = Path(data_dir)
    path = data_dir / 'gain' / variant / f'{map_id}_{tx_id}.png'
    img = Image.open(path)
    return np.array(img)


def load_antenna_positions(
    data_dir: Union[str, Path],
    map_id: int
) -> np.ndarray:
    """
    Load transmitter positions for a map.

    Args:
        data_dir: Path to data/raw directory
        map_id: Map identifier (0-700)

    Returns:
        Antenna positions (80, 2) with [x, y] coordinates
    """
    import json
    data_dir = Path(data_dir)
    path = data_dir / 'antenna' / f'{map_id}.json'
    with open(path) as f:
        positions = json.load(f)
    return np.array(positions)


# Pathloss conversion utilities
PATHLOSS_MIN_DB = -186
PATHLOSS_MAX_DB = -47
PATHLOSS_RANGE_DB = PATHLOSS_MAX_DB - PATHLOSS_MIN_DB


def png_to_db(png_value: np.ndarray) -> np.ndarray:
    """Convert PNG value (0-255) to pathloss in dB."""
    return (png_value / 255.0) * PATHLOSS_RANGE_DB + PATHLOSS_MIN_DB


def db_to_png(db_value: np.ndarray) -> np.ndarray:
    """Convert pathloss in dB to PNG value (0-255)."""
    png = (db_value - PATHLOSS_MIN_DB) / PATHLOSS_RANGE_DB * 255
    return np.clip(png, 0, 255).astype(np.uint8)
