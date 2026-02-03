"""
Trajectory sampling algorithms for RadioMapSeer dataset.

This module implements three trajectory generation strategies:
1. ShortestPathTrajectory: A* pathfinding between random points
2. RandomWalkTrajectory: Momentum-based random walk
3. CorridorBiasedTrajectory: Prefers wide open areas (streets)

All trajectories:
- Respect walkable boundaries (never cross buildings)
- Support position noise injection
- Support RSS measurement noise
- Return structured trajectory data
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import numpy as np
from scipy.ndimage import distance_transform_edt
import heapq


@dataclass
class TrajectoryPoint:
    """Single point on a trajectory."""
    t: float      # Timestamp (seconds)
    x: float      # X coordinate (pixels/meters)
    y: float      # Y coordinate (pixels/meters)
    rss: float    # RSS/pathloss value (dB or raw)

    def to_tuple(self) -> Tuple[float, float, float, float]:
        return (self.t, self.x, self.y, self.rss)


@dataclass
class Trajectory:
    """Complete trajectory with metadata."""
    points: List[TrajectoryPoint] = field(default_factory=list)
    trajectory_type: str = "unknown"
    map_id: Optional[int] = None
    tx_id: Optional[int] = None

    def __len__(self) -> int:
        return len(self.points)

    def to_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Convert to separate arrays (t, x, y, rss)."""
        if not self.points:
            return np.array([]), np.array([]), np.array([]), np.array([])
        t = np.array([p.t for p in self.points])
        x = np.array([p.x for p in self.points])
        y = np.array([p.y for p in self.points])
        rss = np.array([p.rss for p in self.points])
        return t, x, y, rss

    def to_sparse_map(
        self,
        shape: Tuple[int, int] = (256, 256)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert trajectory to sparse RSS map and mask.

        Returns:
            sparse_rss: (H, W) RSS values at sampled locations
            mask: (H, W) binary mask of sampled locations
        """
        sparse_rss = np.zeros(shape, dtype=np.float32)
        mask = np.zeros(shape, dtype=np.float32)

        for p in self.points:
            x_int, y_int = int(round(p.x)), int(round(p.y))
            if 0 <= x_int < shape[1] and 0 <= y_int < shape[0]:
                sparse_rss[y_int, x_int] = p.rss
                mask[y_int, x_int] = 1.0

        return sparse_rss, mask


class TrajectoryGenerator(ABC):
    """Base class for trajectory generation strategies."""

    def __init__(
        self,
        position_noise_std: float = 0.5,
        rss_noise_std: float = 2.0,
        sampling_interval: float = 1.0,
        seed: Optional[int] = None
    ):
        """
        Initialize trajectory generator.

        Args:
            position_noise_std: Gaussian noise std for positions (meters/pixels)
            rss_noise_std: Gaussian noise std for RSS measurements (dB)
            sampling_interval: Distance between samples along path (meters/pixels)
            seed: Random seed for reproducibility
        """
        self.position_noise_std = position_noise_std
        self.rss_noise_std = rss_noise_std
        self.sampling_interval = sampling_interval
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def generate(
        self,
        walkable_mask: np.ndarray,
        radio_map: np.ndarray,
        n_points: Optional[int] = None,
        **kwargs
    ) -> Trajectory:
        """
        Generate a single trajectory.

        Args:
            walkable_mask: Binary mask of walkable areas (H, W)
            radio_map: Ground truth radio/pathloss map (H, W)
            n_points: Target number of points (approximate)

        Returns:
            Generated trajectory
        """
        pass

    def _sample_rss(
        self,
        radio_map: np.ndarray,
        x: float,
        y: float,
        add_noise: bool = True
    ) -> float:
        """
        Sample RSS value from radio map with optional noise.

        Args:
            radio_map: Radio map array
            x: X coordinate
            y: Y coordinate
            add_noise: Whether to add measurement noise

        Returns:
            RSS value
        """
        # Bilinear interpolation
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        x1, y1 = min(x0 + 1, radio_map.shape[1] - 1), min(y0 + 1, radio_map.shape[0] - 1)

        # Clamp to valid range
        x0 = max(0, min(x0, radio_map.shape[1] - 1))
        y0 = max(0, min(y0, radio_map.shape[0] - 1))

        # Interpolation weights
        wx = x - x0
        wy = y - y0

        # Bilinear interpolation
        rss = (radio_map[y0, x0] * (1 - wx) * (1 - wy) +
               radio_map[y0, x1] * wx * (1 - wy) +
               radio_map[y1, x0] * (1 - wx) * wy +
               radio_map[y1, x1] * wx * wy)

        if add_noise and self.rss_noise_std > 0:
            rss += self.rng.normal(0, self.rss_noise_std)

        # Clamp to valid PNG range [0, 255] to prevent out-of-range values
        # after normalization to [-1, 1]
        rss = float(np.clip(rss, 0, 255))

        return rss

    def _add_position_noise(self, x: float, y: float) -> Tuple[float, float]:
        """Add Gaussian noise to position."""
        if self.position_noise_std > 0:
            x += self.rng.normal(0, self.position_noise_std)
            y += self.rng.normal(0, self.position_noise_std)
        return x, y


class ShortestPathTrajectory(TrajectoryGenerator):
    """
    Generate trajectories using A* pathfinding between random points.

    This simulates purposeful navigation (e.g., going from A to B).
    """

    def __init__(
        self,
        min_path_length: float = 30.0,
        max_attempts: int = 50,
        **kwargs
    ):
        """
        Args:
            min_path_length: Minimum path length in pixels/meters
            max_attempts: Maximum attempts to find valid path
        """
        super().__init__(**kwargs)
        self.min_path_length = min_path_length
        self.max_attempts = max_attempts

    def generate(
        self,
        walkable_mask: np.ndarray,
        radio_map: np.ndarray,
        n_points: Optional[int] = None,
        **kwargs
    ) -> Trajectory:
        """Generate trajectory using A* pathfinding."""

        # Get walkable coordinates
        walkable_coords = np.argwhere(walkable_mask)
        if len(walkable_coords) < 2:
            raise ValueError("Not enough walkable area for trajectory")

        # Try to find a valid path
        for _ in range(self.max_attempts):
            # Random start and end
            idx = self.rng.choice(len(walkable_coords), size=2, replace=False)
            start = tuple(walkable_coords[idx[0]])  # (y, x)
            end = tuple(walkable_coords[idx[1]])

            # Check minimum distance
            dist = np.sqrt((start[0] - end[0])**2 + (start[1] - end[1])**2)
            if dist < self.min_path_length:
                continue

            # A* pathfinding
            path = self._astar(walkable_mask, start, end)
            if path is not None and len(path) >= self.min_path_length:
                break
        else:
            # Fallback: just sample random walkable points
            return self._fallback_random_points(walkable_mask, radio_map, n_points or 50)

        # Sample points along path
        trajectory = self._sample_along_path(path, radio_map, n_points)
        trajectory.trajectory_type = "shortest_path"
        return trajectory

    def _astar(
        self,
        walkable_mask: np.ndarray,
        start: Tuple[int, int],
        end: Tuple[int, int]
    ) -> Optional[List[Tuple[int, int]]]:
        """
        A* pathfinding algorithm.

        Args:
            walkable_mask: Binary walkable mask
            start: Start position (y, x)
            end: End position (y, x)

        Returns:
            List of (y, x) positions or None if no path found
        """
        H, W = walkable_mask.shape

        def heuristic(a, b):
            return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

        def neighbors(pos):
            y, x = pos
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and walkable_mask[ny, nx]:
                    yield (ny, nx)

        # Priority queue: (f_score, counter, position)
        counter = 0
        open_set = [(heuristic(start, end), counter, start)]
        came_from = {}
        g_score = {start: 0}

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current == end:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]

            for neighbor in neighbors(current):
                # Cost: 1 for cardinal, sqrt(2) for diagonal
                dy = abs(neighbor[0] - current[0])
                dx = abs(neighbor[1] - current[1])
                move_cost = np.sqrt(dy**2 + dx**2)

                tentative_g = g_score[current] + move_cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, end)
                    counter += 1
                    heapq.heappush(open_set, (f_score, counter, neighbor))

        return None  # No path found

    def _sample_along_path(
        self,
        path: List[Tuple[int, int]],
        radio_map: np.ndarray,
        n_points: Optional[int] = None
    ) -> Trajectory:
        """Sample points along a path at regular intervals."""
        trajectory = Trajectory()

        # Calculate cumulative distances
        path_arr = np.array(path, dtype=float)  # (N, 2) as (y, x)
        diffs = np.diff(path_arr, axis=0)
        distances = np.sqrt(np.sum(diffs**2, axis=1))
        cumulative = np.concatenate([[0], np.cumsum(distances)])
        total_length = cumulative[-1]

        # Determine sampling interval
        if n_points is not None:
            interval = total_length / max(n_points - 1, 1)
        else:
            interval = self.sampling_interval

        # Sample at regular intervals
        t = 0.0
        sample_distances = np.arange(0, total_length + interval/2, interval)

        for dist in sample_distances:
            # Find position on path
            idx = np.searchsorted(cumulative, dist)
            if idx == 0:
                y, x = path_arr[0]
            elif idx >= len(path_arr):
                y, x = path_arr[-1]
            else:
                # Interpolate
                frac = (dist - cumulative[idx-1]) / (cumulative[idx] - cumulative[idx-1] + 1e-8)
                y = path_arr[idx-1, 0] + frac * (path_arr[idx, 0] - path_arr[idx-1, 0])
                x = path_arr[idx-1, 1] + frac * (path_arr[idx, 1] - path_arr[idx-1, 1])

            # Add noise
            x_noisy, y_noisy = self._add_position_noise(x, y)

            # Sample RSS
            rss = self._sample_rss(radio_map, x_noisy, y_noisy)

            trajectory.points.append(TrajectoryPoint(t=t, x=x_noisy, y=y_noisy, rss=rss))
            t += interval  # Assuming 1 m/s walking speed

        return trajectory

    def _fallback_random_points(
        self,
        walkable_mask: np.ndarray,
        radio_map: np.ndarray,
        n_points: int
    ) -> Trajectory:
        """Fallback: sample random walkable points."""
        walkable_coords = np.argwhere(walkable_mask)
        idx = self.rng.choice(len(walkable_coords), size=min(n_points, len(walkable_coords)), replace=False)

        trajectory = Trajectory(trajectory_type="random_points")
        for i, coord in enumerate(walkable_coords[idx]):
            y, x = coord
            x_noisy, y_noisy = self._add_position_noise(float(x), float(y))
            rss = self._sample_rss(radio_map, x_noisy, y_noisy)
            trajectory.points.append(TrajectoryPoint(t=float(i), x=x_noisy, y=y_noisy, rss=rss))

        return trajectory


class RandomWalkTrajectory(TrajectoryGenerator):
    """
    Generate trajectories using momentum-based random walk.

    This simulates wandering behavior with directional persistence.
    """

    def __init__(
        self,
        step_size: float = 2.0,
        momentum: float = 0.8,
        **kwargs
    ):
        """
        Args:
            step_size: Distance per step in pixels/meters
            momentum: Directional persistence (0 = pure random, 1 = straight line)
        """
        super().__init__(**kwargs)
        self.step_size = step_size
        self.momentum = momentum

    def generate(
        self,
        walkable_mask: np.ndarray,
        radio_map: np.ndarray,
        n_points: Optional[int] = None,
        **kwargs
    ) -> Trajectory:
        """Generate trajectory using random walk."""
        if n_points is None:
            n_points = 100

        # Random start
        walkable_coords = np.argwhere(walkable_mask)
        start_idx = self.rng.choice(len(walkable_coords))
        y, x = walkable_coords[start_idx].astype(float)

        # Random initial direction
        angle = self.rng.uniform(0, 2 * np.pi)

        trajectory = Trajectory(trajectory_type="random_walk")
        t = 0.0

        for _ in range(n_points):
            # Add current point
            x_noisy, y_noisy = self._add_position_noise(x, y)
            rss = self._sample_rss(radio_map, x_noisy, y_noisy)
            trajectory.points.append(TrajectoryPoint(t=t, x=x_noisy, y=y_noisy, rss=rss))
            t += self.step_size

            # Try to take a step
            for attempt in range(20):
                # Perturb angle
                angle_delta = self.rng.normal(0, (1 - self.momentum) * np.pi/2)
                new_angle = angle + angle_delta

                # Compute new position
                dx = self.step_size * np.cos(new_angle)
                dy = self.step_size * np.sin(new_angle)
                new_x, new_y = x + dx, y + dy

                # Check if valid
                new_x_int, new_y_int = int(round(new_x)), int(round(new_y))
                if (0 <= new_x_int < walkable_mask.shape[1] and
                    0 <= new_y_int < walkable_mask.shape[0] and
                    walkable_mask[new_y_int, new_x_int]):
                    x, y = new_x, new_y
                    angle = new_angle
                    break
            else:
                # Stuck: random restart nearby
                nearby = walkable_coords[
                    np.linalg.norm(walkable_coords - np.array([y, x]), axis=1) < 20
                ]
                if len(nearby) > 0:
                    idx = self.rng.choice(len(nearby))
                    y, x = nearby[idx].astype(float)
                    angle = self.rng.uniform(0, 2 * np.pi)

        return trajectory


class CorridorBiasedTrajectory(TrajectoryGenerator):
    """
    Generate trajectories biased toward corridor/street centers.

    This simulates realistic pedestrian behavior of preferring
    wide open areas over narrow passages.
    """

    def __init__(
        self,
        corridor_preference: float = 0.7,
        min_corridor_distance: float = 3.0,
        **kwargs
    ):
        """
        Args:
            corridor_preference: Probability of staying in corridor region
            min_corridor_distance: Minimum distance from walls to be "corridor"
        """
        super().__init__(**kwargs)
        self.corridor_preference = corridor_preference
        self.min_corridor_distance = min_corridor_distance

    def generate(
        self,
        walkable_mask: np.ndarray,
        radio_map: np.ndarray,
        n_points: Optional[int] = None,
        **kwargs
    ) -> Trajectory:
        """Generate corridor-biased trajectory."""
        if n_points is None:
            n_points = 100

        # Compute distance transform
        dist_transform = distance_transform_edt(walkable_mask)
        corridor_mask = dist_transform >= self.min_corridor_distance

        # Get corridor and non-corridor walkable points
        corridor_coords = np.argwhere(corridor_mask)
        walkable_coords = np.argwhere(walkable_mask)

        if len(corridor_coords) == 0:
            # No corridors, fall back to random walk
            gen = RandomWalkTrajectory(
                position_noise_std=self.position_noise_std,
                rss_noise_std=self.rss_noise_std
            )
            gen.rng = self.rng
            return gen.generate(walkable_mask, radio_map, n_points)

        # Start in corridor
        start_idx = self.rng.choice(len(corridor_coords))
        y, x = corridor_coords[start_idx].astype(float)
        angle = self.rng.uniform(0, 2 * np.pi)

        trajectory = Trajectory(trajectory_type="corridor_biased")
        t = 0.0
        step_size = 2.0

        for _ in range(n_points):
            # Add current point
            x_noisy, y_noisy = self._add_position_noise(x, y)
            rss = self._sample_rss(radio_map, x_noisy, y_noisy)
            trajectory.points.append(TrajectoryPoint(t=t, x=x_noisy, y=y_noisy, rss=rss))
            t += step_size

            # Bias toward corridor center
            current_dist = dist_transform[int(round(y)), int(round(x))]

            # Try to take a step
            best_pos = None
            best_score = -float('inf')

            for _ in range(10):
                angle_delta = self.rng.normal(0, np.pi/4)
                new_angle = angle + angle_delta

                dx = step_size * np.cos(new_angle)
                dy = step_size * np.sin(new_angle)
                new_x, new_y = x + dx, y + dy
                new_x_int, new_y_int = int(round(new_x)), int(round(new_y))

                if (0 <= new_x_int < walkable_mask.shape[1] and
                    0 <= new_y_int < walkable_mask.shape[0] and
                    walkable_mask[new_y_int, new_x_int]):

                    new_dist = dist_transform[new_y_int, new_x_int]

                    # Score: prefer higher distance from walls
                    if self.rng.random() < self.corridor_preference:
                        score = new_dist
                    else:
                        score = self.rng.random()

                    if score > best_score:
                        best_score = score
                        best_pos = (new_x, new_y, new_angle)

            if best_pos is not None:
                x, y, angle = best_pos

        return trajectory


class MixedTrajectoryGenerator:
    """
    Generate trajectories using a mix of strategies.

    Useful for creating diverse training data.
    """

    def __init__(
        self,
        weights: Optional[dict] = None,
        **kwargs
    ):
        """
        Args:
            weights: Dict mapping strategy name to weight
                     e.g., {'shortest_path': 0.4, 'random_walk': 0.3, 'corridor_biased': 0.3}
        """
        if weights is None:
            weights = {
                'shortest_path': 0.4,
                'random_walk': 0.3,
                'corridor_biased': 0.3
            }

        self.weights = weights
        self.generators = {
            'shortest_path': ShortestPathTrajectory(**kwargs),
            'random_walk': RandomWalkTrajectory(**kwargs),
            'corridor_biased': CorridorBiasedTrajectory(**kwargs),
        }
        self.rng = np.random.default_rng(kwargs.get('seed'))

    def generate(
        self,
        walkable_mask: np.ndarray,
        radio_map: np.ndarray,
        n_points: Optional[int] = None,
        strategy: Optional[str] = None,
        **kwargs
    ) -> Trajectory:
        """
        Generate trajectory using random or specified strategy.

        Args:
            strategy: If None, randomly select based on weights
        """
        if strategy is None:
            strategies = list(self.weights.keys())
            probs = np.array([self.weights[s] for s in strategies])
            probs = probs / probs.sum()
            strategy = self.rng.choice(strategies, p=probs)

        generator = self.generators[strategy]
        return generator.generate(walkable_mask, radio_map, n_points, **kwargs)

    def generate_multiple(
        self,
        walkable_mask: np.ndarray,
        radio_map: np.ndarray,
        n_trajectories: int = 3,
        points_per_trajectory: int = 100,
        **kwargs
    ) -> List[Trajectory]:
        """Generate multiple trajectories for one map."""
        return [
            self.generate(walkable_mask, radio_map, points_per_trajectory, **kwargs)
            for _ in range(n_trajectories)
        ]
