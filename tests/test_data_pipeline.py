"""
Tests for the data pipeline components.

Run with: pytest tests/test_data_pipeline.py -v
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


class TestFloorPlanProcessor:
    """Test floor plan processing utilities."""

    def test_walkable_mask_binary(self):
        """Walkable mask should be binary."""
        from data.floor_plan import FloorPlanProcessor

        # Create synthetic building map
        building_map = np.zeros((256, 256), dtype=np.uint8)
        building_map[100:150, 100:150] = 255  # Walkable area

        processor = FloorPlanProcessor(erosion_radius=0)
        walkable = processor.get_walkable_mask(building_map, erode=False)

        assert walkable.dtype == bool
        assert set(np.unique(walkable)) <= {True, False}

    def test_walkable_mask_erosion(self):
        """Erosion should shrink walkable area."""
        from data.floor_plan import FloorPlanProcessor

        building_map = np.zeros((256, 256), dtype=np.uint8)
        building_map[50:200, 50:200] = 255

        processor_no_erosion = FloorPlanProcessor(erosion_radius=0)
        processor_with_erosion = FloorPlanProcessor(erosion_radius=2)

        mask_no_erosion = processor_no_erosion.get_walkable_mask(building_map, erode=False)
        mask_with_erosion = processor_with_erosion.get_walkable_mask(building_map, erode=True)

        assert mask_with_erosion.sum() < mask_no_erosion.sum()

    def test_distance_transform(self):
        """Distance transform should be positive in walkable areas."""
        from data.floor_plan import FloorPlanProcessor

        building_map = np.zeros((256, 256), dtype=np.uint8)
        building_map[50:200, 50:200] = 255

        processor = FloorPlanProcessor()
        walkable = processor.get_walkable_mask(building_map, erode=False)
        dist = processor.get_distance_transform(walkable)

        # Center should have highest distance
        assert dist[125, 125] > dist[51, 51]

    def test_png_to_db_conversion(self):
        """Test PNG to dB conversion."""
        from data.floor_plan import png_to_db, db_to_png

        png_vals = np.array([0, 127, 255])
        db_vals = png_to_db(png_vals)

        assert db_vals[0] == pytest.approx(-186, abs=1)
        assert db_vals[2] == pytest.approx(-47, abs=1)

        # Round trip
        recovered = db_to_png(db_vals)
        assert np.allclose(recovered, png_vals, atol=1)


class TestTrajectorySampler:
    """Test trajectory sampling algorithms."""

    @pytest.fixture
    def walkable_mask(self):
        """Create simple walkable mask."""
        mask = np.zeros((256, 256), dtype=bool)
        mask[50:200, 50:200] = True  # Large walkable area
        return mask

    @pytest.fixture
    def radio_map(self):
        """Create synthetic radio map."""
        return np.random.randint(0, 256, (256, 256), dtype=np.uint8)

    def test_shortest_path_generates_points(self, walkable_mask, radio_map):
        """Shortest path should generate requested number of points."""
        from data.trajectory_sampler import ShortestPathTrajectory

        gen = ShortestPathTrajectory(seed=42)
        traj = gen.generate(walkable_mask, radio_map, n_points=50)

        assert len(traj) >= 10  # Should have some points
        assert len(traj) <= 100  # Not too many

    def test_random_walk_generates_points(self, walkable_mask, radio_map):
        """Random walk should generate requested number of points."""
        from data.trajectory_sampler import RandomWalkTrajectory

        gen = RandomWalkTrajectory(seed=42)
        traj = gen.generate(walkable_mask, radio_map, n_points=50)

        assert len(traj) == 50

    def test_trajectory_points_in_walkable(self, walkable_mask, radio_map):
        """All trajectory points should be in walkable area."""
        from data.trajectory_sampler import MixedTrajectoryGenerator

        gen = MixedTrajectoryGenerator(seed=42, position_noise_std=0)
        traj = gen.generate(walkable_mask, radio_map, n_points=50)

        for point in traj.points:
            x_int, y_int = int(round(point.x)), int(round(point.y))
            if 0 <= x_int < 256 and 0 <= y_int < 256:
                assert walkable_mask[y_int, x_int], f"Point ({x_int}, {y_int}) not walkable"

    def test_trajectory_to_sparse_map(self, walkable_mask, radio_map):
        """Trajectory should convert to sparse map correctly."""
        from data.trajectory_sampler import MixedTrajectoryGenerator

        gen = MixedTrajectoryGenerator(seed=42)
        traj = gen.generate(walkable_mask, radio_map, n_points=50)

        sparse, mask = traj.to_sparse_map((256, 256))

        assert sparse.shape == (256, 256)
        assert mask.shape == (256, 256)
        assert mask.sum() > 0
        assert mask.sum() <= len(traj)


class TestTransforms:
    """Test data augmentation transforms."""

    def test_rotation_changes_tensor(self):
        """Rotation should modify the tensor."""
        import torch
        from data.transforms import RadioMapTransform

        sample = {
            'building_map': torch.rand(1, 256, 256),
            'radio_map': torch.rand(1, 256, 256),
            'tx_position': torch.tensor([100.0, 150.0]),
        }
        original = sample['radio_map'].clone()

        transform = RadioMapTransform(
            random_rotation=True,
            random_flip=False,
            p_rotation=1.0  # Always rotate
        )

        # Apply multiple times until we get a rotation
        for _ in range(10):
            transformed = transform({k: v.clone() for k, v in sample.items()})
            if not torch.allclose(transformed['radio_map'], original):
                break
        else:
            pytest.fail("Rotation didn't change tensor after 10 attempts")

    def test_flip_changes_tensor(self):
        """Flip should modify the tensor."""
        import torch
        from data.transforms import RadioMapTransform

        sample = {
            'building_map': torch.arange(256).unsqueeze(0).unsqueeze(0).expand(1, 256, 256).float(),
            'radio_map': torch.arange(256).unsqueeze(0).unsqueeze(0).expand(1, 256, 256).float(),
            'tx_position': torch.tensor([100.0, 150.0]),
        }

        transform = RadioMapTransform(
            random_rotation=False,
            random_flip=True,
            p_flip=1.0
        )

        # Some flip should occur
        different = False
        for _ in range(10):
            transformed = transform({k: v.clone() for k, v in sample.items()})
            if not torch.allclose(transformed['radio_map'], sample['radio_map']):
                different = True
                break

        assert different, "Flip didn't change tensor"


class TestBaselines:
    """Test interpolation baselines."""

    @pytest.fixture
    def sparse_data(self):
        """Create sparse sample data."""
        sparse = np.zeros((256, 256), dtype=np.float32)
        mask = np.zeros((256, 256), dtype=np.float32)

        # Add some samples
        np.random.seed(42)
        for _ in range(100):
            x, y = np.random.randint(0, 256, 2)
            sparse[y, x] = np.random.uniform(0, 255)
            mask[y, x] = 1

        return sparse, mask

    def test_idw_interpolates(self, sparse_data):
        """IDW should produce full map."""
        from models.baselines.interpolation import IDWBaseline

        sparse, mask = sparse_data
        baseline = IDWBaseline()
        result = baseline(sparse, mask)

        assert result.shape == (256, 256)
        assert not np.any(np.isnan(result))
        assert result.min() >= 0
        assert result.max() <= 255

    def test_rbf_interpolates(self, sparse_data):
        """RBF should produce full map."""
        from models.baselines.interpolation import RBFBaseline

        sparse, mask = sparse_data
        baseline = RBFBaseline(kernel='thin_plate_spline')
        result = baseline(sparse, mask)

        assert result.shape == (256, 256)
        assert not np.any(np.isnan(result))

    def test_nearest_neighbor_exact_at_samples(self, sparse_data):
        """Nearest neighbor should be exact at sample locations."""
        from models.baselines.interpolation import NearestNeighborBaseline

        sparse, mask = sparse_data
        baseline = NearestNeighborBaseline()
        result = baseline(sparse, mask)

        # At sample locations, should match
        sample_locs = mask > 0
        assert np.allclose(result[sample_locs], sparse[sample_locs])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
