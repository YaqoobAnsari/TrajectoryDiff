"""
Classical interpolation baselines for radio map reconstruction.

These methods don't use learning - they purely interpolate from sparse samples.
Used to establish performance floor and show improvement from learning.

Methods:
1. Nearest Neighbor: Simple but blocky
2. IDW (Inverse Distance Weighting): Smooth but ignores structure
3. RBF (Radial Basis Functions): Flexible kernel-based
4. Kriging (Gaussian Process): Provides uncertainty but slow
"""

from typing import Optional, Tuple
import numpy as np
from scipy.interpolate import RBFInterpolator, NearestNDInterpolator
from scipy.ndimage import distance_transform_edt


class NearestNeighborBaseline:
    """
    Nearest neighbor interpolation.

    Simple but fast. Each unknown pixel takes value of nearest sample.
    """

    def __init__(self):
        self.name = "nearest_neighbor"

    def __call__(
        self,
        sparse_rss: np.ndarray,
        mask: np.ndarray,
        building_map: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Interpolate using nearest neighbor.

        Args:
            sparse_rss: Sparse RSS values (H, W)
            mask: Binary mask of sample locations (H, W)
            building_map: Optional building map (unused)

        Returns:
            Interpolated radio map (H, W)
        """
        H, W = sparse_rss.shape

        # Get sample locations and values
        y_samples, x_samples = np.where(mask > 0)
        values = sparse_rss[y_samples, x_samples]

        if len(values) == 0:
            return np.zeros_like(sparse_rss)

        # Create interpolator
        points = np.column_stack([x_samples, y_samples])
        interpolator = NearestNDInterpolator(points, values)

        # Create query grid
        xx, yy = np.meshgrid(np.arange(W), np.arange(H))
        query_points = np.column_stack([xx.ravel(), yy.ravel()])

        # Interpolate
        result = interpolator(query_points).reshape(H, W)

        return result


class IDWBaseline:
    """
    Inverse Distance Weighting interpolation.

    Smooth interpolation where influence decays with distance.
    """

    def __init__(self, power: float = 2.0, epsilon: float = 1e-10):
        """
        Args:
            power: Distance power (higher = more local)
            epsilon: Small value to avoid division by zero
        """
        self.name = "idw"
        self.power = power
        self.epsilon = epsilon

    def __call__(
        self,
        sparse_rss: np.ndarray,
        mask: np.ndarray,
        building_map: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Interpolate using IDW.

        Args:
            sparse_rss: Sparse RSS values (H, W)
            mask: Binary mask of sample locations (H, W)
            building_map: Optional building map (unused)

        Returns:
            Interpolated radio map (H, W)
        """
        H, W = sparse_rss.shape

        # Get sample locations and values
        y_samples, x_samples = np.where(mask > 0)
        values = sparse_rss[y_samples, x_samples]

        if len(values) == 0:
            return np.zeros_like(sparse_rss)

        if len(values) == 1:
            return np.full_like(sparse_rss, values[0])

        # Create coordinate grids
        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

        # Compute weighted sum
        result = np.zeros((H, W), dtype=np.float64)
        weights_sum = np.zeros((H, W), dtype=np.float64)

        for i in range(len(values)):
            # Distance to this sample
            dist = np.sqrt((xx - x_samples[i])**2 + (yy - y_samples[i])**2)
            weight = 1.0 / (dist**self.power + self.epsilon)

            result += weight * values[i]
            weights_sum += weight

        result /= weights_sum

        return result.astype(np.float32)


class RBFBaseline:
    """
    Radial Basis Function interpolation.

    Flexible kernel-based interpolation with various kernel choices.
    """

    def __init__(
        self,
        kernel: str = 'thin_plate_spline',
        smoothing: float = 0.0,
        max_samples: int = 1000,
        epsilon: Optional[float] = None,
    ):
        """
        Args:
            kernel: RBF kernel type ('thin_plate_spline', 'multiquadric',
                   'inverse_multiquadric', 'gaussian', 'linear', 'cubic')
            smoothing: Smoothing parameter (0 = exact interpolation)
            max_samples: Maximum samples to use (for speed)
            epsilon: Shape parameter required for 'multiquadric', 'gaussian', etc.
        """
        self.name = f"rbf_{kernel}"
        self.kernel = kernel
        self.smoothing = smoothing
        self.max_samples = max_samples
        self.epsilon = epsilon

    def __call__(
        self,
        sparse_rss: np.ndarray,
        mask: np.ndarray,
        building_map: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Interpolate using RBF.

        Args:
            sparse_rss: Sparse RSS values (H, W)
            mask: Binary mask of sample locations (H, W)
            building_map: Optional building map (unused)

        Returns:
            Interpolated radio map (H, W)
        """
        H, W = sparse_rss.shape

        # Get sample locations and values
        y_samples, x_samples = np.where(mask > 0)
        values = sparse_rss[y_samples, x_samples]

        if len(values) == 0:
            return np.zeros_like(sparse_rss)

        # Subsample if too many points
        if len(values) > self.max_samples:
            idx = np.random.choice(len(values), self.max_samples, replace=False)
            x_samples = x_samples[idx]
            y_samples = y_samples[idx]
            values = values[idx]

        # Create interpolator
        points = np.column_stack([x_samples, y_samples])
        rbf_kwargs = dict(kernel=self.kernel, smoothing=self.smoothing)
        if self.epsilon is not None:
            rbf_kwargs['epsilon'] = self.epsilon
        interpolator = RBFInterpolator(points, values, **rbf_kwargs)

        # Create query grid
        xx, yy = np.meshgrid(np.arange(W), np.arange(H))
        query_points = np.column_stack([xx.ravel(), yy.ravel()])

        # Interpolate
        result = interpolator(query_points).reshape(H, W)

        return result.astype(np.float32)


class KrigingBaseline:
    """
    Kriging (Gaussian Process) interpolation.

    Provides uncertainty estimates but is computationally expensive.
    Uses simplified spherical variogram.
    """

    def __init__(
        self,
        variogram_model: str = 'spherical',
        max_samples: int = 500,
        range_param: float = 50.0,
        sill: float = 1.0,
        nugget: float = 0.01,
    ):
        """
        Args:
            variogram_model: Variogram model type
            max_samples: Maximum samples to use (for speed)
            range_param: Range parameter for variogram
            sill: Sill parameter for variogram
            nugget: Nugget parameter for variogram
        """
        self.name = "kriging"
        self.variogram_model = variogram_model
        self.max_samples = max_samples
        self.range_param = range_param
        self.sill = sill
        self.nugget = nugget

    def _spherical_variogram(self, h: np.ndarray) -> np.ndarray:
        """Spherical variogram model."""
        result = np.zeros_like(h)
        mask = h <= self.range_param
        result[mask] = self.sill * (
            1.5 * h[mask] / self.range_param -
            0.5 * (h[mask] / self.range_param) ** 3
        )
        result[~mask] = self.sill
        return result + self.nugget

    def __call__(
        self,
        sparse_rss: np.ndarray,
        mask: np.ndarray,
        building_map: Optional[np.ndarray] = None,
        return_uncertainty: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Interpolate using Kriging.

        Args:
            sparse_rss: Sparse RSS values (H, W)
            mask: Binary mask of sample locations (H, W)
            building_map: Optional building map (unused)
            return_uncertainty: If True, return uncertainty map

        Returns:
            Interpolated radio map (H, W), and optionally uncertainty map
        """
        H, W = sparse_rss.shape

        # Get sample locations and values
        y_samples, x_samples = np.where(mask > 0)
        values = sparse_rss[y_samples, x_samples]

        if len(values) == 0:
            if return_uncertainty:
                return np.zeros_like(sparse_rss), np.ones_like(sparse_rss)
            return np.zeros_like(sparse_rss)

        # Subsample if too many points
        if len(values) > self.max_samples:
            idx = np.random.choice(len(values), self.max_samples, replace=False)
            x_samples = x_samples[idx]
            y_samples = y_samples[idx]
            values = values[idx]

        n_samples = len(values)
        points = np.column_stack([x_samples, y_samples])

        # Build covariance matrix
        dist_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                d = np.sqrt((points[i, 0] - points[j, 0])**2 +
                           (points[i, 1] - points[j, 1])**2)
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

        C = self.sill + self.nugget - self._spherical_variogram(dist_matrix)

        # Add regularization for numerical stability
        C += np.eye(n_samples) * 1e-6

        # Solve for weights
        try:
            C_inv = np.linalg.inv(C)
        except np.linalg.LinAlgError:
            # Fallback to IDW
            idw = IDWBaseline()
            if return_uncertainty:
                return idw(sparse_rss, mask), np.ones_like(sparse_rss) * self.sill
            return idw(sparse_rss, mask)

        # Interpolate each point
        result = np.zeros((H, W), dtype=np.float64)
        uncertainty = np.zeros((H, W), dtype=np.float64) if return_uncertainty else None

        # Process in chunks for efficiency
        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

        for yi in range(H):
            for xi in range(W):
                # Distance to all samples
                dist_to_samples = np.sqrt(
                    (points[:, 0] - xi)**2 + (points[:, 1] - yi)**2
                )
                c0 = self.sill + self.nugget - self._spherical_variogram(dist_to_samples)

                # Weights
                weights = C_inv @ c0

                # Prediction
                result[yi, xi] = np.dot(weights, values)

                # Uncertainty
                if return_uncertainty:
                    uncertainty[yi, xi] = self.sill + self.nugget - np.dot(c0, weights)

        if return_uncertainty:
            return result.astype(np.float32), np.sqrt(np.maximum(uncertainty, 0)).astype(np.float32)
        return result.astype(np.float32)


class DistanceTransformBaseline:
    """
    Simple baseline using distance transform.

    Fills unknown regions with value from nearest sample,
    optionally with distance-based decay.
    """

    def __init__(self, decay_rate: float = 0.0):
        """
        Args:
            decay_rate: Rate at which values decay with distance (0 = no decay)
        """
        self.name = "distance_transform"
        self.decay_rate = decay_rate

    def __call__(
        self,
        sparse_rss: np.ndarray,
        mask: np.ndarray,
        building_map: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Interpolate using distance transform."""
        from scipy.ndimage import distance_transform_edt

        H, W = sparse_rss.shape

        if mask.sum() == 0:
            return np.zeros_like(sparse_rss)

        # Get distance to nearest sample and indices
        dist, indices = distance_transform_edt(1 - mask, return_indices=True)

        # Get value from nearest sample
        result = sparse_rss[indices[0], indices[1]]

        # Apply decay if specified
        if self.decay_rate > 0:
            result = result * np.exp(-self.decay_rate * dist)

        return result.astype(np.float32)


def get_all_baselines() -> dict:
    """Get dictionary of all baseline methods."""
    return {
        'nearest_neighbor': NearestNeighborBaseline(),
        'idw': IDWBaseline(power=2.0),
        'idw_p3': IDWBaseline(power=3.0),
        'rbf_tps': RBFBaseline(kernel='thin_plate_spline'),
        'rbf_multiquadric': RBFBaseline(kernel='multiquadric', epsilon=1.0),
        'distance_transform': DistanceTransformBaseline(),
        # Kriging is slow, use sparingly
        # 'kriging': KrigingBaseline(),
    }


def run_baseline_evaluation(
    sparse_rss: np.ndarray,
    mask: np.ndarray,
    ground_truth: np.ndarray,
    baselines: Optional[dict] = None,
) -> dict:
    """
    Run all baselines and compute metrics.

    Args:
        sparse_rss: Sparse RSS samples
        mask: Sample mask
        ground_truth: Ground truth radio map
        baselines: Optional dict of baselines (uses defaults if None)

    Returns:
        Dict mapping baseline name to metrics dict
    """
    import sys
    sys.path.insert(0, 'src')
    from evaluation.metrics import compute_all_metrics

    if baselines is None:
        baselines = get_all_baselines()

    results = {}

    for name, baseline in baselines.items():
        # Run baseline
        prediction = baseline(sparse_rss, mask)

        # Compute metrics
        metrics = compute_all_metrics(prediction, ground_truth, trajectory_mask=mask)

        results[name] = {
            'prediction': prediction,
            'metrics': metrics,
        }

    return results
