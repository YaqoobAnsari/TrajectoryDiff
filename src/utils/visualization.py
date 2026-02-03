"""
Visualization utilities for TrajectoryDiff.

This module provides publication-quality visualizations for:
1. Trajectories overlaid on floor plans
2. Sparse samples with coverage density
3. Radio map comparisons (pred vs ground truth)
4. Uncertainty maps
5. Metrics visualization
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm


# Custom colormaps
RADIO_CMAP = 'viridis'  # Good for pathloss visualization
ERROR_CMAP = 'RdYlBu_r'  # Red = high error, Blue = low error
COVERAGE_CMAP = 'YlOrRd'  # Yellow = low, Red = high coverage


def set_style():
    """Set publication-quality matplotlib style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'figure.figsize': (12, 8),
        'font.size': 12,
        'font.family': 'sans-serif',
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'image.cmap': RADIO_CMAP,
    })


def plot_building_map(
    building_map: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: str = "Building Map",
    show_colorbar: bool = False,
) -> plt.Axes:
    """
    Plot building map with buildings in dark and walkable areas in light.

    Args:
        building_map: Binary map (0=building, 255=walkable) or normalized
        ax: Matplotlib axes (creates new if None)
        title: Plot title
        show_colorbar: Whether to show colorbar

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # Ensure proper range
    if building_map.max() <= 1:
        building_map = building_map * 255

    im = ax.imshow(building_map, cmap='gray', vmin=0, vmax=255)
    ax.set_title(title)
    ax.axis('off')

    if show_colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    return ax


def plot_trajectory(
    building_map: np.ndarray,
    trajectory_points: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: str = "Trajectory",
    color: str = 'red',
    show_path: bool = True,
    show_points: bool = True,
    point_size: int = 20,
    alpha: float = 0.8,
) -> plt.Axes:
    """
    Plot trajectory overlaid on building map.

    Args:
        building_map: Building map array
        trajectory_points: (N, 2) array of [x, y] or [y, x] coordinates
        ax: Matplotlib axes
        title: Plot title
        color: Trajectory color
        show_path: Draw lines connecting points
        show_points: Draw scatter points
        point_size: Size of scatter points
        alpha: Transparency

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # Plot building map
    if building_map.max() <= 1:
        building_map = building_map * 255
    ax.imshow(building_map, cmap='gray', vmin=0, vmax=255)

    # Extract coordinates
    if trajectory_points.shape[1] == 2:
        x = trajectory_points[:, 0]
        y = trajectory_points[:, 1]
    else:
        # Assume [t, x, y, rss] format
        x = trajectory_points[:, 1]
        y = trajectory_points[:, 2]

    # Plot path
    if show_path and len(x) > 1:
        ax.plot(x, y, color=color, linewidth=1.5, alpha=alpha * 0.7)

    # Plot points
    if show_points:
        ax.scatter(x, y, c=color, s=point_size, alpha=alpha, edgecolors='white', linewidth=0.5)

    ax.set_title(title)
    ax.axis('off')

    return ax


def plot_multiple_trajectories(
    building_map: np.ndarray,
    trajectories: List[np.ndarray],
    labels: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "Multiple Trajectories",
) -> plt.Axes:
    """
    Plot multiple trajectories on same building map.

    Args:
        building_map: Building map array
        trajectories: List of (N, 2+) trajectory arrays
        labels: Labels for legend
        colors: Colors for each trajectory
        ax: Matplotlib axes
        title: Plot title

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))

    if labels is None:
        labels = [f"Trajectory {i+1}" for i in range(len(trajectories))]

    # Plot building map
    if building_map.max() <= 1:
        building_map = building_map * 255
    ax.imshow(building_map, cmap='gray', vmin=0, vmax=255)

    # Plot each trajectory
    for traj, label, color in zip(trajectories, labels, colors):
        if traj.shape[1] >= 2:
            x, y = traj[:, 0], traj[:, 1]
        else:
            continue

        ax.plot(x, y, color=color, linewidth=2, alpha=0.8, label=label)
        ax.scatter(x, y, c=[color], s=15, alpha=0.6)

    ax.legend(loc='upper right')
    ax.set_title(title)
    ax.axis('off')

    return ax


def plot_sparse_samples(
    building_map: np.ndarray,
    sparse_rss: np.ndarray,
    trajectory_mask: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: str = "Sparse Samples",
    cmap: str = RADIO_CMAP,
) -> plt.Axes:
    """
    Plot sparse RSS samples colored by value on building map.

    Args:
        building_map: Building map array
        sparse_rss: Sparse RSS values
        trajectory_mask: Binary mask of sample locations
        ax: Matplotlib axes
        title: Plot title
        cmap: Colormap for RSS values

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    # Plot building map in gray
    if building_map.max() <= 1:
        building_map = building_map * 255
    ax.imshow(building_map, cmap='gray', vmin=0, vmax=255, alpha=0.5)

    # Get sample locations
    y_coords, x_coords = np.where(trajectory_mask > 0)
    rss_values = sparse_rss[trajectory_mask > 0]

    # Normalize RSS values for coloring
    if len(rss_values) > 0:
        vmin, vmax = rss_values.min(), rss_values.max()
        if vmax == vmin:
            vmax = vmin + 1

        scatter = ax.scatter(
            x_coords, y_coords,
            c=rss_values, cmap=cmap,
            s=30, alpha=0.8,
            vmin=vmin, vmax=vmax,
            edgecolors='white', linewidth=0.3
        )
        plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label='RSS Value')

    ax.set_title(f"{title} ({len(x_coords)} samples)")
    ax.axis('off')

    return ax


def plot_coverage_density(
    coverage_density: np.ndarray,
    building_map: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "Coverage Density",
) -> plt.Axes:
    """
    Plot coverage density map.

    Args:
        coverage_density: Coverage density array
        building_map: Optional building map for overlay
        ax: Matplotlib axes
        title: Plot title

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    # Plot building map as background
    if building_map is not None:
        if building_map.max() <= 1:
            building_map = building_map * 255
        ax.imshow(building_map, cmap='gray', vmin=0, vmax=255, alpha=0.3)

    # Plot coverage density
    im = ax.imshow(coverage_density, cmap=COVERAGE_CMAP, alpha=0.7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Coverage Density')

    ax.set_title(title)
    ax.axis('off')

    return ax


def plot_radio_map_comparison(
    building_map: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    trajectory_mask: Optional[np.ndarray] = None,
    tx_position: Optional[Tuple[int, int]] = None,
    figsize: Tuple[int, int] = (16, 5),
    title: str = "Radio Map Comparison",
) -> plt.Figure:
    """
    Plot side-by-side comparison of ground truth and prediction.

    Args:
        building_map: Building map array
        ground_truth: Ground truth radio map
        prediction: Predicted radio map
        trajectory_mask: Optional mask of sampled locations
        tx_position: Optional transmitter position (x, y)
        figsize: Figure size
        title: Overall title

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 4, figsize=figsize)

    # Normalize for display
    vmin = min(ground_truth.min(), prediction.min())
    vmax = max(ground_truth.max(), prediction.max())

    # Building map
    if building_map.max() <= 1:
        building_map = building_map * 255
    axes[0].imshow(building_map, cmap='gray')
    if tx_position is not None:
        axes[0].scatter([tx_position[0]], [tx_position[1]], c='red', s=100, marker='x', linewidths=2)
    axes[0].set_title("Building Map + Tx")
    axes[0].axis('off')

    # Ground truth
    im1 = axes[1].imshow(ground_truth, cmap=RADIO_CMAP, vmin=vmin, vmax=vmax)
    if tx_position is not None:
        axes[1].scatter([tx_position[0]], [tx_position[1]], c='red', s=100, marker='x', linewidths=2)
    axes[1].set_title("Ground Truth")
    axes[1].axis('off')

    # Prediction
    im2 = axes[2].imshow(prediction, cmap=RADIO_CMAP, vmin=vmin, vmax=vmax)
    if tx_position is not None:
        axes[2].scatter([tx_position[0]], [tx_position[1]], c='red', s=100, marker='x', linewidths=2)
    axes[2].set_title("Prediction")
    axes[2].axis('off')

    # Error map
    error = np.abs(ground_truth - prediction)
    im3 = axes[3].imshow(error, cmap=ERROR_CMAP)
    if trajectory_mask is not None:
        # Overlay trajectory
        y_coords, x_coords = np.where(trajectory_mask > 0)
        axes[3].scatter(x_coords, y_coords, c='blue', s=5, alpha=0.3)
    axes[3].set_title(f"Absolute Error (RMSE: {np.sqrt(np.mean(error**2)):.3f})")
    axes[3].axis('off')

    # Colorbars
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04, label='Error')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


def plot_trajectory_type_comparison(
    building_map: np.ndarray,
    radio_map: np.ndarray,
    trajectories: dict,
    figsize: Tuple[int, int] = (16, 8),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Compare different trajectory types side by side.

    Args:
        building_map: Building map array
        radio_map: Radio map for RSS coloring
        trajectories: Dict mapping type name to trajectory array
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    n_types = len(trajectories)
    fig, axes = plt.subplots(2, n_types, figsize=figsize)

    if n_types == 1:
        axes = axes.reshape(-1, 1)

    colors = {'shortest_path': 'red', 'random_walk': 'blue', 'corridor_biased': 'green'}

    for i, (traj_type, traj_data) in enumerate(trajectories.items()):
        color = colors.get(traj_type, 'purple')

        # Top row: trajectory on building map
        ax_top = axes[0, i]
        if building_map.max() <= 1:
            bm = building_map * 255
        else:
            bm = building_map
        ax_top.imshow(bm, cmap='gray')

        if hasattr(traj_data, 'points'):
            x = [p.x for p in traj_data.points]
            y = [p.y for p in traj_data.points]
        else:
            x, y = traj_data[:, 0], traj_data[:, 1]

        ax_top.plot(x, y, color=color, linewidth=1.5, alpha=0.8)
        ax_top.scatter(x, y, c=color, s=15, alpha=0.6)
        ax_top.set_title(traj_type.replace('_', ' ').title())
        ax_top.axis('off')

        # Bottom row: sparse samples colored by RSS
        ax_bot = axes[1, i]
        ax_bot.imshow(bm, cmap='gray', alpha=0.4)

        # Sample RSS at trajectory points
        if hasattr(traj_data, 'points'):
            rss = [p.rss for p in traj_data.points]
        else:
            rss = [radio_map[int(yi), int(xi)] for xi, yi in zip(x, y)]

        scatter = ax_bot.scatter(x, y, c=rss, cmap=RADIO_CMAP, s=25, alpha=0.8)
        ax_bot.set_title(f"RSS Samples ({len(x)} pts)")
        ax_bot.axis('off')

    plt.colorbar(scatter, ax=axes[1, :], shrink=0.6, label='RSS Value', location='right')
    fig.suptitle("Trajectory Type Comparison", fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_uncertainty_map(
    mean_prediction: np.ndarray,
    std_prediction: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    trajectory_mask: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (14, 5),
) -> plt.Figure:
    """
    Plot uncertainty visualization.

    Args:
        mean_prediction: Mean prediction from ensemble
        std_prediction: Standard deviation (uncertainty)
        ground_truth: Optional ground truth for error comparison
        trajectory_mask: Optional mask showing sampled regions

    Returns:
        Matplotlib figure
    """
    n_cols = 3 if ground_truth is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=figsize)

    # Mean prediction
    im0 = axes[0].imshow(mean_prediction, cmap=RADIO_CMAP)
    axes[0].set_title("Mean Prediction")
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Uncertainty
    im1 = axes[1].imshow(std_prediction, cmap='plasma')
    if trajectory_mask is not None:
        # Show trajectory overlay
        y_coords, x_coords = np.where(trajectory_mask > 0)
        axes[1].scatter(x_coords, y_coords, c='white', s=3, alpha=0.3)
    axes[1].set_title("Uncertainty (Std Dev)")
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Error vs uncertainty correlation
    if ground_truth is not None:
        error = np.abs(mean_prediction - ground_truth)
        im2 = axes[2].imshow(error, cmap=ERROR_CMAP)
        axes[2].set_title("Actual Error")
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    return fig


def create_summary_figure(
    building_map: np.ndarray,
    radio_map: np.ndarray,
    sparse_rss: np.ndarray,
    trajectory_mask: np.ndarray,
    coverage_density: np.ndarray,
    tx_position: Optional[Tuple[int, int]] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Create comprehensive summary figure for a single sample.

    Args:
        building_map: Building map
        radio_map: Ground truth radio map
        sparse_rss: Sparse RSS samples
        trajectory_mask: Trajectory mask
        coverage_density: Coverage density
        tx_position: Transmitter position
        save_path: Optional path to save

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Building, Radio Map, Sparse Samples
    if building_map.max() <= 1:
        bm = building_map * 255
    else:
        bm = building_map

    axes[0, 0].imshow(bm, cmap='gray')
    if tx_position is not None:
        axes[0, 0].scatter([tx_position[0]], [tx_position[1]], c='red', s=100, marker='x', linewidths=2)
    axes[0, 0].set_title("Building Map + Transmitter")
    axes[0, 0].axis('off')

    im1 = axes[0, 1].imshow(radio_map, cmap=RADIO_CMAP)
    if tx_position is not None:
        axes[0, 1].scatter([tx_position[0]], [tx_position[1]], c='red', s=100, marker='x', linewidths=2)
    axes[0, 1].set_title("Ground Truth Radio Map")
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # Sparse samples
    axes[0, 2].imshow(bm, cmap='gray', alpha=0.5)
    y_coords, x_coords = np.where(trajectory_mask > 0)
    rss_values = sparse_rss[trajectory_mask > 0]
    if len(rss_values) > 0:
        sc = axes[0, 2].scatter(x_coords, y_coords, c=rss_values, cmap=RADIO_CMAP, s=20, alpha=0.8)
        plt.colorbar(sc, ax=axes[0, 2], fraction=0.046, pad=0.04)
    axes[0, 2].set_title(f"Sparse Samples ({len(x_coords)} points)")
    axes[0, 2].axis('off')

    # Row 2: Coverage Density, Trajectory Visualization, Histogram
    im3 = axes[1, 0].imshow(coverage_density, cmap=COVERAGE_CMAP)
    axes[1, 0].set_title("Coverage Density")
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # Trajectory on map
    axes[1, 1].imshow(bm, cmap='gray')
    axes[1, 1].scatter(x_coords, y_coords, c='red', s=15, alpha=0.6)
    if len(x_coords) > 1:
        axes[1, 1].plot(x_coords, y_coords, 'r-', alpha=0.3, linewidth=0.5)
    axes[1, 1].set_title("Trajectory Path")
    axes[1, 1].axis('off')

    # Value histogram
    axes[1, 2].hist(radio_map.flatten(), bins=50, alpha=0.7, label='Full Map', color='blue')
    if len(rss_values) > 0:
        axes[1, 2].hist(rss_values, bins=30, alpha=0.7, label='Sampled', color='red')
    axes[1, 2].set_xlabel('RSS Value')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].set_title('Value Distribution')
    axes[1, 2].legend()

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
