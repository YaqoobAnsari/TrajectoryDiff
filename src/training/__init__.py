"""
Training modules for TrajectoryDiff.

Contains PyTorch Lightning modules and callbacks for training diffusion models.
"""

from .diffusion_module import (
    DiffusionModule,
    SampleCallback,
    get_diffusion_module,
)

from .inference import (
    DiffusionInference,
    denormalize_radio_map,
    compute_uncertainty,
    sample_interpolation,
)

from .callbacks import (
    WandBSampleLogger,
    MetricsLogger,
    CheckpointEveryNSteps,
    GradientMonitor,
)

__all__ = [
    # Training
    'DiffusionModule',
    'SampleCallback',
    'get_diffusion_module',
    # Inference
    'DiffusionInference',
    'denormalize_radio_map',
    'compute_uncertainty',
    'sample_interpolation',
    # Callbacks
    'WandBSampleLogger',
    'MetricsLogger',
    'CheckpointEveryNSteps',
    'GradientMonitor',
]
