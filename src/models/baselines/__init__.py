"""
Baseline models for TrajectoryDiff.

Contains classical interpolation baselines and deep learning baselines.
"""

from .supervised_unet import SupervisedUNetBaseline
from .radio_unet import RadioUNetBaseline
from .rmdm import RMDMBaseline

__all__ = [
    'SupervisedUNetBaseline',
    'RadioUNetBaseline',
    'RMDMBaseline',
]
