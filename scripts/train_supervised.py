#!/usr/bin/env python3
"""
Train Supervised U-Net Baseline (C3).

This script trains a supervised baseline using the same UNet architecture
as the diffusion model, but with direct MSE loss (no diffusion).

Usage:
    python scripts/train_supervised.py
    python scripts/train_supervised.py experiment=supervised_baseline
    python scripts/train_supervised.py model.unet_size=large training.max_epochs=100
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import hydra
import lightning as L
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
from lightning.pytorch.loggers import WandbLogger

from data import RadioMapDataModule
from models.baselines.supervised_unet import SupervisedUNetBaseline


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Main training function for supervised baseline."""

    # Print config
    print("=" * 60)
    print("TrajectoryDiff: Supervised U-Net Baseline Training")
    print("=" * 60)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))

    # Set seed for reproducibility
    L.seed_everything(cfg.get('seed', 42), workers=True)

    # Setup data
    print("\nSetting up data...")
    datamodule = RadioMapDataModule(
        data_dir=cfg.data.dataset.root,
        batch_size=cfg.data.loader.batch_size,
        num_workers=cfg.data.loader.num_workers,
        train_ratio=cfg.data.splits.train,
        val_ratio=cfg.data.splits.val,
        sampling_strategy=cfg.data.sampling.strategy,
        num_trajectories=cfg.data.sampling.trajectory.num_trajectories,
        points_per_trajectory=cfg.data.sampling.trajectory.points_per_trajectory,
        trajectory_method=cfg.data.sampling.trajectory.method,
        rss_noise_std=cfg.data.sampling.trajectory.rss_noise_std,
        position_noise_std=cfg.data.sampling.trajectory.get('position_noise_std', 0.5),
    )

    # Setup model
    print("Setting up model...")
    model = SupervisedUNetBaseline(
        unet_size=cfg.model.get('unet_size', 'medium'),
        image_size=cfg.data.dataset.get('image_size', 256),
        condition_channels=cfg.model.get('condition_channels', 64),
        use_building_map=cfg.model.get('use_building_map', True),
        use_sparse_rss=cfg.model.get('use_sparse_rss', True),
        use_trajectory_mask=cfg.model.get('use_trajectory_mask', True),
        use_coverage_density=cfg.model.get('use_coverage_density', True),
        use_tx_position=cfg.model.get('use_tx_position', True),
        learning_rate=cfg.training.get('learning_rate', 1e-4),
        weight_decay=cfg.training.get('weight_decay', 0.01),
        warmup_epochs=cfg.training.get('warmup_epochs', 5),
        loss_type=cfg.training.get('loss_type', 'mse'),
        mask_unobserved_weight=cfg.training.get('mask_unobserved_weight', 1.0),
    )

    # Setup callbacks
    callbacks = []

    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(cfg.get('output_dir', 'experiments'), 'checkpoints'),
        filename='supervised-{epoch:03d}-{val_loss:.4f}',
        monitor='val/loss',
        mode='min',
        save_top_k=3,
        save_last=True,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)

    # Learning rate monitor
    callbacks.append(LearningRateMonitor(logging_interval='step'))

    # Early stopping (optional)
    if cfg.training.get('early_stopping', False):
        early_stop = EarlyStopping(
            monitor='val/loss',
            patience=cfg.training.get('early_stopping_patience', 20),
            mode='min',
            min_delta=cfg.training.get('early_stopping_min_delta', 0.0001),
            verbose=True,
        )
        callbacks.append(early_stop)

    # Setup logger
    logger = None
    if cfg.get('use_wandb', True):
        logger = WandbLogger(
            project=cfg.get('wandb_project', 'trajdiff-supervised'),
            name=cfg.get('experiment_name', 'supervised_baseline'),
            save_dir=cfg.get('output_dir', 'experiments'),
            log_model=False,
        )
        # Log config to wandb
        logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True))

    # Setup trainer
    print("Setting up trainer...")
    trainer = L.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator='auto',
        devices=1,
        precision=cfg.training.get('precision', 'bf16-mixed'),
        gradient_clip_val=cfg.training.get('gradient_clip_val', 1.0),
        accumulate_grad_batches=cfg.training.get('accumulate_grad_batches', 1),
        log_every_n_steps=cfg.training.get('log_every_n_steps', 10),
        val_check_interval=cfg.training.get('val_check_interval', 1.0),
        callbacks=callbacks,
        logger=logger,
        deterministic=False,  # Set to True for full reproducibility (slower)
        enable_progress_bar=True,
    )

    # Train
    print("\nStarting training...")
    print(f"Max epochs: {cfg.training.max_epochs}")
    print(f"Batch size: {cfg.data.loader.batch_size}")
    print(f"Learning rate: {cfg.training.get('learning_rate', 1e-4)}")
    print(f"Output dir: {cfg.get('output_dir', 'experiments')}")
    print()

    trainer.fit(model, datamodule=datamodule)

    # Print best checkpoint
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best val/loss: {checkpoint_callback.best_model_score:.4f}")


if __name__ == "__main__":
    main()
