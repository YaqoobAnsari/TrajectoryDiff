#!/usr/bin/env python3
"""
TrajectoryDiff Training Script

Usage:
    python scripts/train.py                           # Default config
    python scripts/train.py experiment=trajectory_baseline  # Named experiment
    python scripts/train.py data.loader.batch_size=32      # Override params
    python scripts/train.py -m model.unet.base_channels=32,64  # Multirun
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import hydra
import lightning as L
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from omegaconf import DictConfig, OmegaConf

from data.datamodule import RadioMapDataModule
from training.trainer import TrajectoryDiffusionModule
from utils.io import save_config


def setup_callbacks(cfg: DictConfig) -> list:
    """Setup training callbacks."""
    callbacks = []
    
    # Checkpointing
    callbacks.append(
        ModelCheckpoint(
            dirpath=os.path.join(cfg.paths.output_dir, "checkpoints"),
            filename="epoch={epoch:03d}-val_rmse={val/rmse:.4f}",
            save_top_k=cfg.training.checkpoint.save_top_k,
            monitor=cfg.training.checkpoint.monitor,
            mode=cfg.training.checkpoint.mode,
            save_last=cfg.training.checkpoint.save_last,
            auto_insert_metric_name=False,
        )
    )
    
    # Early stopping
    if cfg.training.early_stopping.enabled:
        callbacks.append(
            EarlyStopping(
                monitor=cfg.training.early_stopping.monitor,
                patience=cfg.training.early_stopping.patience,
                mode=cfg.training.early_stopping.mode,
                min_delta=cfg.training.early_stopping.min_delta,
            )
        )
    
    # Learning rate monitor
    callbacks.append(LearningRateMonitor(logging_interval="step"))
    
    # Progress bar
    callbacks.append(RichProgressBar())
    
    return callbacks


def setup_loggers(cfg: DictConfig) -> list:
    """Setup experiment loggers."""
    loggers = []
    
    # TensorBoard
    if cfg.logging.tensorboard.enabled:
        loggers.append(
            TensorBoardLogger(
                save_dir=cfg.paths.log_dir,
                name="tensorboard",
                version=cfg.experiment.name,
            )
        )
    
    # Weights & Biases
    if cfg.logging.wandb.enabled:
        loggers.append(
            WandbLogger(
                project=cfg.logging.wandb.project,
                entity=cfg.logging.wandb.entity,
                name=cfg.experiment.name,
                save_dir=cfg.paths.log_dir,
                offline=cfg.logging.wandb.offline,
                tags=cfg.experiment.tags,
                config=OmegaConf.to_container(cfg, resolve=True),
            )
        )
    
    return loggers if loggers else None


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> float:
    """Main training function."""
    
    # Print config
    print(OmegaConf.to_yaml(cfg))
    
    # Set seed for reproducibility
    L.seed_everything(cfg.experiment.seed, workers=True)
    
    # Save config
    save_config(cfg, os.path.join(cfg.paths.output_dir, "config.yaml"))
    
    # Setup data module
    datamodule = RadioMapDataModule(cfg)
    
    # Setup model
    model = TrajectoryDiffusionModule(cfg)
    
    # Setup callbacks and loggers
    callbacks = setup_callbacks(cfg)
    loggers = setup_loggers(cfg)
    
    # Setup trainer
    trainer = L.Trainer(
        accelerator=cfg.hardware.accelerator,
        devices=cfg.hardware.devices,
        precision=cfg.hardware.precision,
        max_epochs=cfg.training.max_epochs,
        max_steps=cfg.training.max_steps or -1,
        val_check_interval=cfg.training.val_check_interval,
        gradient_clip_val=cfg.training.gradient.clip_val,
        accumulate_grad_batches=cfg.training.gradient.accumulation_steps,
        log_every_n_steps=cfg.training.log_every_n_steps,
        deterministic=cfg.training.deterministic,
        callbacks=callbacks,
        logger=loggers,
        default_root_dir=cfg.paths.output_dir,
    )
    
    # Train
    trainer.fit(model, datamodule=datamodule)
    
    # Test with best checkpoint
    if trainer.checkpoint_callback.best_model_path:
        trainer.test(model, datamodule=datamodule, ckpt_path="best")
    
    # Return best validation metric for hyperparameter optimization
    return trainer.callback_metrics.get("val/rmse", float("inf"))


if __name__ == "__main__":
    main()
