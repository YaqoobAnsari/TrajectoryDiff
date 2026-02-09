#!/usr/bin/env python3
"""
TrajectoryDiff Training Script

Usage:
    python scripts/train.py                           # Default config
    python scripts/train.py experiment=trajectory_baseline  # Named experiment
    python scripts/train.py data.loader.batch_size=32      # Override params
    python scripts/train.py -m model.unet.base_channels=32,64  # Multirun

Examples:
    # Quick test run
    python scripts/train.py training.max_epochs=2 data.loader.batch_size=4

    # Full training with W&B
    python scripts/train.py logging.wandb.enabled=true experiment.name=baseline_v1

    # Resume from checkpoint
    python scripts/train.py +ckpt_path=/path/to/checkpoint.ckpt
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
    TQDMProgressBar,
)
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger, WandbLogger
from omegaconf import DictConfig, OmegaConf

from data import RadioMapDataModule
from training import (
    DiffusionModule,
    WandBSampleLogger,
    MetricsLogger,
    GradientMonitor,
    TrainingHealthCheck,
)


def setup_callbacks(cfg: DictConfig) -> list:
    """Setup training callbacks."""
    callbacks = []

    # Checkpointing
    ckpt_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir) / "checkpoints"
    callbacks.append(
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="epoch={epoch:03d}-val_loss={val/loss:.4f}",
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

    # Progress bar: use TQDM on SLURM (clean file output), Rich locally
    is_slurm = cfg.hardware.get('slurm', False) or os.environ.get('SLURM_JOB_ID') is not None
    if is_slurm:
        callbacks.append(TQDMProgressBar(refresh_rate=100))
    else:
        callbacks.append(RichProgressBar())

    # Gradient monitoring
    callbacks.append(GradientMonitor(log_every_n_steps=100))

    # Sample logging (if W&B enabled)
    if cfg.logging.wandb.enabled:
        callbacks.append(
            WandBSampleLogger(
                every_n_epochs=5,
                num_samples=4,
                use_ddim=True,
            )
        )

    # Metrics computation (full sampling is expensive — run sparingly)
    callbacks.append(
        MetricsLogger(
            compute_every_n_epochs=10,
            num_eval_samples=10,
        )
    )

    # Training health check (NaN detection, loss explosion, GPU memory logging)
    callbacks.append(TrainingHealthCheck(log_gpu_every_n_steps=100))

    return callbacks


def setup_loggers(cfg: DictConfig) -> list:
    """Setup experiment loggers."""
    loggers = []
    log_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    # CSV logger (always enabled on SLURM for file-based metric tracking)
    is_slurm = cfg.hardware.get('slurm', False) or os.environ.get('SLURM_JOB_ID') is not None
    if is_slurm:
        loggers.append(
            CSVLogger(
                save_dir=str(log_dir),
                name="csv_logs",
                version="",
            )
        )

    # TensorBoard
    if cfg.logging.tensorboard.enabled:
        loggers.append(
            TensorBoardLogger(
                save_dir=str(log_dir),
                name="tensorboard",
                version="",
            )
        )

    # Weights & Biases
    if cfg.logging.wandb.enabled:
        loggers.append(
            WandbLogger(
                project=cfg.logging.wandb.project,
                entity=cfg.logging.wandb.entity,
                name=cfg.experiment.name,
                save_dir=str(log_dir),
                offline=cfg.logging.wandb.offline,
                tags=list(cfg.experiment.tags) if cfg.experiment.tags else None,
                config=OmegaConf.to_container(cfg, resolve=True),
            )
        )

    return loggers if loggers else None


def create_model(cfg: DictConfig) -> DiffusionModule:
    """Create diffusion model from config."""
    model_cfg = cfg.model

    # U-Net size from config (small/medium/large)
    unet_size = model_cfg.unet.get('size', 'medium')

    # Conditioning flags from config
    cond_cfg = model_cfg.get('conditioning', {})
    use_building_map = cond_cfg.get('use_building_map', True)
    use_sparse_rss = cond_cfg.get('use_sparse_rss', True)
    use_trajectory_mask = cond_cfg.get('use_trajectory_mask', True)
    use_coverage_density = cond_cfg.get('use_coverage_density', True)
    use_tx_position = cond_cfg.get('use_tx_position', True)

    # Physics loss config
    physics_cfg = model_cfg.get('physics', {})
    use_physics_losses = physics_cfg.get('enabled', False)
    traj_consistency_weight = physics_cfg.get('trajectory_consistency', {}).get('weight', 0.1)
    coverage_weighted = physics_cfg.get('coverage_weighted', True)
    distance_decay_weight = physics_cfg.get('distance_decay', {}).get('weight', 0.01)

    # Coverage attention config
    coverage_attn_cfg = model_cfg.get('coverage_attention', {})
    use_coverage_attention = coverage_attn_cfg.get('enabled', False)
    coverage_temperature = coverage_attn_cfg.get('temperature', 1.0)

    return DiffusionModule(
        # Model config
        unet_size=unet_size,
        image_size=cfg.data.image.height,
        condition_channels=model_cfg.get('condition_channels', 64),
        # Diffusion config
        num_timesteps=model_cfg.diffusion.num_timesteps,
        beta_schedule=model_cfg.diffusion.beta_schedule,
        prediction_type=model_cfg.diffusion.prediction_type,
        loss_type=model_cfg.diffusion.loss_type,
        # Conditioning flags
        use_building_map=use_building_map,
        use_sparse_rss=use_sparse_rss,
        use_trajectory_mask=use_trajectory_mask,
        use_coverage_density=use_coverage_density,
        use_tx_position=use_tx_position,
        # Physics losses
        use_physics_losses=use_physics_losses,
        trajectory_consistency_weight=traj_consistency_weight,
        coverage_weighted=coverage_weighted,
        distance_decay_weight=distance_decay_weight,
        # Coverage attention
        use_coverage_attention=use_coverage_attention,
        coverage_temperature=coverage_temperature,
        # Training config
        learning_rate=cfg.training.optimizer.lr,
        weight_decay=cfg.training.optimizer.weight_decay,
        warmup_steps=cfg.training.scheduler.warmup_epochs * 100,  # Approximate
        max_steps=cfg.training.max_steps or cfg.training.max_epochs * 1000,
        use_ema=True,
        ema_decay=0.9999,
        # Sampling config
        ddim_steps=model_cfg.diffusion.ddim_steps,
    )


def create_datamodule(cfg: DictConfig) -> RadioMapDataModule:
    """Create data module from config."""
    return RadioMapDataModule(
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


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> float:
    """Main training function."""

    # Print config
    print("=" * 60)
    print("TrajectoryDiff Training")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))

    # Set seed for reproducibility
    L.seed_everything(cfg.experiment.seed, workers=True)

    # Create output directory
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save resolved config
    config_path = output_dir / "config.yaml"
    with open(config_path, 'w') as f:
        OmegaConf.save(cfg, f)
    print(f"\nConfig saved to: {config_path}")

    # Setup data module
    print("\nSetting up data module...")
    datamodule = create_datamodule(cfg)

    # Setup model
    print("Setting up model...")
    model = create_model(cfg)

    # Optional: torch.compile for GPU acceleration
    if cfg.hardware.get('compile', False) and torch.cuda.is_available():
        print("Compiling model with torch.compile (mode=reduce-overhead)...")
        model.model = torch.compile(model.model, mode='reduce-overhead')

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Setup callbacks and loggers
    callbacks = setup_callbacks(cfg)
    loggers = setup_loggers(cfg)

    # Setup trainer
    print("\nSetting up trainer...")
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
        default_root_dir=str(output_dir),
    )

    # Resume from checkpoint if provided
    ckpt_path = cfg.get('ckpt_path', None)

    # Train
    print("\nStarting training...")
    print("=" * 60)
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)

    # Test with best checkpoint
    best_ckpt = trainer.checkpoint_callback.best_model_path
    if best_ckpt:
        print(f"\nTesting with best checkpoint: {best_ckpt}")
        trainer.test(model, datamodule=datamodule, ckpt_path="best")

    # Return best validation metric for hyperparameter optimization
    val_loss = trainer.callback_metrics.get("val/loss", float("inf"))
    print(f"\nBest validation loss: {val_loss}")

    return float(val_loss)


if __name__ == "__main__":
    main()
