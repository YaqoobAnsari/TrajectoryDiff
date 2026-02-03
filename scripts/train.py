#!/usr/bin/env python3
"""
TrajectoryDiff Training Script

Supports multiple model types:
    - diffusion (default): TrajectoryDiff diffusion model
    - supervised: SupervisedUNetBaseline (same arch, no diffusion)
    - radio_unet: RadioUNet baseline (Levie et al., 2021)
    - rmdm: RMDM baseline (Xu et al., 2025)

Usage:
    python scripts/train.py                           # Default config
    python scripts/train.py experiment=trajectory_baseline  # Named experiment
    python scripts/train.py experiment=supervised_unet      # Supervised baseline
    python scripts/train.py experiment=radio_unet           # RadioUNet baseline
    python scripts/train.py experiment=rmdm_baseline        # RMDM baseline
    python scripts/train.py data.loader.batch_size=32       # Override params
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


def setup_callbacks(cfg: DictConfig, model_type: str = 'diffusion') -> list:
    """Setup training callbacks.

    Args:
        cfg: Full config.
        model_type: Model type — diffusion-specific callbacks are skipped
            for non-diffusion models (supervised, radio_unet).
    """
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

    # Diffusion-specific callbacks (require DiffusionModule APIs like .sample(),
    # ._extract_condition(), .model attribute).  Skip for non-diffusion models.
    is_diffusion = model_type in ('diffusion', 'rmdm')

    if is_diffusion:
        # Gradient monitoring (accesses pl_module.model)
        callbacks.append(GradientMonitor(log_every_n_steps=100))

        # Sample logging (if W&B enabled) — calls pl_module.sample()
        if cfg.logging.wandb.enabled:
            callbacks.append(
                WandBSampleLogger(
                    every_n_epochs=5,
                    num_samples=4,
                    use_ddim=True,
                )
            )

        # Metrics computation (calls pl_module.sample())
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


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

def _create_diffusion_model(cfg: DictConfig) -> DiffusionModule:
    """Create diffusion model (default)."""
    model_cfg = cfg.model
    unet_size = model_cfg.unet.get('size', 'medium')

    cond_cfg = model_cfg.get('conditioning', {})
    physics_cfg = model_cfg.get('physics', {})
    coverage_attn_cfg = model_cfg.get('coverage_attention', {})

    return DiffusionModule(
        unet_size=unet_size,
        image_size=cfg.data.image.height,
        condition_channels=model_cfg.get('condition_channels', 64),
        num_timesteps=model_cfg.diffusion.num_timesteps,
        beta_schedule=model_cfg.diffusion.beta_schedule,
        prediction_type=model_cfg.diffusion.prediction_type,
        loss_type=model_cfg.diffusion.loss_type,
        use_building_map=cond_cfg.get('use_building_map', True),
        use_sparse_rss=cond_cfg.get('use_sparse_rss', True),
        use_trajectory_mask=cond_cfg.get('use_trajectory_mask', True),
        use_coverage_density=cond_cfg.get('use_coverage_density', True),
        use_tx_position=cond_cfg.get('use_tx_position', True),
        use_physics_losses=physics_cfg.get('enabled', False),
        trajectory_consistency_weight=physics_cfg.get('trajectory_consistency', {}).get('weight', 0.1),
        coverage_weighted=physics_cfg.get('coverage_weighted', True),
        distance_decay_weight=physics_cfg.get('distance_decay', {}).get('weight', 0.01),
        physics_warmup_epochs=physics_cfg.get('warmup_epochs', 0),
        physics_rampup_epochs=physics_cfg.get('rampup_epochs', 10),
        use_coverage_attention=coverage_attn_cfg.get('enabled', False),
        coverage_temperature=coverage_attn_cfg.get('temperature', 1.0),
        learning_rate=cfg.training.optimizer.lr,
        weight_decay=cfg.training.optimizer.weight_decay,
        warmup_epochs=cfg.training.scheduler.warmup_epochs,
        use_ema=True,
        ema_decay=0.9999,
        ddim_steps=model_cfg.diffusion.ddim_steps,
    )


def _create_supervised_model(cfg: DictConfig) -> L.LightningModule:
    """Create supervised UNet baseline."""
    from models.baselines import SupervisedUNetBaseline

    model_cfg = cfg.model
    cond_cfg = model_cfg.get('conditioning', {})

    return SupervisedUNetBaseline(
        unet_size=model_cfg.unet.get('size', 'medium'),
        image_size=cfg.data.image.height,
        condition_channels=model_cfg.get('condition_channels', 64),
        use_building_map=cond_cfg.get('use_building_map', True),
        use_sparse_rss=cond_cfg.get('use_sparse_rss', True),
        use_trajectory_mask=cond_cfg.get('use_trajectory_mask', True),
        use_coverage_density=cond_cfg.get('use_coverage_density', True),
        use_tx_position=cond_cfg.get('use_tx_position', True),
        learning_rate=cfg.training.optimizer.lr,
        weight_decay=cfg.training.optimizer.weight_decay,
        warmup_epochs=cfg.training.scheduler.warmup_epochs,
    )


def _create_radio_unet_model(cfg: DictConfig) -> L.LightningModule:
    """Create RadioUNet baseline."""
    from models.baselines import RadioUNetBaseline

    return RadioUNetBaseline(
        image_size=cfg.data.image.height,
        learning_rate=cfg.training.optimizer.lr,
        weight_decay=cfg.training.optimizer.weight_decay,
        warmup_epochs=cfg.training.scheduler.warmup_epochs,
    )


def _create_rmdm_model(cfg: DictConfig) -> L.LightningModule:
    """Create RMDM baseline."""
    from models.baselines import RMDMBaseline

    model_cfg = cfg.model

    return RMDMBaseline(
        image_size=cfg.data.image.height,
        num_timesteps=model_cfg.diffusion.num_timesteps,
        beta_schedule=model_cfg.diffusion.beta_schedule,
        prediction_type=model_cfg.diffusion.prediction_type,
        ddim_steps=model_cfg.diffusion.ddim_steps,
        learning_rate=cfg.training.optimizer.lr,
        weight_decay=cfg.training.optimizer.weight_decay,
        warmup_epochs=cfg.training.scheduler.warmup_epochs,
    )


def create_model(cfg: DictConfig) -> L.LightningModule:
    """Create model from config based on model_type."""
    model_type = cfg.model.get('model_type', 'diffusion')

    factories = {
        'diffusion': _create_diffusion_model,
        'supervised': _create_supervised_model,
        'radio_unet': _create_radio_unet_model,
        'rmdm': _create_rmdm_model,
    }

    if model_type not in factories:
        raise ValueError(
            f"Unknown model_type: {model_type}. "
            f"Choose from {list(factories.keys())}"
        )

    print(f"Model type: {model_type}")
    return factories[model_type](cfg)


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

    # Use TF32 for float32 matmuls (significant speedup on H200 Tensor Cores)
    torch.set_float32_matmul_precision('medium')

    model_type = cfg.model.get('model_type', 'diffusion')

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

    # Optional: torch.compile for GPU acceleration (diffusion model only)
    if cfg.hardware.get('compile', False) and torch.cuda.is_available():
        if model_type == 'diffusion' and hasattr(model, 'model'):
            print("Compiling model with torch.compile (mode=reduce-overhead)...")
            model.model = torch.compile(model.model, mode='reduce-overhead')

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Setup callbacks and loggers
    callbacks = setup_callbacks(cfg, model_type=model_type)
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
