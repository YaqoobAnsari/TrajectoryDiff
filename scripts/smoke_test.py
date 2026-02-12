#!/usr/bin/env python3
"""
Quick smoke test: 3 epochs with 2 maps to verify the full pipeline works.
No Hydra, no file scanning overhead — just direct instantiation.

Usage:
    python scripts/smoke_test.py
    python scripts/smoke_test.py --coverage-attention --physics-losses
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import time

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from data.dataset import RadioMapDataset
from training.diffusion_module import DiffusionModule


def main():
    parser = argparse.ArgumentParser(description="TrajectoryDiff Smoke Test")
    parser.add_argument("--coverage-attention", action="store_true", help="Enable coverage attention")
    parser.add_argument("--physics-losses", action="store_true", help="Enable physics losses")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--num-maps", type=int, default=2, help="Number of maps to use")
    args = parser.parse_args()

    data_dir = Path(__file__).parent.parent / "data" / "raw"
    print(f"Data dir: {data_dir}")
    print(f"Coverage attention: {args.coverage_attention}")
    print(f"Physics losses: {args.physics_losses}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}")

    # Use only a few maps for speed
    map_ids = list(range(args.num_maps))

    print(f"\nCreating dataset with {args.num_maps} maps...")
    t0 = time.time()
    train_dataset = RadioMapDataset(
        data_dir=data_dir,
        map_ids=map_ids,
        variant="IRT2",
        num_trajectories=2,
        points_per_trajectory=50,
        trajectory_method="random_walk",  # fastest method
        seed=42,
    )
    print(f"  Dataset created: {len(train_dataset)} samples in {time.time() - t0:.1f}s")

    val_dataset = RadioMapDataset(
        data_dir=data_dir,
        map_ids=map_ids,
        variant="IRT2",
        num_trajectories=2,
        points_per_trajectory=50,
        trajectory_method="random_walk",
        seed=123,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Test one batch loads correctly
    print("\nLoading first batch...")
    t0 = time.time()
    batch = next(iter(train_loader))
    print(f"  First batch loaded in {time.time() - t0:.1f}s")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={v.shape}, range=[{v.min():.2f}, {v.max():.2f}]")

    # Create model
    print("\nCreating model...")
    model = DiffusionModule(
        unet_size="small",
        image_size=256,
        condition_channels=32,
        num_timesteps=100,  # Fewer steps for speed
        beta_schedule="cosine",
        prediction_type="epsilon",
        loss_type="mse",
        use_building_map=True,
        use_sparse_rss=True,
        use_trajectory_mask=True,
        use_coverage_density=True,
        use_tx_position=True,
        use_physics_losses=args.physics_losses,
        trajectory_consistency_weight=0.1,
        coverage_weighted=True,
        distance_decay_weight=0.01,
        use_coverage_attention=args.coverage_attention,
        coverage_temperature=1.0,
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_epochs=1,
        use_ema=True,
        ema_decay=0.999,
        ddim_steps=10,  # Fewer DDIM steps
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {total_params:,}")

    # Train
    print(f"\nTraining for {args.epochs} epochs...")
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        precision=32,
        max_epochs=args.epochs,
        log_every_n_steps=1,
        enable_checkpointing=True,
        enable_progress_bar=True,
        callbacks=[
            ModelCheckpoint(
                dirpath="experiments/smoke_test",
                filename="smoke-{epoch:02d}-{val/loss:.4f}",
                monitor="val/loss",
                mode="min",
                save_last=True,
                auto_insert_metric_name=False,
            ),
        ],
        logger=False,
        gradient_clip_val=1.0,
    )

    t0 = time.time()
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    elapsed = time.time() - t0
    print(f"\nTraining completed in {elapsed:.1f}s")

    # Check results
    val_loss = trainer.callback_metrics.get("val/loss")
    train_loss = trainer.callback_metrics.get("train/loss")
    print(f"  Final train loss: {train_loss}")
    print(f"  Final val loss: {val_loss}")

    # Quick DDIM sampling test
    print("\nTesting DDIM sampling...")
    model.eval()
    with torch.no_grad():
        sample_batch = next(iter(val_loader))
        condition = {
            k: v for k, v in sample_batch.items()
            if k != "radio_map" and k != "metadata" and isinstance(v, torch.Tensor)
        }
        samples = model.sample(condition=condition, use_ddim=True, progress=True)
        print(f"  Samples shape: {samples.shape}")
        print(f"  Samples range: [{samples.min():.2f}, {samples.max():.2f}]")

    print("\n" + "=" * 50)
    print("SMOKE TEST PASSED!")
    print("=" * 50)


if __name__ == "__main__":
    main()
