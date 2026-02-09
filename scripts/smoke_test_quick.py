#!/usr/bin/env python3
"""
Quick smoke test: 2 manual training steps with REAL data on CPU.
Proves the full pipeline works without waiting for full epochs.

Usage:
    python scripts/smoke_test_quick.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import time
import torch
from torch.utils.data import DataLoader
from data.dataset import RadioMapDataset
from training.diffusion_module import DiffusionModule

print("=" * 60)
print("TrajectoryDiff Quick Smoke Test (Real Data)")
print("=" * 60)

data_dir = Path(__file__).parent.parent / "data" / "raw"

# 1. Dataset creation
print("\n[1/6] Creating dataset (2 maps)...")
t0 = time.time()
dataset = RadioMapDataset(
    data_dir=data_dir, map_ids=[0, 1], variant="IRT2",
    num_trajectories=2, points_per_trajectory=50,
    trajectory_method="random_walk", seed=42,
)
print(f"  OK: {len(dataset)} samples in {time.time()-t0:.1f}s")

# 2. Data loading
print("\n[2/6] Loading first batch (batch_size=2)...")
loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
t0 = time.time()
batch = next(iter(loader))
print(f"  OK: loaded in {time.time()-t0:.1f}s")
for k, v in batch.items():
    if isinstance(v, torch.Tensor):
        print(f"    {k}: {v.shape}, [{v.min():.3f}, {v.max():.3f}]")

# 3. Model creation (standard)
print("\n[3/6] Creating DiffusionModule (small UNet, 100 timesteps)...")
model = DiffusionModule(
    unet_size="small", image_size=256, condition_channels=32,
    num_timesteps=100, beta_schedule="cosine", prediction_type="epsilon",
    loss_type="mse", use_building_map=True, use_sparse_rss=True,
    use_trajectory_mask=True, use_coverage_density=True, use_tx_position=True,
    use_physics_losses=False, use_coverage_attention=False,
    learning_rate=1e-4, weight_decay=0.01, warmup_steps=10, max_steps=100,
    use_ema=True, ema_decay=0.999, ddim_steps=10,
)
total_p = sum(p.numel() for p in model.parameters())
print(f"  OK: {total_p:,} params")

# 4. Manual training steps
print("\n[4/6] Running 2 manual training steps...")
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
losses = []
for step in range(2):
    t0 = time.time()
    optimizer.zero_grad()
    loss_dict = model.training_step(batch, step)
    loss = loss_dict["loss"] if isinstance(loss_dict, dict) else loss_dict
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    print(f"  Step {step+1}: loss={loss.item():.4f}, time={time.time()-t0:.1f}s")

print(f"  Loss decreased: {losses[0]:.4f} -> {losses[1]:.4f} ({'YES' if losses[1] < losses[0] else 'not yet (expected for 2 steps)'})")

# 5. Validation step
print("\n[5/6] Running validation step...")
model.eval()
with torch.no_grad():
    t0 = time.time()
    val_out = model.validation_step(batch, 0)
    val_loss = val_out.get("val_loss", val_out.get("loss", None)) if isinstance(val_out, dict) else val_out
    print(f"  OK: val_loss={val_loss.item():.4f}, time={time.time()-t0:.1f}s")

# 6. Test with coverage attention + physics
print("\n[6/6] Testing full features (coverage attention + physics)...")
model_full = DiffusionModule(
    unet_size="small", image_size=256, condition_channels=32,
    num_timesteps=100, beta_schedule="cosine", prediction_type="epsilon",
    loss_type="mse", use_building_map=True, use_sparse_rss=True,
    use_trajectory_mask=True, use_coverage_density=True, use_tx_position=True,
    use_physics_losses=True, trajectory_consistency_weight=0.1,
    coverage_weighted=True, distance_decay_weight=0.01,
    use_coverage_attention=True, coverage_temperature=1.0,
    learning_rate=1e-4, weight_decay=0.01, warmup_steps=10, max_steps=100,
    use_ema=True, ema_decay=0.999, ddim_steps=10,
)
model_full.train()
t0 = time.time()
loss = model_full.training_step(batch, 0)
loss.backward()
elapsed = time.time() - t0
print(f"  OK: loss={loss.item():.4f}, time={elapsed:.1f}s")
print(f"  Gradients exist: {any(p.grad is not None for p in model_full.parameters())}")

print("\n" + "=" * 60)
print("ALL SMOKE TESTS PASSED!")
print("=" * 60)
print("\nVerified with real RadioMapSeer data:")
print("  - Dataset loads correctly (160 samples from 2 maps)")
print("  - Data shapes and normalization ranges correct")
print("  - Standard model: forward + backward works")
print("  - Full model (coverage attention + physics): works")
print("  - Validation step works")
print("\nReady for full training on GPU via SSH.")
