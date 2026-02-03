# TrajectoryDiff: Complete Technical Documentation

## Table of Contents
1. [The Problem We're Solving](#the-problem)
2. [The Trajectory Sampling System](#trajectory-sampling)
3. [Model Architecture Deep Dive](#model-architecture)
4. [Training Pipeline](#training-pipeline)
5. [What Makes This Novel](#novelty)
6. [Current Limitations](#limitations)

---

## 1. The Problem We're Solving <a name="the-problem"></a>

### Background: Radio Maps

A **radio map** is a 2D image where each pixel represents the signal strength (RSS) at that location:
- Values are in **dBm** (decibels relative to milliwatt)
- Range: typically -120 dBm (very weak) to 0 dBm (very strong)
- Near transmitter (TX): strong signal (~-30 to -50 dBm)
- Far from TX or behind walls: weak signal (~-80 to -120 dBm)

**Why radio maps matter:**
- Indoor localization (find position from signal strength)
- Network planning (where to place access points)
- Coverage optimization (identify dead zones)

### The Data Collection Problem

**Ideal scenario:** Measure RSS at every location (dense grid)
**Reality:** You can only measure where people actually walk

This creates **sampling bias**:
- Corridors: heavily sampled (people walk here)
- Offices/rooms: moderately sampled
- Restricted areas: never sampled (locked doors, storage)

### What Existing Methods Assume

Most radio map prediction methods assume **uniform random sampling**:
```
"We have RSS measurements at 1% of randomly selected locations"
```

But real crowdsourced data follows **trajectories**:
```
"We have RSS measurements along walking paths through the building"
```

### Our Key Insight

**Trajectory sampling is fundamentally different from uniform sampling:**

| Aspect | Uniform Sampling | Trajectory Sampling |
|--------|-----------------|---------------------|
| Spatial distribution | Random, even | Clustered on paths |
| Temporal correlation | None | Sequential, smooth |
| Coverage bias | None | Avoids restricted areas |
| Blind spots | Random gaps | Systematic gaps (rooms) |

**Our contribution:** A diffusion model that understands WHERE samples came from (trajectories) to better extrapolate into unobserved regions.

---

## 2. The Trajectory Sampling System <a name="trajectory-sampling"></a>

### Location: `src/data/trajectory_sampler.py`

### How Trajectories Are Generated

We simulate realistic walking paths on top of ground-truth radio maps from RadioMapSeer.

#### Step 1: Extract Walkable Areas

```python
class FloorPlanProcessor:
    def get_walkable_mask(self, building_map):
        """
        Convert building map to binary walkable/non-walkable.

        Input: building_map (H, W) - walls are 1, air is 0
        Output: walkable_mask (H, W) - True where you can walk

        Process:
        1. Invert: walls become 0, air becomes 1
        2. Erode slightly: can't walk right against walls
        3. Fill small holes: ignore tiny obstacles
        """
```

#### Step 2: Generate Trajectory Points

We implemented THREE trajectory types:

**A. Shortest Path (A\* Algorithm)**
```python
def generate_shortest_path(self, start, end, walkable_mask):
    """
    Find shortest walkable path between two points.

    Uses A* pathfinding:
    - Start and end are random walkable points
    - Path avoids walls
    - Result: list of (x, y) coordinates

    Problem: Paths are very direct (unrealistic)
    """
```

**B. Random Walk**
```python
def generate_random_walk(self, start, walkable_mask, num_steps):
    """
    Simulate wandering movement.

    At each step:
    1. Sample random direction (with momentum from previous direction)
    2. Take step in that direction
    3. If hit wall, bounce back and try again

    Parameters:
    - momentum: 0.8 (80% chance to continue same direction)
    - step_size: 1-2 pixels

    Problem: Can be too jagged without smoothing
    """
```

**C. Corridor-Biased Walk**
```python
def generate_corridor_biased(self, walkable_mask):
    """
    Prefer walking in corridors (like real humans).

    Uses distance transform:
    - Compute distance from each pixel to nearest wall
    - High distance = likely corridor (wide space)
    - Low distance = near walls (less preferred)

    Sampling probability proportional to distance from walls.
    """
```

#### Step 3: Sample RSS Values

```python
def sample_rss_along_trajectory(self, trajectory_points, radio_map, noise_std=2.0):
    """
    For each trajectory point, sample the ground-truth RSS value.

    Args:
        trajectory_points: list of (x, y) positions
        radio_map: ground truth (H, W) in dBm
        noise_std: measurement noise (typically 2-5 dB)

    Returns:
        list of (x, y, rss) tuples

    Process:
    1. For each (x, y), look up radio_map[y, x]
    2. Add Gaussian noise to simulate real measurements
    3. Return noisy RSS values
    """
```

#### Step 4: Create Sparse Map

```python
def trajectory_to_sparse_map(self, trajectory_points, map_shape):
    """
    Convert trajectory points to image format for model input.

    Creates TWO outputs:

    1. sparse_rss (H, W):
       - Non-zero at trajectory points
       - Value = RSS measurement at that point
       - Zero everywhere else

    2. trajectory_mask (H, W):
       - 1 at trajectory points
       - 0 everywhere else
       - Tells model "we have data here"
    """
```

### Why The Trajectories Might Look Bad

Looking at visualizations, you might see:

**Problem 1: Straight Lines**
- A* shortest paths are too direct
- Real humans meander, explore, backtrack
- **Fix needed:** Add waypoints, use splines for smoothing

**Problem 2: Sparse Coverage**
- Only 100-300 points per trajectory
- On a 256x256 map = less than 0.5% coverage
- **This is intentional:** Real crowdsourcing has sparse data

**Problem 3: Disconnected Segments**
- Multiple trajectories don't form connected network
- Each trajectory is independent walk
- **This is realistic:** Different people walk different paths

**Problem 4: No Smoothing**
- Raw pixel positions look jagged
- Real GPS/WiFi positioning has smooth paths
- **Fix needed:** Apply Gaussian smoothing to paths

### The Coverage Density Map

```python
def compute_coverage_density(trajectory_mask, sigma=5.0):
    """
    Create smooth "confidence map" from trajectory mask.

    Process:
    1. Take binary trajectory_mask
    2. Apply Gaussian blur (sigma=5 pixels)
    3. Result: smooth gradient from trajectory

    Meaning:
    - High value (1.0): on trajectory, high confidence
    - Medium value (0.3): near trajectory, some confidence
    - Low value (0.0): far from trajectory, low confidence

    This tells the model: "trust your predictions more near trajectories"
    """
```

---

## 3. Model Architecture Deep Dive <a name="model-architecture"></a>

### Overview Diagram

```
                        INPUT                              OUTPUT
                          │                                  │
    ┌─────────────────────┼──────────────────────┐          │
    │                     ▼                      │          │
    │   ┌─────────────────────────────────────┐  │          │
    │   │         CONDITION ENCODER           │  │          │
    │   │                                     │  │          │
    │   │  building_map ──┐                   │  │          │
    │   │  sparse_rss ────┼──► CONCAT ──► CNN │  │          │
    │   │  trajectory_mask┘       │          │  │          │
    │   │                         │          │  │          │
    │   │  coverage_density ──────┘          │  │          │
    │   │                                     │  │          │
    │   │  tx_position ──► MLP ──► Spatial   │  │          │
    │   │                         Encoding   │  │          │
    │   │                                     │  │          │
    │   │  Output: condition_features (B,C,H,W)│  │          │
    │   └─────────────────────────────────────┘  │          │
    │                     │                      │          │
    │                     ▼                      │          │
    │   ┌─────────────────────────────────────┐  │          │
    │   │            U-NET BACKBONE           │  │          │
    │   │                                     │  │          │
    │   │   noisy_x ──┬──► ENCODER ──► BOTTLENECK ──► DECODER ──► noise_pred
    │   │             │        │                 │          │
    │   │   time_emb ─┴────────┴─────────────────┘          │
    │   │                                     │  │          │
    │   │   (Skip connections from encoder    │  │          │
    │   │    to decoder at each resolution)   │  │          │
    │   └─────────────────────────────────────┘  │          │
    │                                            │          │
    └────────────────────────────────────────────┘          │
                                                            ▼
                                                     Radio Map
```

### Component 1: Gaussian Diffusion (DDPM)

**Location:** `src/models/diffusion/ddpm.py`

**What is diffusion?**
A way to generate images by learning to remove noise.

**Forward Process (Adding Noise):**
```python
def q_sample(self, x_0, t, noise=None):
    """
    Add noise to clean image x_0 at timestep t.

    x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise

    Where:
    - x_0: clean radio map
    - t: timestep (0 = clean, 1000 = pure noise)
    - alpha_bar_t: cumulative noise schedule
    - noise: random Gaussian noise

    At t=0: x_t ≈ x_0 (almost no noise)
    At t=1000: x_t ≈ noise (pure noise)
    """
```

**Reverse Process (Removing Noise):**
```python
def p_sample(self, model, x_t, t, condition):
    """
    Remove one step of noise from x_t.

    1. Model predicts the noise in x_t
    2. We subtract (scaled) predicted noise
    3. Add small amount of new noise (for stochasticity)

    Repeat 1000 times: pure noise → clean image
    """
```

**Noise Schedules:**
```python
def get_beta_schedule(schedule_type, num_timesteps):
    """
    How much noise to add at each step.

    Linear: beta increases linearly (simple, works okay)
    Cosine: beta follows cosine curve (better for images)
    Sigmoid: beta follows sigmoid (smooth transition)

    We use COSINE for best quality.
    """
```

### Component 2: U-Net Backbone

**Location:** `src/models/diffusion/unet.py`

**Purpose:** Predict the noise in a noisy image.

**Architecture:**
```
Input: noisy_x (B, 1, H, W) + condition (B, C, H, W) → Concatenated (B, 1+C, H, W)

ENCODER (downsample):
    Level 1: 64 channels,  H×W      → ResBlock → ResBlock → Downsample
    Level 2: 128 channels, H/2×W/2  → ResBlock → ResBlock → Downsample
    Level 3: 256 channels, H/4×W/4  → ResBlock → Attention → Downsample
    Level 4: 512 channels, H/8×W/8  → ResBlock → Attention → Downsample

BOTTLENECK:
    Level 5: 512 channels, H/16×W/16 → ResBlock → Attention → ResBlock

DECODER (upsample):
    Level 4: 512→256, H/8×W/8   → Upsample → ResBlock + Skip → Attention
    Level 3: 256→128, H/4×W/4   → Upsample → ResBlock + Skip → Attention
    Level 2: 128→64,  H/2×W/2   → Upsample → ResBlock + Skip
    Level 1: 64→1,    H×W       → Upsample → ResBlock + Skip → Output

Output: predicted_noise (B, 1, H, W)
```

**Key Components:**

**ResidualBlock:**
```python
class ResidualBlock(nn.Module):
    """
    Process features with residual connection.

    x → Conv → GroupNorm → SiLU → Conv → GroupNorm → + → SiLU → output
    │                                                 ↑
    └──────────────── (skip connection) ─────────────┘

    Also incorporates time embedding:
    - time_emb → Linear → Add to features after first conv
    - This tells the network "how noisy is this image"
    """
```

**AttentionBlock:**
```python
class AttentionBlock(nn.Module):
    """
    Self-attention at low resolutions (16×16, 8×8).

    Allows long-range dependencies:
    - A pixel at top-left can attend to bottom-right
    - Important for global structure (walls affect entire room)

    We use multi-head attention with 4-8 heads.
    """
```

**Skip Connections:**
```python
# In decoder, concatenate encoder features
h = torch.cat([h, skip], dim=1)  # Double channels
h = conv(h)  # Reduce back to normal
```

### Component 3: Condition Encoder

**Location:** `src/models/encoders/condition_encoder.py`

**Purpose:** Process all conditioning inputs into a single feature map.

```python
class ConditionEncoder(nn.Module):
    """
    Fuse multiple conditioning signals.

    Inputs:
        building_map:     (B, 1, H, W) - where walls are
        sparse_rss:       (B, 1, H, W) - observed RSS values
        trajectory_mask:  (B, 1, H, W) - where we have data
        coverage_density: (B, 1, H, W) - confidence map
        tx_position:      (B, 2)       - transmitter x,y

    Process:
        1. Concatenate spatial inputs: (B, 4, H, W)
        2. Apply CNN to extract features
        3. Encode TX position as spatial map
        4. Add TX position features to spatial features
        5. Output: (B, condition_channels, H, W)
    """
```

**TX Position Encoder:**
```python
class TxPositionEncoder(nn.Module):
    """
    Encode transmitter position as spatial feature map.

    Input: tx_position (B, 2) - normalized (x, y) in [0, 1]

    Process:
    1. Create coordinate grid: (H, W, 2)
    2. Compute distance from TX to each pixel
    3. Compute angle from TX to each pixel
    4. Encode with MLP to get features
    5. Broadcast to (B, channels, H, W)

    Why? Distance from TX is crucial for signal propagation.
    """
```

### Component 4: TrajectoryConditionedUNet

**Location:** `src/models/encoders/condition_encoder.py`

**The complete model:**
```python
class TrajectoryConditionedUNet(nn.Module):
    """
    Full model combining everything.

    Forward pass:
    1. Encode conditions: building_map, sparse_rss, etc.
    2. Concatenate condition with noisy input
    3. Get time embedding for current noise level
    4. Run through U-Net
    5. Output: predicted noise
    """

    def forward(self, x, t, building_map, sparse_rss, trajectory_mask,
                coverage_density, tx_position):
        # Encode all conditions
        condition = self.condition_encoder(
            building_map, sparse_rss, trajectory_mask,
            coverage_density, tx_position
        )

        # Concatenate with noisy input
        x_cond = torch.cat([x, condition], dim=1)

        # Get time embedding
        t_emb = self.time_mlp(t)

        # U-Net forward
        noise_pred = self.unet(x_cond, t_emb)

        return noise_pred
```

### Component 5: DDIM Sampler

**Location:** `src/models/diffusion/ddpm.py`

**Problem:** DDPM needs 1000 steps to generate an image (slow!)

**Solution:** DDIM uses only 50 steps with same quality.

```python
class DDIMSampler:
    """
    Denoising Diffusion Implicit Models - fast sampling.

    Instead of 1000 steps, use 50 evenly spaced steps.
    Uses deterministic updates (no random noise added).

    Speed: 20x faster than DDPM
    Quality: Nearly identical to DDPM
    """

    def sample(self, model, shape, condition):
        # Start with pure noise
        x = torch.randn(shape)

        # 50 denoising steps instead of 1000
        for t in [980, 960, 940, ..., 20, 0]:
            noise_pred = model(x, t, condition)
            x = self.ddim_step(x, t, noise_pred)

        return x
```

---

## 4. Training Pipeline <a name="training-pipeline"></a>

### Location: `src/training/diffusion_module.py`

### Training Step

```python
def training_step(self, batch, batch_idx):
    """
    One training iteration.

    1. Get clean radio map from batch
    2. Sample random timestep t
    3. Add noise to get x_t
    4. Model predicts the noise
    5. Loss = MSE between predicted and actual noise
    """
    x_0 = batch['radio_map']  # Clean image

    # Random timestep for each sample
    t = torch.randint(0, 1000, (batch_size,))

    # Add noise
    noise = torch.randn_like(x_0)
    x_t = self.diffusion.q_sample(x_0, t, noise)

    # Predict noise
    condition = self._extract_condition(batch)
    noise_pred = self.model(x_t, t, **condition)

    # Loss
    loss = F.mse_loss(noise_pred, noise)

    return loss
```

### EMA (Exponential Moving Average)

```python
def _update_ema(self):
    """
    Maintain slow-moving average of model weights.

    EMA weights = 0.9999 * EMA + 0.0001 * current

    Why? EMA produces smoother, more stable samples.
    We use EMA model for inference, regular model for training.
    """
```

### Learning Rate Schedule

```python
# Warmup: 0 → learning_rate over 1000 steps
# Cosine decay: learning_rate → 0.01*learning_rate over remaining steps

warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=1000)
cosine_scheduler = CosineAnnealingLR(optimizer, T_max=remaining_steps)
scheduler = SequentialLR([warmup_scheduler, cosine_scheduler])
```

---

## 5. What Makes This Novel <a name="novelty"></a>

### Novelty 1: Trajectory-Aware Conditioning

**Previous work:** Conditions on sparse measurements as independent points.

**Our approach:** Conditions on trajectory structure:
- `trajectory_mask`: Where did we walk?
- `coverage_density`: How confident should we be at each location?

This lets the model understand: "I have data HERE, so I'm confident here, but uncertain THERE."

### Novelty 2: Explicit Uncertainty Through Sampling

**Previous work:** Single prediction, no uncertainty.

**Our approach:** Generate multiple samples, compute variance.
- High variance = model is uncertain (probably a blind spot)
- Low variance = model is confident (on trajectory)

### Novelty 3: TX Position Encoding

**Previous work:** TX location not explicitly modeled.

**Our approach:** Encode TX position as spatial features:
- Distance from TX at each pixel
- Angle from TX at each pixel

This helps model the physics: signal weakens with distance from TX.

### Novelty 4: Combined Conditioning

We fuse FIVE conditioning signals:
1. Building map (where are walls?)
2. Sparse RSS (what values did we observe?)
3. Trajectory mask (where did we walk?)
4. Coverage density (how densely sampled?)
5. TX position (where is transmitter?)

No previous work combines all of these.

---

## 6. Current Limitations <a name="limitations"></a>

### Limitation 1: Synthetic Trajectories

We SIMULATE trajectories on ground-truth data. This may not perfectly match real human movement patterns.

**Mitigation:** Implement more realistic trajectory models (Bezier curves, real GPS traces).

### Limitation 2: Single TX

Current model handles one transmitter at a time. Real buildings have multiple access points.

**Mitigation:** Future work could extend to multi-TX prediction.

### Limitation 3: No Real-World Validation

All experiments are on RadioMapSeer (simulation). Real indoor environments may differ.

**Mitigation:** Collect small real-world dataset for validation.

### Limitation 4: Computational Cost

Diffusion models are slow:
- Training: 10-20 hours on GPU
- Inference: 1-2 seconds per sample (with DDIM)

**Mitigation:** DDIM already helps. Could explore consistency models for faster sampling.

---

## Quick Reference: File Locations

| Component | File |
|-----------|------|
| Trajectory sampling | `src/data/trajectory_sampler.py` |
| Data module | `src/data/datamodule.py` |
| DDPM/DDIM | `src/models/diffusion/ddpm.py` |
| U-Net | `src/models/diffusion/unet.py` |
| Condition encoder | `src/models/encoders/condition_encoder.py` |
| Training module | `src/training/diffusion_module.py` |
| Inference | `src/training/inference.py` |
| Metrics | `src/evaluation/metrics.py` |
| Train script | `scripts/train.py` |
| Eval script | `scripts/evaluate.py` |
