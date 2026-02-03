# TrajectoryDiff Research Plan

## Executive Summary

**Problem**: Existing radio map diffusion methods assume uniformly random samples. Real-world crowdsourced data follows trajectory patterns with systematic spatial bias.

**Solution**: TrajectoryDiff — a trajectory-conditioned diffusion model that explicitly models WHERE samples come from, enabling proper extrapolation into unobserved regions with calibrated uncertainty.

**Novelty Claims**:
1. First diffusion model for trajectory-to-map generation (vs. random-sparse-to-dense)
2. First to explicitly model spatial sampling bias in radio map prediction
3. Novel trajectory encoder capturing temporal correlation and coverage patterns
4. Uncertainty-aware generation with proper calibration for blind spots

---

## Phase 0: Environment Setup (Week 0)

### Goals
- [ ] Repository structure created
- [ ] Environment configured (conda/pip)
- [ ] RadioMapSeer dataset downloaded and explored
- [ ] Basic data loading working

### Tasks

#### 0.1 Repository Setup
```bash
# Create repo structure
mkdir -p trajectorydiff/{configs,src,scripts,notebooks,tests,docs,data}
mkdir -p trajectorydiff/src/{data,models,training,evaluation,utils}
mkdir -p trajectorydiff/configs/{model,data,training,experiment}
```

#### 0.2 Environment
```yaml
# environment.yaml
name: trajdiff
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  - python=3.10
  - pytorch=2.1
  - pytorch-cuda=12.1
  - torchvision
  - lightning=2.1
  - hydra-core=1.3
  - omegaconf
  - wandb
  - numpy
  - scipy
  - scikit-image
  - matplotlib
  - seaborn
  - pandas
  - h5py
  - tqdm
  - pytest
  - black
  - ruff
  - mypy
  - pip
  - pip:
    - einops
    - timm
```

#### 0.3 Dataset Download
- Download RadioMapSeer from IEEE DataPort
- Verify data integrity
- Document data format and structure

### Deliverables
- [ ] Working conda environment
- [ ] Data loading script that can visualize sample radio maps
- [ ] Understanding of RadioMapSeer data format

---

## Phase 1: Data Pipeline & Trajectory Simulation (Weeks 1-2)

### Goals
- [ ] Trajectory sampling algorithms implemented
- [ ] Multiple trajectory types (shortest-path, random-walk, corridor-biased)
- [ ] Data augmentation pipeline
- [ ] Visualization tools for trajectories + sparse samples

### Technical Specification

#### 1.1 Floor Plan Processing

**Input**: RGB floor plan image from RadioMapSeer
**Output**: Binary free-space mask + material map

```python
class FloorPlanProcessor:
    """Process raw floor plans into usable formats."""
    
    def extract_free_space_mask(self, floor_plan: np.ndarray) -> np.ndarray:
        """
        Args:
            floor_plan: (H, W, 3) RGB image
        Returns:
            mask: (H, W) binary, 1 = walkable, 0 = wall/obstacle
        """
        pass
    
    def extract_material_map(self, floor_plan: np.ndarray) -> np.ndarray:
        """
        Returns:
            materials: (H, W) categorical, 0=air, 1=drywall, 2=concrete, etc.
        """
        pass
```

#### 1.2 Trajectory Generation

**Core Interface**:
```python
@dataclass
class TrajectoryPoint:
    t: float      # timestamp (seconds)
    x: float      # x coordinate (meters or pixels)
    y: float      # y coordinate
    rss: float    # RSS value (dBm)

Trajectory = List[TrajectoryPoint]

class TrajectoryGenerator(ABC):
    """Base class for trajectory generation strategies."""
    
    @abstractmethod
    def generate(
        self,
        free_space_mask: np.ndarray,
        radio_map: np.ndarray,
        n_points: int,
        **kwargs
    ) -> Trajectory:
        """Generate a single trajectory."""
        pass
```

**Trajectory Types**:

1. **ShortestPathTrajectory**
   - Sample random start/end in free space
   - Use A* to find path
   - Sample points along path at fixed interval
   - Add position noise (Gaussian, σ=0.5m)
   - Query RSS from radio map + measurement noise (Gaussian, σ=2dB)

2. **RandomWalkTrajectory**
   - Start at random free-space point
   - At each step: sample direction with momentum
   - Reject steps that hit walls
   - Continue for N steps
   - Natural corridor-following behavior emerges

3. **CorridorBiasedTrajectory**
   - Precompute "corridor score" for each pixel (distance transform from walls)
   - Bias sampling toward high-score regions
   - Simulate realistic human preference for corridors

#### 1.3 Sparse Sample Representation

**From Trajectory to Sparse Map**:
```python
def trajectory_to_sparse_input(
    trajectory: Trajectory,
    map_shape: Tuple[int, int],
    resolution: float  # meters per pixel
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Args:
        trajectory: List of (t, x, y, rss) points
        map_shape: (H, W) of output
        resolution: meters per pixel
    
    Returns:
        sparse_rss: (H, W) RSS values, 0 where no measurement
        mask: (H, W) binary, 1 where measurement exists
    """
    sparse_rss = np.zeros(map_shape)
    mask = np.zeros(map_shape)
    
    for point in trajectory:
        px, py = int(point.x / resolution), int(point.y / resolution)
        if 0 <= px < map_shape[1] and 0 <= py < map_shape[0]:
            sparse_rss[py, px] = point.rss
            mask[py, px] = 1.0
    
    return sparse_rss, mask
```

**Coverage Density Map** (for conditioning):
```python
def compute_coverage_density(
    mask: np.ndarray,
    sigma: float = 5.0  # pixels
) -> np.ndarray:
    """
    Gaussian-smoothed coverage map showing sample density.
    High values = lots of nearby samples (confident region)
    Low values = sparse/no samples (uncertain region)
    """
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(mask.astype(float), sigma=sigma)
```

#### 1.4 Data Augmentation

```python
class RadioMapAugmentation:
    """Augmentations that preserve physics consistency."""
    
    def random_rotation(self, floor_plan, radio_map, trajectory, angle: int):
        """Rotate by 90/180/270 degrees."""
        pass
    
    def random_flip(self, floor_plan, radio_map, trajectory, horizontal: bool):
        """Horizontal or vertical flip."""
        pass
    
    def random_crop(self, floor_plan, radio_map, trajectory, crop_size: int):
        """Random crop (trajectory points outside crop are dropped)."""
        pass
    
    # NOTE: We do NOT scale RSS values - they are physics quantities
```

### Deliverables
- [ ] `src/data/trajectory_sampler.py` with all trajectory types
- [ ] `src/data/transforms.py` with augmentations
- [ ] Visualization notebook showing trajectories on floor plans
- [ ] Statistics: coverage distribution, sample density analysis

---

## Phase 2: Baseline Models (Weeks 3-4)

### Goals
- [ ] Implement non-learning baselines (interpolation)
- [ ] Implement standard diffusion baseline (ignores trajectory structure)
- [ ] Establish performance benchmarks on RadioMapSeer

### Baselines to Implement

#### 2.1 Classical Interpolation

```python
class InterpolationBaseline:
    """Classical interpolation methods."""
    
    def kriging(self, sparse_rss, mask, floor_plan):
        """Gaussian process interpolation."""
        pass
    
    def idw(self, sparse_rss, mask, floor_plan, power=2):
        """Inverse distance weighting."""
        pass
    
    def rbf(self, sparse_rss, mask, floor_plan, kernel='thin_plate'):
        """Radial basis function interpolation."""
        pass
```

#### 2.2 U-Net Regression Baseline

Simple supervised learning baseline:
```
Input: floor_plan (3, H, W) + sparse_rss (1, H, W) + mask (1, H, W)
       → Concatenate → (5, H, W)
Output: full_map (1, H, W)
Loss: MSE on full map
```

#### 2.3 Standard Diffusion Baseline

DDPM-style diffusion that treats sparse samples as conditioning:
```
Input: floor_plan + sparse_rss + mask (concatenated as condition)
Output: denoised radio map
Process: Standard forward/reverse diffusion
Note: This baseline ignores trajectory structure (treats samples as independent)
```

### Evaluation Protocol

**Metrics**:
```python
def evaluate_radio_map(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray):
    """
    Args:
        pred: (H, W) predicted radio map
        gt: (H, W) ground truth radio map
        mask: (H, W) trajectory sampling mask
    
    Returns:
        dict with:
        - rmse_overall: RMSE on all pixels
        - rmse_observed: RMSE on pixels with samples (interpolation)
        - rmse_unobserved: RMSE on pixels without samples (extrapolation)
        - ssim: Structural similarity
        - coverage_weighted_rmse: Higher weight on unobserved regions
    """
```

**Split Strategy**:
- Train: 70% of buildings
- Val: 15% of buildings  
- Test: 15% of buildings
- Within each split, generate multiple trajectories per radio map

### Deliverables
- [ ] `src/models/baselines/` with all baseline implementations
- [ ] Baseline results table on RadioMapSeer
- [ ] Analysis: Where do baselines fail? (likely: extrapolation into blind spots)

---

## Phase 3: TrajectoryDiff Core Model (Weeks 5-7)

### Goals
- [ ] Trajectory encoder design and implementation
- [ ] Trajectory-conditioned diffusion architecture
- [ ] Training pipeline with proper conditioning

### Architecture Overview

```
                                    ┌─────────────────────┐
                                    │   Trajectory Encoder │
                                    │   (GNN/Transformer)  │
                                    └──────────┬──────────┘
                                               │
                                    trajectory_embedding
                                               │
                                               ▼
┌───────────┐   ┌───────────┐   ┌─────────────────────────────┐
│ Floor Plan│ → │  Encoder  │ → │                             │
└───────────┘   └───────────┘   │    Diffusion U-Net          │
                                │    with Cross-Attention     │ → Predicted Map
┌───────────┐   ┌───────────┐   │    to Trajectory Embedding  │
│Sparse RSS │ → │  Encoder  │ → │                             │
└───────────┘   └───────────┘   └─────────────────────────────┘
                                               ▲
                                               │
                                    ┌──────────┴──────────┐
                                    │  Noise Level (t)     │
                                    │  Time Embedding      │
                                    └─────────────────────┘
```

### 3.1 Trajectory Encoder

**Why we need this**: Standard diffusion conditions on images (sparse RSS map). But this loses:
- Temporal order of samples
- Path geometry (which direction was walked)
- Coverage patterns (where is dense vs. sparse)

**Design Options**:

**Option A: Graph Neural Network**
```python
class TrajectoryGNN(nn.Module):
    """
    Encode trajectory as a graph where:
    - Nodes = trajectory points (x, y, rss)
    - Edges = consecutive points on path
    """
    def __init__(self, hidden_dim=256, num_layers=4):
        self.node_encoder = MLP([3, hidden_dim, hidden_dim])  # (x, y, rss) → features
        self.gnn_layers = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.global_pool = GlobalAttentionPool(hidden_dim)
    
    def forward(self, trajectory_points, edge_index):
        # trajectory_points: (N, 3) - (x, y, rss) for each point
        # edge_index: (2, E) - edges between consecutive points
        x = self.node_encoder(trajectory_points)
        for layer in self.gnn_layers:
            x = layer(x, edge_index) + x  # residual
        return self.global_pool(x)  # (hidden_dim,) global embedding
```

**Option B: Transformer on Point Sequence**
```python
class TrajectoryTransformer(nn.Module):
    """
    Encode trajectory as a sequence with positional encoding.
    """
    def __init__(self, hidden_dim=256, num_layers=4, num_heads=8):
        self.point_encoder = MLP([3, hidden_dim, hidden_dim])
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, num_heads),
            num_layers
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
    
    def forward(self, trajectory_points):
        # trajectory_points: (B, N, 3) - batched trajectories
        x = self.point_encoder(trajectory_points)  # (B, N, D)
        x = x + self.pos_encoding(torch.arange(x.shape[1]))
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)  # prepend CLS token
        x = self.transformer(x)
        return x[:, 0]  # CLS token as global embedding
```

**Option C: Hybrid - Spatial + Sequential**
```python
class HybridTrajectoryEncoder(nn.Module):
    """
    Combine:
    1. Sparse map encoding (spatial info)
    2. Sequence encoding (temporal/path info)
    3. Coverage density encoding (where data exists)
    """
    def __init__(self, hidden_dim=256):
        self.sparse_encoder = ConvEncoder()  # Process sparse RSS map
        self.coverage_encoder = ConvEncoder()  # Process coverage density
        self.sequence_encoder = TrajectoryTransformer()  # Process point sequence
        self.fusion = CrossAttentionFusion()
    
    def forward(self, sparse_rss, mask, coverage_density, trajectory_points):
        spatial_feat = self.sparse_encoder(sparse_rss, mask)
        coverage_feat = self.coverage_encoder(coverage_density)
        sequence_feat = self.sequence_encoder(trajectory_points)
        return self.fusion(spatial_feat, coverage_feat, sequence_feat)
```

### 3.2 Diffusion Model

**Base**: Standard DDPM/DDIM with U-Net backbone

**Modifications for trajectory conditioning**:

```python
class TrajectoryDiffusion(nn.Module):
    def __init__(
        self,
        unet: UNet,
        trajectory_encoder: TrajectoryEncoder,
        num_timesteps: int = 1000,
        beta_schedule: str = "linear"
    ):
        self.unet = unet
        self.trajectory_encoder = trajectory_encoder
        self.betas = get_beta_schedule(beta_schedule, num_timesteps)
        # ... standard diffusion setup
    
    def forward(self, x_0, floor_plan, trajectory_data, t=None):
        """
        Training forward pass.
        
        Args:
            x_0: (B, 1, H, W) clean radio maps
            floor_plan: (B, 3, H, W) floor plan features
            trajectory_data: dict with sparse_rss, mask, points
            t: (B,) timesteps (if None, sample randomly)
        """
        # Encode trajectory
        traj_embedding = self.trajectory_encoder(
            trajectory_data['sparse_rss'],
            trajectory_data['mask'],
            trajectory_data['coverage_density'],
            trajectory_data['points']
        )
        
        # Forward diffusion: add noise
        if t is None:
            t = torch.randint(0, self.num_timesteps, (x_0.shape[0],))
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)
        
        # Predict noise with trajectory conditioning
        noise_pred = self.unet(
            x_t, 
            t, 
            floor_plan=floor_plan,
            traj_embedding=traj_embedding
        )
        
        return noise_pred, noise
    
    @torch.no_grad()
    def sample(self, floor_plan, trajectory_data, num_samples=1):
        """
        Generate radio maps conditioned on floor plan and trajectory.
        
        Returns multiple samples for uncertainty estimation.
        """
        traj_embedding = self.trajectory_encoder(...)
        
        # DDIM sampling for speed
        x = torch.randn(num_samples, 1, H, W)
        for t in reversed(range(self.num_timesteps)):
            x = self.p_sample(x, t, floor_plan, traj_embedding)
        
        return x  # (num_samples, 1, H, W)
```

### 3.3 U-Net with Cross-Attention

```python
class TrajectoryConditionedUNet(nn.Module):
    """
    U-Net with cross-attention layers for trajectory conditioning.
    """
    def __init__(self, in_channels=4, out_channels=1, traj_dim=256):
        # Encoder
        self.enc1 = DownBlock(in_channels, 64)
        self.enc2 = DownBlock(64, 128)
        self.enc3 = DownBlock(128, 256)
        self.enc4 = DownBlock(256, 512)
        
        # Cross-attention at bottleneck
        self.bottleneck_attn = CrossAttention(512, traj_dim)
        
        # Decoder with skip connections
        self.dec4 = UpBlock(512, 256)
        self.dec3 = UpBlock(256, 128)
        self.dec2 = UpBlock(128, 64)
        self.dec1 = UpBlock(64, out_channels)
        
        # Time embedding
        self.time_mlp = TimeMLP(256)
    
    def forward(self, x, t, floor_plan, traj_embedding):
        # Concatenate input with floor plan
        x = torch.cat([x, floor_plan], dim=1)  # (B, 4, H, W)
        
        # Time embedding
        t_emb = self.time_mlp(t)
        
        # Encoder
        e1 = self.enc1(x, t_emb)
        e2 = self.enc2(e1, t_emb)
        e3 = self.enc3(e2, t_emb)
        e4 = self.enc4(e3, t_emb)
        
        # Cross-attention with trajectory
        e4 = self.bottleneck_attn(e4, traj_embedding)
        
        # Decoder
        d4 = self.dec4(e4, e3, t_emb)
        d3 = self.dec3(d4, e2, t_emb)
        d2 = self.dec2(d3, e1, t_emb)
        out = self.dec1(d2, None, t_emb)
        
        return out
```

### Deliverables
- [ ] `src/models/encoders/trajectory_encoder.py`
- [ ] `src/models/diffusion/trajectory_diffusion.py`
- [ ] `src/models/unet.py` with cross-attention
- [ ] Training working on small subset of data
- [ ] Visualizations of denoising process

---

## Phase 4: Physics-Informed Components (Week 8)

### Goals
- [ ] Physics-informed loss terms
- [ ] RSS consistency constraints along trajectories
- [ ] Wall attenuation priors

### 4.1 Physics-Informed Loss

```python
class PhysicsInformedLoss(nn.Module):
    """
    Additional loss terms based on radio propagation physics.
    """
    
    def trajectory_consistency_loss(self, pred_map, trajectory_points):
        """
        Predicted RSS along trajectory should match observed RSS.
        
        This is STRONGER than just MSE on sparse points because:
        - We sample intermediate points on the trajectory
        - We check smoothness along the path
        """
        loss = 0
        for (x, y, rss_observed) in trajectory_points:
            rss_pred = bilinear_sample(pred_map, x, y)
            loss += (rss_pred - rss_observed) ** 2
        return loss
    
    def wall_attenuation_loss(self, pred_map, floor_plan, tx_location):
        """
        Signal should drop when crossing walls.
        
        For each wall pixel, check that:
        - RSS on Tx side > RSS on far side (by expected attenuation)
        """
        # Simplified: detect wall crossings, penalize if gradient wrong sign
        pass
    
    def distance_decay_loss(self, pred_map, tx_location):
        """
        Signal should generally decrease with distance from Tx.
        (Soft constraint - can be violated by multipath)
        """
        pass
```

### 4.2 Uncertainty Quantification

**Approach**: Generate multiple samples from diffusion, compute statistics

```python
def estimate_uncertainty(model, floor_plan, trajectory_data, n_samples=10):
    """
    Generate multiple radio map samples and compute uncertainty.
    
    Returns:
        mean_map: (H, W) mean prediction
        std_map: (H, W) standard deviation (uncertainty)
        samples: (n_samples, H, W) all samples
    """
    samples = model.sample(floor_plan, trajectory_data, num_samples=n_samples)
    mean_map = samples.mean(dim=0)
    std_map = samples.std(dim=0)
    return mean_map, std_map, samples
```

**Expected behavior**:
- Low uncertainty along trajectories (we have data)
- High uncertainty in blind spots (extrapolation)
- Uncertainty should correlate with actual error

### Deliverables
- [ ] `src/training/losses.py` with physics-informed losses
- [ ] `src/evaluation/uncertainty.py` for uncertainty calibration
- [ ] Ablation: with/without physics losses

---

## Phase 5: Experiments & Ablations (Weeks 9-10)

### Experiment Matrix

| Experiment | Description | Metric Focus |
|------------|-------------|--------------|
| E1: Baseline comparison | All baselines vs TrajectoryDiff | RMSE, SSIM |
| E2: Sampling type | Uniform vs trajectory sampling | RMSE_observed vs RMSE_unobserved |
| E3: Trajectory encoder ablation | GNN vs Transformer vs Hybrid | RMSE, compute time |
| E4: Coverage level | 1%, 5%, 10%, 20% coverage | RMSE vs coverage curve |
| E5: Multiple trajectories | 1, 3, 5, 10 trajectories | RMSE improvement |
| E6: Trajectory type | Shortest-path vs random-walk vs corridor-biased | RMSE per type |
| E7: Physics loss ablation | With/without physics constraints | RMSE, physical plausibility |
| E8: Uncertainty calibration | Check if uncertainty matches error | Calibration plots |
| E9: Generalization | Train on subset, test on held-out buildings | Cross-building RMSE |

### Key Hypotheses to Test

1. **H1**: TrajectoryDiff > baselines when sampling is trajectory-based
2. **H2**: TrajectoryDiff ≈ baselines when sampling is uniform (our advantage is trajectory handling)
3. **H3**: Uncertainty is higher in blind spots and correlates with error
4. **H4**: Multiple trajectories improve performance (especially coverage)
5. **H5**: Physics losses improve physical plausibility without hurting RMSE

### Visualization Outputs

- Radio map predictions vs ground truth
- Uncertainty maps with error overlays
- Trajectory coverage visualization
- Per-region error breakdown (corridor vs room vs blind spot)

### Deliverables
- [ ] Full experiment results table
- [ ] Ablation study figures
- [ ] Analysis document interpreting results

---

## Phase 6: Paper Writing (Weeks 11-12)

### Target Venue
- **Primary**: ECCV 2025 (or CVPR 2026)
- **Backup**: ICML, NeurIPS workshop

### Paper Outline

1. **Introduction** (1 page)
   - Radio map problem and importance
   - Gap: trajectory vs uniform sampling
   - Our contribution

2. **Related Work** (1 page)
   - Diffusion for radio maps (IRDM, RadioDiff, etc.)
   - Trajectory-based localization
   - Physics-informed deep learning

3. **Method** (2-3 pages)
   - Problem formulation
   - Trajectory encoding
   - TrajectoryDiff architecture
   - Physics-informed training

4. **Experiments** (2-3 pages)
   - Dataset and setup
   - Baselines
   - Main results
   - Ablations
   - Uncertainty analysis

5. **Conclusion** (0.5 page)

### Figures to Prepare

- Fig 1: Problem illustration (uniform vs trajectory sampling)
- Fig 2: Architecture diagram
- Fig 3: Qualitative results (radio maps)
- Fig 4: Uncertainty visualization
- Fig 5: Ablation plots
- Fig 6: Coverage vs performance curves

### Deliverables
- [ ] Complete paper draft
- [ ] Supplementary material
- [ ] Code release preparation

---

## Risk Assessment & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| RadioMapSeer not suitable | Low | High | Have backup: simulate trajectories on other datasets |
| Trajectory encoding doesn't help | Medium | High | Try multiple encoder designs; ablate thoroughly |
| Diffusion too slow | Medium | Medium | Use DDIM; consider consistency models |
| Physics losses hurt performance | Low | Low | Make them optional; tune weights carefully |
| Overfitting to RadioMapSeer | Medium | Medium | Test on held-out buildings; collect small real dataset |

---

## Success Criteria

**Minimum Viable**:
- TrajectoryDiff outperforms baselines on trajectory-sampled data by >10% RMSE
- Uncertainty correlates with error (r > 0.5)
- Paper submitted to workshop

**Target**:
- Clear improvement across all trajectory types
- Well-calibrated uncertainty
- ECCV/CVPR main conference paper

**Stretch**:
- Real-world cellular trajectory dataset collected
- Demonstrate practical localization improvement
- Open-source release with adoption

---

## Timeline Summary

| Week | Phase | Key Deliverable |
|------|-------|-----------------|
| 0 | Setup | Working environment + data loaded |
| 1-2 | Data Pipeline | Trajectory sampling implemented |
| 3-4 | Baselines | Baseline results table |
| 5-7 | Core Model | TrajectoryDiff training |
| 8 | Physics | Physics-informed components |
| 9-10 | Experiments | Full ablation study |
| 11-12 | Paper | Submission-ready draft |
