# TrajectoryDiff CVPR Readiness Audit Report

**Date**: 2026-02-15
**Auditors**: 4 specialized agents (math, architecture, training, evaluation)
**Scope**: ~7,000 lines across 19 key files + 16 experiment configs
**Mode**: Read-only audit (no code changes)

---

## Executive Summary

**Verdict: NOT READY for submission.** The codebase has strong foundations (correct DDPM math, proper train/val/test splits, no data leakage) but contains **8 critical issues** that would likely cause rejection, **13 moderate issues** that reviewers would flag, and **14 suggestions** for strengthening the paper.

The most urgent problems are:
1. The core novel contribution (CoverageAwareAttention) has a mathematical error in its modulation mechanism
2. Physics loss gradients are amplified by ~10,000x at high timesteps, making them unreliable
3. No deep learning baselines (only classical interpolation) — insufficient for CVPR
4. No statistical significance testing — cannot support any comparative claims

Estimated fix time: 7-10 days for all critical issues, 3-5 additional days for moderate issues.

| Category | Count |
|----------|-------|
| Critical (rejection-worthy) | 8 |
| Moderate (reviewer complaints) | 13 |
| Suggestions (improvements) | 14 |

---

## Critical Issues (Rejection-Worthy)

### C1: CoverageAwareAttention Multiplicative Modulation Is Mathematically Inconsistent
- **Source**: architecture-reviewer
- **File**: `src/models/diffusion/attention.py:96-114`
- **Issue**: The core novel mechanism multiplies raw attention logits by `coverage_weight` (in [0,1]) before softmax. Since attention logits can be **negative**, this has inconsistent directional effects: positive logits are reduced (less attention — intended), but negative logits are brought closer to zero (MORE attention — opposite of intended). The mechanism actually makes attention more **uniform** for low-coverage keys rather than suppressing them. The paper's claim "high coverage keys get higher attention" (line 105) is only true when logits happen to be positive.
- **Fix**: Replace multiplicative modulation with **additive log-bias** (standard for attention modulation, cf. ALiBi, Swin relative position bias):
  ```python
  # Instead of: attn = attn * coverage_weight
  coverage_logbias = torch.log(key_coverage + 1e-6) / self.coverage_temperature
  attn = attn + coverage_logbias
  ```
  The existing dead-code class `AdaptiveCoverageAttention` (lines 261-354) already implements additive bias — consider adopting it.

### C2: Physics Loss Gradient Amplification Across Timesteps
- **Source**: math-reviewer
- **File**: `src/training/diffusion_module.py:334-358`
- **Issue**: Physics losses operate on `pred_x0`, recovered via `pred_x0 = sqrt(1/alpha_bar_t) * x_t - sqrt(1/alpha_bar_t - 1) * eps_theta`. The gradient `d(pred_x0)/d(eps_theta) = -sqrt(1/alpha_bar_t - 1)` varies by **~10,000x** across timesteps (100x at t=999 vs 0.01x at t=1). The code applies a batch-mean SNR scalar (`alphas_cumprod[t].mean()` ~= 0.5), which does **not** compensate per-sample. High-t samples with garbage `pred_x0` dominate the physics loss gradient.
- **Fix**: Apply **per-sample** SNR weighting inside the loss computation:
  ```python
  snr_weights = self.diffusion.alphas_cumprod[t]  # (B,) per-sample
  # Weight each sample's physics loss by snr_weights[i] before batch reduction
  ```
  Alternative: detach the `sqrt(1/alpha_bar - 1)` scaling factor to prevent gradient amplification, letting physics losses provide signal only through a straight-through path.

### C3: Missing SOTA Deep Learning Baselines — **RESOLVED** (Feb 17, 2026)
- **Source**: eval-reviewer
- **File**: `scripts/run_baselines.py` (entire file)
- **Issue**: Only compares against classical interpolation (IDW, RBF, Kriging, NN). No comparison to deep learning methods. CVPR reviewers will immediately ask: "Why diffusion over a supervised U-Net, cGAN, conditional VAE, or Neural Processes?"
- **Fix**: Implement at minimum:
  1. **Supervised U-Net baseline**: Reuse existing UNet, train with MSE loss (direct sparse_rss -> radio_map regression). Minimal new code.
  2. **RadioUNet** (Levie et al., 2021): Domain-specific SOTA for radio map completion.
  3. Optionally: conditional VAE or Neural Process baseline for uncertainty comparison.
- **Resolution**: All three baselines implemented:
  1. `src/models/baselines/supervised_unet.py` — SupervisedUNetBaseline (same arch, direct MSE)
  2. `src/models/baselines/radio_unet.py` — RadioUNetBaseline (standalone UNet, Levie 2021)
  3. `src/models/baselines/rmdm.py` — RMDMBaseline (dual-UNet diffusion, Xu 2025)
  - Configs: `experiment=supervised_unet`, `experiment=radio_unet`, `experiment=rmdm_baseline`
  - `train.py` factory dispatches on `model_type` field; `evaluate.py` auto-detects from checkpoint

### C4: No Statistical Significance Testing
- **Source**: eval-reviewer
- **File**: `scripts/evaluate.py` (all), `scripts/run_baselines.py` (all)
- **Issue**: All metrics reported as single point estimates (mean +/- std). No hypothesis testing to determine if differences are statistically significant. Cannot claim "our method outperforms X" without p-values.
- **Fix**: Add to evaluation pipeline:
  ```python
  from scipy.stats import wilcoxon, ttest_rel
  # Per-sample metrics for paired tests
  t_stat, p_value = wilcoxon(rmse_ours, rmse_baseline)
  # Bootstrap 95% CI (1000 resamples) for all reported metrics
  ```
  Add significance markers (*, **, ***) to comparison tables/figures.

### C5: DDIM Timestep Subsequence Skips Highest Noise Level
- **Source**: math-reviewer
- **File**: `src/models/diffusion/ddpm.py:590-591`
- **Issue**: `range(0, 1000, 20)` produces [0, 20, ..., 980]. Maximum is 980, not 999. Sampling starts from pure Gaussian noise (t=infinity) but the first denoising step applies the model at t=980 where `alpha_bar` expects some signal content. This creates a distribution mismatch at inference time.
- **Fix**: Use linspace to span full range:
  ```python
  self.ddim_timesteps = np.linspace(0, diffusion.num_timesteps - 1, ddim_num_steps, dtype=int).tolist()
  ```

### C6: No Sample Diversity Metrics for Generative Model
- **Source**: eval-reviewer
- **File**: Missing from all evaluation scripts
- **Issue**: Diffusion models are generative — they produce diverse outputs. Current eval only measures reconstruction (RMSE, SSIM), not distribution quality. No FID, no intra-sample variance as a quality metric, no mode coverage analysis.
- **Fix**: Add FID computation (pretrained ResNet features) and report intra-sample diversity from multiple DDIM samples per input. At minimum, quantify that diverse samples are meaningful, not just noise.

### C7: SSIM vs PSNR Computed on Different Scales
- **Source**: eval-reviewer
- **File**: `scripts/evaluate.py:202-203`
- **Issue**: SSIM computed on normalized [-1,1] data (`data_range=2.0`) while PSNR computed on dBm data (`max_val=139.0`). Inconsistent comparison basis makes results confusing and hard to interpret together.
- **Fix**: Compute both SSIM and PSNR in dBm scale:
  ```python
  all_ssim.append(compute_ssim(samples_dbm, gt_dbm, data_range=139.0))
  all_psnr.append(compute_psnr(samples_dbm, gt_dbm, max_val=139.0))
  ```

### C8: Architectural Novelty May Be Insufficient as Sole Contribution
- **Source**: architecture-reviewer
- **File**: `src/models/diffusion/attention.py:62-69`
- **Issue**: The coverage_gate is a 2-layer MLP (Linear(1, dim//4) -> SiLU -> Linear(dim//4, num_heads) -> Sigmoid) mapping a scalar to per-head gates. This is essentially SE-Net-style channel attention / FiLM conditioning — a well-explored pattern. The domain application is novel but the mechanism is not. A CVPR reviewer will ask how this differs from scalar multiplication by coverage density.
- **Fix**: Strengthen the contribution through one or more of:
  1. Make coverage modulation **query-key dependent** (coverage contrast between positions)
  2. Add **coverage-conditioned value projection** (modulate V, not just attention weights)
  3. Frame as a **system contribution** (trajectory conditioning + physics losses + coverage attention together), not a single architectural novelty
  4. Adopt the more sophisticated `AdaptiveCoverageAttention` (currently dead code) which adds learned fusion gates

---

## Moderate Issues (Reviewer Complaints)

### M1: CoverageWeightedLoss Breaks the Variational Bound
- **Source**: math-reviewer
- **File**: `src/training/losses.py:102-146`, `src/training/diffusion_module.py:352-355`
- **Issue**: Spatially weighting noise MSE by coverage density violates the ELBO derivation (requires uniform weighting). With `min_weight=0.1`, blind spots get 10x less gradient for noise prediction. Reviewers will ask whether this degrades sample quality in unobserved regions.
- **Fix**: Acknowledge the ELBO deviation in the paper with explicit justification. Consider using uniform noise MSE for diffusion loss and trajectory-specific losses only on pred_x0.

### M2: No DataLoader Worker Seeding
- **Source**: training-reviewer
- **File**: `src/data/datamodule.py:170-199`
- **Issue**: No `worker_init_fn` for DataLoader workers. With `num_workers=8`, all workers inherit the same RNG state, reducing trajectory diversity by ~8x and creating epoch-to-epoch correlations with `persistent_workers=True`.
- **Fix**: Add worker initialization:
  ```python
  def _worker_init_fn(worker_id):
      worker_seed = torch.initial_seed() % 2**32
      np.random.seed(worker_seed)
      random.seed(worker_seed)
  ```

### M3: Incomplete Physics Loss Config in 6 Experiment Files
- **Source**: training-reviewer
- **File**: `configs/experiment/coverage_sweep_{1,5,10,20}pct.yaml`, `num_trajectories_sweep.yaml`, `ablation_small_unet.yaml`
- **Issue**: These configs set `physics.enabled: true` but don't explicitly specify loss weights or warmup schedules, relying on implicit defaults. If base config defaults change, these experiments silently change behavior.
- **Fix**: Add explicit `trajectory_consistency.weight: 0.5`, `distance_decay.weight: 0.1`, `warmup_epochs: 30`, `rampup_epochs: 20` to all 6 configs.

### M4: Data Augmentation Code Exists But Is Never Used
- **Source**: training-reviewer
- **File**: `src/data/transforms.py` (241 lines), `src/data/dataset.py:131-177`
- **Issue**: Complete augmentation module with physics-consistent transforms exists, config has augmentation settings, but transforms are never instantiated or applied. Reviewers will ask "did you try augmentation?"
- **Fix**: Either wire up transforms in the dataset or remove `transforms.py` to avoid confusion.

### M5: Validation/Test Trajectory Sampling Is Non-Deterministic
- **Source**: training-reviewer
- **File**: `src/data/dataset.py:264-271`
- **Issue**: Trajectories are generated on-the-fly each `__getitem__` call. Validation metrics fluctuate due to trajectory sampling variance (not model improvement), making early stopping unreliable.
- **Fix**: Set `trajectory_cache_sets > 0` for val/test datasets to pre-generate and reuse trajectory sets.

### M6: Weight Initialization Inconsistency for Replaced Attention Blocks
- **Source**: architecture-reviewer
- **File**: `src/models/diffusion/coverage_unet.py:44-53`, `src/models/diffusion/unet.py:327-338`
- **Issue**: UNet applies `kaiming_normal_(mode='fan_out')` to all layers, then `_replace_attention_blocks()` creates new CoverageAwareAttentionBlock modules with PyTorch default init (`kaiming_uniform_(mode='fan_in')`). Coverage gate, QKV projections, and output projections have mismatched initialization.
- **Fix**: Call `_init_weights()` again after replacement, or explicitly initialize new blocks.

### M7: QKV Bias Mismatch Between Original and Replacement Attention
- **Source**: architecture-reviewer
- **File**: `src/models/diffusion/unet.py:137` vs `src/models/diffusion/attention.py:56`
- **Issue**: Original `AttentionBlock` uses Conv2d QKV with bias=True (default). Replacement `CoverageAwareAttention` uses Linear QKV with `qkv_bias=False` (explicit default). This silently removes bias when coverage attention is enabled — a confounded variable in ablation studies.
- **Fix**: Set `qkv_bias=True` as default in `CoverageAwareAttention.__init__`.

### M8: Unreachable Attention Resolutions in Config
- **Source**: architecture-reviewer
- **File**: `src/models/diffusion/coverage_unet.py:199`, `src/models/diffusion/unet.py:448-449`
- **Issue**: Medium UNet specifies `attention_resolutions=(32, 16, 8)` but only downsamples to 32x32. Resolutions 16 and 8 are never reached. CoverageAwareAttention operates at exactly one spatial resolution (32x32), limiting its architectural impact.
- **Fix**: Either add deeper downsampling levels, change config to `attention_resolutions=(32,)` for honesty, or add lightweight attention at higher resolutions.

### M9: DDIM Lacks Numerical Clamping for eta > 0
- **Source**: math-reviewer
- **File**: `src/models/diffusion/ddpm.py:670`
- **Issue**: `sqrt(1 - alpha_bar_prev - sigma^2)` can go slightly negative due to floating-point arithmetic when eta > 0, producing NaN.
- **Fix**: `torch.sqrt(torch.clamp(1 - alpha_bar_t_prev - sigma_t**2, min=0))`

### M10: No Per-Region SSIM/PSNR Breakdown
- **Source**: eval-reviewer
- **File**: `scripts/evaluate.py:206-209`
- **Issue**: Code computes per-region RMSE (trajectory vs blind spot) but not per-region SSIM/PSNR. For a trajectory-conditioned model, SSIM in blind spots vs observed regions is highly relevant.
- **Fix**: Add masked SSIM/PSNR computation for observed and unobserved regions.

### M11: Figure Quality Not Publication-Ready
- **Source**: eval-reviewer
- **File**: `scripts/generate_figures.py` (all plotting functions)
- **Issue**: No error bars on bar charts, no significance markers, default matplotlib fonts (not LaTeX), inconsistent colormaps, missing legends on some subplots.
- **Fix**: Add error bars, significance stars, LaTeX font settings, consistent colormaps, and proper legends.

### M12: Cross-Evaluation Configs Don't Specify Evaluation Strategy
- **Source**: training-reviewer
- **File**: `configs/experiment/cross_eval_traj_to_uniform.yaml`, `cross_eval_uniform_to_traj.yaml`
- **Issue**: Config names promise cross-evaluation (train on trajectory, eval on uniform) but no mechanism exists to specify a different sampling strategy during evaluation vs training.
- **Fix**: Implement `eval_sampling_strategy` parameter, or rename configs and document that cross-evaluation is done post-hoc.

### M13: No Cross-Validation or Multiple Seeds
- **Source**: eval-reviewer
- **File**: All training/eval scripts
- **Issue**: Single train/val/test split, single random seed. No variance across seeds reported.
- **Fix**: Train 3-5 models with different seeds, report mean +/- std across runs.

---

## Suggestions (Improvements)

### S1: Consider Min-SNR-gamma Loss Weighting
- **Source**: math-reviewer
- **File**: `src/models/diffusion/ddpm.py:529-545`
- **Suggestion**: Min-SNR-gamma (Hang et al., ICCV 2023) balances gradient magnitudes across timesteps, significantly speeding convergence. Would strengthen the approach.

### S2: Coverage Gate Could Incorporate Spatial Context
- **Source**: architecture-reviewer
- **File**: `src/models/diffusion/attention.py:64-69`
- **Suggestion**: The gate processes each position independently (scalar -> per-head gate). A small conv layer on coverage before the gate would add spatial context (e.g., "edge of covered region" vs "center of coverage").

### S3: ConditionEncoder Hidden Channels May Bottleneck
- **Source**: architecture-reviewer
- **File**: `src/models/encoders/condition_encoder.py:283-287`
- **Suggestion**: Compresses 12 input channels through `hidden_channels=32` before expanding to 64. Could lose fine-grained TX encoding information. Consider increasing to 64 or exposing in config.

### S4: Add Flash Attention Fallback When Coverage Is None
- **Source**: architecture-reviewer
- **File**: `src/models/diffusion/attention.py:96`
- **Suggestion**: CoverageAwareAttention uses manual O(N^2) attention. When `coverage=None`, could fall back to `F.scaled_dot_product_attention` for efficiency.

### S5: Coverage Density Dual Pathway Needs Justification
- **Source**: architecture-reviewer
- **File**: `src/models/encoders/condition_encoder.py:498`
- **Suggestion**: Coverage density enters through both the condition encoder (spatial input) and CoverageAwareUNet (attention modulation). Document this design decision and consider an ablation separating the two pathways.

### S6: DistanceDecayLoss Uses Batch-Global Averaging
- **Source**: math-reviewer
- **File**: `src/training/losses.py:229-236`
- **Suggestion**: `near_rss = pred_map[near_mask].mean()` averages over all near pixels in the entire batch. Per-sample averaging before batch reduction would be more balanced.

### S7: DDIM Wastes One Model Evaluation at t=0
- **Source**: math-reviewer
- **File**: `src/models/diffusion/ddpm.py:620-682`
- **Suggestion**: The final step (t=0) just returns pred_x0 without meaningful denoising. Exclude t=0 or handle without a model forward pass.

### S8: EMA Should Also Sync Buffers
- **Source**: math-reviewer
- **File**: `src/training/diffusion_module.py:177-189`
- **Suggestion**: EMA update only iterates parameters, not buffers. Currently benign (GroupNorm has no running stats) but fragile if BatchNorm is added later.

### S9: Dead Code Cleanup — AdaptiveCoverageAttention
- **Source**: architecture-reviewer
- **File**: `src/models/diffusion/attention.py:126-186`, `261-354`
- **Suggestion**: Two substantial unused classes. Either adopt `AdaptiveCoverageAttention` (better design, addresses C1) or delete to avoid reviewer confusion.

### S10: DDIM V-Prediction Support Incomplete
- **Source**: math-reviewer
- **File**: `src/models/diffusion/ddpm.py:644`
- **Suggestion**: DDIM raises NotImplementedError for v-prediction despite the conversion formula existing elsewhere. Complete the implementation.

### S11: Make Train/Val/Test Splits Versioned
- **Source**: training-reviewer
- **File**: `src/data/dataset.py:296-326`
- **Suggestion**: Save split IDs to a JSON file for reproducibility. Currently deterministic (seed=42) but not explicitly versioned.

### S12: Validation Metric Doesn't Measure Generation Quality
- **Source**: training-reviewer
- **File**: `configs/training/default.yaml:38`
- **Suggestion**: Checkpoint selection uses val/loss (single-step denoising MSE), which doesn't correlate well with DDIM reconstruction quality. Add periodic full-sampling evaluation.

### S13: Add Qualitative Failure Cases
- **Source**: eval-reviewer
- **Suggestion**: Show examples where the model fails (high RMSE) and discuss why (extreme sparsity, unusual geometry). Standard CVPR practice.

### S14: Compute Metrics on Distance Bins
- **Source**: eval-reviewer
- **File**: `src/evaluation/metrics.py` (`rmse_by_distance` exists but unused in main eval)
- **Suggestion**: Show how error grows with distance from trajectory — a key trajectory-aware metric that strengthens the paper's narrative.

---

## Verified Correct (No Issues Found)

The following were audited and confirmed correct:

- **DDPM forward process** q(x_t|x_0): reparameterization trick, noise scheduling
- **DDPM reverse process** p(x_{t-1}|x_t): posterior mean coefficients
- **pred_x0 from epsilon**: `x_0 = sqrt(1/alpha_bar)*x_t - sqrt(1/alpha_bar - 1)*eps`
- **v-prediction formula**: `v = sqrt(alpha_bar)*eps - sqrt(1-alpha_bar)*x_0`
- **Cosine beta schedule**: Matches Nichol & Dhariwal exactly
- **DDIM update equation**: Correct formulation
- **EMA update**: `theta_ema = decay*theta_ema + (1-decay)*theta_model`
- **LR schedule**: LinearLR warmup -> CosineAnnealingLR, epoch-based conversion
- **Physics warmup**: Delay + linear ramp implementation
- **Gradient flow**: Physics losses -> pred_x0 -> model output -> model params (verified chain)
- **Train/val/test split**: By map_id (no data leakage), 70/15/15 split
- **Trajectory generation**: Independent per-split, different seeds
- **Ground truth usage**: Standard diffusion protocol (noised x_0, compare to noise/x0)
- **dBm denormalization**: `((norm + 1) / 2) * 139 + (-186)` — correct everywhere
- **RMSE, MAE, SSIM formulas**: All correct
- **Baseline data fairness**: Same sparse_rss and trajectory_mask from test loader
- **CoverageAwareUNet**: Correctly threads coverage through encoder/middle/decoder
- **num_heads**: Correctly passed from original AttentionBlock during replacement
- **Output conv zero-init**: Standard diffusion practice, correctly applied
- **Conditioning pipeline**: 5 inputs -> 64ch encoder -> concat with x_t -> UNet (dimensions correct)
- **Sinusoidal timestep embedding**: Standard implementation

---

## Action Plan (Priority Order)

### Phase 1: Critical Math/Architecture Fixes (Days 1-3)
1. **Fix C1** (attention modulation) — Replace multiplicative with additive log-bias. Consider adopting AdaptiveCoverageAttention. ~0.5 day
2. **Fix C2** (physics loss gradients) — Per-sample SNR weighting or detach scaling factor. ~0.5 day
3. **Fix C5** (DDIM timesteps) — Use np.linspace for full range. ~1 hour
4. **Fix C7** (SSIM/PSNR scales) — Compute both in dBm. ~1 hour
5. **Fix M9** (DDIM eta clamping) — Add torch.clamp. ~15 minutes
6. **Fix M6** (init mismatch) — Re-initialize after attention replacement. ~1 hour
7. **Fix M7** (QKV bias) — Set qkv_bias=True. ~15 minutes
8. **Fix M3** (explicit configs) — Add physics weights to 6 config files. ~1 hour

### Phase 2: Baselines & Evaluation (Days 3-6)
9. **~~Fix C3~~** ~~(DL baselines)~~ — **DONE**: Supervised UNet, RadioUNet, RMDM all implemented (Feb 17). ~2 days
10. **Fix C4** (statistical tests) — Add paired tests, bootstrap CIs, p-values. ~0.5 day
11. **Fix C6** (diversity metrics) — Add FID or intra-sample diversity. ~1 day
12. **Fix M10** (per-region SSIM/PSNR) — Add masked metrics. ~0.5 day

### Phase 3: Training Pipeline (Days 6-8)
13. **Fix M2** (worker seeding) — Add worker_init_fn. ~1 hour
14. **Fix M5** (val determinism) — Cache val/test trajectories. ~0.5 day
15. **Fix M4** (augmentation) — Wire up or remove transforms.py. ~0.5 day
16. **Fix M13** (multiple seeds) — Run 3 seeds for final experiments. ~compute time only

### Phase 4: Paper Polish (Days 8-10)
17. **Fix C8** (novelty framing) — Frame as system contribution, strengthen attention mechanism. ~paper writing
18. **Fix M1** (ELBO deviation) — Document and justify in paper. ~paper writing
19. **Fix M11** (figures) — Error bars, significance, LaTeX fonts. ~1 day
20. **Fix M8** (attention resolutions) — Decide: more levels or honest config. ~0.5 day
21. **Fix M12** (cross-eval) — Implement eval_sampling_strategy or rename. ~0.5 day

---

## Audit Metadata

| Agent | Files Reviewed | Critical | Moderate | Suggestions |
|-------|---------------|----------|----------|-------------|
| math-reviewer | ddpm.py, losses.py, diffusion_module.py | 2 | 4 | 3 |
| architecture-reviewer | coverage_unet.py, attention.py, condition_encoder.py, unet.py | 2 | 6 | 5 |
| training-reviewer | diffusion_module.py, dataset.py, datamodule.py, trajectory_sampler.py, 16 configs | 2 | 5 | 3 |
| eval-reviewer | metrics.py, evaluate.py, run_baselines.py, generate_figures.py, analyze_uncertainty.py | 4 | 7 | 6 |
| **TOTAL (deduplicated)** | **19 files + 16 configs** | **8** | **13** | **14** |

*Report generated by 4-agent CVPR readiness audit team. All findings include specific file:line references and concrete fix descriptions. No code was modified during this audit.*
