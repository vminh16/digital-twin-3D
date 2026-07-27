# Chair and bonsai candidate validation

**Status:** COMPLETE — theory gate closed; implementation has not started.

**Authority:** this note supplies the evidence behind Phase 2 of
`../specs/2026-07-27-chair-bonsai-deep-optimization.md`.

## 1. What is and is not proved

Evidence labels used below:

- **L — local proof:** directly measured from repository inputs or completed
  artifacts;
- **P — published proof:** demonstrated by a peer-reviewed primary paper on
  its stated datasets;
- **H — project hypothesis:** mechanism is compatible with L and P evidence,
  but a paired local run is still required.

No paper result proves a gain on the private test RGB. Official per-scene
metrics and test RGB are unavailable, so the only valid local selector remains
the fixed internal holdout. “Proved” in this note means that a mechanism is
supported by both local evidence and a primary-source result, not that an
official-score gain is guaranteed.

## 2. New local evidence

### 2.1 Chair has a deterministic sparse-color initialization bug

The physical chair images and current camera model are `720 x 1280`, but
COLMAP `points2D` remain in the original `1080 x 1920` coordinate system.
For observed points, reprojection gives:

```text
points2D_stored = 1.5 * projection_current
```

The current `_infer_observation_scale()` takes the 99.9th-percentile coordinate
ratio and applies `ceil`. On all 205 physical chair images it measures
`(1.49558, 1.49761)` and returns `(2, 2)`. The current sparse initializer
therefore samples:

```text
points2D_stored / 2 = 0.75 * projection_current
```

instead of the correct `points2D_stored / 1.5`.

The error affects all `79,828` points used to initialize chair. Across
`474,038` physical-image observations, median-per-point image colors were
compared with the colors stored in `points3D.bin`:

| Observation scale | Mean RGB MAE | Median RGB MAE | p90 RGB MAE |
|---:|---:|---:|---:|
| correct `1.5` | `33.6835` | `32.5890` | `58.4306` |
| current inferred `2.0` | `77.5610` | `66.5674` | `157.7571` |

Using the true continuous scale lowers mean initialization color error by
`56.57%`. This is **L proof of a harness defect**, not a model hypothesis.
Image dimensions, train intrinsics and train poses are already consistent; the
defect is limited to resampling colors from legacy `points2D`.

The defect is specific to leakage-controlled internal-holdout initialization,
which rebuilds point colors from the internal-train images. Full-data
production currently uses the colors already stored in `points3D.bin` and is
not affected. E3 is therefore a research control that repairs the local
selector; it is not itself a production model candidate.

The defect can amplify the diagnosed density problem during research: early
reconstruction gradients are generated from badly initialized colors before
density decisions become irreversible. How much it changes held-out metrics is
still H and must be measured. Any later model candidate must use the same
continuous mapping as its paired corrected incumbent.

The implemented E3 CPU preflight on the exact v3 internal-train split confirms:

```text
fit observations              346,212
fitted scale                   (1.50001973, 1.50001935)
mapped reprojection p95        1.77976 px
legacy reprojection p95        329.57106 px
common color-comparison points 74,478
mapped mean RGB MAE            33.29314
legacy mean RGB MAE            77.10911
```

Thus the implementation reproduces the diagnosed `1.5` mapping on real chair
data before any L4 training. Metric impact remains an H claim until the paired
15k run finishes.

### 2.2 Corrected SfM coverage

Chair coverage below rescales legacy `points2D` by `1/1.5` before binning.
Bonsai needs no rescaling.

| Scene | Registered images | Sparse points | Median valid observations/image | Median occupied `16 x 9` cells |
|---|---:|---:|---:|---:|
| chair | `263` | `80,491` | `2,188` | `0.7431` |
| bonsai | `276` | `54,422` | `1,413.5` | `0.6389` |

Chair sentinel cell coverage is `0.9097` at frame `525`, `0.6250` at `870`
and `0.6319` at `885`. The close mesh is nevertheless locally weak: points
cluster on carpet and hard chair edges while large dark mesh regions remain
poorly anchored. This is compatible with over-reconstruction by large
Gaussians; it is not a globally sparse-scene problem.

For bonsai frame `390`, the approximate tabletop ROI occupies `15.29%` of the
image but contains only `103/1,289 = 7.99%` of its valid SfM observations. Its
relative point density is `0.523` of the image average. Points occur mainly on
the pot, table boundary and reflected edges; the black tabletop interior has
little geometric support. Therefore sparse COLMAP depth can anchor the table
boundary but cannot by itself supervise the whole reflective plane.

The overlay audit is diagnostic rather than semantic ground truth:

```text
<visualization-root>/aux_sfm_overlays/chair/
<visualization-root>/aux_sfm_overlays/bonsai/
```

### 2.3 Existing local method evidence

The completed 7k HCM screens provide an in-repository prior for AbsGrad:

| Scene | AbsGrad Score50 delta | LPIPS delta | Gaussian ratio | Time ratio |
|---|---:|---:|---:|---:|
| HCM0421 | `+1.3636` | `-0.02193` | `1.28x` | `1.35x` |
| HCM0539 | `+0.8853` | `-0.01558` | `1.29x` | `1.36x` |

This proves the existing implementation can improve local scenes and gives a
usable L4 cost prior. It does not prove transfer to chair or bonsai.

Phase 1 still supplies the causal tail evidence:

- chair: `8,572` Gaussians above `128 px`; diagnostic deletion repairs
  frame `525` but hurts normal views;
- bonsai: `21,088` Gaussians above `128 px`; deleting the tail loses about
  `6.8 dB` PSNR on average;
- bonsai pose gap correlates with LPIPS (`r = 0.647`).

The correct operation is therefore redistribution or stabilization, not global
tail deletion.

## 3. Literature-to-mechanism validation

### 3.1 AbsGS / absolute-gradient densification

Vanilla density control can cancel opposite per-pixel projected-mean
gradients. In simplified form:

```text
g_vanilla(i) = norm(sum_p dL_p / dmu_i)
g_abs(i)     = sum_p norm_1(dL_p / dmu_i)
```

The first statistic can remain small for a large Gaussian that spans several
different structures; the second preserves the evidence that it should split.
AbsGS identifies this exact “gradient collision” mechanism and reports recovery
of fine detail from large over-reconstructed Gaussians
([paper](https://arxiv.org/abs/2404.10484)).

**Fit:** strong for chair mesh and useful for bonsai foliage/table boundaries.
The repository already exposes the exact `gsplat` absolute-gradient path.

**Risk:** it creates more Gaussians and can also grow floaters. It should not be
paired with unconditional low thresholds or global scale pruning.

### 3.2 Spec-Gaussian coarse-to-fine density curriculum

Spec-Gaussian combines absolute densification gradients with training that
starts from images downsampled by factor 4 along each spatial axis and
progressively reaches full resolution. The low-resolution phase prevents
early excessive growth; the absolute-gradient term recovers details as
resolution rises. Its ablation reports LPIPS `0.265 -> 0.180` and fewer
Gaussians when the complete curriculum is restored, and explicitly attributes
the gain to floater removal
([NeurIPS paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/708e0d691a22212e1e373dc8779cbe53-Paper-Conference.pdf)).

**Fit:** strongest deadline-compatible published mechanism for bonsai’s
floater/pose-gap failure. It uses no external data.

**Limit:** the paper explicitly notes that its appearance field still cannot
fully handle reflections without explicit geometry. The curriculum may
stabilize the tabletop representation, but it cannot invent unobserved
table-interior geometry.

### 3.3 Pixel-GS

Pixel-GS replaces view-count averaging with a pixel-participation-weighted
criterion:

```text
sum_k m_ik * f(i,k) * ||g_ik|| / sum_k m_ik > tau_pos
f(i,k) = clip((z_ik / (gamma_depth * R_scene))^2, 0, 1)
```

`m_ik` is the fraction of pixels to which Gaussian `i` actually contributes.
The depth factor suppresses near-camera floaters. The complete method improves
Mip-NeRF360 from `27.71/0.826/0.202` to `27.88/0.834/0.176` and
Tanks & Temples from `24.19/0.844/0.194` to `24.38/0.850/0.178`
(PSNR/SSIM/LPIPS). Critically, pixel-aware growth alone collapses the T&T
ablation to `21.80/0.791/0.239`; the depth-scaled component is necessary
([ECCV paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02926.pdf)).

**Fit:** excellent causal match to large splats, sparse local initialization
and near-camera floaters in both scenes.

**Integration fact:** `gsplat 1.4.0` exposes projected means, radius, depth and
tile intersections, but not the exact post-alpha/transmittance per-Gaussian
pixel-participation count required by Pixel-GS. Exact reproduction needs a
rasterizer change. Ellipse area or touched-tile count would only be an
unpublished proxy and is not authorized as “Pixel-GS”.

**Decision:** keep as a strong second-line research method, not the first
deadline implementation.

### 3.4 Error-driven density and MCMC

Revising Densification uses per-pixel reconstruction error for primitive
allocation and growth control. On the canonical Mip-NeRF360 bonsai scene its
supplement reports `32.040/0.940/0.254 -> 32.478/0.944/0.231`
([ECCV supplement](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08041-supp.pdf)).
It is a good mechanism match but requires a new error-rasterization and
allocation path.

3DGS-MCMC replaces clone/split heuristics with SGLD, relocation and
regularization, improving robustness to initialization
([NeurIPS paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/93be245fce00a9bb2333c17ceae4b732-Abstract-Conference.html)).
`gsplat 1.4.0` contains `MCMCStrategy`, but a faithful candidate also changes
loss regularization, optimizer updates and the entire density policy.

**Decision:** both are valid fallback families, but neither is the minimum
causal change for the first screen.

### 3.5 Sparse depth, 2DGS and planar geometry

A confidence-masked sparse-depth loss can be written using only provided
COLMAP tracks:

```text
L_sparse_depth =
    sum_ij w_ij * rho(D_render_i(project_i(X_j)) - z_ij) / sum_ij w_ij
```

where confidence can depend on track length and reprojection error. This is
contract-compliant. DNGaussian demonstrates that depth constraints can repair
Gaussian positions in sparse-view regimes, but its full method uses dense
monocular depth
([CVPR paper](https://openaccess.thecvf.com/content/CVPR2024/html/Li_DNGaussian_Optimizing_Sparse-View_3D_Gaussian_Radiance_Fields_with_Global-Local_Depth_CVPR_2024_paper.html)).

Two caveats are locally decisive:

1. bonsai’s reflective table interior is under-covered by COLMAP;
2. ordinary alpha-blended expected depth is not an unbiased surface depth.

PGSR derives plane-unbiased depth and geometric consistency, but its ablation
also shows the score trade-off: adding multi-view regularization improves
Meetingroom F1 from `0.15` to `0.29` while PSNR falls from `28.14` to `27.30`
([paper](https://arxiv.org/abs/2406.06521)).

2DGS directly represents oriented disks and adds depth-distortion and
normal-consistency losses
([paper](https://arxiv.org/abs/2403.17888)). It is geometrically well matched
and `gsplat 1.4.0` has a renderer, but the paper’s Mip-NeRF360 NVS table shows
a small appearance regression versus 3DGS:

| Split | Method | PSNR | SSIM | LPIPS |
|---|---|---:|---:|---:|
| outdoor | 3DGS | `24.64` | `0.731` | `0.234` |
| outdoor | 2DGS | `24.34` | `0.717` | `0.246` |
| indoor | 3DGS | `30.41` | `0.920` | `0.189` |
| indoor | 2DGS | `30.40` | `0.916` | `0.195` |

It also changes renderer outputs, density-gradient keys, losses, checkpoint
semantics and inference. **Decision:** sparse depth is a conditional auxiliary
candidate; 2DGS/PGSR are post-deadline geometry fallbacks.

### 3.6 Reflective appearance methods

GaussianShader reports a `1.57 dB` PSNR gain over 3DGS on specular-object
datasets, using a shading function and normals estimated from the shortest
Gaussian axis
([CVPR paper](https://openaccess.thecvf.com/content/CVPR2024/html/Jiang_GaussianShader_3D_Gaussian_Splatting_with_Shading_Functions_for_Reflective_Surfaces_CVPR_2024_paper.html)).
Spec-Gaussian replaces SH with anisotropic spherical Gaussians. HMGS separates
view-direction and reflected-direction colors
([ACCV paper](https://openaccess.thecvf.com/content/ACCV2024/html/Zhang_HMGS_Hybrid_Model_of_Gaussian_Splatting_for_Enhancing_3D_Reconstruction_ACCV_2024_paper.html)).

These methods need reliable normals or geometry. The local bonsai failure is
already geometric while the SH4 screen produced only a modest gain. Increasing
SH again is therefore not causal.

SpecTRe-GS is a closer model of reflected nearby objects, but requires an
integrated secondary-ray tracer, normal-prior guidance and multi-stage joint
geometry optimization
([CVPR 2025 paper](https://openaccess.thecvf.com/content/CVPR2025/html/Tang_SpecTRe-GS_Modeling_Highly_Specular_Surfaces_with_Reflected_Nearby_Objects_by_CVPR_2025_paper.html)).
It is not a deadline-sized extension of this repository.

**Decision:** do not spend the first screen on ASG, higher SH, shading or ray
tracing. Revisit appearance only after geometry/tail gates pass.

## 4. Candidate decision

| Rank | Candidate | Evidence | Code risk | Expected L4 cost | Decision |
|---:|---|---|---|---|---|
| C0 | chair continuous observation-scale research control + incumbent local-Laplacian | L direct | low | `~1.0x` | **run first** |
| B0 | bonsai SH4 + AbsGrad + 4x-downsample-to-full c2f curriculum | L+P strong | medium | `<=~1.4x` prior | **run first** |
| C1 | C0 + AbsGrad `t04` | L+P strong | low | `~1.35x` prior | conditional on remaining tail |
| B1 | B0 + robust sparse COLMAP depth anchors | L+P partial | medium | small renderer/loss overhead | conditional on frame `390` |
| P1 | exact Pixel-GS complete model | P strong, H local | medium-high | paper `1.6–2.0x` | defer |
| M1 | MCMC + incumbent scene appearance/loss | P broad | medium | often `~1.8x` | fallback |
| G1 | 2DGS/PGSR | P geometry, mixed NVS | high | new renderer contract | defer |
| A1 | ASG/GaussianShader/HMGS/SpecTRe | P appearance | high | new appearance/ray path | defer |

Reserved IDs:

```text
E3-chair-observation-scale-v1
E3-bonsai-c2f-absgrad-sh4-v1
E4-chair-observation-scale-absgrad-v1
E4-bonsai-c2f-absgrad-sh4-sparse-depth-v1
```

`E3-chair-observation-scale-v1` is implemented as a chair/research-only
control. It must never be authorized for confirm or production. The other
names remain non-executable until code, contract and tests are merged. The
former `E3-*-scale-guard-*` names are superseded by this evidence and must not
be registered.

## 5. Falsification gates

### Chair C0

The candidate must record:

- fitted continuous observation scale and residual reprojection ratio;
- initial sparse-color error under old and new mappings;
- initial and final Gaussian count;
- paired frame metrics for `525`, `870`, `885`;
- opacity-weighted radius tails.

Reject C0 if initial sparse-color error does not improve, or if mean Score50,
worst-decile LPIPS or any sentinel regresses. C0 cannot become a production
winner. If it repairs the selector but a giant-radius collapse remains, compare
C1 against C0 under the same mapping. If the collapse disappears, retain the
existing full-data production policy and do not add AbsGrad.

### Bonsai B0

The c2f schedule and AbsGrad are treated as one published density curriculum;
neither an AbsGrad-only production jump nor a changed SH degree is allowed.
Require:

- fixed schedule from 4x downsampling on each spatial axis to full resolution,
  ending by step `5,000`;
- same SH4, holdout, seed and 30k horizon/15k stop as the incumbent;
- no regression on frames `290`, `340`, `390`, `420`, `430`, `650`;
- improved hard-stratum Score50 and worst-decile LPIPS;
- controlled Gaussian and opacity-weighted radius growth.

If floaters/tails improve but frame `390` remains a geometric failure, B1 is
causally justified. If B0 fails the tail gate, adding sparse depth or more
steps is not justified.

## 6. Score and deadline reality

For one scene:

```text
delta Score50 = 0.6 * delta PSNR + 30 * delta SSIM - 40 * delta LPIPS
```

Because the official score averages seven scenes, moving `71.2124 -> 75`
using only chair and bonsai requires their combined scene-score gain to be:

```text
7 * (75 - 71.2124) = 26.5132
```

or `13.2566` per open scene if equal. Reaching `80` would require `30.7566`
per open scene. Published average gains from compatible density methods are
usually around `1–1.5` scene-score points, although a catastrophic-tail repair
can be larger. The evidence therefore supports meaningful improvement, not a
promise of `75–80`.

The minimum initial decision path is two 15k screens:

```text
chair C0   about 26 minutes of measured L4 training
bonsai B0 about 36 minutes using the local AbsGrad 1.35x prior
```

Early low-resolution training may offset part of B0’s overhead. One L4 process
at a time remains required. A 70k run, hard tail deletion, lower global growth
threshold, higher SH alone and unpaired production training remain rejected.
