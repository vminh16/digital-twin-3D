# Chair and bonsai deep-optimization authority

**Status:** ACTIVE — Phase 1 implemented; L4 CUDA smoke pending.

**Plan ID:** `scene-opt-v3-chair-bonsai`

This is the only active research authority after submission
`MVP-hybrid-4scene-q99-v1`. `AGENTS.md` remains authoritative for data,
evaluation, production and output contracts.

## 1. Closed starting point

- `B0-submission-q99-v1` is immutable: official Score `70.98330`.
- `MVP-hybrid-4scene-q99-v1` is submitted and closed: official Score
  `71.2124`, PSNR `24.629191`, SSIM `0.807208`, LPIPS `0.194533`, 7/7 scenes.
- Stage A, Stage B1, the five-scene screen, four override productions,
  rerender and hybrid assembly are historical and must not be reopened.
- `HCM0644`, `HCM0674`, `HCM0540`, `HCM0421` and `HCM0539` are frozen during
  this plan.
- Official test RGB and official per-scene metrics are unavailable and cannot
  be used for candidate selection.

## 2. Evidence that opens this plan

The full-image audit covered 453 train images and 86 submission renders.

### Chair

- 5/58 outputs were catastrophic; four form the `855–895` close-mesh cluster.
- Adjacent train images are sharp and pose distance does not predict severity.
- Source disagreement predicts severity, consistent with an occlusion and
  projected-scale instability.
- Primary failure: oversized or excessively elongated Gaussians becoming
  visible over a narrow view interval.
- Secondary failure: seven strong-motion-blur train images cause mild
  softness, not the catastrophic veil.

### Bonsai

- 10/28 outputs were major or catastrophic.
- Nearest-train pose gap strongly predicts severity; all outputs with
  normalized gap at least 2 were non-clear.
- The pot often remains stable while the table, legs, carpet and foreground
  become translucent or geometrically inconsistent.
- Sharp internal-holdout frame `390` already failed at 7k; SH4 did not repair
  geometry.
- Primary failure: weakly constrained depth and multi-view geometry around the
  reflective table under local pose gaps.
- Secondary failure: SH cannot fully represent sharp view-dependent
  reflections.

The repository currently disables world-scale pruning with
`prune_scale3d=inf`. Densification stops at 15k. More steps alone therefore do
not repair missing or malformed topology.

## 3. Active method families

### Method C — chair: scale-controlled frequency densification

Goal: prevent giant near-camera splats while preserving the chair mesh.

Candidate ladder:

1. `E3-chair-scale-guard-v1` — reserved, not executable yet.
   - inherit the current chair incumbent
     `E2-loss-local-laplacian-v1`;
   - enable conservative 3D scale pruning;
   - collect normalized projected radius during densification;
   - split or prune only the extreme projected-scale tail;
   - log removed/split counts and scale/radius quantiles.
2. `E3-chair-scale-fregs-v1` — reserved, not executable yet.
   - inherit the passing scale-guard candidate;
   - add progressive frequency regularization during densification;
   - do not add a second appearance model.

Unconditional anisotropy rewards are forbidden. A valid regularizer caps
absolute/projected size and pathological in-plane elongation; it must not
penalize the thin axis of a legitimate planar or wire-like primitive.

Strong-blur down-weighting is a later isolated ablation. It is not bundled
into the first scale candidate.

### Method B — bonsai: geometry-consistent reflective appearance

Goal: stabilize the table and surrounding depth before increasing specular
appearance capacity.

Candidate ladder:

1. `E3-bonsai-scale-guard-sh4-v1` — reserved, not executable yet.
   - inherit the current bonsai incumbent `E2-appearance-sh4-v1`;
   - apply the same instrumented conservative scale guard;
   - retain SH4 unchanged.
2. `E3-bonsai-geometry-sh4-v1` — reserved, not executable yet.
   - inherit the passing scale-guard candidate;
   - add robust sparse-depth anchors from the provided COLMAP tracks;
   - add weak edge-aware depth/normal or multi-view geometry consistency;
   - use confidence masks derived only from provided train observations;
   - retain SH4 as the appearance component.

External monocular-depth weights, diffusion priors and external images remain
out of scope. Spec-Gaussian or GaussianShader is a later appearance-only
fallback after geometry passes; neither is authorized in the first ladder.

## 4. Unified research harness

Historical `reference` and `screen` stages retain their exact 7k semantics.
The active harness adds:

```text
stage          = research
authorized     = chair, bonsai only
schedule       = max_steps 30000
stop_step      = 15000
holdout        = true
checkpoint     = none
experiment root= runs/scene_opt_v3
```

The 30k schedule with a 15k stop preserves the production learning-rate
trajectory. It replaces the old practice of setting `max_steps=7000`, which
made a screen follow a different means-LR schedule from production.

The only operational wrapper is:

```bash
bash scripts/run_chair_bonsai_research.sh <chair|bonsai> <candidate-id>
```

Reserved E3 candidate IDs must fail before GPU allocation until their code,
contracts and tests are merged into the registry.

## 5. Validation design

Research selection uses train RGB only.

- Chair sentinel coverage must include sharp close-mesh train poses in the
  `840–920` interval, including a held-out interpolation challenge around
  frames `870/885`.
- Bonsai must include sharp frame `390` and algorithmically selected
  high-nearest-pose-gap train cameras.
- The existing generic pose-FPS holdout remains provenance; a v3 targeted
  holdout extension requires a separate hash and must never rewrite old
  `holdout.json`.

Implemented Phase 1 contract:

- artifact: `holdout_research_v3.json`;
- algorithm: `targeted_pose_guard2_v3`;
- chair sentinels: deterministic nearest available frames to 870 and 885
  within 840–920;
- bonsai sentinels: exact frame 390 plus the four cameras with largest
  all-camera nearest-pose gap;
- diagnostic thresholds: normalized max-axis scale `0.1` and projected radius
  `128 px`; these observe tails and do not define pruning policy;
- `gaussian_diagnostics.json` records scale/opacity quantiles, conservative
  max-over-view radius quantiles, opacity mass above both thresholds and exact
  observed net Gaussian-count changes;
- `diagnostic_filtered_renders/` suppresses over-threshold radius primitives
  per validation camera and is marked diagnostic-only;
- schema-2 research reports add worst-decile LPIPS/edge distance and
  deterministic veil/collapse flags. Historical reports remain schema 1.

The current `gsplat` strategy API does not expose separate clone/split/prune
counts. Phase 1 records exact before/after count deltas instead of assigning
unverifiable backend event labels.

Candidate gates:

1. all artifacts, renders, metrics and hashes are complete and finite;
2. mean Score50 improves against the paired incumbent;
3. LPIPS does not worsen;
4. hard-stratum Score50 does not worsen;
5. worst-decile LPIPS and symmetric-edge error improve;
6. catastrophic/veil sentinel count is zero;
7. missing-edge and spurious-edge do not both worsen;
8. peak VRAM remains below 23 GB;
9. paired wall time remains at most 1.25;
10. scale/radius tails and Gaussian growth remain controlled.

Mean improvement cannot override a failed tail gate.

## 6. Step and production policy

- 15k research evidence selects mechanisms only.
- A winner receives a fresh paired 30k internal-holdout confirmation.
- Production is a separate fresh 30k full-data run with no internal holdout.
- A 30k winner may receive one explicit 5–10k low-LR polish experiment only
  when the 15k→30k held-out curve is still improving and scale tails are
  stable.
- A 70k run is not authorized. Changing `max_steps` to 70k changes the LR
  trajectory and is not a continuation of the closed 30k policy.

## 7. Active phase

```text
Phase 0  CLOSED  close old phases; add research stage and unified wrapper
Phase 1  IMPLEMENTED  CPU contract passed; bounded L4 CUDA smoke pending
Phase 2  PENDING implement and screen chair method family
Phase 3  PENDING implement and screen bonsai method family
Phase 4  PENDING paired 30k confirmation
Phase 5  PENDING full-data production, rerender and new submission assembly
```

No Phase 2+ GPU run is authorized until Phase 1 tests and artifact contract
pass.
