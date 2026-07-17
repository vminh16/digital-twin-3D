# Phase 4.6 Accelerated Baseline Design

## Purpose

Freeze a reproducible Phase 4 baseline without committing the production cohort
to the current 50--70 hour sequential runtime.  Performance changes may alter
floating-point ordering, but must not alter the objective, camera sampler,
density thresholds, optimizer hyperparameters, or 30,000-step horizon.

## Scope and sequence

Phase 4.6 is split by an external GPU gate:

1. implement and qualify training backends on HCM0181 for 1,000 full-resolution
   internal-holdout steps;
2. select the fastest passing backend;
3. freeze that backend in the immutable B0 baseline bundle;
4. export and verify the compact inference artifact.

This session implements step 1 and the freeze inputs.  The final baseline ID is
not selected until the NVIDIA L4 qualification report exists.

## Training backend contract

Two independent configuration fields are canonical and hashed:

```text
optimizer_backend: adam | adam-fused
precision: fp32 | amp-fp16
```

The reference is `adam` + `fp32`.  Candidate F1 is `adam-fused` + `fp32`.
Candidate F2 is `adam-fused` + `amp-fp16`.  `amp-fp16` with unfused Adam is
rejected because it adds a redundant qualification branch with no production
value.  Defaults remain the reference backend so existing runs and unit tests do
not silently change behavior.

`adam-fused` means `torch.optim.Adam(..., fused=True)` with the exact existing
learning rates, betas, epsilon, parameter ownership, scheduler, and checkpoint
state.  It is CUDA-only and fails before training on CPU or unsupported CUDA
builds.  No fallback is allowed because a fallback would make the run hash lie
about the executed backend.

## AMP mathematical contract

Gaussian parameters and Adam state remain FP32.  Only renderer and loss execute
inside `torch.autocast(device_type="cuda", dtype=torch.float16)`.

The step order is fixed:

1. forward and loss under autocast;
2. `scaler.scale(loss).backward()`;
3. call `scaler.unscale_()` exactly once for each of the six optimizers;
4. divide retained non-leaf `means2d.grad` by the same current scale;
5. reject every non-finite leaf or projected-means gradient;
6. run density strategy on unscaled gradients;
7. call `scaler.step()` once for each optimizer;
8. call `scaler.update()` once;
9. advance the means scheduler only after all optimizer steps succeed.

Step 4 is mandatory.  gsplat 1.4.0 accumulates densification statistics from
`info["means2d"].grad`, and PyTorch optimizer unscaling only covers gradients
owned by that optimizer.  Passing the scaled non-leaf gradient to gsplat would
change the topology objective by multiplying the effective densification
threshold by the inverse loss scale.

The current trainer raises on non-finite gradients.  AMP preserves this policy;
it does not silently skip only a subset of six optimizer steps.

## L4-specific policy

NVIDIA L4 (Ada, compute capability 8.9) supports FP16, BF16, and TF32 Tensor
Cores.  This phase qualifies FP16 AMP because PyTorch provides gradient scaling
for its reduced exponent range.  BF16 and explicit TF32 are not enabled in the
same change: each changes numerical behavior and would require its own gate.

No `torch.compile`, CUDA graphs, sparse gradients, `radius_clip`, unpacked
rasterization, SelectiveAdam, or dependency upgrade is included.  Those either
change semantics, conflict with dynamic densification, or add complexity before
the measured bottlenecks justify it.

## 1,000-step GPU qualification

A single shell entry point runs three fresh HCM0181 full-resolution
internal-holdout jobs with seed 0 and the already-qualified cached/pinned input
path:

```text
R0: adam       + fp32
F1: adam-fused + fp32
F2: adam-fused + amp-fp16
```

One thousand steps cross the first density refinement at step 600 and include
refinements through step 1,000.  Each run writes a compact gradient audit at
selected boundary steps rather than retaining extra checkpoints.

The audit records finite status, loss scale, projected-gradient magnitude,
leaf-gradient maxima, Gaussian count, CUDA time, wall time, peak VRAM, device
name/capability, and actual tensor dtypes.  It contains no full tensors.

The comparator enforces:

- identical sampled image indices for all 1,000 steps;
- finite losses, parameters, leaf gradients, and projected gradients;
- projected gradients observed by strategy are unscaled;
- matching density-event step numbers;
- final Gaussian-count relative delta at most 1%;
- loss traces all-close before the first topology-changing refinement;
- final rolling-100 mean loss relative delta at most 2%;
- F1 must improve median measured CUDA step time by at least 10%;
- F2 is selected over F1 only if it adds at least 5% speedup;
- peak VRAM remains below the existing 20 GiB gate.

F1 is selected if it passes and F2 does not.  F2 is selected only if both its
correctness and incremental performance gates pass.  If neither passes, R0 is
frozen.  A 1,000-step pass qualifies engine mechanics, not final image quality;
the existing 30k HCM0181 report remains the quality evidence for B0.

## Baseline freeze boundary

The selected `optimizer_backend` and `precision`, the qualification report hash,
PyTorch/CUDA/gsplat versions, and code commit become part of
`phase4_baseline.yaml` and its SHA-256.  Resume and rendering reject mismatched
baseline/config/manifest hashes.  Any later backend change creates a new baseline
ID.

## Documentation basis

- PyTorch documents CUDA fused Adam as stable and normally faster than foreach or
  the per-tensor loop.
- PyTorch AMP requires unscaling each optimizer separately, stepping each
  optimizer separately, and updating one scaler once per iteration.
- NVIDIA specifies L4 throughput support for FP32, TF32, FP16, and BF16 and 24 GB
  VRAM.
- gsplat 1.4.0 DefaultStrategy reads retained projected-means gradients to drive
  duplication and splitting.

## Non-goals

- reducing the 30,000-step horizon;
- changing batch size, loss, SH schedule, density thresholds, or sampling;
- claiming leaderboard comparability;
- selecting AMP merely because L4 advertises high FP16 peak throughput;
- running two full-resolution scenes concurrently on one 24 GB L4.

