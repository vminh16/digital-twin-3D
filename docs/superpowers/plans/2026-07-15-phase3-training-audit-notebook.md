# Phase 3 Training Audit Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and execute a visual, reproducible notebook that audits HCM0181 Run A and Run B now and accepts an optional full-resolution pilot later.

**Architecture:** Keep all audit-only loading, validation, calculation, and plotting functions inside one notebook so Phase 3 production code remains unchanged. Read small JSON/YAML/PNG artifacts and checkpoint metadata only; never load checkpoint tensors. Produce an answer-first scorecard followed by image comparisons, error maps, convergence curves, performance/storage plots, and an evidence-bounded readiness conclusion.

**Tech Stack:** Python 3, Jupyter/nbformat/nbclient, NumPy, Pillow, PyYAML, Matplotlib, pandas.

## Global Constraints

- Do not modify trainer, checkpoint, renderer, dataset, manifest, or run artifacts.
- Do not use official test poses or test ground truth.
- Treat stored masked PSNR/SSIM as authoritative smoke metrics.
- Label PNG-recomputed metrics as unmasked and quantized.
- Treat pilot artifacts as optional until the pilot finishes.
- Do not load `.pt` tensors; inspect checkpoint name and byte size only.
- Preserve `runs.tar.gz` as untracked user data.

---

### Task 1: Reproducible artifact loading and integrity gate

**Files:**
- Create: `notebooks/phase3_training_audit.ipynb`

**Interfaces:**
- Consumes: `runs/HCM0181/run_a_factor4_500`, `runs/HCM0181/run_b_factor2_7000_v2`, optional `runs/HCM0181/pilot_factor1_3000`.
- Produces inside the notebook: `load_run(run_dir: Path, label: str) -> dict`, `validate_run(run: dict) -> list[dict]`, and a normalized `runs: list[dict]`.

- [ ] **Step 1: Scaffold the notebook with reader-facing structure**

Create markdown sections in this order: `TL;DR`, `Context & Methods`, `Artifact Integrity`, `Run Overview`, `Image Reconstruction`, `Training Dynamics`, `Performance & Storage`, and `Takeaways`. Add one parameter cell with repository-relative run paths and `ROLLING_WINDOW = 100`.

- [ ] **Step 2: Implement bounded artifact loading**

Add notebook functions that load YAML/JSON/JSONL, decode the three preview PNGs, list `.pt` checkpoint sizes without calling `torch.load`, and return `None` for a missing optional pilot directory.

```python
def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def checkpoint_metadata(path: Path) -> list[dict]:
    return [
        {"name": item.name, "step": int(item.stem.split("_")[-1]), "bytes": item.stat().st_size}
        for item in sorted(path.glob("step_*.pt"))
    ]
```

- [ ] **Step 3: Implement visible integrity checks**

Check required files, exact step coverage, finite numeric fields, summary agreement, preview decode/resolution, final checkpoint existence, and CUDA/device metadata. Render a compact dataframe with columns `run`, `check`, `status`, and `detail`; blockers use `BLOCK`, caveats use `CAVEAT`, and successful checks use `PASS`.

- [ ] **Step 4: Execute the notebook loading section**

Run:

```powershell
.\.venv\Scripts\python.exe -m jupyter nbconvert --execute --to notebook --inplace notebooks/phase3_training_audit.ipynb
```

Expected: Run A and Run B load successfully, pilot is shown as unavailable, and no integrity row is `BLOCK`.

### Task 2: Visual quality and convergence dashboard

**Files:**
- Modify: `notebooks/phase3_training_audit.ipynb`

**Interfaces:**
- Consumes: normalized `runs` from Task 1.
- Produces: scorecard table, overview table, GT/initial/final panel, error maps, detail crops, and convergence charts.

- [ ] **Step 1: Add run overview and gate scorecard**

Show resolution, steps, elapsed minutes, steps/second, final loss, final/max Gaussian count, peak VRAM, final masked PSNR/SSIM, and checkpoint GiB. Derive gates from artifact integrity, finite completion, non-blank render, positive PSNR/SSIM deltas, VRAM below 23 GiB, and presence of unseen-view evidence.

- [ ] **Step 2: Add image reconstruction panels**

For each available run, render GT, initialization, and final preview with identical dimensions and no axes. Add initialization and final mean-absolute RGB error heatmaps using one shared `vmax` per run, plus three fixed crops at relative image coordinates so factor-4 and factor-2 images remain comparable.

- [ ] **Step 3: Add numerical image consistency checks**

Recompute unmasked PNG PSNR and MAE in float64 and display them beside stored masked metrics. Label the recomputed values `PNG/unmasked`; do not mix them into the readiness gate.

- [ ] **Step 4: Add training dynamics plots**

Plot raw loss with a 100-step rolling median, Gaussian count, and position LR. Annotate densification start at step 500, SH changes at 1k/2k/3k, opacity resets at 3k/6k, and checkpoint steps. Use consistent colors for Run A, Run B, and pilot.

- [ ] **Step 5: Execute and reconcile headline values**

Expected Run B headline values: 7,000 steps, about 14.76 minutes, final loss about 0.03336, 2,472,712 final Gaussians, 4.43 GiB peak VRAM, 23.14 dB masked PSNR, and 0.862 masked SSIM.

### Task 3: Performance, storage, conclusion, and visual QA

**Files:**
- Modify: `notebooks/phase3_training_audit.ipynb`

**Interfaces:**
- Consumes: run metrics, timing, checkpoints, and scorecard.
- Produces: performance/storage charts and final evidence-bounded recommendation.

- [ ] **Step 1: Add performance and storage charts**

Plot rolling total step time and throughput, total-time distribution, Gaussian growth versus checkpoint size, and cumulative checkpoint GiB. Mark per-stage CUDA timings as approximate because the trainer does not synchronize around each segment.

- [ ] **Step 2: Add evidence-bounded takeaways**

State separately that Phase 3.6 engineering convergence passes and novel-view readiness remains unverified. Recommend waiting for factor-1 pilot plus deterministic internal holdout before multi-scene full training. Do not infer leaderboard quality from train camera 0.

- [ ] **Step 3: Execute top to bottom**

Run:

```powershell
.\.venv\Scripts\python.exe -m jupyter nbconvert --execute --to notebook --inplace notebooks/phase3_training_audit.ipynb
```

Expected: exit code 0, saved outputs for every chart/table, no raw unbounded logs, and pilot absence represented as a caveat.

- [ ] **Step 4: Render notebook for visual QA**

Run:

```powershell
.\.venv\Scripts\python.exe -m jupyter nbconvert --to html notebooks/phase3_training_audit.ipynb --output-dir notebooks/_rendered
```

Inspect the HTML for clipped labels, inconsistent heatmap scales, unreadable legends, or misleading axes. Remove the temporary rendered directory after QA; the committed deliverable is the executed notebook.

- [ ] **Step 5: Run repository verification**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
git diff --check
```

Expected: all existing tests pass; CUDA/real-data tests may skip on the local CPU environment; diff check reports no whitespace errors.

- [ ] **Step 6: Commit the executed notebook**

```bash
git add notebooks/phase3_training_audit.ipynb docs/superpowers/plans/2026-07-15-phase3-training-audit-notebook.md
git commit -m "docs: add visual phase 3 training audit"
```
