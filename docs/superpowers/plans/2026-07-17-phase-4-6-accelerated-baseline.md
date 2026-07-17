# Phase 4.6 Accelerated Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add qualified fused-Adam and safe FP16-AMP training backends, then produce a deterministic 1,000-step NVIDIA L4 report that selects the backend eligible for the Phase 4 baseline freeze.

**Architecture:** Keep Gaussian parameters, optimizer state, camera math, loss definition, sampler, and density policy unchanged. Isolate precision mechanics in one small controller, expose two hashed CLI fields, and compare three fresh HCM0181 runs with a compact deterministic report.

**Tech Stack:** Python 3.12, PyTorch 2.13 CUDA, gsplat 1.4.0, pytest, Bash, JSON.

## Global Constraints

- Commit directly to `main`, with design and implementation in separate commits.
- Defaults remain `optimizer_backend=adam` and `precision=fp32`.
- `adam-fused` is CUDA-only and never silently falls back.
- `amp-fp16` requires `adam-fused`, CUDA, FP32 model parameters, and one GradScaler.
- Unscale all six optimizer-owned gradients and retained `means2d.grad` before density strategy.
- Keep 30,000 steps, batch size one, seed, sampler, loss, SH schedule, and density thresholds unchanged.
- Do not freeze a production backend until the L4 1,000-step report passes.

---

### Task 1: Canonical backend configuration

**Files:**
- Modify: `src/bts_nvs/training/run_training.py`
- Modify: `tests/unit/test_run_training.py`

**Interfaces:**
- Consumes: CLI values `--optimizer_backend` and `--precision`.
- Produces: hashed config keys `optimizer_backend: str`, `precision: str`; `validate_training_backend(args) -> None`.

- [ ] **Step 1: Write RED CLI/config tests**

Add tests proving defaults are `adam/fp32`, both keys enter `build_training_config`, `amp-fp16 + adam` is rejected, and the three sanctioned pairs are accepted.

```python
@pytest.mark.parametrize("backend,precision", [
    ("adam", "fp32"),
    ("adam-fused", "fp32"),
    ("adam-fused", "amp-fp16"),
])
def test_training_backend_contract_accepts_supported_pairs(backend, precision):
    run_training.validate_training_backend(
        _args(optimizer_backend=backend, precision=precision)
    )
```

- [ ] **Step 2: Run RED test**

Run: `.venv\Scripts\python -m pytest tests/unit/test_run_training.py -q`

Expected: FAIL because the new arguments and validator do not exist.

- [ ] **Step 3: Implement minimal CLI and validation**

Use argparse choices, call the validator before CUDA preflight, and add both exact strings to `build_training_config`.

- [ ] **Step 4: Run GREEN test**

Run: `.venv\Scripts\python -m pytest tests/unit/test_run_training.py -q`

Expected: all tests pass.

### Task 2: Fused Adam without fallback

**Files:**
- Modify: `src/bts_nvs/models/optimizer.py`
- Modify: `src/bts_nvs/training/trainer.py`
- Modify: `tests/unit/test_loss_optimizer.py`
- Modify: `tests/unit/test_trainer_loop.py`

**Interfaces:**
- Consumes: `setup_optimizers(gaussians, *, backend="adam")`.
- Produces: six standard Adam optimizers with identical groups and `fused=True` only for `adam-fused`.

- [ ] **Step 1: Write RED optimizer tests**

Assert unknown backends fail, reference defaults preserve the current optimizer contract, fused backend sets `optimizer.defaults["fused"] is True`, and Trainer rejects fused on a CPU device before training.

- [ ] **Step 2: Run RED tests**

Run: `.venv\Scripts\python -m pytest tests/unit/test_loss_optimizer.py tests/unit/test_trainer_loop.py -q`

Expected: FAIL because `backend` is unsupported.

- [ ] **Step 3: Implement backend selection**

Pass `fused=True` only for the fused branch. Validate the device in Trainer and pass the config value to `setup_optimizers`; do not catch backend errors or retry with ordinary Adam.

- [ ] **Step 4: Run GREEN tests**

Run the same pytest command; expected: all pass.

### Task 3: AMP controller with density-safe unscaling

**Files:**
- Create: `src/bts_nvs/training/precision.py`
- Modify: `src/bts_nvs/training/trainer.py`
- Create: `tests/unit/test_training_precision.py`
- Modify: `tests/unit/test_trainer_loop.py`

**Interfaces:**
- Produces: `TrainingPrecision(mode: str, device: torch.device)`, `autocast()`, `backward_and_unscale(loss, optimizers, projected_means) -> float`, and `step(optimizers) -> None`.
- Consumes: six optimizers and retained `result.info["means2d"]`.

- [ ] **Step 1: Write RED precision tests**

Use an injectable fake scaler to prove each optimizer is unscaled once, projected non-leaf gradient is divided by the same scale before strategy sees it, each optimizer is stepped once, scaler updates once, and FP32 executes ordinary backward/step unchanged.

```python
scale = controller.backward_and_unscale(loss, optimizers, projected)
assert scale == 8.0
assert torch.equal(projected.grad, expected_unscaled_grad)
controller.step(optimizers)
assert fake_scaler.update_calls == 1
```

- [ ] **Step 2: Run RED test**

Run: `.venv\Scripts\python -m pytest tests/unit/test_training_precision.py -q`

Expected: FAIL because the module does not exist.

- [ ] **Step 3: Implement the minimal controller**

Use `nullcontext()` for FP32 and `torch.autocast("cuda", dtype=torch.float16)` plus `torch.amp.GradScaler("cuda")` for AMP. After backward, unscale every optimizer, manually divide `projected_means.grad`, then return the scale for auditing. Reject missing/non-finite projected gradient in AMP.

- [ ] **Step 4: Integrate trainer in the required order**

Wrap only renderer and loss in `autocast()`. Call controller backward/unscale, existing finite checks, density strategy, controller optimizer steps, then scheduler. Keep the existing FP32 execution order semantically identical.

- [ ] **Step 5: Run GREEN and trainer regression tests**

Run: `.venv\Scripts\python -m pytest tests/unit/test_training_precision.py tests/unit/test_trainer_loop.py -q`

Expected: all pass.

### Task 4: Compact 1,000-step gradient audit and comparator

**Files:**
- Create: `src/bts_nvs/training/backend_qualification.py`
- Create: `src/bts_nvs/training/compare_backend_qualification.py`
- Modify: `src/bts_nvs/training/trainer.py`
- Modify: `src/bts_nvs/training/run_training.py`
- Create: `tests/unit/test_backend_qualification.py`
- Modify: `tests/unit/test_trainer_loop.py`

**Interfaces:**
- Consumes: config flag `backend_qualification`, metrics/timing traces, selected gradient observations.
- Produces: per-run `backend_profile.json`; aggregate `backend_qualification.json`; `compare_backend_profiles(reference, fused, amp) -> dict`.

- [ ] **Step 1: Write RED schema/comparator tests**

Prove JSON rejects NaN, mismatched sample indices, non-finite gradients, wrong event steps, >1% count drift, >2% rolling loss drift, F1 speedup below 10%, and F2 incremental speedup below 5%. Prove selection order `amp-fp16`, `adam-fused`, then reference fallback.

- [ ] **Step 2: Run RED test**

Run: `.venv\Scripts\python -m pytest tests/unit/test_backend_qualification.py -q`

Expected: FAIL because qualification APIs do not exist.

- [ ] **Step 3: Implement deterministic records**

Write atomic, sorted, `allow_nan=False` JSON. Record all 1,000 sample/loss/count traces and gradient summaries only at steps `1, 499, 500, 501, 600, 1000`. Compare the pre-topology prefix through step 599 and rolling final 100 losses.

- [ ] **Step 4: Wire qualification mode**

Add `--backend_qualification`, require HCM0181, factor 1, seed 0, exactly 1,000 steps, cached/pinned input, internal holdout, fresh output, and no checkpoints. Write the profile at run completion.

- [ ] **Step 5: Run GREEN tests**

Run: `.venv\Scripts\python -m pytest tests/unit/test_backend_qualification.py tests/unit/test_run_training.py tests/unit/test_trainer_loop.py -q`

Expected: all pass.

### Task 5: L4 runner and phase documentation

**Files:**
- Create: `scripts/run_phase4_backend_qualification.sh`
- Modify: `tests/unit/test_phase4_shell_scripts.py`
- Modify: `docs/phase_4_spec.md`
- Modify: `docs/phase_3_6_l4_runbook.md`

**Interfaces:**
- Consumes: prepared HCM0181 manifest/holdout and CUDA `.venv`.
- Produces: three run directories and `runs/phase4/backend_qualification/backend_qualification.json`.

- [ ] **Step 1: Write RED script contract test**

Assert the script uses three fresh directories, exactly 1,000 steps, factor 1, cached/pinned input, backend qualification mode, and invokes the comparator only after all profiles exist.

- [ ] **Step 2: Run RED test**

Run: `.venv\Scripts\python -m pytest tests/unit/test_phase4_shell_scripts.py -q`

Expected: FAIL because the script is missing.

- [ ] **Step 3: Implement shell runner and docs**

The script runs R0, F1, F2 sequentially on one GPU, supports only path and Python overrides, refuses non-empty incomplete directories, and prints the selected backend from the aggregate report. Document L4 prerequisites and exact pass conditions.

- [ ] **Step 4: Run full verification**

Run:

```powershell
.venv\Scripts\python -m pytest tests/unit -q
.venv\Scripts\python -m pytest tests/integration -q
git diff --check
```

Expected: unit tests pass; CUDA/data-dependent integration tests skip on CPU local; no whitespace errors.

- [ ] **Step 5: Red review and implementation commit**

Verify AMP projected gradients are unscaled before strategy, all six optimizer/scaler calls obey PyTorch order, old config/checkpoints remain compatible, no fallback hides backend identity, and qualification cannot access official test images. Commit as:

```text
feat: qualify accelerated phase 4 baseline
```

## External GPU gate

On the L4 VM run:

```bash
bash scripts/run_phase4_backend_qualification.sh
```

Phase 4.6a passes only when the aggregate JSON selects a backend and reports
`accepted: true`.  Return that artifact for audit.  Phase 4.6b then writes the
immutable baseline bundle and compact inference artifact using the selected
backend; it must not guess the result locally.

