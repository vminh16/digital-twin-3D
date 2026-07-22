# Candidate Registry and AbsGrad Hooks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a generic, immutable candidate registry and the minimal AbsGrad-capable runtime hooks required to execute the first density candidates without changing legacy B0 behavior.

**Architecture:** Candidate identity and validation live in a new `bts_nvs.experiments` package; renderer, density strategy, precision, and trainer consume only primitive settings from the locked candidate contract. Existing training entry points remain operationally unchanged, and no candidate CLI or experiment orchestration is added until Module 3.

**Tech Stack:** Python 3.10+, dataclasses, hashlib/JSON, PyTorch, pinned `gsplat==1.4.0`, pytest, existing trainer/rendering APIs.

## Global Constraints

- Implement directly on `main`; do not create a branch or worktree.
- Keep `B0-submission-q99-v1` closed and immutable.
- The only executable candidate IDs are `B0-reference`, `E1-density-absgrad-t04-v1`, and `E1-density-scale005-v1`.
- AbsGrad and `grow_scale3d=0.005` remain isolated candidates; do not combine them.
- Do not add revised opacity, `grow_grad2d=0.0008`, C1 IDs, C1 runners, or phase-specific behavior.
- Do not change optimizer learning rates, loss weights, SH schedule, initialization, camera model, normalization, codec, inference, or submission behavior.
- Do not add a Bash runner or a second training loop.
- Old configs with no candidate fields must retain effective B0 renderer and strategy behavior.
- CPU tests use fakes or monkeypatches and must not require CUDA.
- The one real AbsGrad smoke runs explicitly on the NVIDIA L4 and must not run in the default unit suite.
- Follow TDD and commit each task separately, staging only the files named by that task.
- Preserve user-owned `.gitignore` and `docs/research/` changes.

---

### Task 1: Immutable Candidate Contract and Registry

**Files:**
- Create: `src/bts_nvs/experiments/__init__.py`
- Create: `src/bts_nvs/experiments/contracts.py`
- Create: `src/bts_nvs/experiments/candidates.py`
- Create: `tests/unit/test_experiment_candidates.py`

**Interfaces:**
- Produces: `CandidateSettings`, `CANDIDATE_IDS`, `candidate_settings(candidate_id)`, and `candidate_training_overrides(candidate_id)`.
- Consumers: Tasks 3–5 and the later generic experiment runner.

- [x] **Step 1: Write failing registry tests**

Create tests that lock every field, reject mutation, reject unknown IDs, and prove that each experimental candidate changes exactly one B0 mechanism:

```python
from dataclasses import FrozenInstanceError, replace

import pytest

from bts_nvs.experiments.candidates import (
    CANDIDATE_IDS,
    candidate_settings,
    candidate_training_overrides,
)


def test_registry_locks_first_executable_candidates() -> None:
    assert CANDIDATE_IDS == (
        "B0-reference",
        "E1-density-absgrad-t04-v1",
        "E1-density-scale005-v1",
    )
    baseline = candidate_settings("B0-reference")
    absgrad = candidate_settings("E1-density-absgrad-t04-v1")
    scale = candidate_settings("E1-density-scale005-v1")

    assert absgrad == replace(
        baseline,
        candidate_id="E1-density-absgrad-t04-v1",
        absgrad=True,
        grow_grad2d=0.0004,
    )
    assert scale == replace(
        baseline,
        candidate_id="E1-density-scale005-v1",
        grow_scale3d=0.005,
    )


def test_candidate_settings_are_immutable_and_unknown_ids_fail() -> None:
    settings = candidate_settings("B0-reference")
    with pytest.raises(FrozenInstanceError):
        settings.absgrad = True
    with pytest.raises(ValueError, match="unknown candidate"):
        candidate_settings("C1-absgrad-t08-revopacity-v1")


def test_training_overrides_are_complete_plain_values() -> None:
    overrides = candidate_training_overrides("E1-density-absgrad-t04-v1")
    assert overrides == {
        "candidate_id": "E1-density-absgrad-t04-v1",
        "absgrad": True,
        "grow_grad2d": 0.0004,
        "grow_scale3d": 0.01,
        "prune_opa": 0.005,
        "refine_stop_step": 15_000,
        "rasterize_mode": "classic",
        "appearance_mode": "baseline",
        "sampling_mode": "uniform",
    }
```

Add parameterized tests proving `CandidateSettings` rejects booleans in numeric fields, non-finite/non-positive thresholds, `prune_opa >= 1`, non-positive `refine_stop_step`, empty IDs, and unknown rasterize/appearance/sampling modes.

- [x] **Step 2: Run the focused test and verify RED**

Run:

```bash
pytest -q tests/unit/test_experiment_candidates.py
```

Expected: collection fails because `bts_nvs.experiments` does not exist.

- [x] **Step 3: Implement the immutable contract**

Implement this exact dataclass in `contracts.py`:

```python
@dataclass(frozen=True)
class CandidateSettings:
    candidate_id: str
    absgrad: bool
    grow_grad2d: float
    grow_scale3d: float
    prune_opa: float
    refine_stop_step: int
    rasterize_mode: str
    appearance_mode: str
    sampling_mode: str

    def training_overrides(self) -> dict[str, bool | float | int | str]:
        return asdict(self)
```

`__post_init__` must validate all fields without coercing values. Allowed mode values in Module 2 are `rasterize_mode in {"classic", "antialiased"}`, `appearance_mode == "baseline"`, and `sampling_mode == "uniform"`. Allowing the rasterizer enum does not authorize an antialiased candidate; the registry remains the executable authority.

- [x] **Step 4: Implement the closed registry**

Store exactly three preconstructed settings objects in a private mapping. `candidate_settings()` must require a non-empty string and return only a registered object. `candidate_training_overrides()` must return a fresh plain dictionary so callers cannot mutate registry state.

- [x] **Step 5: Run focused tests**

Run:

```bash
pytest -q tests/unit/test_experiment_candidates.py
```

Expected: all candidate tests pass.

- [x] **Step 6: Commit Task 1**

```bash
git add src/bts_nvs/experiments/__init__.py src/bts_nvs/experiments/contracts.py src/bts_nvs/experiments/candidates.py tests/unit/test_experiment_candidates.py
git commit -m "feat: add locked experiment candidates"
```

### Task 2: Canonical Provenance Utilities

**Files:**
- Create: `src/bts_nvs/experiments/provenance.py`
- Create: `tests/unit/test_experiment_provenance.py`

**Interfaces:**
- Produces: `canonical_json_sha256(record)`, `save_json_artifact(record, path)`, and `load_json_artifact(path, expected_sha256=None)`.
- Consumers: Module 3 experiment/decision artifacts; this task does not refactor Module 1 writers.

- [x] **Step 1: Write failing provenance tests**

```python
import json

import pytest

from bts_nvs.experiments.provenance import (
    canonical_json_sha256,
    load_json_artifact,
    save_json_artifact,
)


def test_semantically_equal_records_have_the_same_hash() -> None:
    first = {"candidate_id": "B0-reference", "values": [1, 2]}
    second = {"values": [1, 2], "candidate_id": "B0-reference"}
    assert canonical_json_sha256(first) == canonical_json_sha256(second)


def test_artifact_save_is_atomic_canonical_and_hash_checked(tmp_path) -> None:
    path = tmp_path / "experiment.json"
    record = {"schema_version": 1, "candidate_id": "B0-reference"}
    digest = save_json_artifact(record, path)
    first = path.read_bytes()

    assert load_json_artifact(path, expected_sha256=digest) == record
    assert first.endswith(b"\n") and b"\r\n" not in first
    save_json_artifact(dict(reversed(tuple(record.items()))), path)
    assert path.read_bytes() == first

    with pytest.raises(ValueError, match="SHA-256"):
        load_json_artifact(path, expected_sha256="0" * 64)
```

Also reject non-object top-level JSON, malformed expected hashes, NaN/Infinity, unreadable JSON, and booleans where a digest string is required. Verify a failed serialization leaves an existing artifact unchanged and leaves no `.tmp` file.

- [x] **Step 2: Run the focused test and verify RED**

Run `pytest -q tests/unit/test_experiment_provenance.py`.

Expected: import fails because `provenance.py` is missing.

- [x] **Step 3: Implement canonical hash and atomic I/O**

Hash UTF-8 bytes from:

```python
json.dumps(
    record,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=False,
    allow_nan=False,
).encode("utf-8")
```

Save a human-readable sorted JSON representation to a sibling temporary file, append exactly one LF, and use `os.replace`. Remove the temporary file in a `finally` block only when it still exists. `load_json_artifact` must recompute the semantic hash after parsing and compare with `hmac.compare_digest`.

- [x] **Step 4: Run focused and existing artifact tests**

Run:

```bash
pytest -q tests/unit/test_experiment_provenance.py tests/unit/test_experiment_report.py tests/unit/test_pose_strata.py tests/unit/test_training_profiling.py
```

Expected: all tests pass; existing artifact writers remain unchanged.

- [x] **Step 5: Commit Task 2**

```bash
git add src/bts_nvs/experiments/provenance.py tests/unit/test_experiment_provenance.py
git commit -m "feat: add experiment provenance utilities"
```

### Task 3: Renderer and Density-Strategy Hooks

**Files:**
- Modify: `src/bts_nvs/rendering/gsplat_renderer.py`
- Modify: `src/bts_nvs/rendering/density_strategy.py`
- Modify: `tests/unit/test_renderer.py`
- Modify: `tests/unit/test_strategy.py`

**Interfaces:**
- `render_gaussians(..., absgrad: bool = False, rasterize_mode: str = "classic") -> RenderResult`.
- `GsplatStrategy(..., absgrad: bool = False, ...)`.
- Consumes: primitive values from `CandidateSettings`; no import from the registry is allowed in rendering modules.

- [x] **Step 1: Add failing renderer hook tests**

Extend the fake-rasterization tests:

```python
def test_renderer_forwards_absgrad_and_rasterize_mode(monkeypatch) -> None:
    captured = {}

    def fake_rasterization(**kwargs):
        captured.update(kwargs)
        return (
            torch.zeros((1, 16, 16, 3)),
            torch.zeros((1, 16, 16, 1)),
            {"means2d": torch.zeros((1, 1, 2))},
        )

    monkeypatch.setattr(gsplat_renderer, "rasterization", fake_rasterization)
    gsplat_renderer.render_gaussians(
        _gaussians(),
        torch.eye(4),
        _intrinsics(),
        active_sh_degree=0,
        absgrad=True,
        rasterize_mode="classic",
    )
    assert captured["absgrad"] is True
    assert captured["rasterize_mode"] == "classic"
```

Add rejection tests for non-boolean `absgrad` and rasterizer values outside `classic`/`antialiased`. Keep the existing default assertion that B0 passes `absgrad=False` and `rasterize_mode="classic"`.

- [x] **Step 2: Add failing strategy hook tests**

```python
def test_strategy_forwards_absgrad_without_revised_opacity(monkeypatch) -> None:
    monkeypatch.setattr(density_strategy, "DefaultStrategy", _FakeDefaultStrategy)
    gaussians = _gaussians()
    strategy = GsplatStrategy(
        gaussians,
        setup_optimizers(gaussians),
        grow_grad2d=0.0004,
        absgrad=True,
    )
    assert strategy.backend.config["grow_grad2d"] == pytest.approx(0.0004)
    assert strategy.backend.config["absgrad"] is True
    assert "revised_opacity" not in strategy.backend.config
```

Add a non-boolean `absgrad` rejection test and retain the exact B0 backend dictionary assertion.

- [x] **Step 3: Run focused tests and verify RED**

Run:

```bash
pytest -q tests/unit/test_renderer.py tests/unit/test_strategy.py
```

Expected: new calls fail because the signatures do not yet accept the hook fields.

- [x] **Step 4: Implement minimal primitive hooks**

Add validated keyword-only arguments and forward them directly to gsplat. Do not import candidate IDs into rendering code. Do not add revised opacity. Defaults must remain `False` and `classic`.

- [x] **Step 5: Run focused tests**

Run `pytest -q tests/unit/test_renderer.py tests/unit/test_strategy.py`.

Expected: all renderer and strategy tests pass; real CUDA tests remain skipped unless their existing requirements are available.

- [x] **Step 6: Commit Task 3**

```bash
git add src/bts_nvs/rendering/gsplat_renderer.py src/bts_nvs/rendering/density_strategy.py tests/unit/test_renderer.py tests/unit/test_strategy.py
git commit -m "feat: add generic density runtime hooks"
```

### Task 4: AMP and Trainer Candidate Forwarding

**Files:**
- Modify: `src/bts_nvs/training/precision.py`
- Modify: `src/bts_nvs/training/trainer.py`
- Modify: `tests/unit/test_training_precision.py`
- Modify: `tests/unit/test_trainer_loop.py`

**Interfaces:**
- `TrainingPrecision.backward_and_unscale(...)` continues returning the loss scale and additionally unscales `projected_means.absgrad` when it exists.
- `Trainer` reads optional `absgrad` and `rasterize_mode` config values and forwards them without candidate-specific branches.

- [x] **Step 1: Write the failing AMP tests**

```python
def test_amp_unscales_signed_and_absolute_projected_gradients_once(monkeypatch) -> None:
    fake = _FakeScaler(scale=8.0)
    monkeypatch.setattr(
        precision_module.torch.amp,
        "GradScaler",
        lambda *args, **kwargs: fake,
    )
    parameter = torch.nn.Parameter(torch.tensor([2.0]))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    projected = parameter * 3.0
    projected.retain_grad()
    projected.absgrad = torch.tensor([16.0])
    controller = TrainingPrecision("amp-fp16", torch.device("cuda"))

    controller.backward_and_unscale(projected.sum(), {"p": optimizer}, projected)

    assert torch.equal(projected.grad, torch.ones_like(projected))
    assert torch.equal(projected.absgrad, torch.tensor([2.0]))
```

Also verify FP32 does not alter a supplied `absgrad`, AMP still works when the attribute is absent, and a non-tensor `absgrad` raises a targeted runtime error instead of being ignored.

- [x] **Step 2: Write the failing trainer forwarding test**

Construct a trainer with config overrides from `E1-density-absgrad-t04-v1`. Monkeypatch environment-version lookup where required so this CPU test does not depend on installed gsplat package metadata. Capture the training render call and assert:

```python
assert captured["render_absgrad"] is True
assert captured["rasterize_mode"] == "classic"
assert trainer.strategy.backend.config["absgrad"] is True
assert trainer.strategy.backend.config["grow_grad2d"] == pytest.approx(0.0004)
```

Add a paired B0 test asserting false/classic/0.0002. The test must execute one actual `Trainer.train(..., stop_after_step=1)` path with fake renderer/strategy components rather than inspecting config only.

- [x] **Step 3: Run focused tests and verify RED**

Run:

```bash
pytest -q tests/unit/test_training_precision.py tests/unit/test_trainer_loop.py -k "absgrad or candidate_forwarding"
```

Expected: new assertions fail because AbsGrad is neither unscaled nor forwarded.

- [x] **Step 4: Implement AMP unscale and trainer forwarding**

After `scaler.unscale_()` handles optimizer-owned leaf gradients, divide both projected gradient tensors by the captured scale. Require signed `.grad`; when `.absgrad` exists require it to be a finite tensor before and after division. In `Trainer`, use configuration defaults only:

```python
absgrad = self.config.get("absgrad", False)
rasterize_mode = self.config.get("rasterize_mode", "classic")
```

Forward `absgrad` to both training rasterization and `GsplatStrategy`; forward `rasterize_mode` only to rasterization. Do not branch on candidate ID. Train-view diagnostics remain B0-style because they do not backpropagate or influence density control.

- [x] **Step 5: Run focused and adjacent tests**

Run:

```bash
pytest -q tests/unit/test_training_precision.py tests/unit/test_renderer.py tests/unit/test_strategy.py tests/unit/test_trainer_loop.py
```

Expected on a complete development environment: all tests pass. On the current local Windows environment, record any pre-existing `PackageNotFoundError: gsplat` separately; do not alter production code to hide missing dependency metadata.

- [x] **Step 6: Commit Task 4**

```bash
git add src/bts_nvs/training/precision.py src/bts_nvs/training/trainer.py tests/unit/test_training_precision.py tests/unit/test_trainer_loop.py
git commit -m "feat: forward candidate density settings"
```

### Task 5: Real L4 AbsGrad Densification Gate

**Files:**
- Create: `tests/integration/test_absgrad_density_smoke.py`

**Interfaces:**
- Consumes the registered `E1-density-absgrad-t04-v1` settings and existing CUDA preflight components.
- Produces no run artifacts or checkpoints; success is the pytest result and finite runtime assertions.

- [x] **Step 1: Write the opt-in L4 integration smoke**

Guard the test with `BTS_RUN_ABSGRAD_SMOKE=1`. It must use the real renderer and `DefaultStrategy`, one tiny CUDA Gaussian set, `adam-fused` plus `amp-fp16`, the exact registered candidate settings, and enough one-based steps to execute the first refinement event. At the event assert:

```python
assert isinstance(means2d.absgrad, torch.Tensor)
assert torch.isfinite(means2d.absgrad).all()
assert torch.isfinite(strategy_state["grad2d"]).all()
assert gaussians.num_gaussians > 0
```

The test must call the same `TrainingPrecision`, `render_gaussians`, and `GsplatStrategy` APIs used by `Trainer`. It must not create a second reusable training loop, save checkpoints, decode scene images, or require external data.

- [x] **Step 2: Verify default local behavior**

Run:

```bash
pytest -q tests/integration/test_absgrad_density_smoke.py
```

Expected locally: one skipped test with the environment-variable reason; no CUDA allocation.

Observed locally on 2026-07-22: `1 skipped`; CUDA was not allocated.

- [x] **Step 3: Run the smoke on the NVIDIA L4**

Run on the VM from a clean tracked worktree:

```bash
BTS_RUN_ABSGRAD_SMOKE=1 pytest -q tests/integration/test_absgrad_density_smoke.py
```

Expected: one test passes, the first refinement event completes, and signed/absolute projected gradients remain finite. Any OOM, missing `absgrad`, non-finite state, zero Gaussian count, or unsupported gsplat API fails Module 2 before a 7k run is authorized.

Status on 2026-07-22: user reported that the NVIDIA L4 smoke passed. The exact
terminal transcript was not imported into this workspace; this record preserves
the distinction between user-confirmed VM evidence and locally executed tests.

- [x] **Step 4: Run the complete Module 2 CPU suite**

Run:

```bash
pytest -q \
  tests/unit/test_experiment_candidates.py \
  tests/unit/test_experiment_provenance.py \
  tests/unit/test_renderer.py \
  tests/unit/test_strategy.py \
  tests/unit/test_training_precision.py \
  tests/unit/test_trainer_loop.py \
  tests/unit/test_run_training.py
```

Expected on the VM environment with pinned dependencies: all selected tests pass.

Observed locally on 2026-07-22: `143 passed, 3 skipped`.

- [x] **Step 5: Commit Task 5**

```bash
git add tests/integration/test_absgrad_density_smoke.py
git commit -m "test: gate AbsGrad density path on L4"
```

### Task 6: Module 2 Contract Verification

**Files:**
- Modify: `docs/superpowers/plans/2026-07-22-candidate-registry-and-absgrad-hooks.md`

**Interfaces:**
- Produces the checked implementation record and Module 2 handoff; no runtime behavior.

- [x] **Step 1: Verify the allowed diff scope**

Run `git diff --name-status fb9eb4a..HEAD`.

Expected files are limited to the new `experiments` package/tests, the four runtime hook files and their tests, the opt-in integration smoke, and this plan.

- [x] **Step 2: Verify B0 compatibility and forbidden C1 content**

Run:

```bash
pytest -q tests/unit/test_renderer.py tests/unit/test_strategy.py tests/unit/test_training_precision.py
rg -n "revised_opacity|C1-|0\.0008|phase_[abc]" \
  src/bts_nvs/experiments \
  src/bts_nvs/rendering/gsplat_renderer.py \
  src/bts_nvs/rendering/density_strategy.py \
  src/bts_nvs/training/precision.py \
  src/bts_nvs/training/trainer.py
```

Expected: B0-focused tests pass and `rg` returns no matches.

- [x] **Step 3: Verify formatting and incomplete markers**

Run:

```bash
git diff --check fb9eb4a..HEAD
rg -n "TB[D]|TO[D]O|implement[ ]later|fill[ ]in details|similar[ ]to Task" \
  docs/superpowers/plans/2026-07-22-candidate-registry-and-absgrad-hooks.md \
  src/bts_nvs/experiments \
  tests/unit/test_experiment_candidates.py \
  tests/unit/test_experiment_provenance.py
```

Expected: no whitespace errors and no incomplete markers.

Observed local full unit suite on 2026-07-22: `415 passed, 5 skipped`.
The allowed diff and forbidden-C1 scans passed. The NVIDIA L4 smoke was later
reported as passed by the user and recorded in Task 5 Step 3.

- [x] **Step 4: Record verification and commit only the plan update**

```bash
git add docs/superpowers/plans/2026-07-22-candidate-registry-and-absgrad-hooks.md
git commit -m "docs: record candidate hook verification"
```

The handoff must include exact CPU test counts, the L4 smoke result, commit IDs, peak CUDA allocation if available, unchanged user-owned files, and explicit confirmation that no 7k/30k training or candidate decision was executed.

## Module Gate and Next Authorization

Module 2 passes only after the CPU suite and opt-in L4 smoke both pass. Passing Module 2 authorizes planning Module 3—the generic runner and decision engine—but does not authorize any 7k candidate run. Stage A references and Stage B1 screens remain gated by the later orchestration contract.
