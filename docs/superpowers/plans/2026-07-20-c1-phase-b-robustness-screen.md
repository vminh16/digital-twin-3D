# C1 Phase B Robustness Screen Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the Phase-A winner at 7,000 steps on the four remaining calibration scenes and emit a validated six-scene Phase-B decision without adding a shell runner.

**Architecture:** Extract only the reusable 7k screening mechanics from the Phase-A runner into a small common module. Keep Phase-A and Phase-B orchestration and decisions separate, expose both through one thin `run_c1_screening.py` CLI, and store each Phase-B run directly at `runs/c1/phase_b/<scene>/`. Rename durable `scripts/` entry points by function with no wrappers and no change to historical artifact paths.

**Tech Stack:** Python 3.12, pytest, NumPy, PyYAML, existing gsplat/PyTorch training stack, Bash entry points, JSON/YAML artifacts.

## Global Constraints

- Continue on `ex1/absgrad-revopacity-phase-a`; do not create another branch.
- Locked candidate: `C1-absgrad-t08-revopacity-v1`.
- New scenes in exact order and case: `hcm0031 HCM0181 HNI0131 HNI0265`.
- Reuse Phase-A runs and historical `B0-reference`; never retrain B0.
- Run at factor 1, seed 0, 7,000 steps, internal holdout, cached images, pinned transfer, and the accepted backend/precision.
- Renderer and strategy both receive `absgrad=true`; strategy receives `revised_opacity=true`; `grow_grad2d=0.0008`.
- Do not save checkpoints in Phase B.
- Do not add a Phase-B shell script.
- Do not rename `runs/phase4/...`; those paths identify closed artifacts.
- Do not stage or alter the pre-existing deleted `outputs_auxiliary` files.

---

### Task 1: Rename durable shell entry points by function

**Files:**
- Rename: `scripts/prepare_phase4_artifacts.sh` → `scripts/prepare_scene_manifests.sh`
- Rename: `scripts/run_phase4_qualification.sh` → `scripts/run_baseline_screening.sh`
- Rename: `scripts/run_phase4_30k_dry_run.sh` → `scripts/run_full_length_qualification.sh`
- Rename: `scripts/run_phase4_backend_qualification.sh` → `scripts/qualify_training_backend.sh`
- Rename: `scripts/run_phase4_full_training.sh` → `scripts/train_scene_cohort.sh`
- Rename: `scripts/run_phase4_inference.sh` → `scripts/render_scene_cohort.sh`
- Rename: `tests/unit/test_phase4_shell_scripts.py` → `tests/unit/test_operational_shell_scripts.py`
- Modify: `README.md`
- Modify: `docs/phase_4_spec.md`
- Modify: shell-to-shell references in `scripts/*.sh`

**Interfaces:**
- Consumes: existing shell behavior and environment variables unchanged.
- Produces: six semantic entry-point names; zero additional shell files.

- [ ] **Step 1: Rename scripts and their test with Git history preserved**

```bash
git mv scripts/prepare_phase4_artifacts.sh scripts/prepare_scene_manifests.sh
git mv scripts/run_phase4_qualification.sh scripts/run_baseline_screening.sh
git mv scripts/run_phase4_30k_dry_run.sh scripts/run_full_length_qualification.sh
git mv scripts/run_phase4_backend_qualification.sh scripts/qualify_training_backend.sh
git mv scripts/run_phase4_full_training.sh scripts/train_scene_cohort.sh
git mv scripts/run_phase4_inference.sh scripts/render_scene_cohort.sh
git mv tests/unit/test_phase4_shell_scripts.py tests/unit/test_operational_shell_scripts.py
```

- [ ] **Step 2: Update the test constants before changing call sites**

```python
PREPARE_SCRIPT = REPO_ROOT / "scripts" / "prepare_scene_manifests.sh"
QUALIFICATION_SCRIPT = REPO_ROOT / "scripts" / "run_baseline_screening.sh"
DRY_RUN_SCRIPT = REPO_ROOT / "scripts" / "run_full_length_qualification.sh"
BACKEND_SCRIPT = REPO_ROOT / "scripts" / "qualify_training_backend.sh"
FULL_TRAINING_SCRIPT = REPO_ROOT / "scripts" / "train_scene_cohort.sh"
INFERENCE_SCRIPT = REPO_ROOT / "scripts" / "render_scene_cohort.sh"
```

Change every assertion for the preparation script to:

```python
assert "prepare_scene_manifests.sh" in script
```

- [ ] **Step 3: Run the renamed test and verify it fails on stale internal references**

Run:

```bash
python -m pytest tests/unit/test_operational_shell_scripts.py -q
```

Expected: failures show `prepare_phase4_artifacts.sh` remains inside renamed scripts.

- [ ] **Step 4: Replace active call sites and documentation commands exactly**

Use this mapping only for executable paths; retain `runs/phase4`,
`prepare_phase4_artifacts.py`, phase-numbered historical headings, and artifact
IDs unchanged:

```text
prepare_phase4_artifacts.sh          -> prepare_scene_manifests.sh
run_phase4_qualification.sh          -> run_baseline_screening.sh
run_phase4_30k_dry_run.sh            -> run_full_length_qualification.sh
run_phase4_backend_qualification.sh  -> qualify_training_backend.sh
run_phase4_full_training.sh          -> train_scene_cohort.sh
run_phase4_inference.sh              -> render_scene_cohort.sh
```

- [ ] **Step 5: Verify semantic names and absence of wrappers**

Run:

```bash
python -m pytest tests/unit/test_operational_shell_scripts.py -q
rg -n "scripts/(prepare_phase4|run_phase4)" README.md docs scripts tests
git ls-files "scripts/*.sh"
```

Expected: tests pass; `rg` has no executable old-name reference; the script
count is unchanged.

- [ ] **Step 6: Commit the rename as an isolated mechanical change**

```bash
git add README.md docs/phase_4_spec.md scripts tests/unit/test_operational_shell_scripts.py
git commit -m "refactor: name training scripts by purpose"
```

---

### Task 2: Extract reusable 7k C1 screening mechanics

**Files:**
- Create: `src/bts_nvs/training/c1_screening.py`
- Create: `tests/unit/test_c1_screening.py`
- Modify: `src/bts_nvs/training/c1_phase_a.py:14-31`
- Modify: `src/bts_nvs/training/c1_phase_a_runner.py:30-177`
- Modify: `tests/unit/test_c1_phase_a.py`
- Modify: `tests/unit/test_c1_phase_a_runner.py`

**Interfaces:**
- Produces: `BASELINE_CANDIDATE`, `MAX_VRAM_MB`, `score50`, `atomic_json`, `load_report`, `load_completed_run`, `build_screening_command`, and `diagnostics_for_run`.
- Consumes: `candidate_settings`, `BackendDecision`, `compute_config_sha256`, manifests, holdouts, and existing high-frequency evaluator.

- [ ] **Step 1: Write focused tests for generic output paths and fixed contract**

```python
def test_screening_command_uses_explicit_flat_run_dir(tmp_path) -> None:
    command = build_screening_command(
        repo_root=tmp_path / "repo",
        scenes_root=tmp_path / "scenes",
        manifests_root=tmp_path / "manifests",
        run_dir=tmp_path / "phase_b" / "HCM0181",
        scene_id="HCM0181",
        candidate_id="C1-absgrad-t08-revopacity-v1",
        decision=SimpleNamespace(optimizer_backend="adam", precision="fp32"),
        python_bin="python",
    )
    assert command[command.index("--output_dir") + 1] == str(
        tmp_path / "phase_b" / "HCM0181"
    )
    assert "--max_steps 7000" in " ".join(command)
    assert "--rolling_checkpoint" not in command
    assert "--resume" not in command


def test_score50_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        score50({"psnr_db_mean": float("nan"), "ssim_mean": 0.8, "lpips_mean": 0.2})
```

- [ ] **Step 2: Run the new tests and verify import failure**

```bash
python -m pytest tests/unit/test_c1_screening.py -q
```

Expected: collection fails because `bts_nvs.training.c1_screening` does not exist.

- [ ] **Step 3: Implement the common contract without stage-specific scene lists**

Create constants `BASELINE_CANDIDATE = "B0-reference"` and
`MAX_VRAM_MB = 23 * 1024`, then expose these exact typed interfaces:

- `score50(report: Mapping[str, object]) -> float`
- `atomic_json(path: Path, payload: dict) -> None`
- `load_report(path: Path, scene_id: str, candidate_id: str) -> dict`
- `load_completed_run(run_dir: Path, scene_id: str, candidate_id: str, decision: BackendDecision) -> dict`
- `build_screening_command(*, repo_root: Path, scenes_root: Path, manifests_root: Path, run_dir: Path, scene_id: str, candidate_id: str, decision: BackendDecision, python_bin: str) -> list[str]`
- `diagnostics_for_run(*, scene_id: str, candidate_id: str, scene_root: Path, manifest_root: Path, render_dir: Path) -> dict`

`load_completed_run` validates this exact mapping and rejects `.pt`/`.pth`:

```python
expected = {
    "scene_id": scene_id,
    "qualification_candidate": candidate_id,
    "resize_factor": 1,
    "max_steps": 7000,
    "seed": 0,
    "cache_images": True,
    "pinned_transfer": True,
    "internal_holdout": True,
    "optimizer_backend": decision.optimizer_backend,
    "precision": decision.precision,
    "rolling_checkpoint": False,
    "grow_grad2d": settings.grow_grad2d,
    "absgrad": settings.absgrad,
    "revised_opacity": settings.revised_opacity,
}
```

- [ ] **Step 4: Make Phase A import the shared functions and preserve its API**

`c1_phase_a.py` imports `BASELINE_CANDIDATE`, `MAX_VRAM_MB`, and `score50`.
`c1_phase_a_runner.py` imports all shared run helpers, then calls
`build_screening_command` with `run_dir=output_root / scene_id / candidate_id`
and the existing repo, scene, manifest, candidate, backend, and Python values.
Remove the duplicate helper bodies; do not change Phase-A paths or decision JSON.

- [ ] **Step 5: Verify shared and Phase-A behavior**

```bash
python -m pytest \
  tests/unit/test_c1_screening.py \
  tests/unit/test_c1_phase_a.py \
  tests/unit/test_c1_phase_a_runner.py -q
```

Expected: all pass and Phase A still executes exactly four scene/candidate pairs.

- [ ] **Step 6: Commit the reusable boundary**

```bash
git add src/bts_nvs/training/c1_screening.py \
  src/bts_nvs/training/c1_phase_a.py \
  src/bts_nvs/training/c1_phase_a_runner.py \
  tests/unit/test_c1_screening.py \
  tests/unit/test_c1_phase_a.py \
  tests/unit/test_c1_phase_a_runner.py
git commit -m "refactor: share C1 screening mechanics"
```

---

### Task 3: Implement deterministic six-scene Phase-B decision

**Files:**
- Create: `src/bts_nvs/training/c1_phase_b.py`
- Create: `tests/unit/test_c1_phase_b.py`

**Interfaces:**
- Consumes: validated reports/diagnostics and `score50` from `c1_screening.py`.
- Produces: `PHASE_B_SCENES`, `SCREENING_SCENES`, `LOCKED_CANDIDATE`, `exact_sign_test`, `build_phase_b_decision`, and `save_phase_b_decision`.

- [ ] **Step 1: Write fixtures for six paired scenes and gate outcomes**

```python
def _report(scene: str, candidate: str, psnr: float) -> dict:
    return {
        "schema_version": 1,
        "scene_id": scene,
        "candidate_id": candidate,
        "step": 7000,
        "image_count": 8,
        "psnr_db_mean": psnr,
        "ssim_mean": 0.8,
        "lpips_mean": 0.2,
        "peak_gaussians": 1_000_000,
        "max_vram_mb": 8_000.0,
        "total_time_seconds": 900.0,
    }


def _diagnostic(scene: str, candidate: str, value: float) -> dict:
    return {
        "schema_version": 1,
        "scene_id": scene,
        "candidate_id": candidate,
        "image_count": 8,
        "missing_edge_mean": value,
        "spurious_edge_mean": value,
        "hf_l1_mean": value,
    }


def matrix(positive_scenes: int):
    baselines = []
    candidates = []
    diagnostics = []
    for index, scene in enumerate(SCREENING_SCENES):
        baselines.append(_report(scene, BASELINE_CANDIDATE, 20.0))
        candidate_psnr = 21.0 if index < positive_scenes else 19.0
        candidates.append(_report(scene, LOCKED_CANDIDATE, candidate_psnr))
        diagnostics.append(_diagnostic(scene, BASELINE_CANDIDATE, 0.10))
        diagnostics.append(_diagnostic(scene, LOCKED_CANDIDATE, 0.09))
    return baselines, candidates, diagnostics


def test_phase_b_passes_six_positive_scenes() -> None:
    baselines, candidates, diagnostics = matrix(positive_scenes=6)
    decision = build_phase_b_decision(baselines, candidates, diagnostics)
    assert decision["phase_b_passed"] is True
    assert decision["positive_scene_count"] == 6
    assert decision["sign_test_p_value"] == pytest.approx(0.03125)


def test_phase_b_rejects_three_positive_scenes() -> None:
    decision = build_phase_b_decision(*matrix(positive_scenes=3))
    assert decision["phase_b_passed"] is False
    assert decision["gates"]["at_least_four_positive"] is False


def test_phase_b_four_of_six_is_conditional_pass() -> None:
    decision = build_phase_b_decision(*matrix(positive_scenes=4))
    assert decision["phase_b_passed"] is True
    assert decision["requires_negative_scene_review"] is True
    assert decision["sign_test_p_value"] == pytest.approx(0.6875)
```

Also test duplicate/missing records, NaN/Inf, LPIPS regression, both aggregate
edge errors worsening, report/diagnostic image-count mismatch, and VRAM equal to
or above 23 GB.

- [ ] **Step 2: Run tests and verify module import failure**

```bash
python -m pytest tests/unit/test_c1_phase_b.py -q
```

Expected: collection fails because the module does not exist.

- [ ] **Step 3: Implement constants and exact sign test**

```python
PHASE_B_SCENES = ("hcm0031", "HCM0181", "HNI0131", "HNI0265")
SCREENING_SCENES = ("hcm0031", "HCM0181", "HCM0421", "HCM1439", "HNI0131", "HNI0265")
LOCKED_CANDIDATE = "C1-absgrad-t08-revopacity-v1"

def exact_sign_test(positive_count: int, total: int) -> float:
    tail = min(positive_count, total - positive_count)
    probability = 2.0 * sum(math.comb(total, k) for k in range(tail + 1)) / (2**total)
    return min(1.0, probability)
```

- [ ] **Step 4: Implement scene-level aggregation and explicit gates**

The decision must expose these keys:

```python
gates = {
    "mean_delta_positive": mean_delta > 0.0,
    "at_least_four_positive": positive_count >= 4,
    "aggregate_lpips_not_worse": candidate_lpips <= baseline_lpips,
    "edge_errors_not_both_worse": not (
        missing_edge_delta > 0.0 and spurious_edge_delta > 0.0
    ),
    "resources_valid": all(scene["max_vram_mb"] < MAX_VRAM_MB for scene in scenes.values()),
}
phase_b_passed = all(gates.values())
```

Average per scene without weighting by holdout image count. Save JSON atomically
with `allow_nan=False`, sorted keys, UTF-8, and a trailing newline.

- [ ] **Step 5: Verify decision tests**

```bash
python -m pytest tests/unit/test_c1_phase_b.py -q
```

Expected: all pass, including p-values `0.6875`, `0.21875`, and `0.03125` for
four, five, and six positive scenes.

- [ ] **Step 6: Commit decision logic**

```bash
git add src/bts_nvs/training/c1_phase_b.py tests/unit/test_c1_phase_b.py
git commit -m "feat: decide C1 Phase B robustness"
```

---

### Task 4: Implement the four-scene Phase-B runner

**Files:**
- Create: `src/bts_nvs/training/c1_phase_b_runner.py`
- Create: `tests/unit/test_c1_phase_b_runner.py`

**Interfaces:**
- Consumes: common screening helpers, Phase-A decision/root, backend decision, four B0 roots, and Phase-B decision builder.
- Produces: `build_phase_b_command`, `load_phase_a_lock`, and `run_phase_b`.

- [ ] **Step 1: Write runner tests for lock, flat path, sequence, reuse, and failure**

```python
def test_phase_b_command_writes_directly_to_scene_directory(tmp_path) -> None:
    command = build_phase_b_command(
        repo_root=tmp_path / "repo",
        scenes_root=tmp_path / "scenes",
        manifests_root=tmp_path / "manifests",
        output_root=tmp_path / "phase_b",
        scene_id="HCM0181",
        decision=SimpleNamespace(optimizer_backend="adam", precision="fp32"),
        python_bin="python",
    )
    assert command[command.index("--output_dir") + 1] == str(
        tmp_path / "phase_b" / "HCM0181"
    )


def test_phase_b_rejects_unapproved_phase_a(tmp_path) -> None:
    decision = tmp_path / "phase_a_decision.json"
    decision.write_text('{"phase_a_passed": false, "selected_candidate": null}')
    with pytest.raises(ValueError, match="Phase A did not pass"):
        load_phase_a_lock(decision)
```

The integration-style unit test uses a fake process to create four valid flat
run directories, asserts exact order `PHASE_B_SCENES`, reruns the runner, and
asserts no second process call occurs.

- [ ] **Step 2: Run tests and verify module import failure**

```bash
python -m pytest tests/unit/test_c1_phase_b_runner.py -q
```

Expected: collection fails because the runner does not exist.

- [ ] **Step 3: Implement complete preflight before loading the GPU backend**

`load_phase_a_lock` requires schema 1, `phase == "C1-phase-A"`,
`phase_a_passed is True`, and the locked candidate. `_require_phase_b_inputs`
checks all four scene directories, manifests, holdouts, all six B0 reports and
render directories, the locked winner's two Phase-A reports/renders/diagnostics,
and the training entry point. It accumulates missing paths and raises one
`FileNotFoundError` before any process starts.

- [ ] **Step 4: Implement sequential execution with flat output directories**

```python
for scene_id in PHASE_B_SCENES:
    run_dir = Path(output_root) / scene_id
    report_path = run_dir / "qualification_report.json"
    if report_path.is_file():
        report = load_completed_run(run_dir, scene_id, LOCKED_CANDIDATE, backend)
    else:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise ValueError(f"Phase B run directory is non-empty without a report: {run_dir}")
        command = build_phase_b_command(
            repo_root=repo_root,
            scenes_root=scenes_root,
            manifests_root=manifests_root,
            output_root=output_root,
            scene_id=scene_id,
            decision=backend,
            python_bin=python_bin,
        )
        result = run_process(command, check=False)
        if getattr(result, "returncode", None) != 0:
            raise RuntimeError(f"Phase B training failed for {scene_id}")
        report = load_completed_run(run_dir, scene_id, LOCKED_CANDIDATE, backend)
```

After all four runs validate, compute/write baseline and candidate diagnostics
for the four new scenes, load both baseline and winner diagnostics for the two
Phase-A scenes, build the six-scene decision, and save
`output_root/phase_b_decision.json`.

- [ ] **Step 5: Verify runner behavior**

```bash
python -m pytest tests/unit/test_c1_phase_b_runner.py -q
```

Expected: exact four-run sequence, safe complete-run reuse, flat output paths,
and rejection of partial/mismatched inputs.

- [ ] **Step 6: Commit the runner**

```bash
git add src/bts_nvs/training/c1_phase_b_runner.py tests/unit/test_c1_phase_b_runner.py
git commit -m "feat: orchestrate C1 Phase B screen"
```

---

### Task 5: Replace the Phase-A-only CLI with one C1 screening CLI

**Files:**
- Rename: `src/bts_nvs/training/run_c1_phase_a.py` → `src/bts_nvs/training/run_c1_screening.py`
- Create: `tests/unit/test_run_c1_screening.py`
- Modify: `docs/superpowers/specs/2026-07-19-c1-absgrad-revised-opacity-30k-design.md`
- Modify: `docs/superpowers/specs/2026-07-20-c1-phase-b-robustness-screen-design.md`

**Interfaces:**
- Consumes: `run_phase_a` and `run_phase_b`.
- Produces: one CLI with `--stage phase-a|phase-b`; Phase B additionally requires `--phase_a_root`.

- [ ] **Step 1: Rename the CLI and write dispatch tests**

```bash
git mv src/bts_nvs/training/run_c1_phase_a.py src/bts_nvs/training/run_c1_screening.py
```

```python
def common_args(tmp_path) -> list[str]:
    return [
        "--repo_root", str(tmp_path / "repo"),
        "--scenes_root", str(tmp_path / "scenes"),
        "--manifests_root", str(tmp_path / "manifests"),
        "--backend_root", str(tmp_path / "backend"),
        "--baseline_root", str(tmp_path / "baseline"),
        "--output_root", str(tmp_path / "output"),
    ]


def test_phase_b_requires_phase_a_root(tmp_path) -> None:
    with pytest.raises(SystemExit):
        parse_args(["--stage", "phase-b", *common_args(tmp_path)])


def test_main_dispatches_phase_b(monkeypatch, tmp_path) -> None:
    observed = {}
    monkeypatch.setattr(cli, "run_phase_b", lambda **kwargs: observed.update(kwargs))
    cli.main([
        "--stage", "phase-b",
        "--phase_a_root", str(tmp_path / "phase_a"),
        *common_args(tmp_path),
    ])
    assert observed["phase_a_root"] == tmp_path / "phase_a"
```

- [ ] **Step 2: Run tests and verify Phase-B dispatch is absent**

```bash
python -m pytest tests/unit/test_run_c1_screening.py -q
```

Expected: failure because the renamed CLI still only invokes Phase A.

- [ ] **Step 3: Implement a thin stage-aware parser and dispatcher**

```python
parser.add_argument("--stage", choices=("phase-a", "phase-b"), required=True)
parser.add_argument("--phase_a_root", type=Path)

args = parser.parse_args(argv)
if args.stage == "phase-b" and args.phase_a_root is None:
    parser.error("--phase_a_root is required for phase-b")
return args
```

`main` dispatches without adding training logic:

```python
args = parse_args(argv)
if args.stage == "phase-b":
    run_phase_b(
        repo_root=args.repo_root,
        scenes_root=args.scenes_root,
        manifests_root=args.manifests_root,
        backend_root=args.backend_root,
        baseline_root=args.baseline_root,
        phase_a_root=args.phase_a_root,
        output_root=args.output_root,
        python_bin=args.python_bin,
    )
else:
    run_phase_a(
        repo_root=args.repo_root,
        scenes_root=args.scenes_root,
        manifests_root=args.manifests_root,
        backend_root=args.backend_root,
        baseline_root=args.baseline_root,
        output_root=args.output_root,
        python_bin=args.python_bin,
    )
```

- [ ] **Step 4: Update the operational Phase-A and Phase-B commands in both C1 specs**

Phase B command must be exactly:

```bash
python src/bts_nvs/training/run_c1_screening.py \
  --stage phase-b \
  --repo_root "$PWD" \
  --scenes_root "$PWD/data/bts_scenes" \
  --manifests_root "$PWD/runs/manifests" \
  --backend_root "$PWD/runs/phase4/backend_qualification" \
  --baseline_root "$PWD/runs/phase4/qualification" \
  --phase_a_root "$PWD/runs/c1/phase_a" \
  --output_root "$PWD/runs/c1/phase_b"
```

- [ ] **Step 5: Verify both dispatch paths**

```bash
python -m pytest tests/unit/test_run_c1_screening.py tests/unit/test_c1_phase_a_runner.py tests/unit/test_c1_phase_b_runner.py -q
python src/bts_nvs/training/run_c1_screening.py --help
```

Expected: tests pass; help lists both stages and shared paths.

- [ ] **Step 6: Commit the common CLI**

```bash
git add src/bts_nvs/training/run_c1_screening.py \
  tests/unit/test_run_c1_screening.py \
  docs/superpowers/specs/2026-07-19-c1-absgrad-revised-opacity-30k-design.md \
  docs/superpowers/specs/2026-07-20-c1-phase-b-robustness-screen-design.md
git commit -m "refactor: expose reusable C1 screening CLI"
```

---

### Task 6: Full verification and VM handoff

**Files:**
- Modify only if verification finds a scoped defect in files from Tasks 1–5.

**Interfaces:**
- Consumes: complete Phase-B implementation.
- Produces: verified CPU code and one exact VM command; no GPU run is launched locally.

- [ ] **Step 1: Run focused Phase-B tests**

```bash
python -m pytest \
  tests/unit/test_c1_screening.py \
  tests/unit/test_c1_phase_a.py \
  tests/unit/test_c1_phase_a_runner.py \
  tests/unit/test_c1_phase_b.py \
  tests/unit/test_c1_phase_b_runner.py \
  tests/unit/test_run_c1_screening.py \
  tests/unit/test_operational_shell_scripts.py -q
```

Expected: all pass.

- [ ] **Step 2: Run the entire CPU suite**

```bash
python -m pytest -q
```

Expected: no regressions relative to the Phase-A baseline suite.

- [ ] **Step 3: Verify repository references and artifact isolation**

```bash
rg -n "scripts/(prepare_phase4|run_phase4)" README.md docs scripts tests
rg -n "run_c1_phase_a.py" README.md docs src tests
git diff --check
git status --short
```

Expected: no stale executable references, no whitespace errors, and no staged
or modified `outputs_auxiliary` paths.

- [ ] **Step 4: Review the exact VM preflight and run command**

```bash
python src/bts_nvs/training/run_c1_screening.py \
  --stage phase-b \
  --repo_root "$PWD" \
  --scenes_root "$PWD/data/bts_scenes" \
  --manifests_root "$PWD/runs/manifests" \
  --backend_root "$PWD/runs/phase4/backend_qualification" \
  --baseline_root "$PWD/runs/phase4/qualification" \
  --phase_a_root "$PWD/runs/c1/phase_a" \
  --output_root "$PWD/runs/c1/phase_b"
```

Expected preflight: reports all missing historical B0 inputs before any GPU
process. Expected successful run: four flat scene directories and
`runs/c1/phase_b/phase_b_decision.json`; no `.pt` or `.pth` under Phase B.

- [ ] **Step 5: Close verification without an aggregate commit**

If verification exposes a defect, return to the owning task, add a regression
test there, and amend that task with its explicit file list and commit message.
If verification passes, do not create an empty or catch-all commit.
