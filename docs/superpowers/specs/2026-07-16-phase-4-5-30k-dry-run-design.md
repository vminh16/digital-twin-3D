# Phase 4.5 — 30k Dry Run and Bounded Artifacts

## Purpose

Validate the selected `B0-reference` baseline through the complete 30,000-step
schedule on one representative scene before spending compute on all 18 BTS
scenes. The run must preserve crash recovery and reproducibility without
retaining a sequence of multi-gigabyte optimizer checkpoints.

This phase is a full-length qualification run, not production training.

## Scope

- Run `HCM0181` at native factor 1 for 30,000 steps.
- Use seed 0, cached images, pinned transfer, and the existing internal holdout.
- Train only on `holdout.train_image_names` and split-specific sparse support.
- Evaluate internal validation at initialization and after step 30,000, without
  saving initialization renders.
- Keep the existing `B0-reference` value `grow_grad2d=0.0002`.
- Do not change optimizer, loss, densification, pruning, opacity reset, SH, or
  camera behavior.

Phase 4.5 does not implement the 18-scene production orchestrator or compact
inference export. Those remain Phase 4.7 and Phase 4.6 respectively.

## Runner interface

Create `scripts/run_phase4_30k_dry_run.sh` with these defaults:

```text
scene root       data/bts_scenes/HCM0181
manifest root    runs/manifests/HCM0181
output root      runs/phase4/dry_run_30k/HCM0181
steps            30000
checkpoint       every 3000 steps, rolling
candidate        B0-reference
```

The script may accept path overrides through the established environment
variables `PYTHON_BIN`, `BTS_SCENES_ROOT`, `BTS_MANIFESTS_ROOT`, and a dedicated
`BTS_DRY_RUN_ROOT`. Scientific settings are locked in code and are not exposed
as shell overrides.

Resume is the only operational exception: setting `BTS_RESUME=1` requires an
existing `${BTS_DRY_RUN_ROOT}/HCM0181/checkpoints/recovery.pt` and resumes that
same run. Without this explicit value, any non-empty output directory is an
error. No arbitrary checkpoint path is accepted by the shell runner.

The Python entry point receives one new explicit
`--full_length_qualification` flag. It is mutually exclusive with the existing
`--qualification_candidate` flag and locks factor 1, seed 0, 30,000 steps,
internal holdout, `B0-reference`, cached images, pinned transfer, rolling
checkpoints, and either a fresh run or the fixed recovery resume described
above. This keeps Phase 4.4 behavior unchanged and avoids assembling a
qualification configuration from loosely related CLI switches.

Before training, the script must:

1. prepare or validate the 18-scene artifact pool;
2. smoke-test the pretrained LPIPS backend;
3. reject an incomplete, non-empty output directory unless a valid rolling
   checkpoint is explicitly supplied for resume;
4. fail before GPU work when required scene, manifest, holdout, or dependency
   artifacts are missing.

## Rolling checkpoint contract

Add an explicit rolling-checkpoint mode to the existing trainer. When enabled:

- checkpoint cadence remains every 3,000 completed steps;
- every save targets `checkpoints/recovery.pt`;
- save remains temporary-sibling plus atomic rename;
- an interrupted save leaves the previous valid recovery checkpoint intact;
- the final step also updates `recovery.pt`;
- no numbered full-state checkpoint is retained;
- disk preflight still requires temporary checkpoint space plus the existing
  safety headroom;
- resume continues to validate manifest and config hashes and restores model,
  optimizer, scheduler, strategy, and RNG state.

Ordinary training keeps the current numbered-checkpoint behavior unless rolling
mode is explicitly enabled. Qualification runs from Phase 4.4 remain unchanged.

Only one full optimizer checkpoint is retained after the 30k run. Phase 4.6 may
later export a compact inference artifact; it must not be invented in this phase.

## Controlled evaluation and rendering

The 30k dry run writes only:

- fixed train reference image;
- train preview at step 0;
- train preview at step 30,000;
- initial internal-validation metrics without image files;
- final internal-train metrics without image files;
- one final internal-validation render per validation camera;
- final per-image and per-scene PSNR, SSIM, and LPIPS report;
- `metrics.jsonl`, timing, convergence, environment, config, and identity hashes;
- the single rolling recovery checkpoint.

It does not run validation or save validation renders at intermediate checkpoint
steps. A shared evaluator accepts `save_images=False` for initialization and
final-train passes, so these checks add only small JSON records. Final validation
is the only pass that writes image files. Training loss and Gaussian count remain
recorded every step, which is sufficient to inspect opacity-reset recovery and
resource growth without repeatedly consulting validation.

The run metadata must record the Git commit when the working tree is clean. A
dirty working tree is rejected for the locked 30k dry run because a commit hash
alone would not identify the executed source.

## Test RGB policy

The following public scenes currently contain `test/images`:

- `hcm0031`
- `hcm0034`
- `HCM0181`
- `HCM0193`
- `HCM0204`

Their test RGB is not read during Phase 4.5 and is not used for hyperparameter
selection, early stopping, checkpoint selection, or rerun decisions. Internal
holdout is the only quality signal used to pass the dry run.

After the baseline is frozen, these five public test sets may be evaluated once
as an external benchmark. If their results are used to modify the baseline,
they become development data and must not be reported as unbiased test scores.
The other 13 scenes have no test RGB and receive output-contract validation only
until ground truth is legitimately available.

## Acceptance criteria

The dry run passes only when all conditions hold:

- exactly 30,000 ordered finite metric and timing records;
- no OOM, swap use, NaN/Inf, corrupted checkpoint, or path-capacity failure;
- peak VRAM is below 20 GiB;
- peak Gaussian count is below 10,000,000;
- final internal-validation PSNR improves by more than 3 dB from initialization;
- final internal-validation SSIM improves by more than 0.05 and LPIPS improves
  by more than 0.05;
- mean train-to-validation PSNR gap is below 8 dB;
- final render is non-blank and visual audit finds no collapse or severe
  full-frame floaters;
- `checkpoints/recovery.pt` loads successfully and restores complete training
  state;
- output contains no numbered checkpoint series and no intermediate validation
  render directories;
- rerunning the script without explicit resume never overwrites a completed or
  partial run.

Passing Phase 4.5 authorizes Phase 4.6 baseline freeze. It does not directly
authorize 18-scene training; Phase 4.6 must first pin the selected configuration,
dependency versions, code commit, manifest/holdout identities, and compact
artifact contract.

## Testing strategy

- Unit-test rolling checkpoint path selection and unchanged default behavior.
- Unit-test 30k CLI locks and rejection of dirty source state.
- Unit-test bounded initialization/final validation scheduling and report
  completeness.
- Shell-contract test the runner command, preflight, resume guard, and path
  overrides.
- Run existing checkpoint/resume, qualification, and trainer-loop suites.
- Run the script first on `HCM0181`; no other scene is launched by Phase 4.5.
