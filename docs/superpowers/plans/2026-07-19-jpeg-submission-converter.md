# JPEG Submission Converter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a validated, size-bounded JPEG submission tree from existing PNG renders without changing training or inference artifacts.

**Architecture:** Add one focused Python module under `bts_nvs.submission` and one path-only Bash wrapper. The converter loads existing manifests, verifies the selected PNG source tree, writes exact CSV-derived JPEG names into an atomic staging directory, validates decoded payloads, and writes a deterministic report.

**Tech Stack:** Python 3.12, Pillow, NumPy, pytest, Bash.

## Global Constraints

- Preserve the existing `outputs/` tree unchanged.
- JPEG defaults are quality 99, subsampling 0 (4:4:4), optimize true, progressive false.
- The default maximum payload is 350,000,000 bytes.
- Output names are exact `manifest.test_image_names`, including extension case.
- Only selected scenes are converted.
- No new dependency and no core manifest/schema change.

---

### Task 1: Converter contract and implementation

**Files:**
- Create: `src/bts_nvs/submission/prepare_jpeg.py`
- Create: `tests/unit/test_prepare_jpeg_submission.py`

**Interfaces:**
- Produces: `prepare_jpeg_submission(...) -> dict`
- Produces: `parse_args(argv=None) -> argparse.Namespace`

- [ ] Write failing tests using tiny real RGB images and fake manifest objects.
- [ ] Verify RED because `bts_nvs.submission.prepare_jpeg` does not exist.
- [ ] Implement exact source/target mapping, one-pass JPEG encoding, decoded
      payload validation, byte budget enforcement, atomic output publication,
      and deterministic report creation.
- [ ] Verify tests pass, including failure cleanup and exact 4:4:4 sampling.

### Task 2: Operational wrapper

**Files:**
- Create: `scripts/prepare_jpeg_submission.sh`
- Modify: `tests/unit/test_phase4_shell_scripts.py`

**Interfaces:**
- Consumes: `python -m bts_nvs.submission.prepare_jpeg`.
- Produces: repository-relative defaults with forwarded CLI flags.

- [ ] Write a failing shell-content test for source, output, manifest, report,
      quality, byte budget, and scene forwarding.
- [ ] Verify RED because the wrapper does not exist.
- [ ] Implement the minimal path-only Bash wrapper.
- [ ] Verify shell and converter tests pass.

### Task 3: Real-output verification and commit

**Files:**
- No production file changes.
- Generated artifact: `submission_outputs/` (Git-ignored runtime output).

- [ ] Run focused unit tests.
- [ ] Run the converter against the five current output scenes at quality 99.
- [ ] Independently audit count, exact names, JPEG payload, RGB mode,
      resolution, 4:4:4 sampling, and total bytes below 350,000,000.
- [ ] Run the full unit suite.
- [ ] Review `git diff`, commit directly to `main`, and provide the archive
      command without deleting the lossless PNG source.
