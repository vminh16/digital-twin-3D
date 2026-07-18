# JPEG Submission Converter Design

## Goal

Create a submission-only conversion step that turns the existing validated PNG
renders into JPEG files whose names, suffixes, case, and resolutions exactly
match `test_image_names` from each scene manifest.

## Scope

This step does not change manifest schema, rendering, checkpoints, training, or
the existing `outputs/` tree. It writes a separate tree so the lossless PNG
renders remain the single source for later quality experiments.

## Contract

- Input: `outputs/<scene_id>/<manifest.test_output_names>`.
- Output: `submission_outputs/<scene_id>/<manifest.test_image_names>`.
- Every target suffix must be `.jpg` or `.jpeg`, case-insensitively.
- Preserve the exact target filename and extension case from the manifest.
- Decode input as RGB and verify the manifest's native width and height.
- Encode JPEG once with RGB input, quality 99 by default, 4:4:4 chroma
  (`subsampling=0`), optimized Huffman tables, and non-progressive output.
- Strip optional metadata by not copying EXIF or ICC payloads.
- Reject existing output, missing/extra source files, duplicate target names,
  invalid modes, wrong resolutions, or output above the configured byte limit.
- Build in a temporary sibling directory and publish only after every file
  validates as JPEG/RGB at the exact target resolution.
- Emit a deterministic JSON report containing scenes, counts, bytes, quality,
  and SHA-256 hashes.

## Interface

Python CLI:

```text
python -m bts_nvs.submission.prepare_jpeg \
  --source_root outputs \
  --output_root submission_outputs \
  --scenes_root data/bts_scenes \
  --manifests_root runs/manifests \
  --report_path runs/submission/jpeg_report.json \
  --max_bytes 350000000 \
  --quality 99 \
  --scene_ids HCM0644 HCM0674 HCM0540 HCM0539 HCM0421
```

A thin Bash wrapper supplies repository-relative defaults and forwards flags.

## Deliberate non-goals

- No archive creation because the upload archive layout is still being tested.
- No automatic inclusion of `chair` or `bonsai` before their renders exist.
- No MozJPEG/jpegli dependency for the first baseline submission.
- No adaptive per-image quality or repeated JPEG recompression.
