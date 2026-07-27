# Historical scene-specific optimization program

**Status:** CLOSED. Full detail remains in Git history.

**Current authority:**
`2026-07-27-chair-bonsai-deep-optimization.md`.

## Purpose and durable design

The program introduced per-scene candidate selection around one shared 3DGS
pipeline for:

```text
HCM0644 HCM0674 HCM0540 HCM0539 HCM0421 chair bonsai
```

Because every scene has independent parameters, each scene may retain B0 or
select a different validated policy. Candidate comparisons must use the same
manifest, holdout, seed, resolution, horizon, backend and local metric
configuration.

The program implemented:

- detail metrics: HF-L1, missing/spurious edge and symmetric edge distance;
- easy/medium/hard pose strata;
- immutable candidate settings and provenance hashes;
- one stage-first generic experiment runner;
- deterministic scene decisions and resource gates;
- no checkpoints at 7k;
- one rolling recovery checkpoint at 30k;
- separation between internal-holdout research and full-data production.

## Historical stages

| Stage | Result |
|---|---|
| Modules 1–3 | implemented and tested |
| Stage A | seven valid 7k B0 references |
| Stage B1 | AbsGrad/scale screens on HCM0539/HCM0421 |
| Stage C | not completed as paired 30k science |
| Deadline production | AbsGrad HCM0539/HCM0421 |
| Five-scene follow-up | local-Laplacian chair, SH4 bonsai; AA rejected |
| Submission v2 | closed at official Score `71.2124` |

## Durable scientific boundaries

- A 7k result is mechanism evidence, not production proof.
- Production previews are not model-selection evidence.
- Official aggregate metrics cannot identify per-scene gains.
- Hidden-test RGB is never used for selection.
- No external images, depth weights, segmentation or 3D assets.
- Every behavior change requires a new candidate ID.
- Historical `runs/scene_opt_v1` and `runs/scene_opt_v2` are immutable.

All closed decisions and scores are consolidated in
`../history/2026-07-27-optimization-phase-closure.md`.
