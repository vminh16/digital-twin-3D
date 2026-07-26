# Five-scene MVP screen plan

**Authority:** `../specs/2026-07-26-five-scene-mvp.md`

## Success criteria

1. Candidate config and output hashes identify the new mechanism.
2. Unit tests cover contract validation, loss weighting, SH4 and renderer mode.
3. Existing B0 behavior and artifacts remain compatible.
4. Candidate runs are fresh and sequential.
5. Decisions are generated only from validated 7k reports.

## Execution order

1. Add candidate contracts and tests.
2. Wire the already-supported antialiased renderer mode.
3. Add bounded local-Laplacian maps and weighted L1.
4. Generalize SH storage, training and inference from degree 3 to degree 4.
5. Run unit tests and one CUDA densification smoke for each new mechanism.
6. Revalidate the five existing B0 reports.
7. Run screens in this order:

```text
chair     E2-loss-local-laplacian-v1
bonsai    E2-loss-local-laplacian-v1
bonsai    E2-appearance-sh4-v1
HCM0674   E2-raster-aa-v1
HCM0540   E2-raster-aa-v1
```

8. Generate `runs/scene_opt_v2/decisions/screen/<scene>.json`.
9. Record `HCM0644 -> B0-reference` without retraining.
10. Authorize paired 30k confirmation for no more than two passing winners.

## L4 compute envelope

Stage A was measured on one NVIDIA L4. Its 7k training times were 7.5 minutes
for chair, 8.5 for bonsai, 15.5 for HCM0674 and 16.3 for HCM0540. The five
authorized runs therefore cost about 56 minutes at B0-equivalent throughput,
or at most about 70 minutes at the 1.25 runtime gate, plus validation and
artifact checks. Peak Stage A VRAM for these scenes was 1.4–6.2 GB, leaving
substantial headroom below the locked 23 GB ceiling.

## L4 VM entry point

From the repository root on the L4 instance:

```bash
bash scripts/run_five_scene_mvp_screen.sh
```

The script requires the original Stage A manifests, validates reused B0
reports, runs only missing candidate artifacts, generates deterministic
decisions and appends every command to
`runs/scene_opt_v2/deployment_commands.log`. It deliberately does not
regenerate manifests because the B0 pairing contract pins their byte hashes.
Pass scene IDs to run a smaller recoverable batch, for example:

```bash
bash scripts/run_five_scene_mvp_screen.sh chair bonsai
bash scripts/run_five_scene_mvp_screen.sh HCM0674 HCM0540 HCM0644
```

Override `PYTHON_BIN`, `BTS_BACKEND_ROOT`, `BTS_B0_EXPERIMENT_ROOT`,
`BTS_MVP_EXPERIMENT_ROOT`, `BTS_SCENES_ROOT`, `BTS_MANIFESTS_ROOT`,
`AUX_SCENES_ROOT` or `AUX_MANIFESTS_ROOT` only when the VM layout differs
from the repository defaults.

## Stop conditions

Stop one candidate, preserve its failure ledger and continue with the next
scene when any of these occurs:

- CUDA OOM or non-finite state;
- output/artifact contract failure;
- uncontrolled Gaussian growth;
- wall time exceeds the B0 reference by more than 25%;
- the deterministic decision rejects the candidate.

Do not add a replacement mechanism during this screen. A rejected scene falls
back to B0.

## Deadline production handoff

The completed screen selected chair local-Laplacian and bonsai SH4. Run both
fresh full-data 30k jobs sequentially on the L4:

```bash
bash scripts/run_auxiliary_mvp_production.sh
```

Outputs use `runs/scene_opt_v2/production_mvp/scenes/<scene_id>/`. The wrapper
validates the original v1 B0 references, v2 screen artifacts and hashed scene
decisions before training. A valid partial run resumes only from its rolling
`checkpoints/recovery.pt`.
