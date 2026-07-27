# Historical generic experiment runner

**Status:** IMPLEMENTED and CLOSED. Full detail remains in Git history.

The runner is an integrity layer around the existing training entry point. It
does not contain a second training loop.

## Preserved contract

```text
reference  7k   B0 internal holdout, no checkpoint
screen     7k   candidate internal holdout, no checkpoint
confirm   30k   paired internal holdout, rolling recovery
production 30k  full data, rolling recovery
```

Paths are stage-first so evidence cannot overwrite another horizon:

```text
runs/scene_opt_v1/<stage>/<scene>/<candidate>/
```

The runner rejects unknown identities, illegal stage/horizon combinations,
non-empty output, manifest/config/holdout mismatch, incomplete renders,
non-finite metrics, invalid checkpoints, excessive VRAM and over-budget paired
runtime.

Historical selection gates were:

```text
Score50 delta > 0
LPIPS delta <= 0
hard Score50 delta >= 0
missing/spurious edge do not both worsen
paired time ratio <= 1.25
peak VRAM < 23 GB
```

The active v3 plan reuses this runner and adds a non-destructive `research`
stage with a 30k optimizer horizon stopped at 15k. See
`2026-07-27-chair-bonsai-deep-optimization.md`.
