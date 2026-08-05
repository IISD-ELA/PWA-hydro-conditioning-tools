# pwa-tools integration tests

Drives **Step 0 through its module CLI** (`python -m pwa_tools.run_step0`) on a
real config — the one path `tests/regression/` doesn't cover (regression calls
the Step 0 functions directly; this exercises argument parsing, config loading,
and the `__main__` wiring on real data). **Local-only** — GitHub CI runs `pytest
tests/unit`.

> The deeper Step 0 science (WhiteboxTools + gdalwarp end-to-end vs the
> grassmere baseline) lives in `tests/regression/` — run it with
> `make regression`.

Skips cleanly when its inputs are absent — except the missing-config error-path
check, which needs nothing and always runs.

## Inputs (environment)

| Variable | Purpose |
|---|---|
| `PWA_STEP0_CONFIG` | Path to a filled-in `pwa_config.yml` (`python -m pwa_tools.init_config`) |

A successful run also needs WhiteboxTools + GDAL available (as Step 0 always
does).

## Run

```bash
make integration

# Or directly:
PWA_STEP0_CONFIG=/path/to/pwa_config.yml pytest tests/integration -v
```
