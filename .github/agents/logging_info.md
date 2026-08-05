# Repository Coding Instructions: Logging

All Python modules in this repository use the standard `logging` package. Follow these conventions exactly when writing or modifying code.

## Standard Setup

Every module must declare a module-level logger immediately after imports:

```python
import logging

logger = logging.getLogger(__name__)
```

Do **not** call `logging.basicConfig()` inside library modules or pipeline steps. That call belongs only in CLI entry points (e.g. `run_nc_processing.py:main()`), where it sets up the root handler once for the process:

```python
# CLI entry point only
logging.basicConfig(
    level=getattr(logging, args.log_level),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
```

If no CLI entry point exists, the module-level logger will inherit the root handler from the calling process. This is the standard pattern for Python libraries.

## When to Log

Log at every major pipeline step using the module-level `logger`. Use:

- `logger.info(...)` — pipeline progress, step starts/completions, output paths
- `logger.warning(...)` — recoverable conditions, fallback behaviour, skipped items
- `logger.debug(...)` — verbose detail useful for diagnosis (counts, intermediate values)

Do **not** use `print()` for pipeline status messages. Reserve `print()` for CLI output that is explicitly part of the user-facing interface (e.g. the final path summary in a `main()` function).

## Patterns to Follow

### Pipeline function — start and finish

```python
def run_nc_processing(config: NcProcessingConfig) -> NcProcessingResult:
    logger.info(
        "Starting NetCDF processing for watershed '%s' (data_category=%s)",
        config.watershed_name, config.data_category,
    )

    # ... pipeline steps ...

    logger.info("NetCDF processing complete. Outputs in %s", config.processed_dir)
    return result
```

### Each major step — log the output path

```python
logger.info("Subsetting PHyDAP NetCDF → %s", config.subset_nc_path)
subset_phydap_for_watershed(...)

logger.info("Generating GridWeights → %s", config.grid_weights_path)
generate_grid_weights(...)

logger.info(
    "Applying %d-hour time offset → %s",
    config.time_offset_hours, config.adjusted_nc_path,
)
time_offset(...)
```

### Warnings for fallbacks or unexpected-but-handled conditions

```python
logger.warning(
    "%d subbasin centroids fell outside PHyDAP coverage; "
    "nearest-neighbour fallback applied.",
    n_outside,
)
```

## Format Rules

- Use `%`-style formatting (`logger.info("value: %s", x)`) — **not** f-strings. This avoids building the string when the log level is disabled.
- Keep messages short and factual. Include the output path when a file is written.
- For multi-line messages, keep the first argument as the format string and pass values as positional arguments.

## Reference Files

- `pwa_raven/src/pwa_raven/nc_processing.py` — canonical example of module-level `logger` and per-step `logger.info` calls inside a pipeline function.
- `pwa_raven/src/pwa_raven/run_nc_processing.py` — canonical example of CLI `logging.basicConfig` setup in `main()`.
