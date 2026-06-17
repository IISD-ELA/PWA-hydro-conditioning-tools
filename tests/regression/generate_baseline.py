"""Generate the grassmere regression baseline manifest.

Walks ``<data_dir>/HydroConditioning/Processed/`` and ``<data_dir>/Raven/Processed/``,
computes SHA-256 hashes of the key Step 0 + Step 1–2 outputs, and extracts a
small set of numeric samples (depression depth statistics, aspect ratios) so
that future regression checks can detect drift even when file hashes diverge
due to embedded timestamps or GDAL version differences.

Run from the project root:

    python PWA-hydro-conditioning-tools/tests/regression/generate_baseline.py

Or point at a specific data directory:

    python tests/regression/generate_baseline.py /path/to/data

The output is written to ``tests/regression/grassmere-baseline.json`` (next to
this script). Re-running with unchanged source data should produce a manifest
identical to the committed one — that is what the integrity test verifies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

# Files to hash + sample. Paths are relative to the data root.
# Step 0 outputs (HydroConditioning/Processed)
STEP0_FILES = [
    "HydroConditioning/Processed/CLRH_basins_depression_depths.shp",
    "HydroConditioning/Processed/CLRH_basins_depression_depths.shx",
    "HydroConditioning/Processed/CLRH_basins_depression_depths.dbf",
    "HydroConditioning/Processed/CLRH_basins_depression_depths.prj",
    "HydroConditioning/Processed/CLRH_basins_depression_depths.cpg",
    "HydroConditioning/Processed/merged_average_dem_filled_clip_resample_5m_FillBurn_Deps_Corr.tif",
    "HydroConditioning/Processed/ZonalStats_grassmere.html",
    "HydroConditioning/Processed/aspect_ratios.csv",
]

# Step 1–2 outputs (Raven/Processed)
STEP12_FILES = [
    "Raven/Processed/grassmere.rvi",
    "Raven/Processed/grassmere.rvp",
    "Raven/Processed/grassmere.rvh",
    "Raven/Processed/grassmere.rvc",
    "Raven/Processed/grassmere.rvt",
    "Raven/Processed/grassmere.rvp.tpl",
    "Raven/Processed/grassmere.rvh.tpl",
    "Raven/Processed/channel_properties.rvp",
    "Raven/Processed/channel_properties.rvp.tpl",
    "Raven/Processed/GridWeights.txt",
    "Raven/Processed/grassmere_HRU_areas.csv",
    "Raven/Processed/grassmere_tp_tc.csv",
]


def sha256_of(path: Path, chunk: int = 1024 * 1024) -> str:
    """Compute SHA-256 of a file. Streams in 1 MB chunks to handle large rasters."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def hash_files(data_root: Path, file_list: list[str]) -> dict[str, dict[str, Any]]:
    """Build the {relpath: {sha256, size_bytes}} mapping for a list of files."""
    out: dict[str, dict[str, Any]] = {}
    for relpath in file_list:
        full = data_root / relpath
        if not full.exists():
            print(f"  ⚠️  missing: {relpath}", file=sys.stderr)
            continue
        out[relpath] = {
            "sha256": sha256_of(full),
            "size_bytes": full.stat().st_size,
        }
        print(f"  ✓ {relpath}  ({full.stat().st_size:,} bytes)")
    return out


def sample_depression_depths(shapefile: Path) -> dict[str, Any]:
    """Extract summary statistics from the depression depths shapefile.

    Captures the ``Deps_Depth`` column distribution so that scientifically
    meaningful drift can be detected even if file hashes match (or vice versa).

    Note: the source code in ``calc_depression_depths`` adds ``Deps_Depth_mm``
    and ``Deps_Vol_m3`` columns, but the DBF format that backs shapefiles
    truncates field names to 10 characters — so on disk they appear as
    ``Deps_Depth`` and ``Deps_Vol_m``. The values are still in mm and m³
    respectively despite the truncated names.
    """
    import geopandas as gpd

    gdf = gpd.read_file(shapefile)
    depths = gdf["Deps_Depth"]  # truncated from Deps_Depth_mm; values are mm
    volumes = gdf["Deps_Vol_m"]  # truncated from Deps_Vol_m3; values are m³
    return {
        "n_features": int(len(gdf)),
        "columns": sorted(gdf.columns.tolist()),
        "deps_depth_mm": {
            "min": float(depths.min()),
            "max": float(depths.max()),
            "mean": float(depths.mean()),
            "median": float(depths.median()),
            "std": float(depths.std()),
            "first_5": [float(x) for x in depths.head(5).tolist()],
        },
        "deps_vol_m3": {
            "min": float(volumes.min()),
            "max": float(volumes.max()),
            "mean": float(volumes.mean()),
            "median": float(volumes.median()),
        },
    }


def sample_aspect_ratios(csv: Path) -> dict[str, Any]:
    """Sample aspect_ratios.csv. It's tiny — capture the whole thing."""
    import pandas as pd

    df = pd.read_csv(csv)
    return {
        "n_rows": int(len(df)),
        "columns": df.columns.tolist(),
        "aspect_ratio": {
            "min": float(df["Aspect_Ratio"].min()),
            "max": float(df["Aspect_Ratio"].max()),
            "mean": float(df["Aspect_Ratio"].mean()),
        },
    }


def sample_depression_raster(tif: Path) -> dict[str, Any]:
    """Capture basic statistics of the depression raster."""
    import numpy as np
    import rasterio

    with rasterio.open(tif) as src:
        data = src.read(1, masked=True)
        return {
            "width": int(src.width),
            "height": int(src.height),
            "crs": str(src.crs),
            "nodata": float(src.nodata) if src.nodata is not None else None,
            "values": {
                "min": float(np.ma.min(data)),
                "max": float(np.ma.max(data)),
                "mean": float(np.ma.mean(data)),
                "n_nonzero": int(np.ma.count(data[data > 0])),
            },
        }


def sample_grid_weights(txt: Path) -> dict[str, Any]:
    """GridWeights.txt is a Raven block — capture row count + first 3 lines."""
    lines = txt.read_text().splitlines()
    return {
        "n_lines": len(lines),
        "first_3_lines": lines[:3],
    }


def build_manifest(data_root: Path) -> dict[str, Any]:
    print(f"\nHashing Step 0 outputs in {data_root}/HydroConditioning/Processed/")
    step0_files = hash_files(data_root, STEP0_FILES)
    print(f"\nHashing Step 1–2 outputs in {data_root}/Raven/Processed/")
    step12_files = hash_files(data_root, STEP12_FILES)

    print("\nExtracting numeric samples...")
    samples: dict[str, Any] = {}

    shp = data_root / "HydroConditioning/Processed/CLRH_basins_depression_depths.shp"
    if shp.exists():
        samples["depression_depths"] = sample_depression_depths(shp)
        print(f"  ✓ depression_depths ({samples['depression_depths']['n_features']} features)")

    csv = data_root / "HydroConditioning/Processed/aspect_ratios.csv"
    if csv.exists():
        samples["aspect_ratios"] = sample_aspect_ratios(csv)
        print(f"  ✓ aspect_ratios ({samples['aspect_ratios']['n_rows']} rows)")

    tif = data_root / "HydroConditioning/Processed/merged_average_dem_filled_clip_resample_5m_FillBurn_Deps_Corr.tif"
    if tif.exists():
        samples["depression_raster"] = sample_depression_raster(tif)
        print(f"  ✓ depression_raster ({samples['depression_raster']['width']}x{samples['depression_raster']['height']})")

    gw = data_root / "Raven/Processed/GridWeights.txt"
    if gw.exists():
        samples["grid_weights"] = sample_grid_weights(gw)
        print(f"  ✓ grid_weights ({samples['grid_weights']['n_lines']} lines)")

    return {
        "schema_version": 1,
        "watershed": "grassmere",
        "generated_at": date.today().isoformat(),
        "source_data_dir_relative": "data",
        "files": {
            "step0_processed": step0_files,
            "step12_processed": step12_files,
        },
        "samples": samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "data_root",
        nargs="?",
        type=Path,
        help="Path to the project's data/ directory (defaults to <repo-root>/../../data)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(__file__).parent / "grassmere-baseline.json",
        help="Output manifest path (default: tests/regression/grassmere-baseline.json)",
    )
    args = parser.parse_args()

    if args.data_root is None:
        # Walk up from this script looking for a data/ dir
        here = Path(__file__).resolve()
        for parent in here.parents:
            candidate = parent / "data"
            if candidate.is_dir() and (candidate / "HydroConditioning").is_dir():
                args.data_root = candidate
                break
        if args.data_root is None:
            print("ERROR: data_root not found. Pass it as an argument.", file=sys.stderr)
            return 1

    if not args.data_root.is_dir():
        print(f"ERROR: {args.data_root} is not a directory", file=sys.stderr)
        return 1

    manifest = build_manifest(args.data_root)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"\n✅ Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
