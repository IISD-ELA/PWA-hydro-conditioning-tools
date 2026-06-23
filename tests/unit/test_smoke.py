"""Smoke tests — confirm the package imports and key submodules resolve.

Anything failing here means the dev environment isn't set up correctly
or the package layout has drifted from what callers (orchestrator,
tests, external users) expect.
"""

from __future__ import annotations


def test_package_imports() -> None:
    """The top-level package is importable and exposes __version__."""
    import pwa_tools

    assert hasattr(pwa_tools, "__version__")
    assert isinstance(pwa_tools.__version__, str)


def test_public_submodules_resolve() -> None:
    """Each focused submodule the README points contributors at should
    import cleanly. Catches accidental import-order breaks introduced
    by future refactors."""
    from pwa_tools.config import PwaConfig
    from pwa_tools.runner import run_step0
    from pwa_tools.io.shapefile import read_shapefile
    from pwa_tools.io.raster import resample_lidar_raster
    from pwa_tools.streams import clip_nhn_to_watershed
    from pwa_tools.projection import project_subbasins_to_lidar
    from pwa_tools.depression import gen_depressions_raster
    from pwa_tools.wetlands import gen_wetland_polygons

    # Spot-check that they're callables, not just symbols (catches a
    # broken submodule that imports but exports nothing useful).
    assert callable(run_step0)
    assert callable(read_shapefile)
    assert callable(resample_lidar_raster)
    assert callable(gen_depressions_raster)
    # PwaConfig is a dataclass, also callable as a constructor.
    assert callable(PwaConfig)
