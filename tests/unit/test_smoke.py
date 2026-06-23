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


def test_wetlands_docstring_cites_live_constants() -> None:
    """``gen_wetland_polygons`` describes its thresholds by referring
    to the module-level constants (``_DEPTH_THRESHOLD_M`` etc.) rather
    than hard-coding values in the docstring. The old docstring had
    literal "5 cm" + "100 m²" baked into the prose; when the live
    constants got updated to 10 cm / 4000 m² (Thomas's d984314), the
    docstring silently drifted into lying.

    This test asserts the docstring refers to the constants by name
    so future threshold tweaks don't recreate the drift."""
    from pwa_tools import wetlands

    doc = wetlands.gen_wetland_polygons.__doc__ or ""
    assert "_DEPTH_THRESHOLD_M" in doc
    assert "_AREA_THRESHOLD_M2" in doc
    # As a defence-in-depth, also assert the current literal values
    # surface somewhere in the docstring — if they drift, this test
    # fires and forces the next maintainer to re-read the docstring.
    assert str(wetlands._DEPTH_THRESHOLD_M) in doc, (
        f"docstring lost track of _DEPTH_THRESHOLD_M={wetlands._DEPTH_THRESHOLD_M}"
    )
    assert str(wetlands._AREA_THRESHOLD_M2) in doc, (
        f"docstring lost track of _AREA_THRESHOLD_M2={wetlands._AREA_THRESHOLD_M2}"
    )


def test_pyproject_includes_subpackages_in_wheel() -> None:
    """Regression test for the silent wheel-omission bug: a previous
    pyproject.toml had ``[tool.setuptools] packages = ["pwa_tools"]``
    — an explicit list that excluded ``pwa_tools.io`` and
    ``pwa_tools.WBT``. Editable installs hide it (they walk the source
    tree), but a non-editable wheel ships without those subpackages
    and crashes on the first ``pwa_tools.io.*`` import.

    Asserts the build-system discovery rule is the auto-find form,
    not a manual list that would silently drop sibling packages."""
    import tomllib
    from pathlib import Path

    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text())
    setuptools_cfg = data.get("tool", {}).get("setuptools", {})

    # Either packages.find (auto-discovery) or a list that explicitly
    # names every subpackage. Reject the bare ["pwa_tools"] form.
    assert "packages" not in setuptools_cfg or setuptools_cfg["packages"] != ["pwa_tools"], (
        "pyproject.toml has [tool.setuptools] packages = ['pwa_tools'] — "
        "this silently excludes pwa_tools.io, pwa_tools.WBT, and any "
        "future subpackage from non-editable wheel builds. Use "
        "[tool.setuptools.packages.find] instead."
    )


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
