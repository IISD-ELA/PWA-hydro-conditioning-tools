"""pwa-tools — hydro-conditioning toolkit for the PWA pipeline.

The public API lives in focused submodules; this top-level module is
deliberately minimal. Import from the relevant submodule for what you
need:

    from pwa_tools.config import PwaConfig
    from pwa_tools.runner import run_step0
    from pwa_tools.io.shapefile import read_shapefile
    from pwa_tools.io.raster import resample_raster
    # ...

See README.md for the full module map. The legacy 1,175-line god-file
``__init__.py`` (with its module-level ``state`` singleton, pickled
checkpoints, and 22 mixed-responsibility functions) was split into
``config.py``, ``runner.py``, ``io/``, ``raven/``, ``streams.py``,
``projection.py``, ``depression.py``, ``wetlands.py``, ``_wbt.py``,
and ``ui.py`` during the 2026 refactor.
"""

__version__ = "1.1.3"
