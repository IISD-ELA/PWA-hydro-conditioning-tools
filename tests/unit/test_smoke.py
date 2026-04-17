"""Smoke tests — confirm the package imports and the most trivial pure helpers
behave correctly. Anything failing here means the dev environment isn't set up.
"""

from __future__ import annotations


def test_package_imports() -> None:
    """The package and its global state singleton are importable."""
    import pwa_tools

    assert hasattr(pwa_tools, "state")
    assert pwa_tools.state.WATERSHED_NAME is None  # default before project_setup runs


def test_snake_case_pure_helper() -> None:
    """snake_case is one of the few pure functions in the package — sanity-check it."""
    from pwa_tools import snake_case

    assert snake_case("Cypress River") == "cypress_river"
    assert snake_case("manning-canal") == "manning_canal"
    assert snake_case("Already_Snake") == "already_snake"
