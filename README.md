# PWA-hydro-conditioning-tools
Python module for hydro conditioning Prairie watersheds.

## For developers

This package is built for use in the PWA-hydro-conditioning-main product repository and the PWA developer repository. Updates to functions in this package must be tested for backwards compatibility with PWA-hydro-conditioning-main before being merged.

### Branches

- **main** : branch used only for tested publishable code
- **dev** : branch used for development and testing

### Versioning Note
Versions follow the Semver convention:
- Patch x.x.Y. - bug fix, no interface change (Not currently tracked)
- Minor x.Y.x. - new feature added, fully backward compatible
- Major Y.x.x. - breaking change to the public interface

Process note: Major or minor commits to this repo should be tested for compatibility with PWA dependencies

### Developer Steps

1. Make sure local versions of PWA, -main and -tools repos are up to date and dev branch is up to date with main.
2. Use setup functions and pwa.save_state() for developing major adjustments to main workflow (to avoid rerunning entire workflow during dev & testing). Patches can be developed directly in a debugging mode.
3. If making several different improvements, run steps below in separate series:
4. Using dev branch, add functions to package (-tools) and insert them in the workflow (-main). Best practice is to do this separately for each repository using your IDE.
5. Re-install the package (-tools) using pip install -e . and test the hydro_condition.py file
6. If successful, update version info in pyproject.toml and merge dev branch to main


### Repository Structure
```bash
PWA-hydro-conditioning-tools/
├── pwa_tools/                            # Main Python module
│   ├── __init__.py                       # Initializes the package and defines key functions and classes
│   └──  WBT                              # Contains all WhiteBox tools used in the package
│       ├── __init__.py                   # Marks WhiteBox as a package (it's just an empty Python file)
│       └── ...WhiteBox related files...
├──  .gitignore                           # Git ignore rules (e.g., __pycache__, *.egg-info)
├──  README.md                            # This file
├──  requirements.txt                     # Python dependencies (except GDAL, which is installed separately using conda)
└──  setup.py                             # Install configuration (used for pip install -e .)
```
