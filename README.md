# PWA-hydro-conditioning-tools
Python module for hydro conditioning Prairie watersheds. Part of the Prairie Watershed Analytics (PWA) - a broader IISD-ELA GIS and modeling project for watershed analysis.

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

### State Management System

The package includes a robust state management system for efficient development workflows:

- **Global State Object**: `state` - tracks project paths, filenames, and workflow progress
- **Save Function**: `save_state()` - persists state to `.pkl` file in interim directory
- **Recovery Function**: `recover_state()` - restores state from `.pkl` file for resumed sessions
- **Last Function Tracking**: `state.LAST_FUNCTION_RUN` - enables workflow continuation from specific points

This system allows developers to avoid rerunning entire workflows during development and testing phases.

### Key Features

- **Setup Functions**: Automated project directory structure creation and file organization
- **State Management**: Save and recover workflow state using `.pkl` files for efficient development and testing
- **GIS Processing**: Comprehensive hydro conditioning workflow including:
  - LiDAR DEM processing and merging
  - Watershed delineation and stream conditioning
  - Depression analysis and wetland polygon generation
  - Coordinate reference system management
- **WhiteBox Integration**: Seamless integration with WhiteboxTools for advanced spatial analysis

### Developer Steps

1. Make sure local versions of PWA, -main and -tools repos are up to date and dev branch is up to date with main.
2. Use setup functions (`project_setup()`, `set_directory_structure()`) and `state.save_state()` for developing major adjustments to main workflow (to avoid rerunning entire workflow during dev & testing). Patches can be developed directly in a debugging mode.
3. Leverage the state recovery system (`recover_state()`) to resume work from saved `.pkl` files when testing iteratively.
4. If making several different improvements, run steps below in separate series:
5. Using dev branch, add functions to package (-tools) and insert them in the workflow (-main). Best practice is to do this separately for each repository using your IDE.
6. Re-install the package (-tools) using `pip install -e .` and test the hydro_condition.py file
7. If successful, update version info in pyproject.toml and merge dev branch to main


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
└──  pyproject.toml                          # Install configuration (used for pip install -e .)
```
