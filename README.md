# PWA-hydro-conditioning-tools
Python module for hydro conditioning Prairie watersheds.

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
