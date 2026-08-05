# Terminal & Environment Notes for Coding Agents (Windows / VS Code)

## Shell environment

The integrated terminal is **PowerShell 5** (not bash, not PowerShell 7).

The conda environment `pwa_ref` may be pre-activated (if the prompt shows `(pwa_ref) PS …`). Assume it is active, but if python or conda commands fail, run conda activate pwa_ref, or use the full path to the python executable in that environment (see below).

## What does NOT work

| Bash habit | PowerShell equivalent |
|---|---|
| `source script.sh` | N/A — not needed; env is already active |
| `cd path` | `Push-Location "path"` / `Pop-Location` |
| `cmd1 && cmd2` | `cmd1 ; cmd2` (semicolons only) |
| `head -n 20` | `Select-Object -First 20` |
| `python …` | See below — PATH is unreliable |
| `conda run -n pwa_ref python …` | Produces no captured output — avoid |

## Invoking Python reliably

Even when the prompt shows `(pwa_ref)`, `python` may not resolve because VS Code
spawns terminal sessions without fully re-running the conda init script.
Use the full executable path instead. For example:

```powershell
$pyexe = Join-Path $HOME ".conda\envs\pwa_ref\python.exe"
& $pyexe -c "import sys; print(sys.version)"
& $pyexe -m pytest tests/ --tb=short
& $pyexe -m pip install some-package
```

## Capturing long command output

pytest and similar tools print to a console buffer that VS Code truncates.
Redirect to a temp file and read it back:

```powershell
$pyexe = Join-Path $HOME ".conda\envs\pwa_ref\python.exe"
& $pyexe -m pytest tests/ --tb=short 2>&1 |
    Out-File -FilePath "C:\Temp\results.txt" -Encoding utf8 -Force
type "C:\Temp\results.txt"
```

## Running git from any working directory

`git` is on PATH. Use `-C` to avoid changing directory:

```powershell
git -C "c:\Users\tsaleh\Code_Workspace\GIS-Projects\PWA" status --short
git -C "c:\Users\tsaleh\Code_Workspace\GIS-Projects\PWA" log --oneline -5
```

Or bracket with `Push-Location` / `Pop-Location`:

```powershell
Push-Location "c:\Users\tsaleh\Code_Workspace\GIS-Projects\PWA"
git add .
git commit -m "…"
Pop-Location
```

## Invoking R scripts

R 4.3.0 has the required packages (`lmomco`, `MGBT`, `dplyr`, `jsonlite`, `DescTools`).
Later R versions (4.4+) do **not** have these packages installed.

```powershell
$r = "C:\Program Files\R\R-4.3.0\bin\Rscript.exe"
& $r "path\to\script.R" arg1 arg2
```

## conda environments

| Env | Purpose |
|---|---|
| `pwa_ref` | Main Python env — all pwa_raven work |
| `hydrocon_env` | Step 0 hydro-conditioning only |

Check which env is active: `$env:CONDA_DEFAULT_ENV`  
Check the Python executable: `Get-Command python | Select-Object -ExpandProperty Source`
```