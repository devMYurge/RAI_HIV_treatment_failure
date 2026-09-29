# Local Environment Setup Guide

## Prerequisites

| Tool | Check | Install |
|---|---|---|
| **Python 3.10.x** | `python --version` | [python.org](https://www.python.org/downloads/) or `brew install python@3.10` (macOS) |
| **pip** | `pip --version` | Comes with Python |
| **Git** | `git --version` | [git-scm.com](https://git-scm.com/) |

> ⚠️ **Python 3.10 is required.** The RAI packages (`responsibleai`, `raiwidgets`) have strict compatibility — 3.11+ may cause install failures. If you have multiple Python versions, use `python3.10` explicitly below.

---

## Option A — pip + venv (Recommended)

### 1. Clone or create your project folder

```bash
mkdir actg175-rai-project && cd actg175-rai-project
```

Place these files inside:
```
actg175-rai-project/
├── data/
│   └── ACTG175.csv            ← download or auto-fetched by notebook
├── ML4HL_ACTG175_RAI_toolbox.ipynb
├── utils.py
├── requirements.txt
├── environment.yml
├── README.md
└── SETUP_GUIDE.md             ← this file
```

### 2. Create the virtual environment

```bash
# macOS / Linux
python3.10 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Windows (CMD)
python -m venv .venv
.\.venv\Scripts\activate.bat
```

### 3. Upgrade pip & install dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

> ☕ This may take 3–5 minutes (RAI packages pull many sub-dependencies).

### 4. Verify the install

```bash
python -c "
import sklearn, responsibleai, fairlearn, econml
print(f'scikit-learn   : {sklearn.__version__}')
print(f'responsibleai  : {responsibleai.__version__}')
print(f'fairlearn      : {fairlearn.__version__}')
print(f'econml         : {econml.__version__}')
print('✅ All packages OK')
"
```

### 5. Launch Jupyter

```bash
jupyter lab
# or
jupyter notebook
```

Then open `ML4HL_ACTG175_RAI_toolbox.ipynb` and **Run All**.

---

## Option B — Conda / Mamba

If you prefer conda (handles compiled C dependencies more reliably on some systems):

```bash
conda env create -f environment.yml
conda activate actg175_rai
jupyter lab
```

---

## Dataset Setup

The notebook automatically attempts to load the dataset. You have two options:

### Option 1 — Automatic Download (recommended)
The notebook's first data cell tries `pd.read_csv("data/ACTG175.csv")`. If the file doesn't exist, it auto-downloads from UCI using `ucimlrepo`:

```python
from ucimlrepo import fetch_ucirepo
data = fetch_ucirepo(id=890)
df = pd.concat([data.data.features, data.data.targets], axis=1)
df.to_csv("data/ACTG175.csv", index=False)
```

This requires `pip install ucimlrepo` (included in requirements.txt).

### Option 2 — Manual Download
1. Go to https://archive.ics.uci.edu/dataset/890/aids+clinical+trials+group+study+175
2. Download the CSV file
3. Place it at `data/ACTG175.csv` in your project folder

---

## Troubleshooting

### ❌ `error: Microsoft Visual C++ 14.0 or greater is required` (Windows)

Some RAI dependencies need a C compiler on Windows.

1. Install [Build Tools for Visual Studio](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. Select **"Desktop development with C++"**
3. Retry `pip install -r requirements.txt`

### ❌ `No matching distribution found for responsibleai==0.34.0`

You're likely on Python 3.12+. Downgrade:
```bash
# If using pyenv
pyenv install 3.10.14
pyenv local 3.10.14
```

### ❌ RAI Dashboard doesn't render in Jupyter

```bash
# Ensure widgets are enabled
jupyter labextension list   # should show ipywidgets
# If blank, try:
pip install --upgrade jupyterlab ipywidgets
jupyter lab build
```

Also: **File → Trust Notebook** in JupyterLab.

### ❌ `ModuleNotFoundError: No module named 'utils'`

Jupyter's working directory must be the project root (where `utils.py` lives). Check:
```python
import os
print(os.getcwd())  # Should show your project folder
```

### ❌ `ModuleNotFoundError: No module named 'ucimlrepo'`

```bash
pip install ucimlrepo
```

Or simply place the CSV manually in `data/ACTG175.csv`.

### ❌ Slow install / timeout

```bash
pip install -r requirements.txt --default-timeout=120
```

### ❌ M1/M2 Mac issues with scipy or numpy

```bash
# Use conda instead — it ships pre-built ARM64 binaries
conda env create -f environment.yml
```

---

## Quick Sanity Check

After setup, run this in a notebook cell or Python shell:

```python
import pandas as pd

# Try local file first, then UCI download
try:
    df = pd.read_csv("data/ACTG175.csv")
except FileNotFoundError:
    from ucimlrepo import fetch_ucirepo
    data = fetch_ucirepo(id=890)
    df = pd.concat([data.data.features, data.data.targets], axis=1)

print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Target (cid): {df['cid'].value_counts().to_dict()}")
print(f"Treatment arms: {df['trt'].value_counts().sort_index().to_dict()}")
```

Expected output (approximately):
```
Dataset loaded: 2139 rows, 27 columns
Target (cid): {0: 1469, 1: 670}
Treatment arms: {0: 532, 1: 522, 2: 524, 3: 561}
```

---

## Deactivating / Removing the Environment

```bash
# venv
deactivate
rm -rf .venv

# conda
conda deactivate
conda env remove -n actg175_rai
```
