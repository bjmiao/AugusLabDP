# AugusLab Data Preprocessing Dashboard

A Python Qt-based dashboard application for Neuropixels data preprocessing. The app scans data folders, detects available sources (Neuropixels AP/LFP, Kilosort outputs, NIDQ, face camera, pupil CSVs, and probe location MAT files), and exports extracted outputs to `cachedata/` (or a custom output folder).

## Features

- Multi-folder dataset loading from the GUI
- Automatic source detection for:
  - Neuropixels AP/LFP streams
  - Kilosort spike-sorting folders
  - NIDQ files
  - Face camera `.npy` files
  - Pupil `.csv` files
  - Probe location `.mat` files
- Per-modality extraction options from the dashboard
- Output export as NumPy arrays and CSV files (depending on modality)
- Optional analysis utilities for notebooks and batch scripts (`AugusLabDP.utils`)

## Requirements

- Python 3.9 or higher (3.11 recommended for the `utils` extra)

### Dashboard app (`pip install .`)

- PyQt6
- NumPy
- SciPy
- pandas
- matplotlib

### Analysis utilities (`pip install ".[utils]"`)

Adds the optional dependencies used by modules under `AugusLabDP/utils/`:

- seaborn
- scikit-learn
- pyqtgraph
- braian
- mne

## Installation

1. Clone this repository:

```bash
git clone <repository-url>
cd AugusLabDP
```

2. Create and activate a virtual environment (recommended):

```bash
conda create -n AugusLabDP-py311 python=3.11
conda activate AugusLabDP-py311
```

3. Install the package:

**Dashboard app only** (GUI + core scientific dependencies):

```bash
pip install .
```

**Analysis / notebook utilities** (adds `utils` optional dependencies on top of the base install):

```bash
pip install ".[utils]"
```

**Editable install for development:**

```bash
pip install -e ".[utils]"
```

**Developer tools** (formatting, linting, tests):

```bash
pip install -e ".[dev]"
```

On Windows PowerShell, keep the quotes around `".[utils]"`.

### Alternative: requirements files

You can still install dependencies directly from the plain requirements files:

```bash
pip install -r requirements.txt
pip install -r requirements-utils.txt
```

For normal use, `pip install .` / `pip install ".[utils]"` is preferred because it also registers the Python packages for import.

## Usage

### Run the dashboard

After `pip install .`, launch the GUI with:

```bash
auguslabdp
```

Or, from a source checkout:

```bash
python main.py
```

Basic workflow:

1. Click **Add Folder(s)** to add one or more dataset folders.
2. Select a folder to preview detected data sources.
3. Configure extraction options in the right panel.
4. Confirm the output folder (default: `cachedata/`).
5. Click **Start Extracting Data**.

### Use the analysis utilities

After `pip install ".[utils]"`, import helpers in notebooks or scripts:

```python
from AugusLabDP.utils.readout_utils import load_dataset, get_all_probe_mapping
from AugusLabDP.utils.brain_region_utils import get_meta_region_coarse
from AugusLabDP.utils.eeg_utils import multitaper_spectrogram
```

See `AugusLabDP/utils/example_usage.ipynb` for a full workflow example.

The GUI code remains importable as the top-level `app` package (for example, `from app.main_window import MainWindow`) so existing in-repo imports continue to work after installation.

## Project Structure

```
AugusLabDP/                     # Repository root
├── main.py                     # Local entry point (source checkout)
├── pyproject.toml              # Package metadata and optional extras
├── requirements.txt            # Core app dependencies (alternative install)
├── requirements-utils.txt      # Utils dependencies (alternative install)
├── AugusLabDP/                 # Python package root
│   ├── __init__.py
│   ├── app/                    # Dashboard GUI (installed as `app`)
│   │   ├── __main__.py         # Console entry point (`auguslabdp`)
│   │   ├── main_window.py
│   │   ├── data_detector.py
│   │   ├── data_extractor.py
│   │   ├── readutil/           # SpikeGLX / Kilosort readers
│   │   └── ...
│   └── utils/                  # Analysis helpers (installed as `AugusLabDP.utils`)
│       ├── brain_region_utils.py
│       ├── readout_utils.py
│       ├── example_usage.ipynb
│       └── ...
├── sample_data/                # Example inputs
├── cachedata/                  # Default extraction output location
├── LICENSE
└── README.md
```

## License

MIT License — see LICENSE file for details.
