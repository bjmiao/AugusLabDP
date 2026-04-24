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

## Requirements

- Python 3.9 or higher
- Core dependencies:
  - PyQt6
  - NumPy
  - SciPy
  - pandas
  - matplotlib
- Optional utility/analysis dependencies (for scripts under `utils/`):
  - seaborn
  - scikit-learn
  - pyqtgraph
  - allensdk

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd AugusLabDP
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Optional utility dependencies (used by scripts under `utils/`):
```bash
pip install -r requirements-utils.txt
```

For package-based installs, you can also use:
```bash
pip install .
```

Or include utility extras from `pyproject.toml`:
```bash
pip install ".[utils]"
```

## Usage

Run the main application:
```bash
python main.py
```

Basic workflow:
1. Click **Add Folder(s)** to add one or more dataset folders.
2. Select a folder to preview detected data sources.
3. Configure extraction options in the right panel.
4. Confirm the output folder (default: `cachedata/`).
5. Click **Start Extracting Data**.

## Project Structure

```
AugusLabDP/
├── main.py              # Application entry point
├── app/                 # Main application package
│   ├── __init__.py
│   ├── main_window.py   # Main GUI window
│   ├── data_detector.py # Source detection logic
│   ├── data_analyzer.py # Source overview logic
│   ├── data_extractor.py # Extraction pipeline
│   ├── readutil/         # SpikeGLX / Kilosort readers
│   └── ...
├── utils/               # Optional analysis/helper scripts
├── requirements.txt     # Python dependencies
├── pyproject.toml       # Project metadata and optional extras
├── LICENSE              # MIT License
└── README.md           # This file
```

## License

MIT License - see LICENSE file for details.

