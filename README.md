# AugusLab Data Preprocessing Dashboard

A Python Qt-based dashboard application for Neuropixels data preprocessing. This software loads SpikeGLX binary files, runs preprocessing steps, integrates behavioral camera and pupil physiology metrics, and organizes all outputs as NumPy `.npy` files.

## Features

- Load SpikeGLX binary files
- Run preprocessing pipeline
- Load behavioral camera data
- Load pupil physiology metrics
- Export organized NumPy arrays (.npy files)

## Requirements

- Python 3.8 or higher
- PyQt6
- NumPy

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

## Usage

Run the main application:
```bash
python main.py
```

## Project Structure

```
AugusLabDP/
├── main.py              # Application entry point
├── app/                 # Main application package
│   ├── __init__.py
│   ├── main_window.py  # Main GUI window
│   └── ...
├── requirements.txt     # Python dependencies
├── LICENSE             # MIT License
└── README.md           # This file
```

## License

MIT License - see LICENSE file for details.

