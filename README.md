<!-- markdownlint-disable MD033 MD041 -->

<div align="center">

# AA-SI Utils

**Utility functions for NOAA Fisheries Active Acoustics Strategic Initiative data processing**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[Overview](#overview) •
[Installation](#installation) •
[Usage](#usage) •
[Development](#development)

</div>

---

## Overview

`aa_si_utils` provides shared utility functions for NOAA Fisheries AA-SI active acoustics projects. It includes tools for:

- **NCEI data retrieval** — query and download sonar data from the NOAA NCEI Water Column Sonar Data archive
- **Raw file I/O** — open, convert, and combine raw sonar files via echopype
- **Depth analysis** — find depth ranges and closest indices for sonar data
- **Distance calculations** — haversine distance between geographic coordinates
- **Data masking** — seafloor removal, surface masking, sparse bin masking, and frequency channel masking
- **Dive profile integration** — align dive profile CSV data to MVBS datasets
- **Visualization helpers** — color generation utilities

### Requirements

- Python 3.10 or higher
- numpy, pandas, xarray, matplotlib, echopype, requests

---

## Installation

```bash
# Clone the repository
git clone https://github.com/nmfs-ost/AA-SI_Utils.git
cd AA-SI_Utils

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

---

## Usage

### NCEI Data Retrieval

```python
import aa_si_utils

# Query the NCEI Water Column Sonar Data archive
results = aa_si_utils.query_ncei_data(
    dataset_name="DY2207_EK80",
    frequencies=["38WKHZ", "120WKHZ"],
    collection_start="2022-06-04",
    collection_end="2022-06-04",
    max_files=5,
)

# Download the queried files (with optional companion files)
paths = aa_si_utils.download_ncei_data(
    results,
    output_dir="./downloads",
    companion_extensions=[".bot", ".idx"],
)
```

### Raw File I/O

```python
import aa_si_utils

# Set up output folders and resolve raw file paths
raw_file_paths = aa_si_utils.initial_setup_and_validation(
    raw_input_folder="./raw_data",
    netcdf_output_folder_string="./output/netcdf",
    sv_output_folder_string="./output/sv",
    output_logs_folder_string="./output/logs",
)

# Open raw files, convert to netCDF, and combine into one echodata object
echodata = aa_si_utils.open_and_combine_raw_files(
    raw_file_paths,
    netcdf_output_folder="./output/netcdf",
    sonar_model="EK80",
)
```

### Depth & Distance

```python
import aa_si_utils

# Find the range_sample index closest to a target depth
idx = aa_si_utils.get_closest_index_for_depth(sv_data, target_depth=100.0)

# Find the depth range where valid data exists
min_depth, max_depth = aa_si_utils.find_data_depth_range(sv_data)

# Calculate distance between two GPS points (in meters)
dist = aa_si_utils.haversine_distance(lat1, lon1, lat2, lon2)
```

### Data Masking

```python
import aa_si_utils

# One-step: create a combined mask (seafloor + surface + frequency exclusion)
mask = aa_si_utils.create_data_mask(
    echodata, ds_Sv,
    seafloor_buffer_m=10.0,
    surface_depth_m=10.0,
    frequencies_to_mask=[200],
)
ds_Sv = aa_si_utils.apply_mask_to_sv(ds_Sv, mask)

# Or build the mask step by step
mask = aa_si_utils.createSvMask(ds_Sv)
mask = aa_si_utils.remove_seafloor_from_mask(ed_raw, ds_Sv, mask)
mask = aa_si_utils.remove_surface_from_mask(ds_Sv, mask, depth_threshold_m=10.0)
mask = aa_si_utils.mask_frequency_channels(ds_Sv, mask, [200])
ds_Sv = aa_si_utils.apply_mask_to_sv(ds_Sv, mask)
```

---

## Development

### Running Tests

```bash
pytest
pytest --cov=aa_si_utils
```

### Code Quality

```bash
black src/ tests/
pylint src/aa_si_utils
pre-commit run --all-files
```

### Building

```bash
pip install build
python -m build
```

---

## Project Structure

```
├── .gitignore
├── .pre-commit-config.yaml
├── .pylintrc
├── CHANGELOG.md
├── LICENSE
├── NOTICE
├── pyproject.toml
├── README.md
├── src/
│   └── aa_si_utils/
│       ├── __init__.py
│       ├── data_retrieval.py
│       └── utils.py
└── tests/
    ├── conftest.py
    └── test_package.py
```

---

## License

This project uses the Apache License 2.0. See [LICENSE](LICENSE) for details.

---

## Disclaimer

This repository is a scientific product and is not official communication of the National Oceanic and Atmospheric Administration, or the United States Department of Commerce. All NOAA GitHub project code is provided on an ‘as is’ basis and the user assumes responsibility for its use. Any claims against the Department of Commerce or Department of Commerce bureaus stemming from the use of this GitHub project will be governed by all applicable Federal law. Any reference to specific commercial products, processes, or services by service mark, trademark, manufacturer, or otherwise, does not constitute or imply their endorsement, recommendation or favoring by the Department of Commerce. The Department of Commerce seal and logo, or the seal and logo of a DOC bureau, shall not be used in any manner to imply endorsement of any commercial product or activity by DOC or the United States Government.
