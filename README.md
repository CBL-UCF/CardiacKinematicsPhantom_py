# CardiacKinematicsPhantom_py (analyticalPhantom): Synthetic DENSE MRI Phantom Generator
Python and extended version of the mid-ventricular cardiac kinematics phantom available at https://github.com/luigiemp/CardiacKinematicsPhantom

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository provides a tool for generating and exporting synthetic cardiac DENSE (Displacement Encoding with Stimulated Echoes) MRI phantoms. It is highly configurable and features a robust, built-in DICOM exporter that produces ready-to-use DENSE series (AveMag, x/y/z phases), making it ideal for testing DENSE analysis pipelines and DICOM viewers.

---

## ✨ Features

* **Analytical Phantom Model**: Generates a synthetic left-ventricular phantom with configurable geometry, motion, and torsion.
* **Realistic DENSE Simulation**: Simulates DENSE imaging physics, including configurable encoding frequencies and signal-to-noise ratios (SNR).
* **Built-in DICOM Exporter**: The primary feature is its ability to export the generated data as a complete, multi-series DICOM set that is compatible with standard medical imaging software.
* **Multiple Output Formats**: In addition to DICOM, it saves visualizations (PNG), raw phase data (`.npy`, `.mat`), and geometry/displacement data for advanced visualization (VTK/VTU).

---

## 🚀 Quickstart Guide

### 1. Setup Environment

First, clone the repository, create a Python virtual environment, and install the required dependencies.

```bash
# Clone the repository
git clone <https://github.com/CBL-UCF/CardiacKinematicsPhantom_py>
cd AnalyticalPhantom

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Phantom

Open **`config.py`** in a text editor or python. All parameters for geometry, motion, imaging, and output are located here. Adjust them as needed for your simulation.

### 3. Run the Generator

Execute the main script to generate the phantom and all associated output files.

```bash
python main.py
```
After the script finishes, all outputs will be located in the `results/` directory.

---

## ⚙️ Configuration (`config.py`)

All runtime parameters are centralized in `config.py`. Key settings are summarized below.

| Category             | Parameters                                           | Description                                      |
| -------------------- | ---------------------------------------------------- | ------------------------------------------------ |
| **Paths** | `BASE_OUTPUT_PATH`                                   | The root directory for all generated files.      |
| **Geometry** | `R_ENDO`, `R_EPI`, `Z_BOTTOM`, `Z_TOP`               | Sets the endocardial/epicardial radii and axial extent. |
| **Motion** | `COMPUTE_PHANTOM`, `AFINAL`, `OPT_BETA`               | Toggles motion optimization and sets torsion parameters. |
| **Imaging** | `XLIM`, `YLIM`, `HX`, `HY`, `HZ`                       | Defines the Field of View (FOV) and voxel resolution. |
| **DENSE Encoding** | `KE_X`, `KE_Y`, `KE_Z`                                 | Sets the DENSE encoding frequency (cycles/mm).   |
| **Noise** | `SNR`, `REPS`                                        | A list of SNRs and repetitions to simulate.      |
| **Outputs** | `SAVE_PHANTOM_VTK`, `TIME_STEPS`                       | Toggles VTK output and sets temporal sampling.   |

---

## 📁 Generated Outputs

The `main.py` script populates the `results/` directory with the following outputs:

* `results/dicom/`: **Primary Output**. Contains the generated DICOM series (`AveMag`, `x-encPha`, `y-encPha`, `z-encPha`).
* `results/images/`: PNG visualizations of the magnitude and wrapped phase images for each time step.
* `results/phasedata/`: Raw phase data saved as `.npy` and `.mat` files, along with a `phantom_data.json` summary.
* `results/vtk/`: VTK (`.vtu`) files containing the phantom geometry and displacement vectors for visualization in software like ParaView.

---

## 🏗️ Project Structure

```
AnalyticalPhantom/
├── main.py                   # Main executable script to run the generator
├── config.py                 # All runtime parameters and configuration
├── export_to_dicom.py        # DICOM writer and export logic
├── requirements.txt          # List of Python dependencies
├── templates/                # JSON templates for DICOM headers (required)
├── models/                   # Analytical, optimization, and noise models
├── utils/                    # Helper functions for mesh, displacement, etc.
└── results/                  # (It will be generated once the main.py is run)
```

---

## 📦 Dependencies

The core dependencies required to run the simulation are:

* `numpy`
* `scipy`
* `matplotlib`
* `Pillow`
* `pydicom` (Crucial for the DICOM export feature)

To generate a pinned `requirements.txt` file from your configured virtual environment, run:
```bash
pip freeze > requirements.txt
```

