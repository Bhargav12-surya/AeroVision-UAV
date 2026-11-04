# AIR PROJECT – UAV Part 1

A collection of scripts and models for aerial‑image‑recognition (YOLO, SSD, LSTM, etc.).

## Folder structure

```
.
├─ .vscode/                # VS Code workspace settings (ignored)
├─ Uav_part1/
│   ├─ compare_models.py   # Compare YOLO vs SSD performance
│   ├─ convert_to_yolo.py  # Convert annotations to YOLO format
│   ├─ dataset*.yaml       # Dataset definitions (RGB / thermal)
│   ├─ train_*.py          # Training scripts for different modalities
│   ├─ test_ssd.py         # SSD inference script
│   ├─ models/             # Model checkpoints (tracked with Git LFS)
│   ├─ data/               # Raw NumPy data (ignored)
│   ├─ runs/               # Generated results (ignored)
│   └─ uav_env/            # Local Python virtual‑env (ignored)
└─ README.md               # This file
```

## Setup

```powershell
# 1️⃣ Clone the repo
git clone <repo‑url>
cd <repo‑dir>

# 2️⃣ Create a virtual environment (recommended)
python -m venv venv
.\venv\Scripts\activate   # Windows PowerShell

# 3️⃣ Install dependencies
pip install -r requirements.txt   # (or `conda env create -f environment.yml`)

# 4️⃣ (Optional) Pull large files via Git LFS
git lfs pull
```

## Running a script

```powershell
# Example: train the RGB YOLO model
python Uav_part1/train_rgb.py --epochs 50 --batch-size 16
```

Adjust command‑line arguments as needed; each script prints its own usage help.

## License & credits

*Add license information here*
*Mention any third‑party datasets/models used.*

