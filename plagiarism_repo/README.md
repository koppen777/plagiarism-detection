# Plagiarism Detection Pipeline

This repository contains the code, dependencies, and lightweight Docker setup for the generative‐plagiarism‑detection project originally developed on Kaggle.

## Contents

```
├── notebooks/
│   └── plagiarismdetection-improve.ipynb   # original notebook
├── scripts/
│   └── detection_pipeline.py               # exported python script
├── run.py                                  # wrapper entry‑point
├── requirements.txt                        # Python dependencies
└── Dockerfile                              # container recipe
```

## Quick start

```bash
# create a virtual environment
python -m venv .venv && source .venv/bin/activate

# install deps
pip install -r requirements.txt

# run pipeline
python run.py --data_dir path/to/dataset --output_dir out_xml
```

## Build & run with Docker

```bash
docker build -t plagiarism-detection .
docker run --rm -v $PWD:/workspace plagiarism-detection        python run.py --data_dir /workspace/data --output_dir /workspace/pred_xml
```

## Notes

* `scripts/detection_pipeline.py` is a linear export of the notebook; each cell is separated by `# %%`.
* Large embedding caches (e.g. `.npy`, `.npz`) are **not** included—upload them separately or mount them into the container.
