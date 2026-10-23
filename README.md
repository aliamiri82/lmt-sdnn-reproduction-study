# LMT-SDNN Reproduction Study

PyTorch coursework investigating lightweight IoT traffic classification, multiteacher distillation, ablations, and evaluation protocol choices.

Coursework by **Ali Amiri**, Sharif University of Technology. Semester 8 — Low-Power Systems Design Project.

## Implementation

`src/` contains data preparation, two teacher models, student and baseline models, distillation losses, training, complexity analysis, and an experiment runner. `results/` and `figures/` retain selected outputs from the original local study; these results were not retrained during repository preparation.

This is a course reproduction/adaptation of the LMT-SDNN paper identified in `src/config.py`. It uses public flow-level CSV records to build an `(N, 32, 8)` context representation. This differs from a raw-packet reconstruction. Assumed and paper-specified settings are labeled in the source.

## Run

```bash
pip install -r requirements.txt
python src/data.py
python src/run_experiments.py all
python src/figures.py
```

Before data preparation, obtain the datasets under their distribution terms and place them at:

```text
data/raw/ton_iot_train_test_network.csv
data/raw/dataset/UNSW_2018_IoT_Botnet_Full5pc_*.csv
```

The preparation script reads these local CSVs; it does not download them. Raw/processed datasets and model checkpoints are excluded. Figures can use the included historical metrics; training requires the datasets and may require substantial compute. Rendering helpers expect `report/tables/`, included as an empty output directory.

Original third-party paper PDFs and university submission documents are excluded. Model architecture and loss interpretations are research choices, not claims of independent verification of the paper.

## Scope and attribution

This repository is a curated coursework archive. Original exercise structure and existing credits are retained; course-provided interfaces and testbenches may be present. Bundled textbooks, assignment PDFs, vendor IP implementations, binaries, and IDE caches are excluded. No blanket open-source license has been added to material with mixed course provenance.

See `VALIDATION.md` for checks and known limits.
