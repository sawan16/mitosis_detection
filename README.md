# Code and results: audit of evaluation leakage and scanner shift in mitosis detection

Files
- `prepare_patches.py` - optional ICPR 2012 patch preparation (not needed by the on-the-fly pipeline).
- `mitosis_experiment.py` - ICPR 2012 experiments (`run`, `aggregate`).
- `analyze_results.py` - ICPR analysis (old-vs-new, scanner shift, attention, stain augmentation, leakage control).
- `midog_prepare_kaggle.ipynb` - downloads MIDOG 2021 (Zenodo 10.5281/zenodo.4643381), converts TIFF to JPEG, writes annotation tables.
- `midog_experiment.py` - MIDOG 2021 experiments (`run`, `aggregate`).
- `analyze_midog.py` - reproduces every MIDOG number in the manuscript from `results/` (tables, contrasts, image-bootstrap intervals, stain-augmentation comparison).
- `results/*.jsonl` - one line per training run (hyperparameters, threshold, per-image TP/FP/FN, AP, precision/recall/F1).
  Raw detection files (`preds/*.pkl`) are not included.

Example commands (Kaggle, NVIDIA T4)
    python mitosis_experiment.py run --root <ICPR2012 folder with .bmp/.csv> --out runs --train A,H,Mixed --attn none,dual,cbam --folds 0,1,2,3,4 --seeds 42 --iters 2000
    python mitosis_experiment.py run ... --leaky        # naive-protocol control
    python mitosis_experiment.py run ... --aug hed      # stain augmentation
    python midog_experiment.py run --root <midog2021 folder> --out runs --train XR,S360,CS2,Mixed --attn none --folds 0,1,2,3,4 --seeds 42 --iters 2000
    python midog_experiment.py run ... --train XR,S360,CS2 --aug hed
    python midog_experiment.py aggregate --out runs

Seeds used: 42, 100, 2024. Software versions: [TO BE ADDED by the authors from the Kaggle environment].


Reproduce the MIDOG numbers (no GPU needed):
    python analyze_midog.py results/midog2021_seed42.jsonl,results/midog2021_seed100.jsonl,results/midog2021_seed2024.jsonl results/midog2021_stain_aug_seed42.jsonl
Reproduce the ICPR numbers: python analyze_results.py results/icpr2012_all_runs.jsonl
