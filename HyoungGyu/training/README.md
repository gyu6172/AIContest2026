# Training Versions

Training artifacts are grouped by version folder.

- `v1/`: original LightGBM ranker notebook and outputs.
- `v2/`: improved ranker script/notebook and outputs.

For future changes, create a new folder instead of modifying an older version in place:

- next experiment: `v3/`
- following experiment: `v4/`

Recommended folder contents:

- `train_lgbm_ranker_vN.py` or `train_lgbm_ranker_vN.ipynb`
- `metrics.json`
- `feature_importance.csv`
- `submission_lgbm_ranker_vN.csv`
- `artifacts/` for model bundles, validation predictions, and error samples
