# tq23414 — Trexquant Earnings Return Prediction Challenge

A clean, progressive Kaggle pipeline for the
[Trexquant Earnings Return Prediction Challenge](https://www.kaggle.com/competitions/earnings-return-prediction-challenge-2025-q-4).

---

## Notebook: `trexquant_progressive_pipeline.ipynb`

The notebook is organized into five phases that build from lightweight baselines
to strong ensembles and an optional advanced sequence branch.

| Phase | Content |
|-------|---------|
| 0 | Setup, reproducibility, data loading, schema checks, EDA-lite |
| 1 | Leakage-safe time-based CV with embargo, Pearson IC utilities, Ridge / ElasticNet / LightGBM baselines |
| 2 | Feature engineering: cross-sectional rank/z-score, grouped context features, lag/rolling by `si` |
| 3 | Strong LightGBM, optional XGBoost & CatBoost, weighted blend optimization |
| 4 | Optional lightweight LSTM sequence branch + meta-feature stacking |
| 5 | Final train, submission export, submission guard checks |

---

## FAST vs FULL mode

Set `FAST_MODE` in the first code cell to control runtime vs quality trade-off.

| Toggle | `FAST_MODE = True` | `FAST_MODE = False` |
|--------|--------------------|---------------------|
| CV folds | 3 | 5 |
| LGB `num_leaves` | 31–63 | 63–127 |
| LGB `n_estimators` | 200–400 | 500–1500 |
| XS rank features | top-50 | top-50 |
| Sequence epochs | 5 | 20 |
| Typical Kaggle runtime | ~15–30 min | ~60–120 min |

Additional feature toggles (set independently of `FAST_MODE`):

| Toggle | Default | Description |
|--------|---------|-------------|
| `USE_FE` | `True` | Phase 2 feature engineering |
| `USE_GBDT_STACK` | `True` | Phase 3 XGBoost / CatBoost stack |
| `USE_XGB` | `False` | Requires `xgboost` package |
| `USE_CAT` | `False` | Requires `catboost` package |
| `USE_SEQ_BRANCH` | `False` | Requires `torch` (PyTorch) |

---

## Kaggle data paths

```
/kaggle/input/competitions/earnings-return-prediction-challenge-2025-q-4/train.csv
/kaggle/input/competitions/earnings-return-prediction-challenge-2025-q-4/test.csv
```

---

## Submission

The final cell writes `submission.csv` with columns `id, target`.
Before writing, the notebook asserts:
- All predictions are finite (no NaN / Inf).
- At least 10 % of predictions are non-zero.
