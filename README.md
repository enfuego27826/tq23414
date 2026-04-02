# tq23414 — Trexquant Earnings Return Prediction Challenge

Progressive, hypothesis-driven Kaggle pipeline for the
[Trexquant Earnings Return Prediction Challenge 2025 Q4](https://www.kaggle.com/competitions/earnings-return-prediction-challenge-2025-q-4).

---

## Strategy

Each experiment tests one hypothesis, prints a verdict, and updates a global `results` dict.
The notebook automatically selects the best model at the end.

| Experiment | Hypothesis tested |
|-----------|-------------------|
| **Exp 1** — Ridge + ElasticNet | Linear models provide a positive IC floor |
| **Exp 2** — XS rank + linear | Per-date rank normalization improves linear IC |
| **Exp 3** — LightGBM raw | Tree model beats linear by capturing non-linearities |
| **Exp 4** — LightGBM XS-rank | XS rank is (mostly) neutral for tree models |
| **Exp 5** — LightGBM + full FE | Group z-scores + lag/rolling features add lift |
| **Exp 6** — Equal-weight blend | Averaging diverse models beats best single model |
| **Exp 7** — Optimized blend | Weight optimization (Nelder-Mead) beats equal-weight |
| **Exp 8** — Tuned LightGBM | Stronger params + regularization push IC higher |
| **Exp 9** — XGBoost *(optional)* | XGB complements LGB with different regularization |
| **Exp 10** — CatBoost *(optional)* | Native categoricals (sector/industry) add signal |
| **Exp 11** — GBDT meta-stack | LGB meta-learner on OOF predictions beats blend |
| **Exp 12** — LSTM sequence *(optional)* | Temporal sequences capture momentum LGB misses |

---

## Notebook: `trexquant_progressive_pipeline.ipynb`

```
Section 0  Setup, config, imports, data loading, EDA-lite, walk-forward CV, metric utils
Section 1  Exp 1: Ridge + ElasticNet baseline
Section 2  Exp 2: Cross-sectional rank normalization
Section 3  Exp 3: LightGBM raw features
Section 4  Exp 4: LightGBM on XS-rank features (rank-invariance test)
Section 5  Exp 5: Full FE (XS rank + group z-scores + lag/roll by si)
Section 6  Exp 6: Equal-weight blend
Section 7  Exp 7: Nelder-Mead optimized blend
Section 8  Exp 8: Tuned LightGBM
Section 9  Exp 9/10: Optional XGBoost / CatBoost
Section 10 Exp 11: GBDT meta-stacking
Section 11 Exp 12: Optional LSTM sequence branch (Pearson loss)
Section 12 Final: Auto-select best model → submission export
```

The notebook also includes a **synthetic data fallback** so every cell
runs correctly even when Kaggle data paths are unavailable (e.g., local testing).

---

## FAST vs FULL mode

Set `FAST_MODE` in the first code cell:

| Setting | `FAST_MODE = True` | `FAST_MODE = False` |
|---------|-------------------|---------------------|
| CV folds | 3 | 5 |
| LGB `num_leaves` | 31 / 63 | 63 / 127 |
| LGB `n_estimators` | 200 / 300 | 500 / 1000 |
| LSTM epochs | 5 | 20 |
| Typical Kaggle runtime | ~20–40 min | ~90–180 min |

Additional toggles (all independent of `FAST_MODE`):

| Toggle | Default | Requires |
|--------|---------|---------|
| `USE_FE_XS_RANK` | `True` | — |
| `USE_FE_GROUP` | `True` | — |
| `USE_FE_LAG` | `True` | — |
| `USE_GBDT_STACK` | `True` | — |
| `USE_XGB` | `False` | `pip install xgboost` |
| `USE_CAT` | `False` | `pip install catboost` |
| `USE_SEQ_BRANCH` | `False` | `pip install torch` |

---

## Kaggle data paths

```
/kaggle/input/competitions/earnings-return-prediction-challenge-2025-q-4/train.csv
/kaggle/input/competitions/earnings-return-prediction-challenge-2025-q-4/test.csv
```

---

## Submission

The final cell writes `submission.csv` (`id, target`).
Guards applied before writing:
- All predictions replaced with 0 where NaN or Inf.
- If < 10 % non-zero, tiny noise is added to satisfy the competition requirement.
- Best model auto-selected by highest OOF Pearson IC.

---

## References

Strategy derived from `Earnings Return Prediction Challenge Research.txt` in this repo.
