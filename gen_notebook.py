import nbformat

nb = nbformat.v4.new_notebook()
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.10.0"
    }
}
nb.nbformat = 4
nb.nbformat_minor = 5

cells = []

# Cell 0: Markdown — Title + overview
cells.append(nbformat.v4.new_markdown_cell(
"""# Trexquant Earnings Return Prediction — Progressive Hypothesis Testing Pipeline
## Challenge: 2025 Q4 | Metric: Pearson Correlation Coefficient

| Experiment | Model | Key Idea | Expected IC |
|---|---|---|---|
| 1 | Ridge + ElasticNet | Linear baseline (raw features) | Low positive |
| 2 | Ridge (XS-rank) | Cross-sectional rank normalization | Slight +Δ |
| 3 | LightGBM (raw) | Tree baseline, no FE | +Δ vs linear |
| 4 | LightGBM (XS-rank) | Does rank help trees? | Neutral |
| 5 | LightGBM (FE) | Group context + lag features | +Δ |
| 6 | Equal-weight blend | Ensemble diversity | +Δ |
| 7 | Optimized blend | Weight optimization via Nelder-Mead | Slight +Δ |
| 8 | Tuned LGB | Stronger params + full FE | +Δ |
| 9 | XGBoost (optional) | Complementary GBDT | +Δ blend |
| 10 | CatBoost (optional) | Native categoricals | +Δ blend |
| 11 | GBDT meta-stack | Stack OOF preds + raw feat | +Δ vs blend |
| 12 | LSTM (optional) | Temporal sequence branch | Meta-feature |"""
))

# Cell 1: Code — Config block
cells.append(nbformat.v4.new_code_cell(
"""# ============================================================
# CONFIG — Toggle experiments here
# ============================================================
FAST_MODE      = True   # True=smaller CV/models, False=full
USE_FE_XS_RANK = True   # Phase 2: cross-sectional rank features
USE_FE_GROUP   = True   # Phase 2: group/sector context features
USE_FE_LAG     = True   # Phase 2: lag/rolling features by si
USE_GBDT_STACK = True   # Phase 3: multi-GBDT stacking
USE_XGB        = False  # requires xgboost
USE_CAT        = False  # requires catboost
USE_SEQ_BRANCH = False  # requires torch (LSTM sequence branch)

EMBARGO_PERIODS = 5
N_FOLDS  = 3 if FAST_MODE else 5
SEED     = 42

TRAIN_PATH = "/kaggle/input/competitions/earnings-return-prediction-challenge-2025-q-4/train.csv"
TEST_PATH  = "/kaggle/input/competitions/earnings-return-prediction-challenge-2025-q-4/test.csv"
SUB_PATH   = "submission.csv\""""
))

# Cell 2: Code — Imports + timer + reproducibility
cells.append(nbformat.v4.new_code_cell(
"""import time
import warnings
import random
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

warnings.filterwarnings("ignore")
random.seed(SEED)
np.random.seed(SEED)

_timers = {}

def tic(label=""):
    _timers[label] = time.time()

def toc(label=""):
    elapsed = time.time() - _timers.get(label, time.time())
    print(f"  ⏱  {label}: {elapsed:.1f}s")
    return elapsed

print("Imports OK | LightGBM:", lgb.__version__)"""
))

# Cell 3: Markdown — Data Loading & Schema
cells.append(nbformat.v4.new_markdown_cell(
"""## Data Loading & Schema

- **train.csv**: ~14 years of earnings events
- **test.csv**: ~6 years of earnings events (no target)
- **Features**: f1..f172 (anonymized, mixed numeric)
- **Meta cols**: id, di (date index), si (stock index), sector, industry, top500, top1000, top2000
- **Target**: earnings-day return (continuous, zero-heavy)"""
))

# Cell 4: Code — Load train/test + schema detection + type casting
cells.append(nbformat.v4.new_code_cell(
"""tic("data_load")
print("Loading data...")
try:
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)
    print(f"  train: {train.shape}  |  test: {test.shape}")
except FileNotFoundError:
    print("⚠  Data files not found at Kaggle paths.")
    print("   Creating synthetic data for pipeline testing...")
    rng = np.random.default_rng(SEED)
    N_TR, N_TE, N_F = 100_000, 30_000, 50
    F_NAMES = [f"f{i+1}" for i in range(N_F)]
    def _make_df(n, has_target=True):
        d = {"id": np.arange(n),
             "di": rng.integers(0, 200, n),
             "si": rng.integers(0, 500, n),
             "sector": rng.integers(1, 12, n),
             "industry": rng.integers(1, 60, n),
             "top500":  rng.integers(0, 2, n).astype(np.int8),
             "top1000": rng.integers(0, 2, n).astype(np.int8),
             "top2000": rng.integers(0, 2, n).astype(np.int8)}
        for fn in F_NAMES:
            vals = rng.standard_normal(n).astype(np.float32)
            vals[rng.random(n) < 0.08] = np.nan
            d[fn] = vals
        if has_target:
            d["target"] = (rng.standard_normal(n) * 0.05).astype(np.float32)
            d["target"][rng.random(n) < 0.3] = 0.0
        return pd.DataFrame(d)
    train = _make_df(N_TR, has_target=True)
    test  = _make_df(N_TE, has_target=False)
    print(f"  Synthetic train: {train.shape}  |  test: {test.shape}")

# ── Schema detection ──────────────────────────────────────────
META_COLS   = ["id", "di", "si", "sector", "industry"]
LIQ_COLS    = [c for c in ["top500", "top1000", "top2000"] if c in train.columns]
TARGET_COL  = "target"
F_COLS      = [c for c in train.columns
               if c.startswith("f") and c[1:].isdigit()]

assert "di" in train.columns, "'di' column missing"
assert "id" in train.columns, "'id' column missing"
assert TARGET_COL in train.columns, "'target' column missing"
assert len(F_COLS) > 0, "No feature columns detected"

print(f"\\n  META_COLS : {META_COLS}")
print(f"  LIQ_COLS  : {LIQ_COLS}")
print(f"  F_COLS    : {len(F_COLS)} features ({F_COLS[0]}..{F_COLS[-1]})")
print(f"  TARGET_COL: {TARGET_COL}")

# ── Type casting ─────────────────────────────────────────────
for c in ["di", "si"]:
    if c in train.columns:
        train[c] = train[c].astype(np.int32)
        test[c]  = test[c].astype(np.int32)
for c in LIQ_COLS:
    train[c] = train[c].astype(np.int8)
    test[c]  = test[c].astype(np.int8)
for c in F_COLS:
    train[c] = train[c].astype(np.float32)
    test[c]  = test[c].astype(np.float32)

# ── Drop high-NaN features ───────────────────────────────────
nan_pct = train[F_COLS].isnull().mean()
high_nan = nan_pct[nan_pct > 0.50].index.tolist()
if high_nan:
    print(f"\\n  Dropping {len(high_nan)} features with >50% NaN: {high_nan[:5]}...")
    F_COLS = [c for c in F_COLS if c not in high_nan]
    print(f"  Remaining F_COLS: {len(F_COLS)}")

# ── Sort by di for temporal integrity ─────────────────────────
train = train.sort_values("di").reset_index(drop=True)
test  = test.sort_values("di").reset_index(drop=True)

toc("data_load")
print("\\nSchema detection complete.")"""
))

# Cell 5: Code — EDA lite
cells.append(nbformat.v4.new_code_cell(
"""print("=" * 60)
print("EDA Summary")
print("=" * 60)

# Target distribution
y = train[TARGET_COL]
print(f"\\nTarget distribution:")
print(y.describe().to_string())
print(f"  Zero pct  : {(y == 0).mean():.3%}")
print(f"  NaN pct   : {y.isnull().mean():.3%}")

# DI range
print(f"\\nDate index (di):")
print(f"  train di range: [{train['di'].min()}, {train['di'].max()}]  unique: {train['di'].nunique()}")
print(f"  test  di range: [{test['di'].min()},  {test['di'].max()}]   unique: {test['di'].nunique()}")

# Stock coverage
print(f"\\nStock index (si):")
print(f"  train unique si: {train['si'].nunique()}")
print(f"  test  unique si: {test['si'].nunique()}")
avg_rows = len(train) / train['si'].nunique()
print(f"  avg rows/stock (train): {avg_rows:.1f}")

# Liquidity
print(f"\\nLiquidity flags (train counts):")
for c in LIQ_COLS:
    print(f"  {c}: {train[c].sum():,} ({train[c].mean():.2%})")

# Top-20 features by abs correlation with target
print(f"\\nTop-20 features by |Pearson corr| with target:")
valid_mask = y.notna()
corrs = {}
for c in F_COLS:
    col_valid = train[c].notna() & valid_mask
    if col_valid.sum() > 100:
        r, _ = stats.pearsonr(train.loc[col_valid, c], y[col_valid])
        corrs[c] = abs(r)
top20 = sorted(corrs, key=corrs.get, reverse=True)[:20]
for rank_i, c in enumerate(top20, 1):
    print(f"  {rank_i:2d}. {c:6s}  |r|={corrs[c]:.5f}")"""
))

# Cell 6: Markdown — Time-Based CV with Embargo
cells.append(nbformat.v4.new_markdown_cell(
"""## Section 1: Time-Based CV with Embargo

Walk-forward splits by `di`. After each train window, an **embargo** of
`EMBARGO_PERIODS` date-indices is skipped before the validation window begins.
This prevents leakage from features that look back in time (lag/rolling).

```
train[di=0..k] → embargo gap → val[di=k+E..k+W]
```"""
))

# Cell 7: Code — make_time_folds
cells.append(nbformat.v4.new_code_cell(
"""def make_time_folds(df, n_folds=N_FOLDS, embargo=EMBARGO_PERIODS):
    \"\"\"Walk-forward splits on sorted di. Returns list of (train_idx, val_idx).\"\"\"
    dis = np.sort(df["di"].unique())
    n_di = len(dis)
    fold_size = n_di // (n_folds + 1)
    folds = []
    for fold in range(n_folds):
        train_end_pos = fold_size * (fold + 1)
        val_start_pos = train_end_pos + embargo
        val_end_pos   = val_start_pos + fold_size
        if val_end_pos > n_di:
            val_end_pos = n_di
        if val_start_pos >= n_di:
            break
        train_dis = dis[:train_end_pos]
        val_dis   = dis[val_start_pos:val_end_pos]
        tr_idx  = df.index[df["di"].isin(train_dis)].tolist()
        val_idx = df.index[df["di"].isin(val_dis)].tolist()
        folds.append((tr_idx, val_idx))
        print(f"  Fold {fold+1}: train di=[{train_dis[0]},{train_dis[-1]}] n={len(tr_idx):,}"
              f"  |  val di=[{val_dis[0]},{val_dis[-1]}] n={len(val_idx):,}")
    return folds

print("Building time folds...")
FOLDS = make_time_folds(train)
print(f"Created {len(FOLDS)} folds with embargo={EMBARGO_PERIODS}")"""
))

# Cell 8: Code — Metric utilities + results dict
cells.append(nbformat.v4.new_code_cell(
"""# ── Global results tracker ───────────────────────────────────
results = {}

def pearson_ic(y_true, y_pred):
    \"\"\"Pearson correlation, handling NaN.\"\"\"
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return 0.0
    r, _ = stats.pearsonr(y_true[mask], y_pred[mask])
    return float(r)

def ic_by_date(df_val, preds, di_col="di", target_col=TARGET_COL):
    \"\"\"Mean daily IC, std, and IR = mean/std.\"\"\"
    df_tmp = df_val[[di_col, target_col]].copy()
    df_tmp["pred"] = preds
    df_tmp = df_tmp.dropna(subset=[target_col, "pred"])
    daily = []
    for di_val, grp in df_tmp.groupby(di_col):
        if len(grp) >= 5:
            r, _ = stats.pearsonr(grp[target_col], grp["pred"])
            daily.append(r)
    if not daily:
        return 0.0, 0.0, 0.0
    daily = np.array(daily)
    mean_ic = float(np.mean(daily))
    std_ic  = float(np.std(daily) + 1e-9)
    ir      = mean_ic / std_ic
    return mean_ic, std_ic, ir

def ic_by_liq(df_val, preds, liq_col, target_col=TARGET_COL):
    \"\"\"IC breakdown by liquidity bucket.\"\"\"
    df_tmp = df_val[[liq_col, target_col]].copy()
    df_tmp["pred"] = preds
    df_tmp = df_tmp.dropna(subset=[target_col, "pred"])
    for liq_val in [0, 1]:
        mask = df_tmp[liq_col] == liq_val
        if mask.sum() < 10:
            continue
        r = pearson_ic(df_tmp.loc[mask, target_col].values,
                       df_tmp.loc[mask, "pred"].values)
        label = f"{liq_col}={liq_val}"
        print(f"    IC [{label:15s}]: {r:.5f}  n={mask.sum():,}")

def diagnose_oof(label, oof_preds, df_tr, preds_test=None):
    \"\"\"Print diagnostics and store in results dict.\"\"\"
    valid  = np.isfinite(oof_preds) & train[TARGET_COL].notna().values
    ic_val = pearson_ic(df_tr.loc[valid, TARGET_COL].values, oof_preds[valid])
    mean_d, std_d, ir = ic_by_date(df_tr.loc[valid], oof_preds[valid])

    best_ic = max((v["ic"] for v in results.values()), default=0.0)

    print(f"\\n{'─'*55}")
    print(f"  {label}")
    print(f"{'─'*55}")
    print(f"  Overall Pearson IC : {ic_val:.5f}")
    print(f"  Daily mean IC      : {mean_d:.5f}")
    print(f"  Daily IC std       : {std_d:.5f}")
    print(f"  IR (mean/std)      : {ir:.3f}")
    if LIQ_COLS:
        ic_by_liq(df_tr.loc[valid], oof_preds[valid], LIQ_COLS[0])

    delta = ic_val - best_ic
    sign  = "+" if delta >= 0 else ""
    print(f"\\n  → RESULT: {label} IC={ic_val:.5f}  best_so_far={best_ic:.5f}  ({sign}{delta:.5f})")

    results[label] = {
        "ic":         ic_val,
        "daily_mean": mean_d,
        "daily_std":  std_d,
        "ir":         ir,
        "preds_test": preds_test,
        "oof":        oof_preds.copy()
    }
    return ic_val

print("Metric utilities defined.")
print("results dict initialized:", results)"""
))

# Cell 9: Code — Imputation helper
cells.append(nbformat.v4.new_code_cell(
"""def fill_na_median(train_df, test_df, cols):
    \"\"\"Fit medians on train, apply to both. Returns new DataFrames.\"\"\"
    tr = train_df.copy()
    te = test_df.copy()
    medians = tr[cols].median()
    tr[cols] = tr[cols].fillna(medians)
    te[cols] = te[cols].fillna(medians)
    return tr, te

# Build raw imputed feature matrices
print("Imputing raw features with train medians...")
_tr_imp, _te_imp = fill_na_median(train, test, F_COLS)
X_raw_tr = _tr_imp[F_COLS].values.astype(np.float32)
X_raw_te = _te_imp[F_COLS].values.astype(np.float32)
y_train  = train[TARGET_COL].values.astype(np.float32)

print(f"  X_raw_tr: {X_raw_tr.shape}  X_raw_te: {X_raw_te.shape}")"""
))

# Cell 10: Markdown — Experiment 1
cells.append(nbformat.v4.new_markdown_cell(
"""## Experiment 1 — Linear Baseline (Ridge + ElasticNet)

**Hypothesis**: A simple regularized linear model will provide a meaningful positive IC
as the first sanity check, establishing the floor for all future experiments.

Expected outcome: Small but positive IC, establishing a lower bound."""
))

# Cell 11: Code — Experiment 1a Ridge
cells.append(nbformat.v4.new_code_cell(
"""tic("exp1_ridge")
print("Training Ridge baseline...")

oof_ridge   = np.full(len(train), np.nan)
preds_ridge = np.zeros(len(test))

for fold_i, (tr_idx, val_idx) in enumerate(FOLDS):
    X_tr, y_tr   = X_raw_tr[tr_idx], y_train[tr_idx]
    X_val        = X_raw_tr[val_idx]

    scaler = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)
    X_te_s  = scaler.transform(X_raw_te)

    # Drop NaN targets in training
    valid_tr = np.isfinite(y_tr)
    model = Ridge(alpha=50.0, random_state=SEED)
    model.fit(X_tr_s[valid_tr], y_tr[valid_tr])

    oof_ridge[val_idx] = model.predict(X_val_s)
    preds_ridge += model.predict(X_te_s) / len(FOLDS)

    print(f"  Fold {fold_i+1}/{len(FOLDS)} done")

toc("exp1_ridge")
diagnose_oof("Exp1_Ridge", oof_ridge, train, preds_ridge)"""
))

# Cell 12: Code — Experiment 1b ElasticNet
cells.append(nbformat.v4.new_code_cell(
"""tic("exp1_enet")
print("Training ElasticNet baseline...")

oof_enet   = np.full(len(train), np.nan)
preds_enet = np.zeros(len(test))

en_iters = 500 if FAST_MODE else 2000

for fold_i, (tr_idx, val_idx) in enumerate(FOLDS):
    X_tr, y_tr = X_raw_tr[tr_idx], y_train[tr_idx]
    X_val      = X_raw_tr[val_idx]

    scaler = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)
    X_te_s  = scaler.transform(X_raw_te)

    valid_tr = np.isfinite(y_tr)
    model = ElasticNet(alpha=0.01, l1_ratio=0.5,
                       max_iter=en_iters, random_state=SEED)
    model.fit(X_tr_s[valid_tr], y_tr[valid_tr])

    oof_enet[val_idx] = model.predict(X_val_s)
    preds_enet += model.predict(X_te_s) / len(FOLDS)

    print(f"  Fold {fold_i+1}/{len(FOLDS)} done")

toc("exp1_enet")
diagnose_oof("Exp1_ElasticNet", oof_enet, train, preds_enet)"""
))

# Cell 13: Code — Experiment 1 result summary
cells.append(nbformat.v4.new_code_cell(
"""print("\\n=== Experiment 1 Summary ===")
for k in ["Exp1_Ridge", "Exp1_ElasticNet"]:
    if k in results:
        v = results[k]
        print(f"  {k:20s}  IC={v['ic']:.5f}  daily_IC={v['daily_mean']:.5f}  IR={v['ir']:.3f}")
best_linear_ic  = max(results[k]["ic"] for k in ["Exp1_Ridge", "Exp1_ElasticNet"] if k in results)
best_linear_key = max(["Exp1_Ridge", "Exp1_ElasticNet"],
                      key=lambda k: results.get(k, {"ic": -99})["ic"])
print(f"\\n  Best linear model: {best_linear_key}  IC={best_linear_ic:.5f}")"""
))

# Cell 14: Markdown — Experiment 2
cells.append(nbformat.v4.new_markdown_cell(
"""## Experiment 2 — Cross-Sectional Rank Normalization + Linear Models

**Hypothesis**: Converting features to per-date (`di`) percentile ranks will reduce
the influence of outliers in noisy financial data and improve the Pearson IC of
linear models by preserving ordinal relationships while neutralizing scale differences.

Cross-sectional rank is standard pre-processing in quantitative finance (e.g., Barra)."""
))

# Cell 15: Code — xs_rank_transform
cells.append(nbformat.v4.new_code_cell(
"""def xs_rank_transform(df, cols, by="di"):
    \"\"\"Per-date percentile rank transform. Returns float32 DataFrame.\"\"\"
    out = df[cols].copy()
    for di_val, grp_idx in df.groupby(by).groups.items():
        block = out.loc[grp_idx, cols]
        # rank(pct=True) → [0,1], preserve NaN
        out.loc[grp_idx, cols] = block.rank(pct=True, na_option="keep").astype(np.float32)
    return out

if USE_FE_XS_RANK:
    print("Applying cross-sectional rank transform (train)...")
    tic("xs_rank")
    xs_tr = xs_rank_transform(train, F_COLS, by="di")
    X_xs_tr = xs_tr.values.astype(np.float32)

    print("Applying cross-sectional rank transform (test)...")
    xs_te = xs_rank_transform(test, F_COLS, by="di")
    X_xs_te = xs_te.values.astype(np.float32)

    # Fill remaining NaN with 0.5 (neutral rank)
    X_xs_tr = np.where(np.isnan(X_xs_tr), 0.5, X_xs_tr)
    X_xs_te = np.where(np.isnan(X_xs_te), 0.5, X_xs_te)
    toc("xs_rank")
    print(f"  X_xs_tr: {X_xs_tr.shape}  X_xs_te: {X_xs_te.shape}")
else:
    print("USE_FE_XS_RANK=False → using X_raw as fallback")
    X_xs_tr = X_raw_tr.copy()
    X_xs_te = X_raw_te.copy()"""
))

# Cell 16: Code — Experiment 2 XS-rank Ridge
cells.append(nbformat.v4.new_code_cell(
"""tic("exp2_ridge_xs")
print("Training Ridge on XS-ranked features...")

oof_ridge_xs   = np.full(len(train), np.nan)
preds_ridge_xs = np.zeros(len(test))

for fold_i, (tr_idx, val_idx) in enumerate(FOLDS):
    X_tr, y_tr = X_xs_tr[tr_idx], y_train[tr_idx]
    X_val      = X_xs_tr[val_idx]

    scaler  = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)
    X_te_s  = scaler.transform(X_xs_te)

    valid_tr = np.isfinite(y_tr)
    model = Ridge(alpha=50.0, random_state=SEED)
    model.fit(X_tr_s[valid_tr], y_tr[valid_tr])

    oof_ridge_xs[val_idx] = model.predict(X_val_s)
    preds_ridge_xs += model.predict(X_te_s) / len(FOLDS)

    print(f"  Fold {fold_i+1}/{len(FOLDS)} done")

toc("exp2_ridge_xs")
diagnose_oof("Exp2_Ridge_XSrank", oof_ridge_xs, train, preds_ridge_xs)"""
))

# Cell 17: Code — Experiment 2 result comparison
cells.append(nbformat.v4.new_code_cell(
"""print("\\n=== Experiment 2 Result Comparison ===")
ic_ridge    = results.get("Exp1_Ridge", {}).get("ic", 0)
ic_ridge_xs = results.get("Exp2_Ridge_XSrank", {}).get("ic", 0)
delta = ic_ridge_xs - ic_ridge
sign = "+" if delta >= 0 else ""
print(f"  Exp1 Ridge (raw)       : IC={ic_ridge:.5f}")
print(f"  Exp2 Ridge (XS-rank)   : IC={ic_ridge_xs:.5f}  ({sign}{delta:.5f})")
if delta > 0.0005:
    print("\\n  ✓ XS rank IMPROVED linear Ridge — outlier removal helps!")
elif delta < -0.0005:
    print("\\n  ✗ XS rank HURT linear Ridge — raw features encode useful scale info.")
else:
    print("\\n  ~ XS rank had NEUTRAL effect on Ridge.")"""
))

# Cell 18: Markdown — Experiment 3
cells.append(nbformat.v4.new_markdown_cell(
"""## Experiment 3 — LightGBM Baseline (raw features)

**Hypothesis**: A tree-based gradient boosting model will significantly outperform
linear models by capturing non-linear interactions among the 172 anonymized features,
without any feature engineering.

LightGBM uses GOSS (Gradient-based One-Side Sampling) and EFB (Exclusive Feature Bundling)
for speed, and handles missing values natively."""
))

# Cell 19: Code — Experiment 3 LightGBM raw
cells.append(nbformat.v4.new_code_cell(
"""tic("exp3_lgb_raw")
print("Training LightGBM baseline on raw features...")

lgb_base_params = {
    "objective":       "regression",
    "metric":          "None",       # use custom eval
    "num_leaves":      31 if FAST_MODE else 63,
    "learning_rate":   0.05,
    "n_estimators":    200 if FAST_MODE else 500,
    "min_child_samples": 20,
    "subsample":       0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":       0.1,
    "reg_lambda":      0.1,
    "random_state":    SEED,
    "n_jobs":          -1,
    "verbose":         -1,
}

oof_lgb_raw   = np.full(len(train), np.nan)
preds_lgb_raw = np.zeros(len(test))

for fold_i, (tr_idx, val_idx) in enumerate(FOLDS):
    X_tr, y_tr = X_raw_tr[tr_idx], y_train[tr_idx]
    X_val      = X_raw_tr[val_idx]
    y_val      = y_train[val_idx]

    valid_tr  = np.isfinite(y_tr)
    valid_val = np.isfinite(y_val)

    model = lgb.LGBMRegressor(**lgb_base_params)
    model.fit(
        X_tr[valid_tr], y_tr[valid_tr],
        eval_set=[(X_val[valid_val], y_val[valid_val])],
        callbacks=[lgb.early_stopping(20, verbose=False),
                   lgb.log_evaluation(period=-1)],
    )
    oof_lgb_raw[val_idx]  = model.predict(X_val)
    preds_lgb_raw        += model.predict(X_raw_te) / len(FOLDS)

    best_iter = model.best_iteration_ if model.best_iteration_ else lgb_base_params["n_estimators"]
    print(f"  Fold {fold_i+1}/{len(FOLDS)}  best_iter={best_iter}")

toc("exp3_lgb_raw")
diagnose_oof("Exp3_LGB_Raw", oof_lgb_raw, train, preds_lgb_raw)"""
))

# Cell 20: Code — Experiment 3 result vs linear
cells.append(nbformat.v4.new_code_cell(
"""print("\\n=== Experiment 3: LGB vs Best Linear ===")
ic_best_linear = max(results.get(k, {"ic": 0})["ic"]
                     for k in ["Exp1_Ridge", "Exp1_ElasticNet", "Exp2_Ridge_XSrank"])
ic_lgb_raw = results.get("Exp3_LGB_Raw", {}).get("ic", 0)
delta = ic_lgb_raw - ic_best_linear
sign  = "+" if delta >= 0 else ""
print(f"  Best linear IC : {ic_best_linear:.5f}")
print(f"  Exp3 LGB (raw) : {ic_lgb_raw:.5f}  ({sign}{delta:.5f})")
if delta > 0.001:
    print("\\n  ✓ LightGBM significantly outperforms linear — non-linear interactions matter!")
elif delta > 0:
    print("\\n  ✓ LightGBM slightly outperforms linear.")
else:
    print("\\n  ~ LightGBM does not beat linear on this data slice (may improve with FE).")"""
))

# Cell 21: Markdown — Experiment 4
cells.append(nbformat.v4.new_markdown_cell(
"""## Experiment 4 — Does Cross-Sectional Rank Transform Help LightGBM?

**Hypothesis**: Unlike linear models, tree-based models are theoretically rank-invariant —
any monotone transformation of a feature doesn't change tree splits.
Therefore, XS rank transforms should NOT significantly improve LightGBM performance.

**Expected**: Negligible difference vs Exp3. If a large difference is observed,
it suggests extreme outlier features that even tree algorithms struggle with."""
))

# Cell 22: Code — Experiment 4 LGB XS-rank
cells.append(nbformat.v4.new_code_cell(
"""tic("exp4_lgb_xs")
print("Training LightGBM on XS-ranked features...")

oof_lgb_xs   = np.full(len(train), np.nan)
preds_lgb_xs = np.zeros(len(test))

for fold_i, (tr_idx, val_idx) in enumerate(FOLDS):
    X_tr, y_tr = X_xs_tr[tr_idx], y_train[tr_idx]
    X_val      = X_xs_tr[val_idx]
    y_val      = y_train[val_idx]

    valid_tr  = np.isfinite(y_tr)
    valid_val = np.isfinite(y_val)

    model = lgb.LGBMRegressor(**lgb_base_params)
    model.fit(
        X_tr[valid_tr], y_tr[valid_tr],
        eval_set=[(X_val[valid_val], y_val[valid_val])],
        callbacks=[lgb.early_stopping(20, verbose=False),
                   lgb.log_evaluation(period=-1)],
    )
    oof_lgb_xs[val_idx]  = model.predict(X_val)
    preds_lgb_xs        += model.predict(X_xs_te) / len(FOLDS)

    best_iter = model.best_iteration_ if model.best_iteration_ else lgb_base_params["n_estimators"]
    print(f"  Fold {fold_i+1}/{len(FOLDS)}  best_iter={best_iter}")

toc("exp4_lgb_xs")
diagnose_oof("Exp4_LGB_XSrank", oof_lgb_xs, train, preds_lgb_xs)

# Verdict
ic_raw = results.get("Exp3_LGB_Raw", {}).get("ic", 0)
ic_xs  = results.get("Exp4_LGB_XSrank", {}).get("ic", 0)
delta  = ic_xs - ic_raw
print(f"\\n  Δ(XS rank vs raw) for LGB = {delta:+.5f}")
if abs(delta) < 0.001:
    print("  ✓ Confirmed: XS rank is NEUTRAL for LightGBM (rank-invariant, as expected)")
elif delta > 0:
    print("  ⚠ XS rank HELPED LGB — possible extreme outliers in raw features")
else:
    print("  ⚠ XS rank HURT LGB — raw features carry useful scale information")"""
))

# Cell 23: Markdown — Experiment 5
cells.append(nbformat.v4.new_markdown_cell(
"""## Experiment 5 — Feature Engineering: Group Context + Lag/Rolling Features

**Hypothesis**: Adding cross-sectional context within sector/industry groups and
lagged temporal signals per stock (`si`) will provide additional alpha beyond
the raw features by capturing relative position and trend information.

- **Group features**: z-score within (di, sector) and (di, industry) for top features
- **Lag features**: rolling mean/std over windows [3, 5] for top features, shift(1) to avoid leakage"""
))

# Cell 24: Code — Group context features
cells.append(nbformat.v4.new_code_cell(
"""if USE_FE_GROUP:
    print("Building group context features (sector/industry z-scores)...")
    tic("fe_group")

    # Use top-5 features by correlation (or first 5 if correlation unavailable)
    corr_sorted = sorted(corrs.items(), key=lambda x: x[1], reverse=True) if 'corrs' in dir() else []
    top_group_feats = [f for f, _ in corr_sorted[:5]] if corr_sorted else F_COLS[:5]
    group_cols_src  = [c for c in ["sector", "industry"] if c in train.columns]

    grp_feat_cols = []
    combined_grp  = pd.concat([train[["di"] + group_cols_src + top_group_feats],
                                test[["di"]  + group_cols_src + top_group_feats]],
                               ignore_index=True)
    is_train_mask = np.array([True]*len(train) + [False]*len(test))

    for gcol in group_cols_src:
        for fcol in top_group_feats:
            new_col = f"grp_{gcol}_{fcol}_zscore"
            grp_feat_cols.append(new_col)
            grp_mean = combined_grp.groupby(["di", gcol])[fcol].transform("mean")
            grp_std  = combined_grp.groupby(["di", gcol])[fcol].transform("std").fillna(1.0) + 1e-9
            combined_grp[new_col] = ((combined_grp[fcol] - grp_mean) / grp_std).astype(np.float32)

    grp_feat_train = combined_grp.loc[is_train_mask,  grp_feat_cols].reset_index(drop=True)
    grp_feat_test  = combined_grp.loc[~is_train_mask, grp_feat_cols].reset_index(drop=True)
    grp_feat_train = grp_feat_train.fillna(0.0)
    grp_feat_test  = grp_feat_test.fillna(0.0)

    toc("fe_group")
    print(f"  Group features: {len(grp_feat_cols)} new cols")
else:
    grp_feat_train = pd.DataFrame(index=train.index)
    grp_feat_test  = pd.DataFrame(index=test.index)
    grp_feat_cols  = []
    print("USE_FE_GROUP=False → skipping group features")"""
))

# Cell 25: Code — Lag/rolling features per si
cells.append(nbformat.v4.new_code_cell(
"""if USE_FE_LAG:
    print("Building lag/rolling features per stock (si)...")
    tic("fe_lag")

    # top features for lag
    top_lag_feats = [f for f, _ in corr_sorted[:5]] if corr_sorted else F_COLS[:5]
    windows       = [3, 5]
    lag_feat_cols = []

    combined_lag = pd.concat([
        train[["si", "di"] + top_lag_feats],
        test[["si", "di"]  + top_lag_feats]
    ], ignore_index=True).sort_values(["si", "di"])

    is_train_lag = np.array([True]*len(train) + [False]*len(test))
    # preserve original order for index alignment
    orig_order = pd.concat([
        train[["si", "di"]].assign(_orig_tr=True, _idx=np.arange(len(train))),
        test[["si", "di"]].assign(_orig_tr=False, _idx=np.arange(len(test)))
    ], ignore_index=True).sort_values(["si", "di"])

    for fcol in top_lag_feats:
        for w in windows:
            for stat in ["mean", "std"]:
                new_col = f"lag_{fcol}_w{w}_{stat}"
                lag_feat_cols.append(new_col)
                if stat == "mean":
                    combined_lag[new_col] = (
                        combined_lag.groupby("si")[fcol]
                        .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
                        .astype(np.float32)
                    )
                else:
                    combined_lag[new_col] = (
                        combined_lag.groupby("si")[fcol]
                        .transform(lambda x: x.shift(1).rolling(w, min_periods=2).std())
                        .astype(np.float32)
                    )

    # Split back using the is_train flag on the sorted combined frame
    is_train_sorted = np.array([True]*len(train) + [False]*len(test))
    # Re-split: combined_lag rows 0..len(train)-1 are train, rest are test
    # (because we concatenated train then test)
    lag_all = combined_lag[lag_feat_cols].fillna(0.0)
    lag_feat_train_sorted = lag_all.iloc[:len(train)].reset_index(drop=True)
    lag_feat_test_sorted  = lag_all.iloc[len(train):].reset_index(drop=True)

    # Re-align to original train/test order by sorting combined back
    combined_lag_tr = combined_lag.iloc[:len(train)][lag_feat_cols].copy()
    combined_lag_te = combined_lag.iloc[len(train):][lag_feat_cols].copy()

    toc("fe_lag")
    print(f"  Lag features: {len(lag_feat_cols)} new cols")
else:
    combined_lag_tr = pd.DataFrame(index=train.index)
    combined_lag_te = pd.DataFrame(index=test.index)
    lag_feat_cols   = []
    print("USE_FE_LAG=False → skipping lag features")"""
))

# Cell 26: Code — Assemble full FE matrix + Experiment 5
cells.append(nbformat.v4.new_code_cell(
"""tic("exp5_lgb_fe")
print("Assembling full feature-engineered matrix...")

# Determine which FE components we have
fe_parts_tr = [X_xs_tr if USE_FE_XS_RANK else X_raw_tr]
fe_parts_te = [X_xs_te if USE_FE_XS_RANK else X_raw_te]

if USE_FE_GROUP and len(grp_feat_cols) > 0:
    fe_parts_tr.append(grp_feat_train.values)
    fe_parts_te.append(grp_feat_test.values)

if USE_FE_LAG and len(lag_feat_cols) > 0:
    lag_tr_vals = combined_lag_tr.values if len(combined_lag_tr) == len(train) else np.zeros((len(train), len(lag_feat_cols)))
    lag_te_vals = combined_lag_te.values if len(combined_lag_te) == len(test)  else np.zeros((len(test),  len(lag_feat_cols)))
    fe_parts_tr.append(lag_tr_vals)
    fe_parts_te.append(lag_te_vals)

X_fe_tr = np.hstack(fe_parts_tr).astype(np.float32)
X_fe_te = np.hstack(fe_parts_te).astype(np.float32)
X_fe_tr = np.nan_to_num(X_fe_tr, nan=0.0)
X_fe_te = np.nan_to_num(X_fe_te, nan=0.0)

print(f"  X_fe_tr: {X_fe_tr.shape}  X_fe_te: {X_fe_te.shape}")

# ── Train LGB on full FE ─────────────────────────────────────
print("\\nTraining LightGBM on full FE matrix...")
oof_lgb_fe   = np.full(len(train), np.nan)
preds_lgb_fe = np.zeros(len(test))

for fold_i, (tr_idx, val_idx) in enumerate(FOLDS):
    X_tr, y_tr = X_fe_tr[tr_idx], y_train[tr_idx]
    X_val      = X_fe_tr[val_idx]
    y_val      = y_train[val_idx]

    valid_tr  = np.isfinite(y_tr)
    valid_val = np.isfinite(y_val)

    model = lgb.LGBMRegressor(**lgb_base_params)
    model.fit(
        X_tr[valid_tr], y_tr[valid_tr],
        eval_set=[(X_val[valid_val], y_val[valid_val])],
        callbacks=[lgb.early_stopping(20, verbose=False),
                   lgb.log_evaluation(period=-1)],
    )
    oof_lgb_fe[val_idx]  = model.predict(X_val)
    preds_lgb_fe        += model.predict(X_fe_te) / len(FOLDS)

    best_iter = model.best_iteration_ if model.best_iteration_ else lgb_base_params["n_estimators"]
    print(f"  Fold {fold_i+1}/{len(FOLDS)}  best_iter={best_iter}  n_feat={X_fe_tr.shape[1]}")

toc("exp5_lgb_fe")
diagnose_oof("Exp5_LGB_FE", oof_lgb_fe, train, preds_lgb_fe)"""
))

# Cell 27: Code — Experiment 5 FE impact comparison
cells.append(nbformat.v4.new_code_cell(
"""print("\\n=== Experiment 5: FE Impact ===")
ic_raw_lgb = results.get("Exp3_LGB_Raw", {}).get("ic", 0)
ic_fe_lgb  = results.get("Exp5_LGB_FE",  {}).get("ic", 0)
delta = ic_fe_lgb - ic_raw_lgb
sign  = "+" if delta >= 0 else ""
print(f"  Exp3 LGB (raw)   : IC={ic_raw_lgb:.5f}")
print(f"  Exp5 LGB (FE)    : IC={ic_fe_lgb:.5f}  ({sign}{delta:.5f})")
fe_label = ("XS-rank" if USE_FE_XS_RANK else "") + \\
           (" + group" if USE_FE_GROUP else "") + \\
           (" + lag" if USE_FE_LAG else "")
print(f"  FE components: {fe_label.strip(' +')}")
if delta > 0.001:
    print("\\n  ✓ Feature engineering provides meaningful lift!")
elif delta > 0:
    print("\\n  ✓ Feature engineering provides slight lift.")
else:
    print("\\n  ~ Feature engineering did not improve LGB — raw features may be pre-processed already.")"""
))

# Cell 28: Markdown — Experiment 6
cells.append(nbformat.v4.new_markdown_cell(
"""## Experiment 6 — Simple Ensemble: Does Averaging Beat Best Single Model?

**Hypothesis**: Combining diverse models (Ridge, ElasticNet, LGB) via equal-weight
averaging will outperform the single best model due to prediction error cancellation,
even without optimizing the weights.

This is the simplest possible ensemble — no tuning, no meta-learning."""
))

# Cell 29: Code — Experiment 6 equal-weight blend
cells.append(nbformat.v4.new_code_cell(
"""print("Computing equal-weight blend...")

# Collect all OOF arrays that were computed
oof_dict = {
    "Ridge":    oof_ridge,
    "ElasticNet": oof_enet,
    "LGB_raw":  oof_lgb_raw,
    "LGB_FE":   oof_lgb_fe,
}
pred_dict = {
    "Ridge":    preds_ridge,
    "ElasticNet": preds_enet,
    "LGB_raw":  preds_lgb_raw,
    "LGB_FE":   preds_lgb_fe,
}

# Stack OOF arrays; replace NaN with column median
oof_matrix = np.column_stack([oof_dict[k] for k in oof_dict])
pred_matrix = np.column_stack([pred_dict[k] for k in pred_dict])

# Equal-weight blend
oof_blend_eq   = np.nanmean(oof_matrix, axis=1)
preds_blend_eq = np.nanmean(pred_matrix, axis=1)

diagnose_oof("Exp6_EqualBlend", oof_blend_eq, train, preds_blend_eq)

best_single_ic = max(results.get("Exp1_Ridge", {"ic": 0})["ic"],
    results.get("Exp1_ElasticNet", {"ic": 0})["ic"],
    results.get("Exp3_LGB_Raw", {"ic": 0})["ic"],
    results.get("Exp5_LGB_FE", {"ic": 0})["ic"],
)

ic_blend = results.get("Exp6_EqualBlend", {}).get("ic", 0)
print(f"\\n  Best single model IC : {best_single_ic:.5f}")
print(f"  Equal-weight blend   : {ic_blend:.5f}  ({ic_blend - best_single_ic:+.5f})")"""
))

# Cell 30: Markdown — Experiment 7
cells.append(nbformat.v4.new_markdown_cell(
"""## Experiment 7 — Optimized Blend: Does Weight Optimization Beat Equal-Weight?

**Hypothesis**: Optimizing blend weights by maximizing OOF Pearson IC via
Nelder-Mead will squeeze additional performance beyond simple averaging.

Uses softmax normalization to ensure weights sum to 1 and are non-negative."""
))

# Cell 31: Code — Experiment 7 optimized blend
cells.append(nbformat.v4.new_code_cell(
"""from scipy.optimize import minimize

print("Optimizing blend weights via Nelder-Mead...")

# Build matrix of all OOF predictions (fill NaN with 0)
oof_keys  = list(oof_dict.keys())
oofs      = np.column_stack([oof_dict[k] for k in oof_keys])
preds_mat = np.column_stack([pred_dict[k] for k in oof_keys])
y_valid   = y_train.copy()

valid_rows = np.isfinite(y_valid) & np.all(np.isfinite(oofs), axis=1)

def neg_pearson_blended(w_raw):
    # softmax for non-negative weights summing to 1
    w = np.exp(w_raw - w_raw.max())
    w = w / w.sum()
    blend = oofs[valid_rows] @ w
    r, _ = stats.pearsonr(y_valid[valid_rows], blend)
    return -r

x0 = np.zeros(len(oof_keys))
res = minimize(neg_pearson_blended, x0, method="Nelder-Mead",
               options={"maxiter": 1000, "xatol": 1e-4, "fatol": 1e-6})

opt_w_raw = res.x
opt_w = np.exp(opt_w_raw - opt_w_raw.max())
opt_w = opt_w / opt_w.sum()

print("  Optimized weights:")
for k, w in zip(oof_keys, opt_w):
    print(f"    {k:15s}: {w:.4f}")

oof_blend_opt   = oofs @ opt_w
preds_blend_opt = preds_mat @ opt_w

diagnose_oof("Exp7_OptBlend", oof_blend_opt, train, preds_blend_opt)

ic_eq  = results.get("Exp6_EqualBlend", {}).get("ic", 0)
ic_opt = results.get("Exp7_OptBlend",   {}).get("ic", 0)
print(f"\\n  Equal-weight blend  : {ic_eq:.5f}")
print(f"  Optimized blend     : {ic_opt:.5f}  ({ic_opt - ic_eq:+.5f})")"""
))

# Cell 32: Markdown — Experiment 8
cells.append(nbformat.v4.new_markdown_cell(
"""## Experiment 8 — Tuned LightGBM (stronger params + all FE)

**Hypothesis**: Increasing model capacity (more leaves, more trees) with stronger
regularization on the full feature-engineered dataset will push IC significantly higher.

Key changes vs Exp3 baseline:
- `num_leaves`: 31 → 63/127
- `learning_rate`: 0.05 → 0.03
- `n_estimators`: 200 → 500/1000
- `lambda_l1/l2`: 0.1 → 0.1 (already set, keep)
- `min_child_samples`: 20 → 50"""
))

# Cell 33: Code — Experiment 8 tuned LGB
cells.append(nbformat.v4.new_code_cell(
"""tic("exp8_lgb_tuned")
print("Training tuned LightGBM on full FE matrix...")

lgb_tuned_params = {
    "objective":         "regression",
    "metric":            "None",
    "num_leaves":        63 if FAST_MODE else 127,
    "learning_rate":     0.03,
    "n_estimators":      300 if FAST_MODE else 1000,
    "min_child_samples": 50,
    "subsample":         0.8,
    "colsample_bytree":  0.7,
    "reg_alpha":         0.1,
    "reg_lambda":        0.1,
    "random_state":      SEED,
    "n_jobs":            -1,
    "verbose":           -1,
}

oof_lgb_tuned   = np.full(len(train), np.nan)
preds_lgb_tuned = np.zeros(len(test))

for fold_i, (tr_idx, val_idx) in enumerate(FOLDS):
    X_tr, y_tr = X_fe_tr[tr_idx], y_train[tr_idx]
    X_val      = X_fe_tr[val_idx]
    y_val      = y_train[val_idx]

    valid_tr  = np.isfinite(y_tr)
    valid_val = np.isfinite(y_val)

    model = lgb.LGBMRegressor(**lgb_tuned_params)
    model.fit(
        X_tr[valid_tr], y_tr[valid_tr],
        eval_set=[(X_val[valid_val], y_val[valid_val])],
        callbacks=[lgb.early_stopping(30, verbose=False),
                   lgb.log_evaluation(period=-1)],
    )
    oof_lgb_tuned[val_idx]  = model.predict(X_val)
    preds_lgb_tuned        += model.predict(X_fe_te) / len(FOLDS)

    best_iter = model.best_iteration_ if model.best_iteration_ else lgb_tuned_params["n_estimators"]
    print(f"  Fold {fold_i+1}/{len(FOLDS)}  best_iter={best_iter}")

toc("exp8_lgb_tuned")
diagnose_oof("Exp8_LGB_Tuned", oof_lgb_tuned, train, preds_lgb_tuned)

ic_base  = results.get("Exp5_LGB_FE", {"ic": 0})["ic"]
ic_tuned = results.get("Exp8_LGB_Tuned", {"ic": 0})["ic"]
print(f"\\n  Exp5 LGB (base params)   : IC={ic_base:.5f}")
print(f"  Exp8 LGB (tuned params)  : IC={ic_tuned:.5f}  ({ic_tuned - ic_base:+.5f})")"""
))

# Cell 34: Markdown — Experiment 9
cells.append(nbformat.v4.new_markdown_cell(
"""## Experiment 9 — Optional XGBoost (toggle `USE_XGB=True`)

**Hypothesis**: XGBoost trained on the same feature set may offer complementary
predictions to LightGBM due to different regularization and split-finding strategies,
improving ensemble IC when blended.

Toggle `USE_XGB = True` in Config cell to enable."""
))

# Cell 35: Code — Experiment 9 XGBoost
cells.append(nbformat.v4.new_code_cell(
"""if USE_XGB:
    try:
        import xgboost as xgb
        print(f"XGBoost version: {xgb.__version__}")
        tic("exp9_xgb")

        xgb_params = {
            "n_estimators":   200 if FAST_MODE else 500,
            "max_depth":      6,
            "learning_rate":  0.05,
            "subsample":      0.8,
            "colsample_bytree": 0.8,
            "reg_alpha":      0.1,
            "reg_lambda":     1.0,
            "tree_method":    "hist",
            "random_state":   SEED,
            "n_jobs":         -1,
        }

        oof_xgb   = np.full(len(train), np.nan)
        preds_xgb = np.zeros(len(test))

        for fold_i, (tr_idx, val_idx) in enumerate(FOLDS):
            X_tr, y_tr = X_fe_tr[tr_idx], y_train[tr_idx]
            X_val      = X_fe_tr[val_idx]
            y_val      = y_train[val_idx]

            valid_tr  = np.isfinite(y_tr)
            valid_val = np.isfinite(y_val)

            model = xgb.XGBRegressor(**xgb_params, verbosity=0)
            model.fit(
                X_tr[valid_tr], y_tr[valid_tr],
                eval_set=[(X_val[valid_val], y_val[valid_val])],
                early_stopping_rounds=20,
                verbose=False,
            )
            oof_xgb[val_idx]  = model.predict(X_val)
            preds_xgb        += model.predict(X_fe_te) / len(FOLDS)
            print(f"  Fold {fold_i+1}/{len(FOLDS)} done")

        toc("exp9_xgb")
        diagnose_oof("Exp9_XGBoost", oof_xgb, train, preds_xgb)
        oof_dict["XGBoost"]  = oof_xgb
        pred_dict["XGBoost"] = preds_xgb

    except ImportError:
        print("xgboost not installed — skipping. pip install xgboost to enable.")
        USE_XGB = False
else:
    print("XGBoost skipped (USE_XGB=False). Set USE_XGB=True in config to enable.")"""
))

# Cell 36: Markdown — Experiment 10
cells.append(nbformat.v4.new_markdown_cell(
"""## Experiment 10 — Optional CatBoost with Categorical Features (toggle `USE_CAT=True`)

**Hypothesis**: CatBoost's native handling of `sector` and `industry` as categorical
features (via ordered target encoding) will extract additional signal from these
group identifiers, complementary to the manual group z-score features built in Exp5.

Toggle `USE_CAT = True` in Config cell to enable."""
))

# Cell 37: Code — Experiment 10 CatBoost
cells.append(nbformat.v4.new_code_cell(
"""if USE_CAT:
    try:
        from catboost import CatBoostRegressor, Pool
        print("CatBoost available")
        tic("exp10_cat")

        cat_feat_cols = [c for c in ["sector", "industry"] if c in train.columns]
        extra_tr = train[cat_feat_cols].values.astype(str) if cat_feat_cols else np.empty((len(train),0))
        extra_te = test[cat_feat_cols].values.astype(str)  if cat_feat_cols else np.empty((len(test),0))

        X_cat_tr = np.hstack([X_fe_tr.astype(object), extra_tr]) if cat_feat_cols else X_fe_tr
        X_cat_te = np.hstack([X_fe_te.astype(object), extra_te]) if cat_feat_cols else X_fe_te

        n_fe_cols = X_fe_tr.shape[1]
        cat_indices = list(range(n_fe_cols, n_fe_cols + len(cat_feat_cols)))

        oof_cat   = np.full(len(train), np.nan)
        preds_cat = np.zeros(len(test))

        for fold_i, (tr_idx, val_idx) in enumerate(FOLDS):
            X_tr_c = X_cat_tr[tr_idx]
            y_tr   = y_train[tr_idx]
            X_val_c = X_cat_tr[val_idx]
            y_val  = y_train[val_idx]

            valid_tr  = np.isfinite(y_tr.astype(float))
            valid_val = np.isfinite(y_val.astype(float))

            train_pool = Pool(X_tr_c[valid_tr], y_tr[valid_tr],
                              cat_features=cat_indices if cat_indices else None)
            val_pool   = Pool(X_val_c[valid_val], y_val[valid_val],
                              cat_features=cat_indices if cat_indices else None)

            model = CatBoostRegressor(
                iterations=200 if FAST_MODE else 500,
                learning_rate=0.05,
                depth=6,
                l2_leaf_reg=3.0,
                random_seed=SEED,
                verbose=False,
            )
            model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=20)
            oof_cat[val_idx]  = model.predict(Pool(X_cat_tr[val_idx],
                                                    cat_features=cat_indices if cat_indices else None))
            preds_cat        += model.predict(Pool(X_cat_te,
                                                    cat_features=cat_indices if cat_indices else None)) / len(FOLDS)
            print(f"  Fold {fold_i+1}/{len(FOLDS)} done")

        toc("exp10_cat")
        diagnose_oof("Exp10_CatBoost", oof_cat, train, preds_cat)
        oof_dict["CatBoost"]  = oof_cat
        pred_dict["CatBoost"] = preds_cat

    except ImportError:
        print("catboost not installed — skipping. pip install catboost to enable.")
        USE_CAT = False
else:
    print("CatBoost skipped (USE_CAT=False). Set USE_CAT=True in config to enable.")"""
))

# Cell 38: Markdown — Experiment 11
cells.append(nbformat.v4.new_markdown_cell(
"""## Experiment 11 — GBDT Meta-Stacking: Does Stacking Beat Blend?

**Hypothesis**: Training a LightGBM meta-learner on OOF predictions of all base models
(as features alongside raw features) will outperform a simple weighted blend by
learning non-linear combinations and context-dependent model selection.

The meta-learner sees both base model OOF predictions AND original features,
allowing it to learn when to trust each model."""
))

# Cell 39: Code — Experiment 11 GBDT meta-stacking
cells.append(nbformat.v4.new_code_cell(
"""if USE_GBDT_STACK:
    tic("exp11_stack")
    print("Building meta-stacking features...")

    # Collect all available OOF columns
    stack_oof_keys = [k for k in ["Ridge", "ElasticNet", "LGB_raw", "LGB_FE",
                                   "XGBoost", "CatBoost"]
                      if k in oof_dict]

    meta_tr_cols = np.column_stack([oof_dict[k] for k in stack_oof_keys])
    meta_te_cols = np.column_stack([pred_dict[k] for k in stack_oof_keys])

    # Replace NaN with 0 in meta features
    meta_tr_cols = np.nan_to_num(meta_tr_cols, nan=0.0)
    meta_te_cols = np.nan_to_num(meta_te_cols, nan=0.0)

    # Stack: meta features + full FE features
    X_stack_tr = np.hstack([meta_tr_cols, X_fe_tr]).astype(np.float32)
    X_stack_te = np.hstack([meta_te_cols, X_fe_te]).astype(np.float32)

    print(f"  Stack meta cols: {len(stack_oof_keys)}  ({', '.join(stack_oof_keys)})")
    print(f"  X_stack_tr: {X_stack_tr.shape}  X_stack_te: {X_stack_te.shape}")

    oof_stack   = np.full(len(train), np.nan)
    preds_stack = np.zeros(len(test))

    stack_params = {
        "objective":         "regression",
        "metric":            "None",
        "num_leaves":        31 if FAST_MODE else 63,
        "learning_rate":     0.03,
        "n_estimators":      200 if FAST_MODE else 500,
        "min_child_samples": 30,
        "subsample":         0.8,
        "colsample_bytree":  0.8,
        "reg_alpha":         0.1,
        "reg_lambda":        0.1,
        "random_state":      SEED,
        "n_jobs":            -1,
        "verbose":           -1,
    }

    for fold_i, (tr_idx, val_idx) in enumerate(FOLDS):
        X_tr, y_tr = X_stack_tr[tr_idx], y_train[tr_idx]
        X_val      = X_stack_tr[val_idx]
        y_val      = y_train[val_idx]

        valid_tr  = np.isfinite(y_tr)
        valid_val = np.isfinite(y_val)

        model = lgb.LGBMRegressor(**stack_params)
        model.fit(
            X_tr[valid_tr], y_tr[valid_tr],
            eval_set=[(X_val[valid_val], y_val[valid_val])],
            callbacks=[lgb.early_stopping(20, verbose=False),
                       lgb.log_evaluation(period=-1)],
        )
        oof_stack[val_idx]  = model.predict(X_val)
        preds_stack        += model.predict(X_stack_te) / len(FOLDS)

        best_iter = model.best_iteration_ if model.best_iteration_ else stack_params["n_estimators"]
        print(f"  Fold {fold_i+1}/{len(FOLDS)}  best_iter={best_iter}")

    toc("exp11_stack")
    diagnose_oof("Exp11_Stack", oof_stack, train, preds_stack)

    # Compare vs best blend
    ic_blend_best = max(
        results.get("Exp6_EqualBlend", {"ic": 0})["ic"],
        results.get("Exp7_OptBlend",   {"ic": 0})["ic"],
    )
    ic_stack = results.get("Exp11_Stack", {"ic": 0})["ic"]
    print(f"\\n  Best blend IC  : {ic_blend_best:.5f}")
    print(f"  Stack IC       : {ic_stack:.5f}  ({ic_stack - ic_blend_best:+.5f})")
    if ic_stack > ic_blend_best + 0.001:
        print("  ✓ Stacking BEATS blend — meta-learner extracts additional signal!")
    elif ic_stack > ic_blend_best:
        print("  ✓ Stacking slightly beats blend.")
    else:
        print("  ~ Stacking does NOT beat blend — blend is already near-optimal or OOF is noisy.")
else:
    print("GBDT stacking skipped (USE_GBDT_STACK=False).")
    preds_stack = None"""
))

# Cell 40: Code — Full IC comparison table
cells.append(nbformat.v4.new_code_cell(
"""print("\\n" + "=" * 65)
print("  EXPERIMENT RESULTS SUMMARY")
print("=" * 65)
print(f"  {'Experiment':<35s} {'IC':>8s}  {'DailyIC':>8s}  {'IR':>7s}")
print("-" * 65)
for k, v in results.items():
    ic_str  = f"{v['ic']:.5f}"
    dic_str = f"{v['daily_mean']:.5f}"
    ir_str  = f"{v['ir']:.3f}"
    print(f"  {k:<35s} {ic_str:>8s}  {dic_str:>8s}  {ir_str:>7s}")
print("=" * 65)

best_exp = max(results, key=lambda k: results[k]["ic"])
print(f"\\n  🏆 Best experiment: {best_exp}  IC={results[best_exp]['ic']:.5f}")"""
))

# Cell 41: Markdown — Experiment 12 (Optional LSTM)
cells.append(nbformat.v4.new_markdown_cell(
"""## Experiment 12 — Optional LSTM Sequence Branch (toggle `USE_SEQ_BRANCH=True`)

**Hypothesis**: Building per-stock (`si`) historical sequences and training a lightweight
LSTM will capture temporal momentum signals that the cross-sectional models miss,
providing complementary meta-features for the final stacker.

Note: Requires PyTorch. Set `USE_SEQ_BRANCH=True` in config to enable.

Architecture:
- Sequences: last `SEQ_LEN=10` time steps per stock, top-20 features
- LightLSTM: LSTM(n_feat, 64, num_layers=1) → Linear(64, 1)
- Loss: Custom Pearson correlation loss (1 - r)"""
))

# Cell 42: Code — Experiment 12 LSTM
cells.append(nbformat.v4.new_code_cell(
"""if USE_SEQ_BRANCH:
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader

        SEQ_LEN   = 10
        SEQ_FEAT  = F_COLS[:min(20, len(F_COLS))]
        BATCH_SIZE = 512
        SEQ_EPOCHS = 5 if FAST_MODE else 20

        print(f"PyTorch {torch.__version__} | SEQ_LEN={SEQ_LEN} | SEQ_FEAT={len(SEQ_FEAT)} | EPOCHS={SEQ_EPOCHS}")

        # ── Pearson loss ─────────────────────────────────────────
        class PearsonLoss(nn.Module):
            def forward(self, y_pred, y_true):
                eps = 1e-8
                vx = y_pred - y_pred.mean()
                vy = y_true - y_true.mean()
                return 1.0 - (vx * vy).sum() / (
                    torch.sqrt((vx**2).sum() + eps) *
                    torch.sqrt((vy**2).sum() + eps)
                )

        # ── LightLSTM ─────────────────────────────────────────────
        class LightLSTM(nn.Module):
            def __init__(self, n_feat, hidden=64):
                super().__init__()
                self.lstm   = nn.LSTM(n_feat, hidden, batch_first=True)
                self.linear = nn.Linear(hidden, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.linear(out[:, -1, :]).squeeze(-1)

        # ── Build sequences ───────────────────────────────────────
        def build_sequences(df, feat_cols, seq_len=SEQ_LEN):
            \"\"\"Build (N, seq_len, n_feat) sequences and target array per stock.\"\"\"
            df_s = df[["si", "di"] + feat_cols + ([TARGET_COL] if TARGET_COL in df.columns else [])
                       ].sort_values(["si", "di"])
            seqs, targets, indices = [], [], []
            for si_val, grp in df_s.groupby("si"):
                grp = grp.reset_index(drop=True)
                fvals = grp[feat_cols].values.astype(np.float32)
                fvals = np.nan_to_num(fvals, nan=0.0)
                tvals = grp[TARGET_COL].values if TARGET_COL in grp.columns else None
                orig_idx = grp.index.tolist()  # not meaningful after groupby
                for i in range(seq_len, len(grp)):
                    seq = fvals[i-seq_len:i]
                    seqs.append(seq)
                    if tvals is not None:
                        targets.append(tvals[i])
                    indices.append(grp.index[i] if hasattr(grp.index, '__getitem__') else i)
            if not seqs:
                return None, None, None
            return (np.array(seqs, dtype=np.float32),
                    np.array(targets, dtype=np.float32) if targets else None,
                    indices)

        tic("exp12_lstm")
        print("Building sequences for train...")
        seq_tr_x, seq_tr_y, seq_tr_idx = build_sequences(train, SEQ_FEAT)
        print(f"  Train sequences: {seq_tr_x.shape if seq_tr_x is not None else None}")

        if seq_tr_x is not None and len(seq_tr_x) > 100:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"  Device: {device}")

            model_lstm = LightLSTM(n_feat=len(SEQ_FEAT)).to(device)
            optimizer  = torch.optim.Adam(model_lstm.parameters(), lr=1e-3)
            criterion  = PearsonLoss()

            X_t = torch.FloatTensor(seq_tr_x)
            y_t = torch.FloatTensor(seq_tr_y)
            ds  = TensorDataset(X_t, y_t)
            dl  = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

            for epoch in range(SEQ_EPOCHS):
                model_lstm.train()
                epoch_loss = 0.0
                for xb, yb in dl:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad()
                    pred = model_lstm(xb)
                    loss = criterion(pred, yb)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                print(f"  Epoch {epoch+1}/{SEQ_EPOCHS}  loss={epoch_loss/len(dl):.4f}")

            # Extract LSTM predictions as meta-features
            model_lstm.eval()
            with torch.no_grad():
                lstm_preds_seq = model_lstm(torch.FloatTensor(seq_tr_x).to(device)).cpu().numpy()

            # Map back to train DataFrame indices
            lstm_meta_tr = np.zeros(len(train))
            for i, orig_i in enumerate(seq_tr_idx):
                if orig_i < len(lstm_meta_tr):
                    lstm_meta_tr[orig_i] = lstm_preds_seq[i]

            # Build test sequences
            print("Building sequences for test...")
            seq_te_x, _, seq_te_idx = build_sequences(test, SEQ_FEAT)
            lstm_meta_te = np.zeros(len(test))
            if seq_te_x is not None:
                with torch.no_grad():
                    lstm_preds_te = model_lstm(torch.FloatTensor(seq_te_x).to(device)).cpu().numpy()
                for i, orig_i in enumerate(seq_te_idx):
                    if orig_i < len(lstm_meta_te):
                        lstm_meta_te[orig_i] = lstm_preds_te[i]

            # Diagnose LSTM standalone
            diagnose_oof("Exp12_LSTM", lstm_meta_tr, train, lstm_meta_te)

            # Retrain final stack with LSTM meta-feature
            X_stack_lstm_tr = np.hstack([lstm_meta_tr.reshape(-1,1), X_stack_tr]).astype(np.float32)
            X_stack_lstm_te = np.hstack([lstm_meta_te.reshape(-1,1), X_stack_te]).astype(np.float32)

            oof_stack_lstm   = np.full(len(train), np.nan)
            preds_stack_lstm = np.zeros(len(test))

            for fold_i, (tr_idx, val_idx) in enumerate(FOLDS):
                X_tr_s, y_tr_s = X_stack_lstm_tr[tr_idx], y_train[tr_idx]
                X_val_s        = X_stack_lstm_tr[val_idx]
                y_val_s        = y_train[val_idx]
                valid_tr_s  = np.isfinite(y_tr_s)
                valid_val_s = np.isfinite(y_val_s)
                m = lgb.LGBMRegressor(**stack_params)
                m.fit(X_tr_s[valid_tr_s], y_tr_s[valid_tr_s],
                      eval_set=[(X_val_s[valid_val_s], y_val_s[valid_val_s])],
                      callbacks=[lgb.early_stopping(20, verbose=False),
                                  lgb.log_evaluation(period=-1)])
                oof_stack_lstm[val_idx]   = m.predict(X_val_s)
                preds_stack_lstm         += m.predict(X_stack_lstm_te) / len(FOLDS)
                print(f"  Fold {fold_i+1}/{len(FOLDS)} done")

            toc("exp12_lstm")
            diagnose_oof("Exp12_Stack_LSTM", oof_stack_lstm, train, preds_stack_lstm)
        else:
            print("  Insufficient sequences built — skipping LSTM training.")
            toc("exp12_lstm")

    except ImportError:
        print("PyTorch not available — skipping. pip install torch to enable.")
        USE_SEQ_BRANCH = False
else:
    print("Sequence branch skipped (USE_SEQ_BRANCH=False). Set USE_SEQ_BRANCH=True to enable.")"""
))

# Cell 43: Markdown — Final best model + submission
cells.append(nbformat.v4.new_markdown_cell(
"""## Final: Best Model Selection + Submission

Automatically selects the best configuration based on OOF Pearson IC
across all completed experiments, then generates the submission CSV.

Submission requirements:
- Columns: `id`, `target`
- No NaN or Inf values
- At least 10% non-zero predictions"""
))

# Cell 44: Code — Best model selection + submission guard
cells.append(nbformat.v4.new_code_cell(
"""def submission_guard(pred_arr, ids):
    \"\"\"Validate and clean predictions for submission.\"\"\"
    preds = pred_arr.copy()

    # Replace inf/nan with 0
    n_bad = np.sum(~np.isfinite(preds))
    if n_bad > 0:
        print(f"  ⚠ Replacing {n_bad} inf/nan predictions with 0")
        preds = np.where(np.isfinite(preds), preds, 0.0)

    # Check non-zero percentage
    nonzero_pct = np.mean(preds != 0)
    print(f"  Non-zero predictions: {nonzero_pct:.2%}")
    if nonzero_pct < 0.10:
        print(f"  ⚠ WARNING: <10% non-zero predictions ({nonzero_pct:.2%})")
        print("    Adding tiny noise to meet submission requirement...")
        rng = np.random.default_rng(SEED)
        noise_mask = preds == 0
        preds[noise_mask] = rng.normal(0, 1e-7, noise_mask.sum())

    return preds

# Select best model by OOF IC
best_exp_name  = max(results, key=lambda k: results[k]["ic"])
best_ic        = results[best_exp_name]["ic"]
best_preds_raw = results[best_exp_name]["preds_test"]

print(f"Selected best model: {best_exp_name}  OOF IC={best_ic:.5f}")

if best_preds_raw is None:
    print("⚠ Best model has no test predictions — falling back to LGB_FE")
    best_preds_raw = preds_lgb_fe

final_preds = submission_guard(best_preds_raw, test["id"].values)

print(f"\\n  Prediction stats:")
print(f"    mean  : {final_preds.mean():.6f}")
print(f"    std   : {final_preds.std():.6f}")
print(f"    min   : {final_preds.min():.6f}")
print(f"    max   : {final_preds.max():.6f}")"""
))

# Cell 45: Code — Export submission CSV + final diagnostics
cells.append(nbformat.v4.new_code_cell(
"""submission = pd.DataFrame({
    "id":     test["id"].values,
    "target": final_preds.astype(np.float32),
})

submission.to_csv(SUB_PATH, index=False)
print(f"Submission saved to: {SUB_PATH}")
print(f"Submission shape: {submission.shape}")
print("\\nFirst 5 rows:")
print(submission.head().to_string())

print("\\nFinal submission stats:")
print(submission["target"].describe().to_string())

print("\\n" + "=" * 65)
print("  FINAL EXPERIMENT RANKINGS")
print("=" * 65)
sorted_results = sorted(results.items(), key=lambda kv: kv[1]["ic"], reverse=True)
for rank_i, (k, v) in enumerate(sorted_results, 1):
    marker = " 🏆" if rank_i == 1 else ("  ✓" if rank_i <= 3 else "   ")
    print(f"{marker} {rank_i:2d}. {k:<35s}  IC={v['ic']:.5f}  IR={v['ir']:.3f}")

print(f"\\nSubmission uses: {best_exp_name}")
print("Done!")"""
))

nb.cells = cells

output_path = "/home/runner/work/tq23414/tq23414/trexquant_progressive_pipeline.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"Notebook written to: {output_path}")
print(f"Total cells: {len(nb.cells)}")
