"""
Flipkart Gridlock Hackathon 2.0 — Traffic Demand Prediction
============================================================
Author      : Shivangi Beniwal (shivangibeniwal338)
Best Score  : 90.73638 (Public Leaderboard)
Submission  : NEW_60best_40wtd.csv
Pipeline    : Two-model ensemble (LightGBM + CatBoost) with rich
              Day-48 geohash-lag features, blended across two
              hyperparameter configurations.

Usage
-----
    python solution_final.py          # writes submission.csv

Requirements
------------
    pandas, numpy, lightgbm, catboost, scipy
    train.csv and test.csv in the working directory
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import lightgbm as lgb
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# 1.  LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("Loading data...")
train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')

# ══════════════════════════════════════════════════════════════════════════════
# 2.  FEATURE HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def add_time_features(df):
    """Parse timestamp → hour / minute / slot (0-95) + cyclic encodings."""
    df = df.copy()
    df['hour']   = df.timestamp.apply(lambda x: int(x.split(':')[0]))
    df['minute'] = df.timestamp.apply(lambda x: int(x.split(':')[1]))
    df['slot']   = df['hour'] * 4 + df['minute'] // 15   # 96 slots / day

    # Cyclic time features
    df['sin_slot'] = np.sin(2 * np.pi * df['slot'] / 96)
    df['cos_slot'] = np.cos(2 * np.pi * df['slot'] / 96)

    # Binary flags from raw columns
    df['is_highway']    = (df['RoadType']      == 'Highway').astype(int)
    df['is_high_lanes'] = (df['NumberofLanes'] >= 4).astype(int)
    df['lv_enc']        = (df['LargeVehicles'] == 'Allowed').astype(int)
    df['lm_enc']        = (df['Landmarks']     == 'Yes').astype(int)

    # Ordinal encodings
    df['road_enc']    = df['RoadType'].fillna('MISSING').map(
                            {'Highway': 0, 'Street': 1,
                             'Residential': 2, 'MISSING': 3})
    df['weather_enc'] = df['Weather'].fillna('MISSING').map(
                            {'Sunny': 0, 'Rainy': 1,
                             'Foggy': 2, 'Snowy': 3, 'MISSING': 4})

    # Geohash spatial prefixes (neighbourhood aggregation)
    df['p4'] = df.geohash.str[:4]
    df['p5'] = df.geohash.str[:5]

    # Fill categorical nulls
    for col in ['RoadType', 'LargeVehicles', 'Landmarks', 'Weather']:
        df[col] = df[col].fillna('MISSING')

    return df


def smooth_target_encode(df, col, target, alpha=30):
    """
    Bayesian smoothed mean encoding.
    alpha controls shrinkage toward the global mean (higher → more shrinkage).
    """
    global_mean = df[target].mean()
    stats = df.groupby(col)[target].agg(['mean', 'count'])
    return (stats['mean'] * stats['count'] + global_mean * alpha) / \
           (stats['count'] + alpha)


def compute_stats(ref):
    """
    Build all aggregate lookup tables from a reference DataFrame.
    Called on Day 48 data (and on OOF halves for within-day48 training).

    The 'golden feature' is geo_slot_mean:
        For a 1-day reference, geo_slot_mean[(g, s)] == demand48[g, s] exactly,
        i.e., it is the direct Day-48 demand at the same geohash and time slot.
    """
    ref = ref.copy()
    geo_slot_enc = ref.geohash + '_' + ref.slot.astype(str)
    ref['_gs']   = geo_slot_enc

    ss = {
        # ── Geohash-level aggregates ─────────────────────────────────────────
        'geo_mean'        : ref.groupby('geohash').demand.mean(),
        'geo_std'         : ref.groupby('geohash').demand.std().fillna(0),
        'geo_median'      : ref.groupby('geohash').demand.median(),
        'geo_max'         : ref.groupby('geohash').demand.max(),
        'geo_q75'         : ref.groupby('geohash').demand.quantile(0.75),
        'geo_smooth'      : smooth_target_encode(ref, 'geohash', 'demand', alpha=30),

        # ── PRIMARY LAG: (geohash, slot) from Day 48 ─────────────────────────
        'geo_slot_mean'   : ref.groupby(['geohash', 'slot']).demand.mean(),
        'geo_slot_smooth' : smooth_target_encode(
                                ref.rename(columns={'_gs': '_key'}),
                                '_key', 'demand', alpha=15),
        'geo_hour_mean'   : ref.groupby(['geohash', 'hour']).demand.mean(),

        # ── Road-type aggregates ─────────────────────────────────────────────
        'road_mean'       : ref.groupby('RoadType').demand.mean(),
        'road_slot_mean'  : ref.groupby(['RoadType', 'slot']).demand.mean(),
        'roadlanes_mean'  : ref.groupby(['RoadType', 'NumberofLanes']).demand.mean(),
        'roadlanes_slot'  : ref.groupby(
                                ['RoadType', 'NumberofLanes', 'slot']).demand.mean(),

        # ── Spatial prefix aggregates (geohash neighbourhood) ────────────────
        'p4_mean'         : ref.groupby(ref.geohash.str[:4]).demand.mean(),
        'p4_slot'         : ref.groupby(
                                [ref.geohash.str[:4], 'slot']).demand.mean(),
        'p5_mean'         : ref.groupby(ref.geohash.str[:5]).demand.mean(),
        'p5_slot'         : ref.groupby(
                                [ref.geohash.str[:5], 'slot']).demand.mean(),

        # ── Global temporal means ────────────────────────────────────────────
        'slot_mean'       : ref.groupby('slot').demand.mean(),
        'hour_mean'       : ref.groupby('hour').demand.mean(),
        'temp_by_slot'    : ref.groupby('slot').Temperature.mean(),
    }
    return ss


def featurize(df, ss):
    """
    Apply all aggregate statistics to produce the 35-column feature matrix.
    Missing lookups fall back through the hierarchy:
        geo_slot → geo_hour → geo_mean → global slot mean
    """
    df  = df.copy()
    gsk = df.geohash + '_' + df.slot.astype(str)
    fb  = df.slot.map(ss['slot_mean'])   # global fallback

    # ── Geohash features ─────────────────────────────────────────────────────
    df['geo_mean']        = df.geohash.map(ss['geo_mean'])
    df['geo_std']         = df.geohash.map(ss['geo_std'])
    df['geo_median']      = df.geohash.map(ss['geo_median'])
    df['geo_max']         = df.geohash.map(ss['geo_max'])
    df['geo_q75']         = df.geohash.map(ss['geo_q75'])
    df['geo_smooth']      = df.geohash.map(ss['geo_smooth'])

    # ── Primary lag (Day-48 exact lookup) ────────────────────────────────────
    df['geo_slot_mean']   = [ss['geo_slot_mean'].get((g, s), np.nan)
                             for g, s in zip(df.geohash, df.slot)]
    df['geo_slot_smooth'] = gsk.map(ss['geo_slot_smooth'])
    df['geo_hour_mean']   = [ss['geo_hour_mean'].get((g, h), np.nan)
                             for g, h in zip(df.geohash, df.hour)]

    # Hierarchical fallback chain
    df['lag1']       = (df['geo_slot_mean']
                          .fillna(df['geo_hour_mean'])
                          .fillna(df['geo_mean'])
                          .fillna(fb))
    df['lag1_smooth']= df['geo_slot_smooth'].fillna(df['geo_mean']).fillna(fb)

    # ── Road-type features ────────────────────────────────────────────────────
    df['road_mean']       = df['RoadType'].map(ss['road_mean'])
    df['road_slot_mean']  = [ss['road_slot_mean'].get((r, s), np.nan)
                             for r, s in zip(df.RoadType, df.slot)]
    df['roadlanes_mean']  = [ss['roadlanes_mean'].get((r, l), np.nan)
                             for r, l in zip(df.RoadType, df.NumberofLanes)]
    df['roadlanes_slot']  = [ss['roadlanes_slot'].get((r, l, s), np.nan)
                             for r, l, s in zip(df.RoadType,
                                                df.NumberofLanes, df.slot)]

    # ── Spatial prefix features ───────────────────────────────────────────────
    df['p4_mean'] = df.p4.map(ss['p4_mean'])
    df['p4_slot'] = [ss['p4_slot'].get((p, s), np.nan)
                     for p, s in zip(df.p4, df.slot)]
    df['p5_mean'] = df.p5.map(ss['p5_mean'])
    df['p5_slot'] = [ss['p5_slot'].get((p, s), np.nan)
                     for p, s in zip(df.p5, df.slot)]

    # ── Global temporal ───────────────────────────────────────────────────────
    df['slot_mean']   = df.slot.map(ss['slot_mean'])
    df['hour_mean']   = df.hour.map(ss['hour_mean'])
    df['temp_filled'] = df['Temperature'].fillna(df.slot.map(ss['temp_by_slot']))

    # ── Interaction features ──────────────────────────────────────────────────
    df['geo_x_slot_ratio'] = df['geo_slot_mean'] / (df['geo_mean'] + 1e-9)
    df['lag1_x_road']      = df['lag1'] * df['road_mean'].fillna(df['lag1'])
    df['geo_x_lanes']      = df['geo_mean'] * df['NumberofLanes']
    df['smooth_x_road']    = df['geo_smooth'] * df['road_mean'].fillna(1)

    return df


FEAT_COLS = [
    'hour', 'slot', 'sin_slot', 'cos_slot',
    'road_enc', 'NumberofLanes', 'lv_enc', 'lm_enc', 'weather_enc',
    'temp_filled', 'is_highway', 'is_high_lanes',
    # geohash aggregates
    'geo_mean', 'geo_std', 'geo_median', 'geo_max', 'geo_q75', 'geo_smooth',
    # core lag features
    'geo_slot_mean', 'geo_slot_smooth', 'geo_hour_mean', 'lag1', 'lag1_smooth',
    # road aggregates
    'road_mean', 'road_slot_mean', 'roadlanes_mean', 'roadlanes_slot',
    # spatial prefix
    'p4_mean', 'p4_slot', 'p5_mean', 'p5_slot',
    # temporal global
    'slot_mean', 'hour_mean',
    # interactions
    'geo_x_slot_ratio', 'lag1_x_road', 'geo_x_lanes', 'smooth_x_road',
]

# ══════════════════════════════════════════════════════════════════════════════
# 3.  PREPARE FEATURE MATRICES
# ══════════════════════════════════════════════════════════════════════════════
print("Building features...")
train = add_time_features(train)
test  = add_time_features(test)

train48 = train[train.day == 48].copy()
train49 = train[train.day == 49].copy()

# Compute reference statistics from Day 48
s48 = compute_stats(train48)

# OOF split within Day 48 to avoid within-sample target leakage
# First-half stats → used to featurize second half, and vice versa
s_h1 = compute_stats(train48[train48.slot < 48])
s_h2 = compute_stats(train48[train48.slot >= 48])

h2_feat      = featurize(train48[train48.slot >= 48].copy(), s_h1)
h1_feat      = featurize(train48[train48.slot <  48].copy(), s_h2)
train48_feat = pd.concat([h1_feat, h2_feat], ignore_index=True)

val49_feat   = featurize(train49.copy(), s48)   # validation
test_feat    = featurize(test.copy(),    s48)   # test (no leakage: uses Day-48 stats)
train49_feat = featurize(train49.copy(), s48)   # for full-data retraining

X_tr   = train48_feat[FEAT_COLS].fillna(-999);  y_tr   = train48_feat['demand']
X_val  = val49_feat[FEAT_COLS].fillna(-999);    y_val  = train49['demand'].values
X_test = test_feat[FEAT_COLS].fillna(-999)

# Full training set: Day 48 + Day 49 combined
X_full = pd.concat([X_tr, train49_feat[FEAT_COLS].fillna(-999)], ignore_index=True)
y_full = pd.concat([y_tr, pd.Series(y_val)], ignore_index=True)

print(f"  OOF train: {X_tr.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
print(f"  Full train: {X_full.shape}")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  HELPER: TRAIN ONE LGBM + CATBOOST PAIR
# ══════════════════════════════════════════════════════════════════════════════
def train_ensemble(X_tr, y_tr, X_val, y_val, X_full, y_full, X_test,
                   lgb_lr=0.015, lgb_leaves=200, lgb_depth=8,
                   cb_lr=0.05,   cb_depth=7,
                   w_lgb=0.70,   w_cb=0.30,
                   tag=""):
    """
    Train LightGBM + CatBoost on OOF data (early stopping on val),
    then retrain both on full data for the final test predictions.
    Returns clipped [0,1] test predictions.
    """
    # ── LightGBM ─────────────────────────────────────────────────────────────
    lgb_cv = lgb.LGBMRegressor(
        n_estimators=3000, learning_rate=lgb_lr,
        max_depth=lgb_depth, num_leaves=lgb_leaves,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
        min_child_samples=10, reg_alpha=0.05, reg_lambda=1.0,
        random_state=42, verbose=-1,
    )
    lgb_cv.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
               callbacks=[lgb.early_stopping(80, verbose=False),
                           lgb.log_evaluation(9999)])
    best_lgb_iter = lgb_cv.best_iteration_
    sp_lgb = spearmanr(y_val, np.clip(lgb_cv.predict(X_val), 0, 1))[0]
    print(f"  [{tag}] LGB val Spearman={sp_lgb:.4f}  best_iter={best_lgb_iter}")

    # ── CatBoost ─────────────────────────────────────────────────────────────
    cb_cv = CatBoostRegressor(
        iterations=2000, learning_rate=cb_lr,
        depth=cb_depth, l2_leaf_reg=3,
        random_seed=42, verbose=0, early_stopping_rounds=50,
    )
    cb_cv.fit(X_tr, y_tr, eval_set=(X_val, y_val))
    best_cb_iter = cb_cv.best_iteration_
    sp_cb = spearmanr(y_val, np.clip(cb_cv.predict(X_val), 0, 1))[0]
    sp_blend_val = spearmanr(y_val,
        np.clip(w_lgb*lgb_cv.predict(X_val) + w_cb*cb_cv.predict(X_val), 0, 1))[0]
    print(f"  [{tag}] CB  val Spearman={sp_cb:.4f}  best_iter={best_cb_iter}")
    print(f"  [{tag}] Blend ({w_lgb:.2f}LGB+{w_cb:.2f}CB) val Spearman={sp_blend_val:.4f}")

    # ── Retrain on full data ─────────────────────────────────────────────────
    lgb_full = lgb.LGBMRegressor(
        n_estimators=best_lgb_iter + 100, learning_rate=lgb_lr,
        max_depth=lgb_depth, num_leaves=lgb_leaves,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
        min_child_samples=10, reg_alpha=0.05, reg_lambda=1.0,
        random_state=42, verbose=-1,
    )
    lgb_full.fit(X_full, y_full)

    cb_full = CatBoostRegressor(
        iterations=best_cb_iter + 100, learning_rate=cb_lr,
        depth=cb_depth, l2_leaf_reg=3,
        random_seed=42, verbose=0,
    )
    cb_full.fit(X_full, y_full)

    tp = np.clip(w_lgb * lgb_full.predict(X_test) +
                 w_cb  * cb_full.predict(X_test), 0, 1)
    return tp


# ══════════════════════════════════════════════════════════════════════════════
# 5.  TWO-CONFIGURATION ENSEMBLE  (reproduces the 90.73638 submission)
#
#     Config A  →  LGB 0.70 + CB 0.30  (matched submission A10, LB 90.31)
#     Config B  →  LGB 0.75 + CB 0.25  (matched submission A13, LB 90.73)
#
#     Final prediction = 0.60 × Config_A  +  0.40 × Config_B
#     File name hint: "60best_40wtd"
#       "best" = A10 (first strong single-config run)
#       "wtd"  = A13 (second run with slightly different blend weights)
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Config A: LGB 0.70 + CB 0.30 ────────────────────────────────────────")
preds_A = train_ensemble(
    X_tr, y_tr, X_val, y_val, X_full, y_full, X_test,
    lgb_lr=0.015, lgb_leaves=200, lgb_depth=8,
    cb_lr=0.05,   cb_depth=7,
    w_lgb=0.70,   w_cb=0.30,
    tag="Config-A",
)

print("\n── Config B: LGB 0.75 + CB 0.25 ────────────────────────────────────────")
preds_B = train_ensemble(
    X_tr, y_tr, X_val, y_val, X_full, y_full, X_test,
    lgb_lr=0.015, lgb_leaves=200, lgb_depth=8,
    cb_lr=0.05,   cb_depth=7,
    w_lgb=0.75,   w_cb=0.25,
    tag="Config-B",
)

# Final blend
final_pred = np.clip(0.60 * preds_A + 0.40 * preds_B, 0, 1)

# ══════════════════════════════════════════════════════════════════════════════
# 6.  GENERATE & VALIDATE SUBMISSION
# ══════════════════════════════════════════════════════════════════════════════
submission = pd.DataFrame({
    'Index' : test['Index'].values,
    'demand': final_pred,
})

assert submission.shape        == (41778, 2),         f"Wrong shape: {submission.shape}"
assert list(submission.columns)== ['Index', 'demand'],"Wrong columns"
assert not submission.demand.isna().any(),             "NaN found"
assert not np.isinf(submission.demand).any(),          "Inf found"
assert (submission.demand >= 0).all(),                 "Negative values"
assert (submission.demand <= 1).all(),                 "Values > 1"

submission.to_csv('submission.csv', index=False)

print(f"\n✅  submission.csv  |  shape={submission.shape}")
print(f"   demand  mean={final_pred.mean():.4f}  std={final_pred.std():.4f}"
      f"  min={final_pred.min():.4f}  max={final_pred.max():.4f}")
print(f"\n   Best public LB score achieved with this pipeline: 90.73638")
