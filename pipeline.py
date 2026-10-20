"""
RECS Energy Consumption Prediction Pipeline
Covers: Preprocessing -> Feature Engineering -> Modeling -> Export
"""
#pip install -r requirements.txt
#python app.py
# recalculation python pipeline.py

import os
import warnings
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

BASE = r'c:\Users\TirthPatel\Documents\innovative\ml'
OUT  = os.path.join(BASE, 'model_artifacts')
os.makedirs(OUT, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 1.  COLUMN MAPS  (raw col -> standard name, per survey year)
# ─────────────────────────────────────────────────────────────
RECS_MISSING = [-2, -8, -9, -99]

COL_MAP = {
    2005: {
        'TOTROOMS':   'rooms',
        'BEDROOMS':   'bedrooms',
        'TOTSQFT':    'sqft',
        'NHSLDMEM':   'occupants',
        'NUMFRIG':    'num_fridge',
        'CWASHER':    'has_washer',
        'DRYER':      'has_dryer',
        'FREEZER':    'has_freezer',
        'AIRCOND':    'has_ac',
        'EQUIPM':     'heat_equip',
        'FUELHEAT':   'heat_fuel',
        'TYPEHUQ':    'housing_type',
        'REGIONC':    'region',
        'DIVISION':   'division',
        'KWH':        'kwh',
        'DOLLAREL':   'dollar_el',
        'HDD65':      'hdd65',
        'CDD65':      'cdd65',
    },
    2009: {
        'TOTROOMS':           'rooms',
        'BEDROOMS':           'bedrooms',
        'TOTSQFT_EN':         'sqft',
        'NHSLDMEM':           'occupants',
        'NUMFRIG':            'num_fridge',
        'CWASHER':            'has_washer',
        'DRYER':              'has_dryer',
        'FREEZER':            'has_freezer',
        'EQUIPM':             'heat_equip',
        'FUELHEAT':           'heat_fuel',
        'TYPEHUQ':            'housing_type',
        'REGIONC':            'region',
        'Climate_Region_Pub': 'climate_region',
        'HDD65':              'hdd65',
        'CDD65':              'cdd65',
        'KWH':                'kwh',
        'DOLLAREL':           'dollar_el',
        'TOTALDOL':           'total_cost',
    },
    2015: {
        'TOTROOMS':   'rooms',
        'BEDROOMS':   'bedrooms',
        'TOTSQFT_EN': 'sqft',
        'NHSLDMEM':   'occupants',
        'NUMFRIG':    'num_fridge',
        'CWASHER':    'has_washer',
        'DRYER':      'has_dryer',
        'UPRTFRZR':   'has_freezer',
        'LGTINNUM':   'num_lights',
        'LGTINCFL':   'num_cfl',
        'LGTINLED':   'num_led',
        'EQUIPM':     'heat_equip',
        'FUELHEAT':   'heat_fuel',
        'TYPEHUQ':    'housing_type',
        'REGIONC':    'region',
        'HDD65':      'hdd65',
        'CDD65':      'cdd65',
        'KWH':        'kwh',
        'DOLLAREL':   'dollar_el',
        'TOTALDOL':   'total_cost',
    },
    2020: {
        'TOTROOMS':       'rooms',
        'BEDROOMS':       'bedrooms',
        'TOTSQFT_EN':     'sqft',
        'NHSLDMEM':       'occupants',
        'NUMFRIG':        'num_fridge',
        'CWASHER':        'has_washer',
        'DRYER':          'has_dryer',
        'FREEZER':        'has_freezer',
        'UPRTFRZR':       'has_freezer',
        'LGTINLED':       'num_led',
        'LGTINCFL':       'num_cfl',
        'LGTIN1TO4':      'lights_1to4',
        'LGTIN4TO8':      'lights_4to8',
        'LGTINMORE8':     'lights_more8',
        'ACEQUIPM_PUB':   'ac_type',
        'EQUIPM':         'heat_equip',
        'FUELHEAT':       'heat_fuel',
        'TYPEHUQ':        'housing_type',
        'REGIONC':        'region',
        'BA_climate':     'climate_region',
        'HDD65':          'hdd65',
        'CDD65':          'cdd65',
        'KWH':            'kwh',
        'DOLLAREL':       'dollar_el',
        'TOTALDOL':       'total_cost',
    },
}

FILES = {
    2005: 'RECS05alldata.csv',
    2009: 'recs2009_public.csv',
    2015: 'recs2015_public_v4.csv',
    2020: 'recs2020_public_v7.csv',
}


# ─────────────────────────────────────────────────────────────
# 2.  LOAD + HARMONIZE
# ─────────────────────────────────────────────────────────────
def load_year(year):
    path = os.path.join(BASE, FILES[year])
    print(f"  Loading {year} ...", end=' ', flush=True)
    df = pd.read_csv(path, low_memory=False)
    print(f"{len(df):,} rows")
    col_map = COL_MAP[year]
    rename = {k: v for k, v in col_map.items() if k in df.columns}
    df = df[list(rename.keys())].rename(columns=rename)
    df = df.replace(RECS_MISSING, np.nan)
    df['survey_year'] = year
    # De-duplicate standard col names (e.g. has_freezer mapped twice)
    df = df.loc[:, ~df.columns.duplicated()]
    return df


def build_combined():
    frames = [load_year(y) for y in sorted(FILES)]
    combined = pd.concat(frames, ignore_index=True)
    print(f"\nCombined shape: {combined.shape}")
    return combined


# ─────────────────────────────────────────────────────────────
# 3.  PREPROCESSING
# ─────────────────────────────────────────────────────────────
CATEGORICAL_COLS = ['heat_equip', 'heat_fuel', 'housing_type',
                    'region', 'climate_region', 'ac_type', 'division']

label_encoders = {}

def preprocess(df):
    df = df.dropna(subset=['kwh']).copy()
    df = df[df['kwh'] > 0]

    num_cols = ['rooms', 'bedrooms', 'sqft', 'occupants',
                'num_fridge', 'num_lights', 'num_cfl', 'num_led',
                'hdd65', 'cdd65', 'kwh', 'dollar_el', 'total_cost',
                'lights_1to4', 'lights_4to8', 'lights_more8']
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    bool_cols = ['has_washer', 'has_dryer', 'has_freezer', 'has_ac']
    for c in bool_cols:
        if c in df.columns:
            df[c] = df[c].apply(lambda x: 1 if x == 1 else (0 if pd.notna(x) else np.nan))

    # Outlier cap (1st-99th pct)
    lo, hi = df['kwh'].quantile([0.01, 0.99])
    df = df[(df['kwh'] >= lo) & (df['kwh'] <= hi)]

    # Encode categoricals
    for c in CATEGORICAL_COLS:
        if c in df.columns:
            df[c] = df[c].astype(str).replace('nan', 'unknown')
            le = LabelEncoder()
            df[c] = le.fit_transform(df[c])
            label_encoders[c] = le

    # Feature engineering
    df['sqft_per_person']   = df['sqft'] / df['occupants'].clip(lower=1)
    df['rooms_per_person']  = df['rooms'] / df['occupants'].clip(lower=1)
    df['bed_to_room_ratio'] = df['bedrooms'] / df['rooms'].clip(lower=1)

    # Compute num_lights from bracket columns if missing
    if 'num_lights' not in df.columns or df['num_lights'].isna().all():
        lt_cols = [c for c in ['lights_1to4', 'lights_4to8', 'lights_more8'] if c in df.columns]
        if lt_cols:
            df['num_lights'] = df[lt_cols].sum(axis=1)
        else:
            df['num_lights'] = np.nan

    df['lights_per_room'] = df['num_lights'] / df['rooms'].clip(lower=1)

    if 'num_led' in df.columns:
        df['led_ratio'] = df['num_led'] / df['num_lights'].replace(0, np.nan)
    else:
        df['led_ratio'] = np.nan

    if 'hdd65' in df.columns:
        df['hdd_x_heat'] = df['hdd65'] * df.get('heat_equip', pd.Series(0, index=df.index))
    else:
        df['hdd_x_heat'] = np.nan

    if 'cdd65' in df.columns:
        df['cdd_x_region'] = df['cdd65'] * df['region'].fillna(0)
    else:
        df['cdd_x_region'] = np.nan

    # Fill remaining NaN with column median
    for c in df.select_dtypes(include='number').columns:
        df[c] = df[c].fillna(df[c].median())

    print(f"After preprocessing: {df.shape}")
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# 4.  FEATURE LIST
# ─────────────────────────────────────────────────────────────
FEATURES = [
    'rooms', 'bedrooms', 'sqft', 'occupants',
    'num_fridge', 'has_washer', 'has_dryer', 'has_freezer', 'has_ac',
    'num_lights', 'num_cfl', 'num_led', 'led_ratio',
    'heat_equip', 'heat_fuel', 'ac_type',
    'housing_type', 'region', 'climate_region', 'division',
    'hdd65', 'cdd65',
    'sqft_per_person', 'rooms_per_person', 'bed_to_room_ratio',
    'lights_per_room', 'hdd_x_heat', 'cdd_x_region',
    'survey_year',
]
TARGET = 'kwh'


def get_feature_matrix(df):
    avail = [f for f in FEATURES if f in df.columns]
    X = df[avail].fillna(df[avail].median())
    y = df[TARGET].copy()
    return X, y, avail


# ─────────────────────────────────────────────────────────────
# 5.  TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────
def year_split(df, test_years=(2020,)):
    train = df[~df['survey_year'].isin(test_years)]
    test  = df[ df['survey_year'].isin(test_years)]
    return train, test


# ─────────────────────────────────────────────────────────────
# 6.  METRICS
# ─────────────────────────────────────────────────────────────
def evaluate(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    print(f"  {name:30s}  MAE={mae:,.0f}  RMSE={rmse:,.0f}  R2={r2:.3f}  MAPE={mape:.1f}%")
    return {'model': name, 'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}


# ─────────────────────────────────────────────────────────────
# 7.  MODELING
# ─────────────────────────────────────────────────────────────
def train_models(X_train, y_train, X_test, y_test, feat_names):
    results = []

    # Naive baseline
    mean_pred = np.full(len(y_test), y_train.mean())
    results.append(evaluate("Baseline (mean)", y_test.values, mean_pred))

    # Ridge
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(X_train)
    Xte_s = scaler.transform(X_test)
    ridge = Ridge(alpha=10)
    ridge.fit(Xtr_s, y_train)
    results.append(evaluate("Ridge Regression", y_test.values, ridge.predict(Xte_s)))

    # Random Forest
    rf = RandomForestRegressor(n_estimators=300, max_depth=15,
                               min_samples_leaf=5, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    results.append(evaluate("Random Forest", y_test.values, rf.predict(X_test)))

    # XGBoost + RandomizedSearch
    xgb_base = xgb.XGBRegressor(objective='reg:squarederror',
                                 tree_method='hist', n_jobs=-1,
                                 random_state=42, verbosity=0)
    param_dist = {
        'n_estimators':     [300, 500, 700],
        'max_depth':        [4, 6, 8],
        'learning_rate':    [0.03, 0.05, 0.1],
        'subsample':        [0.7, 0.85, 1.0],
        'colsample_bytree': [0.7, 0.85, 1.0],
        'min_child_weight': [1, 3, 5],
        'reg_alpha':        [0, 0.1, 1],
    }
    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    search = RandomizedSearchCV(xgb_base, param_dist, n_iter=20,
                                cv=cv, scoring='neg_mean_absolute_error',
                                n_jobs=-1, random_state=42, verbose=1)
    print("\n  XGBoost RandomizedSearch (20 iters x 3 folds) ...")
    search.fit(X_train, y_train)
    best_xgb = search.best_estimator_
    results.append(evaluate("XGBoost (tuned)", y_test.values, best_xgb.predict(X_test)))
    print(f"  Best params: {search.best_params_}")

    # Save model
    model_path = os.path.join(OUT, 'best_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': best_xgb,
            'feature_names': feat_names,
            'label_encoders': label_encoders,
        }, f)
    print(f"\n  Model saved -> {model_path}")

    # Feature importance plot
    imp = pd.Series(best_xgb.feature_importances_, index=feat_names).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 7))
    imp.head(20).plot(kind='barh', ax=ax, color='steelblue')
    ax.invert_yaxis()
    ax.set_title('XGBoost Feature Importances (Top 20)', fontsize=14)
    ax.set_xlabel('Importance')
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, 'feature_importance.png'), dpi=120)
    plt.close(fig)

    # SHAP
    try:
        print("  Computing SHAP values ...")
        explainer = shap.TreeExplainer(best_xgb)
        sample = X_test.sample(min(500, len(X_test)), random_state=42)
        shap_vals = explainer.shap_values(sample)
        fig2 = plt.figure()
        shap.summary_plot(shap_vals, sample, show=False)
        plt.tight_layout()
        fig2.savefig(os.path.join(OUT, 'shap_summary.png'), dpi=120)
        plt.close(fig2)

        mean_shap = pd.Series(np.abs(shap_vals).mean(axis=0),
                              index=feat_names).sort_values(ascending=False)
        mean_shap.to_json(os.path.join(OUT, 'shap_importance.json'))
        print("  SHAP saved.")
    except Exception as e:
        print(f"  SHAP skipped: {e}")

    pd.DataFrame(results).to_csv(os.path.join(OUT, 'metrics.csv'), index=False)
    print("\n  All metrics saved -> metrics.csv")
    return best_xgb, results


# ─────────────────────────────────────────────────────────────
# 8.  MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("  RECS Energy Prediction Pipeline")
    print("=" * 60)

    print("\n[1] Loading & harmonizing data ...")
    raw = build_combined()
    raw.to_csv(os.path.join(OUT, 'combined_raw.csv'), index=False)

    print("\n[2] Preprocessing ...")
    df = preprocess(raw)
    df.to_csv(os.path.join(OUT, 'combined_clean.csv'), index=False)

    print("\n[3] Year-aware split (test = 2020) ...")
    train_df, test_df = year_split(df, test_years=(2020,))
    print(f"  Train: {len(train_df):,}  |  Test: {len(test_df):,}")

    X_train, y_train, feat_names = get_feature_matrix(train_df)
    X_test,  y_test,  _          = get_feature_matrix(test_df)
    X_test = X_test[feat_names]

    print(f"\n  Features ({len(feat_names)}): {feat_names}")

    # Save feature stats for website form defaults
    stats = X_train.describe().T[['mean', '50%', 'min', 'max']]
    stats.columns = ['mean', 'median', 'min', 'max']
    stats.to_json(os.path.join(OUT, 'feature_stats.json'))

    meta = {'features': feat_names, 'target': TARGET,
            'train_years': [2005, 2009, 2015], 'test_years': [2020]}
    with open(os.path.join(OUT, 'model_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print("\n[4] Training models ...")
    best_model, metrics = train_models(X_train, y_train, X_test, y_test, feat_names)

    print("\n[5] Done! Artifacts saved to:", OUT)
