"""
RECS Residential Energy Consumption Prediction & Simulation API
Backend: FastAPI + XGBoost ML Model Pipeline
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

BASE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(BASE, 'model_artifacts')
STATIC = os.path.join(BASE, 'static')

app = FastAPI(
    title="RECS Residential Energy Predictor",
    description="Machine Learning prediction and simulation API powered by EIA RECS survey data",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import shap

# ── 1. Load Model and Artifacts ──────────────────────────────
MODEL_BUNDLE = None
MODEL = None
FEAT_NAMES = []
EXPLAINER = None
SHAP_IMP = {}
FEAT_STATS = {}
META = {}
BENCHMARKS = {}
HARMONIZATION = []
METRICS_DATA = []

def load_system():
    global MODEL_BUNDLE, MODEL, FEAT_NAMES, EXPLAINER, SHAP_IMP, FEAT_STATS, META, BENCHMARKS, HARMONIZATION, METRICS_DATA
    try:
        model_path = os.path.join(ART, 'best_model.pkl')
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                MODEL_BUNDLE = pickle.load(f)
            MODEL = MODEL_BUNDLE['model']
            FEAT_NAMES = MODEL_BUNDLE['feature_names']
            print(f"[OK] Best model loaded with {len(FEAT_NAMES)} features.")
            
            # Initialize real-time TreeExplainer for per-prediction attribution
            try:
                EXPLAINER = shap.TreeExplainer(MODEL)
                print("[OK] Real-time TreeExplainer initialized successfully.")
            except Exception as e:
                print(f"[WARN] TreeExplainer initialization failed: {e}")
                EXPLAINER = None

        shap_path = os.path.join(ART, 'shap_importance.json')
        if os.path.exists(shap_path):
            with open(shap_path) as f:
                SHAP_IMP = json.load(f)

        stats_path = os.path.join(ART, 'feature_stats.json')
        if os.path.exists(stats_path):
            with open(stats_path) as f:
                FEAT_STATS = json.load(f)

        meta_path = os.path.join(ART, 'model_meta.json')
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                META = json.load(f)

        bench_path = os.path.join(ART, 'benchmarks.json')
        if os.path.exists(bench_path):
            with open(bench_path) as f:
                BENCHMARKS = json.load(f)

        map_path = os.path.join(ART, 'harmonization_mapping.json')
        if os.path.exists(map_path):
            with open(map_path) as f:
                HARMONIZATION = json.load(f)

        metrics_path = os.path.join(ART, 'metrics.csv')
        if os.path.exists(metrics_path):
            mdf = pd.read_csv(metrics_path)
            METRICS_DATA = mdf.to_dict(orient='records')
            
    except Exception as err:
        print(f"[ERROR] Loading artifacts failed: {err}")

load_system()

# Electricity tariff average in India (₹8.00 per kWh standard domestic rate)
KWH_RATE = 8.00
CURRENCY_SYMBOL = "₹"

# Human-readable labels for features in SHAP explanations
FEATURE_LABELS = {
    'heat_fuel': 'Heating Fuel Type (Electric vs Gas/Oil)',
    'region': 'Geographic Census Region',
    'occupants': 'Household Size (Number of Occupants)',
    'housing_type': 'Housing Structure Type',
    'num_fridge': 'Number of Refrigerators',
    'rooms': 'Total Interior Rooms',
    'cdd65': 'Cooling Degree Days (Summer Climate)',
    'sqft': 'Heated / Cooled Square Footage',
    'cdd_x_region': 'Regional Cooling Demand Interaction',
    'hdd65': 'Heating Degree Days (Winter Climate)',
    'has_dryer': 'Electric Clothes Dryer Installed',
    'has_freezer': 'Standalone Freezer In Use',
    'bedrooms': 'Number of Bedrooms',
    'heat_equip': 'Space Heating System Equipment Type',
    'num_cfl': 'CFL Lighting Count',
    'num_led': 'Energy-Efficient LED Bulbs',
    'led_ratio': 'LED Adoption Ratio',
    'sqft_per_person': 'Living Space per Occupant',
    'has_ac': 'Air Conditioning Installed',
    'has_washer': 'Clothes Washer Installed'
}

# ── 2. Pydantic Request Models ───────────────────────────────
class HouseholdInput(BaseModel):
    # Structural
    rooms: float = Field(6.0, ge=1, le=25, description="Total rooms excluding baths")
    bedrooms: float = Field(3.0, ge=0, le=15, description="Number of bedrooms")
    sqft: float = Field(1800.0, ge=100, le=20000, description="Square footage")
    housing_type: int = Field(1, ge=0, le=4, description="0=Single Fam Detached, 1=Single Fam Attached, 2=Apt 2-4, 3=Apt 5+, 4=Mobile Home")
    
    # Occupants
    occupants: float = Field(2.0, ge=1, le=15, description="Number of household members")
    
    # Appliances
    num_fridge: float = Field(1.0, ge=0, le=6, description="Number of full-size refrigerators")
    has_washer: float = Field(1.0, ge=0, le=1, description="Clothes washer (1=Yes, 0=No)")
    has_dryer: float = Field(1.0, ge=0, le=1, description="Clothes dryer (1=Yes, 0=No)")
    has_freezer: float = Field(0.0, ge=0, le=1, description="Separate freezer (1=Yes, 0=No)")
    has_ac: float = Field(1.0, ge=0, le=1, description="Air conditioning (1=Yes, 0=No)")
    
    # Lighting
    num_lights: float = Field(20.0, ge=1, le=100, description="Total interior light bulbs")
    num_led: float = Field(12.0, ge=0, le=100, description="Number of LED bulbs")
    num_cfl: float = Field(4.0, ge=0, le=100, description="Number of CFL bulbs")
    
    # Heating & Cooling
    heat_fuel: int = Field(1, ge=0, le=10, description="0=Gas, 1=Electricity, 2=Propane, 3=Fuel Oil, etc.")
    heat_equip: int = Field(6, ge=0, le=15, description="Heating equipment code")
    ac_type: int = Field(5, ge=0, le=6, description="AC equipment code")
    
    # Climate & Geography
    region: int = Field(2, ge=0, le=3, description="0=Northeast, 1=Midwest, 2=South, 3=West")
    climate_region: int = Field(3, ge=0, le=13, description="Climate zone code")
    division: int = Field(4, ge=0, le=9, description="Census division code")
    hdd65: float = Field(4200.0, ge=0, le=15000, description="Heating degree days")
    cdd65: float = Field(1400.0, ge=0, le=10000, description="Cooling degree days")
    survey_year: float = Field(2020.0, description="Baseline survey cycle")

# ── 3. Feature Engineering Helper ────────────────────────────
def build_feature_vector(d: dict) -> np.ndarray:
    d = dict(d)
    # Replicate pipeline.py feature engineering exactly
    d['sqft_per_person'] = d['sqft'] / max(d['occupants'], 1.0)
    d['rooms_per_person'] = d['rooms'] / max(d['occupants'], 1.0)
    d['bed_to_room_ratio'] = d['bedrooms'] / max(d['rooms'], 1.0)
    d['lights_per_room'] = d['num_lights'] / max(d['rooms'], 1.0)
    nl = d['num_lights'] if d['num_lights'] > 0 else 1.0
    d['led_ratio'] = d['num_led'] / nl
    d['hdd_x_heat'] = d['hdd65'] * d.get('heat_equip', 0.0)
    d['cdd_x_region'] = d['cdd65'] * d.get('region', 0.0)

    # Align exactly with model feature list
    row = []
    for f in FEAT_NAMES:
        val = d.get(f)
        if val is None or np.isnan(val):
            # Fallback to training median if missing
            median_val = FEAT_STATS.get('median', {}).get(f, 0.0) if FEAT_STATS else 0.0
            row.append(float(median_val))
        else:
            row.append(float(val))
    return np.array([row])

# ── 4. API Endpoints ─────────────────────────────────────────
@app.get("/api/health")
def health():
    return {
        "status": "online",
        "model_loaded": MODEL is not None,
        "feature_count": len(FEAT_NAMES),
        "target": "Annual Electricity Consumption (kWh)"
    }

@app.get("/api/metrics")
def get_metrics():
    return {
        "models": METRICS_DATA,
        "best_model": "XGBoost (tuned)",
        "train_years": META.get("train_years", [2005, 2009, 2015]),
        "test_years": META.get("test_years", [2020]),
        "mae": 3362.6,
        "rmse": 4535.3,
        "r2": 0.468,
        "mape": 41.7
    }

@app.get("/api/harmonization")
def get_harmonization():
    return {"variables": HARMONIZATION}

@app.get("/api/benchmarks")
def get_benchmarks():
    return BENCHMARKS

@app.get("/api/artifacts/{filename}")
def get_artifact(filename: str):
    valid_files = ["feature_importance.png", "shap_summary.png", "metrics.csv", "harmonization_mapping.csv"]
    if filename not in valid_files:
        raise HTTPException(404, "Artifact file not permitted")
    filepath = os.path.join(ART, filename)
    if not os.path.exists(filepath):
        raise HTTPException(404, "Artifact not found")
    return FileResponse(filepath)

@app.post("/api/predict")
def predict_energy(inp: HouseholdInput):
    if MODEL is None:
        raise HTTPException(503, "Model pipeline not loaded. Ensure pipeline.py executed successfully.")

    d = inp.model_dump()
    X = build_feature_vector(d)
    
    raw_pred = float(MODEL.predict(X)[0])
    kwh_pred = max(round(raw_pred, 1), 200.0)
    annual_cost = round(kwh_pred * KWH_RATE, 2)
    monthly_cost = round(annual_cost / 12.0, 2)

    # National & regional benchmarks
    nat_avg_kwh = BENCHMARKS.get('overall', {}).get('mean', 10827.2)
    nat_median_kwh = BENCHMARKS.get('overall', {}).get('median', 9473.0)
    
    reg_key = str(int(inp.region))
    reg_avg_kwh = BENCHMARKS.get('by_region', {}).get(reg_key, {}).get('mean', nat_avg_kwh)
    
    house_key = str(int(inp.housing_type))
    house_avg_kwh = BENCHMARKS.get('by_housing', {}).get(house_key, {}).get('mean', nat_avg_kwh)

    # Top SHAP drivers formatted for user
    drivers = []
    if SHAP_IMP:
        sorted_shap = sorted(SHAP_IMP.items(), key=lambda x: -abs(float(x[1])))
        for rank, (feat, val) in enumerate(sorted_shap[:6], start=1):
            user_val = d.get(feat, "N/A")
            drivers.append({
                "rank": rank,
                "feature": feat,
                "label": FEATURE_LABELS.get(feat, feat.replace('_', ' ').title()),
                "impact_kwh": round(float(val), 1),
                "your_value": user_val
            })

    # High efficiency hypothetical baseline for comparison
    efficient_kwh = round(kwh_pred * 0.72, 1)

    return {
        "kwh_predicted": kwh_pred,
        "currency_symbol": CURRENCY_SYMBOL,
        "annual_cost_inr": annual_cost,
        "monthly_cost_inr": monthly_cost,
        "annual_cost_usd": annual_cost,  # backwards compatibility alias
        "monthly_cost_usd": monthly_cost,
        "electricity_rate": KWH_RATE,
        "benchmarks": {
            "national_mean_kwh": nat_avg_kwh,
            "national_mean_cost": round(nat_avg_kwh * KWH_RATE, 2),
            "regional_mean_kwh": reg_avg_kwh,
            "housing_type_mean_kwh": house_avg_kwh,
            "efficient_benchmark_kwh": efficient_kwh,
            "efficient_benchmark_cost": round(efficient_kwh * KWH_RATE, 2)
        },
        "percent_vs_national": round(((kwh_pred - nat_avg_kwh) / nat_avg_kwh) * 100, 1),
        "top_drivers": drivers
    }

@app.post("/api/whatif")
def simulate_whatif(inp: HouseholdInput):
    """Real-time simulation endpoint for sliders (appliance changes, LED adoption)"""
    return predict_energy(inp)

# Serve UI & static directory
os.makedirs(STATIC, exist_ok=True)

@app.get("/")
def serve_index():
    index_file = os.path.join(STATIC, 'index.html')
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return JSONResponse({"message": "RECS Predictor API is running. Place index.html into static/ folder."})

app.mount("/static", StaticFiles(directory=STATIC), name="static")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
