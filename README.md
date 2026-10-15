# RECS Energy Intelligence Studio ⚡
**Residential Electricity Demand Prediction & What-If Retrofit Simulator**
*Trained on 39,833 U.S. Household Records across 4 EIA RECS Survey Cycles (2005, 2009, 2015, 2020)*

---

## 📌 Overview

The **RECS Energy Intelligence Studio** is a machine learning pipeline and interactive dashboard that predicts residential annual electricity consumption (kWh) and utility costs in **Indian Rupees (₹)**. Built with **FastAPI**, **XGBoost**, and a **monochrome minimalist dark dashboard (Black / Grey / White)**, it provides:

1. **⚡ Household Energy Predictor**: Dynamic questionnaire capturing interior square footage, room density, occupancy, appliances, heating fuel, and climate severity.
2. **🎛️ Live "What-If" Retrofit Simulator**: Real-time sliders adjusting appliance counts, LED lighting adoption, and freezer/dryer status with instant savings delta calculations in ₹.
3. **📊 Rigorous Evaluation**: Strictly time-aware test split (trained on 2005–2015 cycles; tested on held-out **2020 cycle**).
4. **🔍 Explainability (SHAP)**: Feature attribution ranking the top structural and behavioral factors driving electrical demand.
5. **📑 Data Harmonization Crosswalk**: Searchable variable mapping table showing how features were harmonized across 15 years of EIA RECS survey instruments.

---

## 🏆 Model Benchmark Results

Models were evaluated on a held-out test cohort of **18,155 households from the 2020 RECS survey cycle**:

| Model Architecture | MAE (kWh) | RMSE (kWh) | $R^2$ Score | MAPE (%) | Status |
|---|---|---|---|---|---|
| **Baseline (Sample Mean)** | 4,950 | 6,223 | -0.003 | 73.3% | Reference Baseline |
| **Ridge Regression** | 6,888 | 9,441 | -1.307 | 72.2% | Linear Baseline |
| **Random Forest (300 Trees)** | 3,425 | 4,627 | 0.446 | 42.2% | Non-linear Ensemble |
| **Tuned XGBoost (Hist)** | **3,363** | **4,535** | **0.468** | **41.7%** | 🏆 **Champion Model** |

---

## 🛠️ Architecture & Pipeline Workflow

```
[Raw RECS 2005, 2009, 2015, 2020 CSVs]
                   │
                   ▼
     [1. Harmonization Crosswalk]
   • Standardized 17 core dwelling features
   • Mapped missing codes (-2, -8, -9, -99) to NaN
   • Excluded survey replicate weights (BRRWT1-96)
                   │
                   ▼
        [2. Preprocessing & Outlier Capping]
   • 1st to 99th percentile consumption trimming
   • Categorical label encoding
                   │
                   ▼
        [3. Domain Feature Engineering]
   • Space density: sqft/occupants, rooms/occupants
   • Lighting efficiency: LED adoption ratio
   • Climate interaction: HDD × heating equipment, CDD × region
                   │
                   ▼
     [4. Time-Aware Split (2005-2015 Train | 2020 Test)]
                   │
                   ▼
    [5. Hyperparameter-Tuned XGBoost (RandomizedSearchCV)]
                   │
                   ▼
      [6. Model Bundle & SHAP Explainability Export]
                   │
                   ▼
    [7. FastAPI Backend & Minimalist Dark Dashboard UI]
```

---

## 🚀 Quickstart

### 1. Clone Repository & Install Dependencies
```bash
git clone https://github.com/tirth1356/RECS-Energy-Intelligence-Studio.git
cd RECS-Energy-Intelligence-Studio
pip install -r requirements.txt
```

### 2. Run the Machine Learning Pipeline (Optional if using pre-trained model)
```bash
python pipeline.py
```

### 3. Launch the Web Application
```bash
python -m uvicorn app:app --reload --host 127.0.0.1 --port 8000
```
Open **`http://127.0.0.1:8000`** in your browser.

---

## 💡 Key Findings from SHAP Explainability
- **Heating Fuel (#1 Impact)**: Space heating fuel type (electric resistance vs. natural gas/oil) causes up to ±1,729 kWh/yr variation.
- **Cooling Demand**: Cooling degree days interacting with geographic climate zone dictates summer compressor runtime.
- **Secondary Refrigerators**: Operating a second refrigerator adds ~800–1,200 kWh/yr (~₹6,400–₹9,600 / year).
- **LED Lighting**: Upgrading incandescent/CFL fixtures to 100% solid-state LEDs significantly reduces continuous electrical baseload.

---

## 📄 License
MIT License. Public microdata courtesy of U.S. Energy Information Administration (EIA).
