# Walmart Sales Analysis & Machine Learning Prediction Pipeline

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-orange.svg)](https://scikit-learn.org/)
[![Vercel Ready](https://img.shields.io/badge/Vercel-Serverless%20API-black.svg)](https://vercel.com/)
[![Model Accuracy](https://img.shields.io/badge/R%C2%B2%20Score-97.2%25-brightgreen.svg)](#-machine-learning-model-benchmarks)

An end-to-end data analytics and predictive machine learning system for Walmart retail sales. This repository contains complete data preprocessing, exploratory time-series analysis (moving averages, seasonality), a gradient-boosted regression pipeline achieving **97.2% $R^2$ accuracy**, and a lightweight **Vercel Serverless Web Application** (`Webpage/`) for real-time sales forecasting.

---

## 📌 Project Overview

Forecasting retail sales accurately across diverse geographic stores and merchandise departments is vital for retail enterprises to:
- **Prevent Stockouts & Overstock:** Align inventory replenishment with demand spikes.
- **Optimize Workforce Scheduling:** Scale store staffing during peak shopping weeks.
- **Maximize Promotional ROI:** Measure the impact of promotional MarkDowns and major holiday events (Super Bowl, Labor Day, Thanksgiving, Christmas).

This project delivers:
1. **In-depth Exploratory Analysis:** Weekly-to-monthly aggregation, store profiling, moving averages (3 & 6 months), and departmental revenue breakdown.
2. **Machine Learning Pipeline:** Leakage-free target encoding, calendar feature engineering, chronological time-series splitting, and hyperparameter tuning.
3. **Interactive Web Application:** A responsive dashboard deployed as a serverless web app on Vercel for instant on-demand sales predictions.

---

## 🗂️ Repository Structure

```bash
├── Webpage/                      # Production Vercel Serverless Web Application
│   ├── api/
│   │   ├── predict.py            # Vercel Python serverless handler (/api/predict)
│   │   └── best_walmart_model.joblib # Serialized ML model bundle (HistGradientBoosting)
│   ├── public/
│   │   └── index.html            # Responsive Tailwind CSS frontend dashboard
│   ├── index.html                # Root fallback frontend UI
│   ├── requirements.txt          # Python dependencies for Vercel runtime
│   └── vercel.json               # Vercel routing and serverless configuration
│
├── data/                         # Raw and processed datasets
│   ├── features.csv              # Macroeconomic indicators (CPI, Fuel, Markdowns, etc.)
│   ├── stores.csv                # Store metadata (Store Type, Square Footage)
│   ├── train.csv                 # Historical weekly sales records (421,570 rows)
│   └── test.csv                  # Test set for out-of-sample evaluation
│
├── notebooks/                    # Interactive Jupyter notebooks for EDA & prototyping
│
├── outputs/                      # High-resolution diagnostic visualizations
│   ├── 1_Walmart Sales.png       # Store-level and aggregate monthly sales trends
│   ├── 2_Walmart Sales.png       # 3-month & 6-month rolling averages (trend smoothing)
│   └── 3_Walmart Sales.png       # Seasonality surge & Top 5 revenue-generating departments
│
├── scripts/                      # Data pipelines and ML model training scripts
│   ├── Merge_data_sets.py        # Dataset joiner and merger
│   ├── Code.py                   # Exploratory analysis & visualization generation
│   └── ml_model.py               # ML training, evaluation, and benchmark pipeline
│
├── LICENSE                       # Repository license
└── README.md                     # Comprehensive project documentation
```

---

## 📊 Exploratory Data Analysis & Key Visualizations

The analytics scripts generate three key visual reports stored in `outputs/`:

1. **Store-Level & Aggregate Monthly Trends (`outputs/1_Walmart Sales.png`)**
   - Resamples high-frequency weekly transactions into monthly store trajectories.
   - Contrasts high-performing regional superstores against smaller retail formats.

2. **Trend Smoothing with Moving Averages (`outputs/2_Walmart Sales.png`)**
   - Applies 3-month and 6-month rolling averages to separate structural growth from seasonal volatility.
   - Highlights positive year-over-year revenue expansion.

3. **Seasonality & Departmental Breakdown (`outputs/3_Walmart Sales.png`)**
   - Identifies consistent annual demand surges in **December** (holiday shopping) and **February** (Super Bowl / Presidents' Day).
   - Pinpoints **Department 92** as the single largest revenue driver across all stores.

---

## ⚙️ Data Preprocessing & Feature Engineering

To ensure robust generalization without lookahead bias or data leakage:

* **Temporal Extraction:** Extracted `Year`, `Month`, `Week` (ISO week number), and `Day` from transaction dates.
* **Missing Value Imputation:**
  - `MarkDown1` to `MarkDown5`: Missing values were imputed with `0`, complemented by binary `MarkDown_Active` flags.
  - `CPI` and `Unemployment`: Missing values were imputed using training set medians.
* **Store Categorical Mapping:** Converted store `Type` (A, B, C) into numerical representations and configured `Store` and `Dept` as native categorical features for tree-based algorithms.
* **Target Encoding (`Store_Dept_Avg`):** Historical sales baselines per Store-Department pair were fitted **strictly on the training split** to prevent data leakage into validation/test periods.
* **Chronological Time-Series Split:**
  - **Training Split (80%):** 337,256 rows (February 2010 to April 2012).
  - **Testing Split (20%):** 84,314 rows (April 2012 to October 2012).

---

## 🤖 Machine Learning Model Benchmarks

Models were evaluated on the chronological out-of-time test set. In addition to standard regression metrics (**MAE**, **RMSE**, and **$R^2$**), performance was measured using Walmart's competition metric:
$$\text{WMAE} = \frac{\sum w_i |y_i - \hat{y}_i|}{\sum w_i}$$
where $w_i = 5$ for holiday weeks and $1$ for regular weeks.

| Model | MAE ($) | RMSE ($) | $R^2$ Score | WMAE ($) | Performance Tier |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Ridge Regression** (Standardized Baseline) | \$2,849.31 | \$5,349.86 | 94.06% | \$2,919.11 | Linear Baseline |
| **Random Forest Regressor** (50 trees, depth 16) | \$2,009.32 | \$4,035.18 | 96.62% | \$2,033.86 | Ensemble |
| **HistGradientBoostingRegressor** (Tuned) | **\$1,889.68** | **\$3,676.67** | **97.20%** | **\$1,937.20** | 🏆 **Best Model** |

### Best Model: HistGradientBoostingRegressor
- **Accuracy:** Explains **97.2%** of sales variance on unseen future dates ($R^2 = 0.972$).
- **Low Error:** Mean Absolute Error of **\$1,889.68** on a dataset with an average weekly sales baseline of ~\$16,000.
- **Weighted Holiday Robustness:** Achieves a **\$1,937.20 WMAE**, confirming superior performance during high-volatility holiday periods.
- **Inference Efficiency:** Lightweight 1.4 MB serialized bundle with sub-second prediction latency.

---

## Deploying to Vercel (Step-by-Step)

The `Webpage/` directory is self-contained and pre-configured for **1-click Vercel Serverless deployment**:

1. **Log in to Vercel:** Go to [vercel.com](https://vercel.com) and click **"Add New..."** → **"Project"**.
2. **Import Repository:** Select this GitHub repository (`MKS-24/Walmart-Sales` or your repo name).
3. **Configure Root Directory (IMPORTANT):**
   - Under **Root Directory**, click **"Edit"**.
   - Type or select: `Webpage`
4. **Deploy:** Click **"Deploy"**.
5. Within 30–60 seconds, your interactive sales forecasting dashboard will be live at `https://<your-project>.vercel.app`!

---

## Local Usage & Inference

### 1. Install Dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

### 2. Run the Analysis Scripts
```bash
# Generate exploratory visualizations
python scripts/Code.py

# Retrain and benchmark all ML models
python scripts/ml_model.py
```

### 3. Real-Time Python Inference Example
```python
from scripts.ml_model import predict_single

# Define input scenario
sample_payload = {
    'Store': 1,
    'Dept': 92,
    'Date': '2012-11-23',   # Thanksgiving / Black Friday
    'IsHoliday': 1,
    'Type': 'A',
    'Size': 151315,
    'Temperature': 55.0,
    'Fuel_Price': 3.50,
    'CPI': 220.0,
    'Unemployment': 7.2,
    'MarkDown1': 8000,
    'MarkDown2': 200,
    'MarkDown3': 5000,
    'MarkDown4': 1500,
    'MarkDown5': 4000
}

predicted_sales = predict_single(sample_payload)
print(f"Predicted Weekly Sales: ${predicted_sales:,.2f}")
# Output: Predicted Weekly Sales: $177,609.19
```

---

## Strategic Business Recommendations

1. **Advance Holiday Inventory Buffers:** Predictable seasonal surges in December and February necessitate inventory ramp-up starting 4–6 weeks prior.
2. **Prioritize Tier-1 Departments:** Direct supply chain and marketing resources toward dominant departments (such as **Dept 92** and **Dept 95**) to maximize top-line revenue.
3. **Optimized Markdown Scheduling:** Coordinated promotional markdowns (particularly MD1 and MD3) show the highest positive correlation with sales expansion during key holiday retail windows.
