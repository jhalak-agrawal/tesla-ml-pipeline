# 🚗 Tesla Deliveries & Production — End-to-End ML Pipeline

An end-to-end Machine Learning pipeline built on Tesla's global deliveries and production data (2015–2025), covering preprocessing, EDA, feature engineering, regression modeling, hyperparameter tuning, and time series forecasting.

---

## 📦 Dataset

[Tesla EA Deliveries and Production Data (2015–2025)](https://www.kaggle.com/datasets/nalisha/tesla-ea-deliveries-and-production-data20152025)

| Property | Value |
|---|---|
| Rows | 2,640 |
| Columns | 12 |
| Years Covered | 2015 – 2025 |
| Models | Model S, Model X, Model 3, Model Y, Cybertruck |
| Regions | North America, Europe, Asia, Middle East |

---

## 🗂️ Repository Structure

```
tesla-ml-pipeline/
│
├── tesla_ml_pipeline.ipynb                  ← Main notebook (all code + charts)
├── tesla_deliveries_dataset_2015_2025.csv   ← Dataset
└── README.md
```

---

## 🔬 Pipeline Overview

### 1. Preprocessing
- Parsed `Year` + `Month` into a unified `Date` column
- Verified zero missing values and zero duplicate rows
- Label-encoded categorical columns (`Region`, `Model`, `Source_Type`)
- Created 5 derived business-logic features:
  - `Delivery_Efficiency` = Deliveries / Production
  - `Revenue_Proxy_M` = Deliveries × Avg Price
  - `Price_per_kWh`, `CO2_per_Delivery`, `Stations_per_Delivery`

### 2. Exploratory Data Analysis (EDA)
- Annual delivery trends broken down by model
- Regional market share (pie chart)
- Average vehicle price trajectory (2015–2025)
- Delivery efficiency comparison across models
- Full feature correlation heatmap
- Seasonality analysis, CO₂ savings by region, revenue proxy trends

### 3. Feature Engineering
21 total features including:

| Feature | Type |
|---|---|
| `Roll3_Deliveries` | 3-month rolling mean |
| `Roll6_Price` | 6-month rolling mean of price |
| `Lag1_Deliveries` | Previous month deliveries |
| `YoY_Delivery_Pct` | Year-over-year % change |
| `Battery_Range_Ratio` | Battery kWh ÷ Range km |
| `Market_Maturity` | Year offset × charging infrastructure |

### 4. Regression Modeling
6 models trained on 2 targets — **Estimated Deliveries** and **Avg Price USD**:

| Model | Deliveries R² | Price R² |
|---|---|---|
| Linear Regression | 0.9985 | 0.9123 |
| Ridge | 0.9982 | 0.9112 |
| Lasso | 0.9984 | 0.9128 |
| Decision Tree | 0.9978 | 0.9948 |
| Random Forest | 0.9977 | 0.9990 |
| **Gradient Boosting** ⭐ | **0.9985** | **0.9994** |

### 5. Hyperparameter Tuning
- `GridSearchCV` on Random Forest with 4-fold cross-validation
- 36 parameter combinations × 4 folds = **144 fits**
- Best params: `n_estimators=250`, `max_depth=16`, `max_features='sqrt'`
- Tuned R²: **0.9834**

### 6. Time Series Forecasting
- Supervised Gradient Boosting with lag (1, 2, 3, 6, 12 months) + rolling mean features
- Trained on all months except last 12 → validated on hold-out
- **12-month forward forecast** generated for 2025/26

---

## 📊 Output Charts

| # | Chart |
|---|---|
| 01 | EDA Overview — deliveries, price trend, region share, correlation matrix |
| 02 | EDA Deep — distributions, seasonality, CO₂, revenue proxy |
| 03 | Model Comparison — R² across all models and targets |
| 04 | Actual vs Predicted — best model per target |
| 05 | Feature Importances — tuned Random Forest |
| 06 | GridSearchCV Heatmap — n_estimators × max_depth |
| 07 | Time Series Forecast — history + 12-month ahead |
| 08 | Residual Analysis — residuals vs predicted + distribution |

---

## ⚙️ Requirements

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

---

## 🚀 How to Run

1. Clone the repo and place the CSV in the same folder as the notebook
2. Open `tesla_ml_pipeline.ipynb` in Jupyter or VS Code
3. Run all cells top to bottom

```bash
git clone https://github.com/YOUR_USERNAME/tesla-ml-pipeline
cd tesla-ml-pipeline
jupyter notebook tesla_ml_pipeline.ipynb
```
