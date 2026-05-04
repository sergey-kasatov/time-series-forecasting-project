# Time Series Forecasting Project

**Author:** Sergey Kasatov  
**Live App:** [FreshMart Demand Planner](https://freshmart-demand-planner.streamlit.app)  
**Repository:** [time-series-forecasting-project](https://github.com/sergey-kasatov/time-series-forecasting-project)

## Project Overview

This project focuses on time series forecasting for daily unit sales of a selected store item.

The dataset is based on a simplified version of the Corporación Favorita Grocery Sales Forecasting data. The goal of the project is to analyze sales patterns, build and compare multiple forecasting models, select the best-performing champion model, and turn it into an interactive Streamlit application.

The final app, **FreshMart Demand Planner**, is designed as a short-term sales forecast planning prototype for a small grocery store manager.

---

## Business Problem

A small grocery store manager needs to understand expected sales demand for the next available planning period after the latest known sales date.

The app helps answer questions such as:

- What sales volume should be expected in the selected planning horizon?
- What is the total expected demand?
- What is the average daily forecast?
- Which day is expected to have the highest demand?
- How can this information support inventory and staffing planning?

---

## Dataset

The project uses a simplified time series dataset derived from the Corporación Favorita Grocery Sales Forecasting competition.

The main target variable is:

- `unit_sales`

Additional datasets such as oil prices and holidays were used during feature engineering to enrich the forecasting dataset.

---

## Project Workflow

The project is organized into five main notebooks:

| Notebook | Purpose |
|---|---|
| `01_eda_and_cleaning.ipynb` | Exploratory data analysis, missing value handling, outlier review, decomposition, stationarity checks, and cleaned time series creation |
| `02_feature_engineering.ipynb` | Creation of calendar, lag, rolling, oil, and holiday-based features |
| `03_statistical_models.ipynb` | Training and evaluation of statistical forecasting models |
| `04_model_training_tuning_mlflow.ipynb` | Baseline machine learning models, HyperOpt tuning, MLflow tracking, LSTM comparison, and champion model selection |
| `05_app_test.ipynb` | Testing model loading, prediction logic, app-ready outputs, and Streamlit app structure |

---

## Models Tested

The project compares several forecasting approaches:

### Statistical Models

- SARIMAX
- Holt-Winters
- Prophet

### Machine Learning Models

- Linear Regression
- Ridge Regression
- Random Forest
- Gradient Boosting
- XGBoost

### Advanced Model

- LSTM
- HyperOpt-tuned LSTM

All models were evaluated on the same test period:

- **Test period:** January 2014 to March 2014

The main evaluation metric used for champion model selection was:

- **RMSE**

---

## Champion Model

The final champion model is:

**HyperOpt-tuned Random Forest**

It was selected because it achieved the lowest RMSE among the compared statistical, machine learning, and advanced models.

Final champion model performance on the historical test period:

| Metric | Value |
|---|---:|
| MAE | 93.14 |
| RMSE | 138.01 |
| MAPE | 21.29% |
| R² | 0.4339 |

The champion model and supporting files are saved in the `models/` folder:

| File | Purpose |
|---|---|
| `champion_model_random_forest.pkl` | Saved champion model |
| `feature_columns.json` | Feature list used by the model |
| `champion_model_metadata.json` | Metadata about the champion model |
| `champion_model_test_predictions.csv` | Saved predictions for the test period |

---

## Streamlit App

The final Streamlit app is called **FreshMart Demand Planner** and is implemented in:

Live app: [FreshMart Demand Planner](https://freshmart-demand-planner.streamlit.app)

```text
app.py
```

FreshMart Demand Planner is designed as a short-term sales forecast planning prototype for a small grocery store manager.

The user can:

- select the last known sales date,
- choose a forecast horizon,
- choose how many recent historical days to display before the cutoff date,
- optionally show historical actual sales for comparison,
- generate a forecast simulation,
- review forecast planning KPIs,
- download the forecast as CSV.

The app displays:

- total forecasted sales,
- average daily forecast,
- peak forecast day,
- peak forecasted sales,
- forecast demand level,
- recent sales before cutoff,
- predicted sales after cutoff,
- forecast table,
- planning interpretation,
- model validation in an expandable section,
- champion model details in an expandable section.

---

## How to Use the App

1. Open the deployed FreshMart Demand Planner app.
2. Select the cutoff date, which represents the last known sales date.
3. Choose the forecast horizon and the number of recent historical days to display.
4. Optionally enable historical actual sales for comparison.
5. Click **Generate forecast**.
6. Review the planning KPIs, forecast chart, forecast table, and planning interpretation.
7. Download the forecast table as CSV if needed.

---

## Repository Structure

```text
time-series-forecasting-project/
│
├── app.py
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── champion_model_random_forest.pkl
│   ├── feature_columns.json
│   ├── champion_model_metadata.json
│   └── champion_model_test_predictions.csv
│
├── notebooks/
│   ├── 01_eda_and_cleaning.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_statistical_models.ipynb
│   ├── 04_model_training_tuning_mlflow.ipynb
│   └── 05_app_test.ipynb
│
└── reports/
```

---

## How to Run the Project Locally

### 1. Clone the repository

```bash
git clone <repository-url>
cd time-series-forecasting-project
```

### 2. Create and activate a virtual environment

Example for Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install requirements

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

If Streamlit is not available as a direct command, run:

```bash
python -m streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

---

## Key Results

The model comparison showed that the HyperOpt-tuned Random Forest model achieved the strongest overall performance.

The final app translates the model output into planning-oriented business information. Instead of only showing technical metrics, the app provides actionable forecast summaries such as total expected demand, average daily forecast, peak forecast day, and peak forecasted sales day.

---

## Limitations

This app is a portfolio forecasting prototype.

The current version simulates forecasts using prepared feature-engineered data from the historical test period. It does not generate forecasts beyond the available feature-engineered dataset.

A production version would require a future feature generation pipeline for future dates, including calendar features, holiday features, oil price assumptions, and recursive lag/rolling features.

---

## Author

**Sergey Kasatov**  
Data Analytics / Data Science portfolio project  
GitHub: [sergey-kasatov](https://github.com/sergey-kasatov)