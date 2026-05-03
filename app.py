# Core libraries
import os
import json
import joblib

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# Page configuration
st.set_page_config(
    page_title="Sales Forecast Planning App",
    page_icon="📈",
    layout="wide"
)


# Project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

CHAMPION_MODEL_PATH = os.path.join(MODELS_DIR, "champion_model_random_forest.pkl")
FEATURE_COLUMNS_PATH = os.path.join(MODELS_DIR, "feature_columns.json")
METADATA_PATH = os.path.join(MODELS_DIR, "champion_model_metadata.json")
TEST_PREDICTIONS_PATH = os.path.join(MODELS_DIR, "champion_model_test_predictions.csv")

FEATURE_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "feature_engineered_timeseries.csv")


# Data loading functions
@st.cache_resource
def load_champion_model(model_path):
    return joblib.load(model_path)


@st.cache_data
def load_json_file(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


@st.cache_data
def load_csv_file(file_path, parse_date_columns=None):
    if parse_date_columns is None:
        return pd.read_csv(file_path)

    return pd.read_csv(file_path, parse_dates=parse_date_columns)


# Load saved model files and data
try:
    champion_model = load_champion_model(CHAMPION_MODEL_PATH)
    feature_columns = load_json_file(FEATURE_COLUMNS_PATH)
    champion_metadata = load_json_file(METADATA_PATH)

    test_predictions_df = load_csv_file(
        TEST_PREDICTIONS_PATH,
        parse_date_columns=["date"]
    )

    feature_df = load_csv_file(
        FEATURE_DATA_PATH,
        parse_date_columns=["date"]
    )

except FileNotFoundError as error:
    st.error(f"Required file not found: {error}")
    st.stop()


# Prediction function
def generate_predictions(input_df, model, feature_cols):
    """
    Generate predictions using a trained model and a saved feature column list.
    """
    missing_cols = [
        col for col in feature_cols
        if col not in input_df.columns
    ]

    if missing_cols:
        raise ValueError(f"Missing required feature columns: {missing_cols}")

    X_input = input_df[feature_cols].copy()

    predictions = model.predict(X_input)
    predictions = np.clip(predictions, 0, None)

    output_df = pd.DataFrame({
        "date": input_df["date"].values,
        "predicted_unit_sales": predictions
    })

    if "unit_sales" in input_df.columns:
        output_df["actual_unit_sales"] = input_df["unit_sales"].values
        output_df = output_df[
            ["date", "actual_unit_sales", "predicted_unit_sales"]
        ]

    return output_df


def calculate_metrics(actual_values, predicted_values):
    """
    Calculate standard regression metrics.
    """
    mae = np.mean(np.abs(actual_values - predicted_values))

    rmse = np.sqrt(
        np.mean((actual_values - predicted_values) ** 2)
    )

    mape = np.mean(
        np.abs((actual_values - predicted_values) / actual_values)
    ) * 100

    r2 = 1 - (
        np.sum((actual_values - predicted_values) ** 2)
        / np.sum((actual_values - actual_values.mean()) ** 2)
    )

    return mae, rmse, mape, r2


# Prepare test-period data for validation and simulation
TEST_START = champion_metadata["test_period"]["start"]
TEST_END = champion_metadata["test_period"]["end"]

test_df = feature_df[
    (feature_df["date"] >= TEST_START) &
    (feature_df["date"] <= TEST_END)
].copy()

test_prediction_df = generate_predictions(
    input_df=test_df,
    model=champion_model,
    feature_cols=feature_columns
)

test_mae, test_rmse, test_mape, test_r2 = calculate_metrics(
    actual_values=test_prediction_df["actual_unit_sales"],
    predicted_values=test_prediction_df["predicted_unit_sales"]
)


# Sidebar controls
st.sidebar.header("Forecast Controls")

st.sidebar.markdown(
    """
    Select the last known date and forecast horizon.

    The app will simulate a forecast for the next available days after the selected cutoff date.
    """
)

test_dates = test_df["date"].sort_values().reset_index(drop=True)

min_cutoff_date = test_dates.min().date()
max_cutoff_date = (test_dates.max() - pd.Timedelta(days=7)).date()

st.sidebar.markdown(
    f"""
    **Available simulation period:**  
    {test_dates.min().date()} to {test_dates.max().date()}
    """
)

selected_cutoff_date = st.sidebar.date_input(
    "Select cutoff date",
    value=min_cutoff_date,
    min_value=min_cutoff_date,
    max_value=max_cutoff_date,
    key="cutoff_date_input"
)

selected_cutoff_datetime = pd.to_datetime(selected_cutoff_date)

available_days_after_cutoff = test_df[
    test_df["date"] > selected_cutoff_datetime
]["date"].nunique()

max_available_horizon = min(30, available_days_after_cutoff)

if max_available_horizon < 7:
    st.sidebar.warning(
        "Not enough prepared data is available after this cutoff date. "
        "Please select an earlier cutoff date."
    )
    selected_horizon = 0

elif max_available_horizon == 7:
    selected_horizon = 7
    st.sidebar.info(
        "Only 7 forecast days are available after this cutoff date."
    )

else:
    selected_horizon = st.sidebar.slider(
        "Select forecast horizon in days",
        min_value=7,
        max_value=max_available_horizon,
        value=min(14, max_available_horizon),
        step=1,
        key="forecast_horizon_slider"
    )

history_window = st.sidebar.slider(
    "Show recent history before cutoff",
    min_value=7,
    max_value=30,
    value=14,
    step=1,
    key="history_window_slider"
)

show_actuals = st.sidebar.checkbox(
    "Show historical actual sales for forecast period",
    value=False,
    key="show_actuals_checkbox"
)

run_forecast = st.sidebar.button(
    "Generate forecast",
    key="generate_forecast_button"
)


st.sidebar.divider()


# App title and introduction
st.title("Sales Forecast Planning App")

st.markdown(
    """
    This app demonstrates a sales forecasting prototype based on the selected champion model.

    Use the sidebar controls to simulate expected unit sales for the next available days after a selected cutoff date.
    """
)

# Forecast simulation section
st.subheader("Forecast Simulation")

selected_cutoff_date = selected_cutoff_datetime
forecast_start_date = selected_cutoff_date + pd.Timedelta(days=1)
forecast_end_date = selected_cutoff_date + pd.Timedelta(days=selected_horizon)

history_start_date = selected_cutoff_date - pd.Timedelta(days=history_window - 1)

history_context_df = feature_df[
    (feature_df["date"] >= history_start_date) &
    (feature_df["date"] <= selected_cutoff_date)
].copy()

forecast_input_df = test_df[
    (test_df["date"] >= forecast_start_date) &
    (test_df["date"] <= forecast_end_date)
].copy()

if not run_forecast:
    st.info("Use the controls in the sidebar and click 'Generate forecast' to create a forecast simulation.")

else:
    if forecast_input_df.empty:
        st.warning("No feature-engineered data is available after the selected cutoff date.")

    else:
        forecast_df = generate_predictions(
            input_df=forecast_input_df,
            model=champion_model,
            feature_cols=feature_columns
        )

        st.write(
            f"Cutoff date: **{selected_cutoff_date.date()}**"
        )

        st.write(
            f"Forecast period: **{forecast_start_date.date()} to "
            f"{forecast_input_df['date'].max().date()}**"
        )

        st.write(f"Number of forecasted days: **{len(forecast_df)}**")

        total_predicted_sales = forecast_df["predicted_unit_sales"].sum()
        average_daily_forecast = forecast_df["predicted_unit_sales"].mean()
        peak_forecast_row = forecast_df.loc[
            forecast_df["predicted_unit_sales"].idxmax()
        ]
        peak_forecast_date = pd.to_datetime(peak_forecast_row["date"]).date()
        peak_forecast_value = peak_forecast_row["predicted_unit_sales"]

        if average_daily_forecast >= 600:
            demand_level = "High"
        elif average_daily_forecast >= 400:
            demand_level = "Medium"
        else:
            demand_level = "Low"

        kpi_1, kpi_2, kpi_3, kpi_4, kpi_5 = st.columns(5)

        kpi_1.metric(
            "Total Forecasted Sales",
            f"{total_predicted_sales:,.0f}"
        )

        kpi_2.metric(
            "Average Daily Forecast",
            f"{average_daily_forecast:,.0f}"
        )

        kpi_3.metric(
            "Peak Forecast Day",
            str(peak_forecast_date)
        )

        kpi_4.metric(
            "Peak Forecasted Sales",
            f"{peak_forecast_value:,.0f}"
        )

        kpi_5.metric(
            "Forecast Demand Level",
            demand_level
        )

        fig_forecast, ax_forecast = plt.subplots(figsize=(14, 5))

        if not history_context_df.empty:
            ax_forecast.plot(
                history_context_df["date"],
                history_context_df["unit_sales"],
                label="Recent Sales Before Cutoff",
                linewidth=2
            )

            ax_forecast.axvline(
                selected_cutoff_date,
                linestyle=":",
                linewidth=2,
                label="Cutoff Date"
            )

        ax_forecast.plot(
            forecast_df["date"],
            forecast_df["predicted_unit_sales"],
            label="Predicted Sales After Cutoff",
            linewidth=2,
            linestyle="--"
        )

        if show_actuals and "actual_unit_sales" in forecast_df.columns:
            ax_forecast.plot(
                forecast_df["date"],
                forecast_df["actual_unit_sales"],
                label="Actual Sales After Cutoff",
                linewidth=2,
                alpha=0.6
        )

        ax_forecast.set_title("Forecast Simulation — Recent History and Forecast Horizon")
        ax_forecast.set_xlabel("Date")
        ax_forecast.set_ylabel("Unit Sales")
        ax_forecast.legend()
        ax_forecast.grid(True, alpha=0.3)

        st.pyplot(fig_forecast)

        st.markdown("#### Forecast Table")

        if show_actuals and "actual_unit_sales" in forecast_df.columns:
            display_forecast_df = forecast_df.copy()
        else:
            display_forecast_df = forecast_df[
                ["date", "predicted_unit_sales"]
            ].copy()

        display_forecast_df["predicted_unit_sales"] = (
            display_forecast_df["predicted_unit_sales"]
            .round(0)
            .astype(int)
        )

        if "actual_unit_sales" in display_forecast_df.columns:
            display_forecast_df["actual_unit_sales"] = (
                display_forecast_df["actual_unit_sales"]
                .round(0)
                .astype(int)
        )

        st.dataframe(
            display_forecast_df,
            width="stretch"
        )

        csv_data = display_forecast_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download forecast as CSV",
            data=csv_data,
            file_name="forecast_simulation.csv",
            mime="text/csv"
        )

        st.markdown("#### Planning Interpretation")

        st.markdown(
            f"""
            Based on the selected cutoff date, the app simulates a forecast for the next available period.

            For the selected horizon, the model predicts a total of approximately **{total_predicted_sales:,.0f} unit sales**.

            The average expected daily sales level is approximately **{average_daily_forecast:,.0f} units**, which indicates a **{demand_level.lower()} demand level** for the selected period.

            The highest predicted demand occurs on **{peak_forecast_date}**, with approximately **{peak_forecast_value:,.0f} predicted unit sales**.

            This information can help a shop owner or planning team prepare inventory, staffing, and operational capacity for the selected period.
            """
        )


# Technical validation section
with st.expander("Show model validation on the historical test period"):
    st.subheader("Model Validation on Historical Test Period")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("MAE", f"{test_mae:.2f}")
    col2.metric("RMSE", f"{test_rmse:.2f}")
    col3.metric("MAPE", f"{test_mape:.2f}%")
    col4.metric("R²", f"{test_r2:.4f}")

    st.markdown("#### Actual vs Predicted Unit Sales")

    fig_validation, ax_validation = plt.subplots(figsize=(14, 5))

    ax_validation.plot(
        test_prediction_df["date"],
        test_prediction_df["actual_unit_sales"],
        label="Actual Unit Sales",
        linewidth=2
    )

    ax_validation.plot(
        test_prediction_df["date"],
        test_prediction_df["predicted_unit_sales"],
        label="Predicted Unit Sales",
        linewidth=2,
        linestyle="--"
    )

    ax_validation.set_title("Champion Model — Actual vs Predicted Unit Sales")
    ax_validation.set_xlabel("Date")
    ax_validation.set_ylabel("Unit Sales")
    ax_validation.legend()
    ax_validation.grid(True, alpha=0.3)

    st.pyplot(fig_validation)

    st.dataframe(
        test_prediction_df,
        width="stretch"
    )


# Model details section
with st.expander("Show champion model details"):
    st.subheader("Champion Model Summary")

    model_summary = {
        "Champion Model": champion_metadata["champion_model_name"],
        "Main Metric": champion_metadata["main_metric"],
        "Champion RMSE": champion_metadata["champion_rmse"],
        "Train Period": f"{champion_metadata['train_period']['start']} to {champion_metadata['train_period']['end']}",
        "Test Period": f"{champion_metadata['test_period']['start']} to {champion_metadata['test_period']['end']}",
        "Feature Count": champion_metadata["feature_count"],
        "Model File": champion_metadata["model_file"]
    }

    model_summary_df = pd.DataFrame(
        model_summary.items(),
        columns=["Item", "Value"]
    )

    model_summary_df["Value"] = model_summary_df["Value"].astype(str)

    st.dataframe(
        model_summary_df,
        width="stretch"
    )


# Business note
st.info(
    """
    Note: This app is a portfolio forecasting prototype. It simulates forecasts using prepared feature-engineered data from the historical test period. It is not a production demand planning system and does not generate forecasts beyond the available feature-engineered dataset.
    """
)