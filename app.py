# Core libraries
import os
import json
import joblib

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# Page configuration
st.set_page_config(
    page_title="FreshMart Demand Planner",
    page_icon="🛒",
    layout="wide"
)

st.markdown("""
<style>
.metric-card {
    background-color: #161b22;
    border: 1px solid #263238;
    border-radius: 14px;
    text-align: center;
    padding: 0.85rem 0.4rem;
    min-height: 105px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.18);
}

.metric-label {
    font-size: 0.82rem;
    font-weight: 600;
    color: #b7c7bd;
    margin-bottom: 0.45rem;
}

.metric-value {
    font-size: 1.9rem;
    font-weight: 700;
    color: #ffffff;
}

.use-case-card {
    background-color: #123524;
    border-left: 5px solid #2ecc71;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin: 1.2rem 0 1.5rem 0;
}

.use-case-card h4 {
    margin-top: 0;
    margin-bottom: 0.4rem;
    color: #ffffff;
}

.use-case-card p {
    margin-bottom: 0;
    color: #e8f5e9;
    font-size: 1rem;
    line-height: 1.5;
}
            
/* Sidebar polish */
section[data-testid="stSidebar"] {
    background-color: #1f242d;
    border-right: 1px solid #2f3b35;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #ffffff;
}

section[data-testid="stSidebar"] label {
    color: #e8f5e9;
    font-weight: 600;
}

/* Primary button styling */
div[data-testid="stButton"] > button {
    background-color: #2ecc71;
    color: #0b1117;
    border: none;
    border-radius: 10px;
    font-weight: 700;
    padding: 0.65rem 1.2rem;
    width: auto;
}

div[data-testid="stButton"] > button:hover {
    background-color: #27ae60;
    color: #ffffff;
    border: none;
}

.footer-card {
    margin-top: 2rem;
    padding: 1rem 1.2rem;
    border-top: 1px solid #263238;
    color: #b7c7bd;
    font-size: 0.9rem;
    line-height: 1.5;
}

.footer-card strong {
    color: #ffffff;
}

.forecast-table-container {
    width: 100%;
    max-width: 760px;
    margin-top: 0.5rem;
    margin-bottom: 1.2rem;
}

.forecast-table {
    width: 100%;
    border-collapse: collapse;
    background-color: #0e1117;
    border: 1px solid #263238;
    border-radius: 10px;
    overflow: hidden;
    font-size: 0.95rem;
}

.forecast-table th {
    background-color: #1c2129;
    color: #b7c7bd;
    font-weight: 700;
    text-align: center;
    padding: 0.65rem;
    border: 1px solid #263238;
}

.forecast-table td {
    color: #ffffff;
    text-align: center;
    padding: 0.65rem;
    border: 1px solid #263238;
    font-weight: 600;
}

@media (max-width: 900px) {
    .forecast-table-container {
        max-width: 100%;
    }

    .forecast-table {
        font-size: 0.85rem;
    }

    .forecast-table th,
    .forecast-table td {
        padding: 0.5rem;
    }
}

</style>
""", unsafe_allow_html=True)


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
    required_base_cols = ["date"]

    missing_base_cols = [
        col for col in required_base_cols
        if col not in input_df.columns
    ]

    if missing_base_cols:
        raise ValueError(f"Missing required base columns: {missing_base_cols}")

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
    actual_values = np.asarray(actual_values)
    predicted_values = np.asarray(predicted_values)

    mae = np.mean(np.abs(actual_values - predicted_values))

    rmse = np.sqrt(
        np.mean((actual_values - predicted_values) ** 2)
    )

    non_zero_mask = actual_values != 0

    if non_zero_mask.sum() == 0:
        mape = np.nan
    else:
        mape = np.mean(
            np.abs(
                (actual_values[non_zero_mask] - predicted_values[non_zero_mask])
                / actual_values[non_zero_mask]
            )
        ) * 100

    denominator = np.sum((actual_values - actual_values.mean()) ** 2)

    if denominator == 0:
        r2 = np.nan
    else:
        r2 = 1 - (
            np.sum((actual_values - predicted_values) ** 2)
            / denominator
        )

    return mae, rmse, mape, r2


def calculate_demand_level(selected_average_forecast, reference_predictions, horizon):
    """
    Classify demand level by comparing the selected period average forecast
    with rolling average forecasts of the same horizon.
    """
    rolling_averages = (
        reference_predictions["predicted_unit_sales"]
        .rolling(window=horizon)
        .mean()
        .dropna()
    )

    if rolling_averages.empty:
        return "Medium"

    low_threshold = rolling_averages.quantile(0.33)
    high_threshold = rolling_averages.quantile(0.66)

    if selected_average_forecast >= high_threshold:
        return "High"
    elif selected_average_forecast >= low_threshold:
        return "Medium"
    else:
        return "Low"    


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
    **Prepared data period:**  
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
        "Selectable forecast horizon after cutoff: 7 days."
    )

else:
    st.sidebar.info(
        f"Selectable forecast horizon after cutoff: 7 to {max_available_horizon} days."
    )
    
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

button_left, button_center, button_right = st.sidebar.columns([0.35, 1.3, 0.35])

with button_center:
    run_forecast = st.button(
        "Generate forecast",
        key="generate_forecast_button"
    )


st.sidebar.divider()


# App title and introduction
st.title("FreshMart Demand Planner")

st.markdown(
    """
    An interactive forecasting prototype for short-term grocery sales planning.
    """
)

st.markdown(
    """
    <div class="use-case-card">
        <h4>Business Use Case</h4>
        <p>
            FreshMart Demand Planner is designed for a small grocery store manager who needs to plan
            short-term sales demand. The forecast helps estimate expected sales, identify high-demand
            days, and support inventory, shelf replenishment, and staffing decisions.
        </p>
    </div>
    """,
    unsafe_allow_html=True
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

        demand_level = calculate_demand_level(
            selected_average_forecast=average_daily_forecast,
            reference_predictions=test_prediction_df,
            horizon=len(forecast_df)
        )

        kpi_1, kpi_2, kpi_3, kpi_4, kpi_5 = st.columns(5)

        with kpi_1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Total Forecasted Sales</div>
                    <div class="metric-value">{total_predicted_sales:,.0f}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with kpi_2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Average Daily Forecast</div>
                    <div class="metric-value">{average_daily_forecast:,.0f}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with kpi_3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Peak Forecast Day</div>
                    <div class="metric-value">{peak_forecast_date}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with kpi_4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Peak Forecasted Sales</div>
                    <div class="metric-value">{peak_forecast_value:,.0f}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with kpi_5:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Relative Demand Level</div>
                    <div class="metric-value">{demand_level}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<div style='margin-bottom: 0.8rem;'></div>", unsafe_allow_html=True)

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
                color="#8e44ad",
                linestyle=":",
                linewidth=2,
                label="Cutoff Date"
            )

            ax_forecast.axvline(
                forecast_start_date,
                color="#c0392b",
                linestyle=":",
                linewidth=2,
                label="Forecast Start"
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

        # Format x-axis dates dynamically
        plot_start_date = history_context_df["date"].min()
        plot_end_date = forecast_df["date"].max()

        total_days_on_plot = len(
            pd.date_range(
                start=plot_start_date,
                end=plot_end_date,
                freq="D"
            )
        )

        if total_days_on_plot <= 15:
            tick_interval = 1
        elif total_days_on_plot <= 30:
            tick_interval = 2
        else:
            tick_interval = 3

        ax_forecast.xaxis.set_major_locator(
            mdates.DayLocator(interval=tick_interval)
        )

        ax_forecast.xaxis.set_major_formatter(
            mdates.DateFormatter("%Y-%m-%d")
        )

        plt.setp(
            ax_forecast.get_xticklabels(),
            rotation=45,
            ha="right"
        )

        plt.tight_layout()

        st.pyplot(fig_forecast)

        st.markdown("### Forecast Table")

        display_forecast_df = forecast_df.copy()

        display_forecast_df["date"] = pd.to_datetime(
            display_forecast_df["date"]
        ).dt.strftime("%Y-%m-%d")

        display_forecast_df["predicted_unit_sales"] = (
            display_forecast_df["predicted_unit_sales"]
            .round(0)
            .astype(int)
        )

        if show_actuals and "actual_unit_sales" in display_forecast_df.columns:
            display_forecast_df["actual_unit_sales"] = (
                display_forecast_df["actual_unit_sales"]
                .round(0)
                .astype(int)
            )

            display_forecast_df = display_forecast_df[
                ["date", "actual_unit_sales", "predicted_unit_sales"]
            ]
        else:
            display_forecast_df = display_forecast_df[
                ["date", "predicted_unit_sales"]
            ]

        forecast_table_html = display_forecast_df.to_html(
            index=False,
            classes="forecast-table",
            border=0
        )

        st.markdown(
            f"""
            <div class="forecast-table-container">
                {forecast_table_html}
            </div>
            """,
            unsafe_allow_html=True
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

            The average expected daily sales level is approximately **{average_daily_forecast:,.0f} units**, which indicates a **{demand_level.lower()} relative demand level** for the selected period.

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

    ax_validation.set_title(
        f"{champion_metadata['champion_model_name']} — Actual vs Predicted Unit Sales"
    )
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
        "Champion RMSE": f"{champion_metadata['champion_rmse']:.2f}",
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
    Note: This app is a portfolio forecasting prototype. It simulates short-term forecasts using prepared feature-engineered rows from the historical test period. It is not a production demand planning system and does not generate forecasts beyond the available feature-engineered dataset.
    """
)


st.markdown(
    f"""
    <div class="footer-card">
        <strong>FreshMart Demand Planner</strong><br>
        Prototype purpose: short-term grocery sales planning<br>
        Model: {champion_metadata["champion_model_name"]}<br>
        Built by Sergey Kasatov as part of the Time Series Forecasting project
    </div>
    """,
    unsafe_allow_html=True
)