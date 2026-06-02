"""
app.py
------
SolarCast — Streamlit Dashboard (光伏出力预测系统 可视化平台)

Displays:
  - Data overview: raw time series, distribution plots.
  - Model performance: metrics table and prediction curve (interactive).
  - Feature importance: SHAP bar chart.
  - Sensitivity analysis: marginal effects of irradiation and temperature.

Design language: professional, minimalist. No emojis. Pure text and structure.
"""

import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

MODEL_DIR = os.path.join(BASE_DIR, "outputs", "models")
FIG_DIR = os.path.join(BASE_DIR, "outputs", "figures")

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SolarCast | PV Power Forecasting",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global CSS overrides — minimalist professional style
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Typography */
    html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }

    /* Remove Streamlit header/footer chrome */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }

    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #f7f8fa; border-right: 1px solid #e8eaed; }
    [data-testid="stSidebar"] .stMarkdown h2 { color: #1a1a2e; font-size: 0.85rem; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.08em; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #ffffff;
        border: 1px solid #e8eaed;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    [data-testid="stMetricLabel"] { font-size: 0.75rem; color: #5f6368; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.06em; }
    [data-testid="stMetricValue"] { font-size: 1.6rem; color: #1a1a2e; font-weight: 700; }

    /* Section headers */
    h1 { font-size: 1.5rem; font-weight: 700; color: #1a1a2e; margin-bottom: 0; }
    h2 { font-size: 1.1rem; font-weight: 600; color: #1a1a2e;
        border-bottom: 2px solid #e94560; padding-bottom: 0.3rem; margin-top: 1.5rem; }
    h3 { font-size: 0.95rem; font-weight: 600; color: #5f6368; }

    /* Divider */
    hr { border: none; border-top: 1px solid #e8eaed; margin: 1.2rem 0; }

    /* Table */
    .stDataFrame { border: 1px solid #e8eaed; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_features():
    path = os.path.join(MODEL_DIR, "df_features.parquet")
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_metrics():
    path = os.path.join(MODEL_DIR, "metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_resource(show_spinner=False)
def load_lgbm():
    path = os.path.join(MODEL_DIR, "lgbm_model.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner=False)
def load_scaler():
    path = os.path.join(MODEL_DIR, "scaler.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------
PALETTE = {"actual": "#1a1a2e", "lgbm": "#e94560", "lstm": "#0f3460"}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="white",
    plot_bgcolor="white",
    font=dict(family="Inter, Segoe UI, sans-serif", color="#1a1a2e", size=12),
    margin=dict(l=50, r=30, t=50, b=40),
    xaxis=dict(showgrid=True, gridcolor="#f0f0f0", linecolor="#e8eaed", showline=True),
    yaxis=dict(showgrid=True, gridcolor="#f0f0f0", linecolor="#e8eaed", showline=True),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"),
)


def fig_time_series(df, date_range=None):
    """Interactive actual AC_POWER time series."""
    if date_range:
        mask = (df["DATE_TIME"] >= pd.Timestamp(date_range[0])) & \
               (df["DATE_TIME"] <= pd.Timestamp(date_range[1]))
        df = df[mask]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["DATE_TIME"], y=df["AC_POWER"],
        mode="lines", line=dict(color=PALETTE["actual"], width=1),
        name="AC Power",
    ))
    fig.add_trace(go.Scatter(
        x=df["DATE_TIME"], y=df["IRRADIATION"] * df["AC_POWER"].max() / max(df["IRRADIATION"].max(), 1e-9),
        mode="lines", line=dict(color="#f5a623", width=0.8, dash="dot"),
        name="Irradiation (scaled)", yaxis="y2", opacity=0.6,
    ))
    fig.update_layout(PLOTLY_LAYOUT)
    fig.update_layout(
        title="AC Power Output and Irradiation — Plant 1",
        yaxis=dict(title="AC Power (kW)"),
        yaxis2=dict(overlaying="y", side="right", title="Irradiation (scaled)",
                    showgrid=False, linecolor="#e8eaed", showline=True),
        legend=dict(orientation="h", y=1.08),
        height=320,
    )
    return fig


def fig_prediction_overlay(df, model_name: str):
    """
    Re-generate predictions on the test split and plot overlay.
    Requires models to be loaded.
    """
    from src.data_processing import FEATURE_COLS, TARGET_COL
    import torch
    from src.models import LSTMForecaster, SequenceDataset
    from torch.utils.data import DataLoader

    scaler = load_scaler()
    lgbm = load_lgbm()
    if scaler is None or lgbm is None:
        return None

    n = len(df)
    test_start = int(n * 0.70) + int(n * 0.15)
    df_test = df.iloc[test_start:].copy()

    X_test = scaler.transform(df_test[FEATURE_COLS].values)
    y_test = df_test[TARGET_COL].values
    dates = df_test["DATE_TIME"].values

    if model_name == "LightGBM":
        y_pred = lgbm.predict(X_test)
    else:
        lstm_path = os.path.join(MODEL_DIR, "lstm_model.pt")
        if not os.path.exists(lstm_path):
            return None
        ckpt = torch.load(lstm_path, map_location="cpu")
        seq_len = ckpt["seq_len"]
        input_size = ckpt["input_size"]
        y_scaler = ckpt.get("y_scaler")
        model = LSTMForecaster(input_size=input_size)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        
        y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).flatten() if y_scaler else y_test
        ds = SequenceDataset(X_test, y_test_scaled, seq_len=seq_len)
        loader = DataLoader(ds, batch_size=128, shuffle=False)
        preds = []
        with torch.no_grad():
            for xb, _ in loader:
                preds.append(model(xb).numpy())
        y_pred = np.concatenate(preds)
        if y_scaler:
            y_pred = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
            y_pred = np.clip(y_pred, 0, None)
            
        y_test = y_test[seq_len:]
        dates = dates[seq_len:]

    color = PALETTE["lgbm"] if model_name == "LightGBM" else PALETTE["lstm"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=y_test, mode="lines",
                             line=dict(color=PALETTE["actual"], width=1), name="Actual"))
    fig.add_trace(go.Scatter(x=dates, y=y_pred, mode="lines",
                             line=dict(color=color, width=1, dash="dash"), name=model_name))
    fig.update_layout(PLOTLY_LAYOUT)
    fig.update_layout(
        title=f"Actual vs. Predicted AC Power — {model_name} (Test Set)",
        yaxis=dict(title="AC Power (kW)"),
        xaxis_title="Date",
        legend=dict(orientation="h", y=1.08),
        height=340,
    )
    return fig


def fig_shap_bar():
    path = os.path.join(FIG_DIR, "shap_importance.png")
    if not os.path.exists(path):
        return None
    return path


def fig_image(filename):
    path = os.path.join(FIG_DIR, filename)
    if os.path.exists(path):
        return path
    return None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## SolarCast")
    st.markdown(
        "**Photovoltaic Power Forecasting System**  \n"
        "Plant 1 · Kaggle Solar Generation Dataset"
    )
    st.markdown("---")

    st.markdown("## Navigation")
    section = st.radio(
        label="Section",
        options=["Data Overview", "Model Performance", "Feature Analysis"],
        label_visibility="collapsed",
    )
    st.markdown("---")

    df_full = load_features()
    if df_full is not None:
        min_date = df_full["DATE_TIME"].min().date()
        max_date = df_full["DATE_TIME"].max().date()
        st.markdown("## Date Filter")
        date_range = st.date_input(
            "Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            label_visibility="collapsed",
        )
    else:
        date_range = None

    st.markdown("---")
    st.markdown("## Model")
    model_choice = st.selectbox(
        "Select model for prediction curve",
        ["LightGBM", "LSTM"],
        label_visibility="collapsed",
    )

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
col_title, col_status = st.columns([6, 2])
with col_title:
    st.markdown("# SolarCast — Photovoltaic Power Forecasting")
    st.markdown(
        "A time-series prediction system built on LightGBM and LSTM, "
        "integrating SHAP-based interpretability for rigorous model evaluation."
    )
st.markdown("---")

# ---------------------------------------------------------------------------
# Section: Data Overview
# ---------------------------------------------------------------------------
if section == "Data Overview":
    df_full = load_features()
    if df_full is None:
        st.warning("Dataset not found. Please run `python src/train.py` first.")
        st.stop()

    st.markdown("## Data Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{len(df_full):,}")
    c2.metric("Date Range", f"{df_full['DATE_TIME'].dt.date.min()} to {df_full['DATE_TIME'].dt.date.max()}")
    c3.metric("Daytime Records", f"{df_full['is_daytime'].sum():,}")
    c4.metric("Features Engineered", str(len([c for c in df_full.columns if c not in ["DATE_TIME", "AC_POWER", "DC_POWER", "DAILY_YIELD", "TOTAL_YIELD", "is_night", "is_anomalous"]])))

    st.markdown("### AC Power Time Series")
    dr = date_range if isinstance(date_range, (list, tuple)) and len(date_range) == 2 else None
    st.plotly_chart(fig_time_series(df_full, dr), width='stretch')

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### AC Power Distribution")
        fig_hist = px.histogram(
            df_full[df_full["is_daytime"] == 1], x="AC_POWER", nbins=60,
            color_discrete_sequence=[PALETTE["lgbm"]],
            title="AC Power Distribution (Daytime Only)",
        )
        fig_hist.update_layout(**PLOTLY_LAYOUT, height=280)
        st.plotly_chart(fig_hist, width='stretch')

    with col_b:
        st.markdown("### Irradiation vs AC Power")
        fig_scatter = px.scatter(
            df_full[df_full["is_daytime"] == 1].sample(min(1000, len(df_full))),
            x="IRRADIATION", y="AC_POWER",
            color="MODULE_TEMPERATURE",
            color_continuous_scale="RdYlBu_r",
            title="Irradiation vs AC Power (coloured by Module Temperature)",
        )
        fig_scatter.update_traces(marker=dict(size=4, opacity=0.6))
        fig_scatter.update_layout(**PLOTLY_LAYOUT, height=280)
        st.plotly_chart(fig_scatter, width='stretch')

    st.markdown("### Data Sample (first 20 rows)")
    display_cols = ["DATE_TIME", "AC_POWER", "DC_POWER", "AMBIENT_TEMPERATURE",
                    "MODULE_TEMPERATURE", "IRRADIATION", "is_daytime"]
    st.dataframe(df_full[display_cols].head(20), width='stretch')

# ---------------------------------------------------------------------------
# Section: Model Performance
# ---------------------------------------------------------------------------
elif section == "Model Performance":
    st.markdown("## Model Performance")

    metrics = load_metrics()
    if metrics is None:
        st.warning("Metrics not found. Please run `python src/train.py` first.")
        st.stop()

    # --- Metrics table ---
    df_metrics = pd.DataFrame(metrics).T.reset_index().rename(columns={"index": "Model"})
    df_metrics.columns = ["Model", "MAE", "RMSE", "MAPE (%)", "R²"]

    st.markdown("### Test Set Evaluation Metrics")
    c1, c2, c3, c4 = st.columns(4)
    lgbm_m = metrics.get("LightGBM", {})
    lstm_m = metrics.get("LSTM", {})

    c1.metric("LightGBM MAE", f"{lgbm_m.get('MAE', '-'):.3f} kW",
              delta=f"LSTM: {lstm_m.get('MAE', '-'):.3f}")
    c2.metric("LightGBM RMSE", f"{lgbm_m.get('RMSE', '-'):.3f} kW",
              delta=f"LSTM: {lstm_m.get('RMSE', '-'):.3f}")
    c3.metric("LightGBM MAPE", f"{lgbm_m.get('MAPE', '-'):.2f}%",
              delta=f"LSTM: {lstm_m.get('MAPE', '-'):.2f}%")
    c4.metric("LightGBM R²", f"{lgbm_m.get('R2', '-'):.4f}",
              delta=f"LSTM: {lstm_m.get('R2', '-'):.4f}")

    st.markdown("")
    st.dataframe(df_metrics.set_index("Model"), width='stretch')

    # --- Prediction curve ---
    st.markdown(f"### Prediction Curve — {model_choice}")
    df_full = load_features()
    if df_full is not None:
        try:
            pfig = fig_prediction_overlay(df_full, model_choice)
            if pfig:
                st.plotly_chart(pfig, width='stretch')
        except Exception as e:
            st.info(f"Interactive chart unavailable. Loading static image. ({e})")
            fname = "lgbm_prediction_curve.png" if model_choice == "LightGBM" else "lstm_prediction_curve.png"
            img = fig_image(fname)
            if img:
                st.image(img, use_column_width=True)

    # --- Comparison bar chart ---
    st.markdown("### Comparison — All Metrics")
    img_cmp = fig_image("metrics_comparison.png")
    img_overlay = fig_image("comparison_overlay.png")
    cA, cB = st.columns(2)
    if img_cmp:
        cA.image(img_cmp, use_column_width=True)
    if img_overlay:
        cB.image(img_overlay, use_column_width=True)

    # --- Scatter plots ---
    st.markdown("### Scatter — Actual vs. Predicted")
    imgA = fig_image("lgbm_scatter.png")
    imgB = fig_image("lstm_scatter.png")
    sA, sB = st.columns(2)
    if imgA:
        sA.image(imgA, caption="LightGBM", use_column_width=True)
    if imgB:
        sB.image(imgB, caption="LSTM", use_column_width=True)

    # --- LSTM loss curve ---
    st.markdown("### LSTM Training Dynamics")
    img_loss = fig_image("lstm_loss_curve.png")
    if img_loss:
        st.image(img_loss, use_column_width=True)

# ---------------------------------------------------------------------------
# Section: Feature Analysis
# ---------------------------------------------------------------------------
elif section == "Feature Analysis":
    st.markdown("## Feature Analysis and Interpretability")

    st.markdown("### SHAP Feature Importance (LightGBM)")
    shap_bar = fig_image("shap_importance.png")
    shap_bee = fig_image("shap_beeswarm.png")

    colA, colB = st.columns(2)
    if shap_bar:
        colA.image(shap_bar, caption="Mean |SHAP| per feature", use_column_width=True)
    else:
        colA.info("SHAP bar chart not found. Run train.py to generate.")
    if shap_bee:
        colB.image(shap_bee, caption="SHAP Beeswarm Summary", use_column_width=True)

    st.markdown("### Sensitivity Analysis — Marginal Effects")
    st.markdown(
        "The following plots show the isolated marginal effect of each environmental variable "
        "on AC Power prediction, with all other features held at their median value "
        "(ceteris paribus analysis using the LightGBM model)."
    )

    imgI = fig_image("irradiation_sensitivity.png")
    imgT = fig_image("temperature_sensitivity.png")
    cI, cT = st.columns(2)
    if imgI:
        cI.image(imgI, caption="Marginal Effect — Irradiation", use_column_width=True)
    if imgT:
        cT.image(imgT, caption="Marginal Effect — Module Temperature", use_column_width=True)

    if not any([shap_bar, shap_bee, imgI, imgT]):
        st.warning("No analysis figures found. Please run `python src/train.py` first.")
