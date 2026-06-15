"""
app.py
------
SolarCast — Streamlit Dashboard (光伏出力预测系统 可视化平台)
Improved Version 2.0

Displays:
  - Data overview: raw time series, distribution plots, EDA.
  - Model performance: metrics table, prediction curves (interactive).
  - Feature analysis: SHAP bar chart, beeswarm.
  - Sensitivity analysis: marginal effects of irradiation and temperature.
  - Probabilistic prediction: prediction intervals.
  - Multi-step forecasting results.

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

# Import feature column definitions directly for accuracy
from src.data_processing import FEATURE_COLS, TARGET_COL

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SolarCast | 光伏出力预测系统",
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
def load_features():
    path = os.path.join(MODEL_DIR, "df_features.parquet")
    if not os.path.exists(path):
        st.warning(f"诊断信息：路径 {path} 不存在。")
        if os.path.exists(MODEL_DIR):
            st.warning(f"当前 outputs/models 目录下包含的文件：{os.listdir(MODEL_DIR)}")
        else:
            st.warning("outputs/models 目录不存在，请确认代码是否推送成功或在 Streamlit 中 Reboot 重新构建。")
        return None
    return pd.read_parquet(path)


def load_metrics():
    path = os.path.join(MODEL_DIR, "metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_train_info():
    path = os.path.join(MODEL_DIR, "train_info.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_lgbm():
    path = os.path.join(MODEL_DIR, "lgbm_model.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


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
        name="AC 功率",
    ))
    fig.add_trace(go.Scatter(
        x=df["DATE_TIME"], y=df["IRRADIATION"] * df["AC_POWER"].max() / max(df["IRRADIATION"].max(), 1e-9),
        mode="lines", line=dict(color="#f5a623", width=0.8, dash="dot"),
        name="辐照度（已归一化）", yaxis="y2", opacity=0.6,
    ))
    fig.update_layout(PLOTLY_LAYOUT)
    fig.update_layout(
        title="AC 功率输出与辐照度 — 电站 1",
        yaxis=dict(title="AC 功率 (kW)"),
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
    from src.models import LSTMForecaster, SequenceDataset
    from torch.utils.data import DataLoader
    import torch

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
    elif model_name == "LSTM":
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
    else:
        return None

    color = PALETTE["lgbm"] if model_name == "LightGBM" else PALETTE["lstm"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=y_test, mode="lines",
                             line=dict(color=PALETTE["actual"], width=1), name="Actual"))
    fig.add_trace(go.Scatter(x=dates, y=y_pred, mode="lines",
                             line=dict(color=color, width=1, dash="dash"), name=model_name))
    fig.update_layout(PLOTLY_LAYOUT)
    fig.update_layout(
        title=f"真实值 vs 预测值 AC 功率 — {model_name}（测试集）",
        yaxis=dict(title="AC 功率 (kW)"),
        xaxis_title="日期",
        legend=dict(orientation="h", y=1.08),
        height=340,
    )
    return fig


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
        "**光伏出力预测系统 v2.0**  \n"
        "Kaggle 光伏数据集 · 电站 1"
    )
    st.markdown("---")

    st.markdown("## 导航")
    section = st.radio(
        label="Section",
        options=["数据概览", "模型性能", "特征分析", "高级分析"],
        label_visibility="collapsed",
    )
    st.markdown("---")

    df_full = load_features()
    if df_full is not None:
        min_date = df_full["DATE_TIME"].min().date()
        max_date = df_full["DATE_TIME"].max().date()
        st.markdown("## 日期筛选")
        date_range = st.date_input(
            "Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            label_visibility="collapsed",
        )
    else:
        date_range = None

    if section == "模型性能":
        st.markdown("---")
        st.markdown("## 模型")
        model_choice = st.selectbox(
            "Select model for prediction curve",
            ["LightGBM", "LSTM"],
            label_visibility="collapsed",
        )
    else:
        model_choice = "LightGBM"

    st.markdown("---")
    # Show training info if available
    train_info = load_train_info()
    if train_info:
        st.markdown("## 训练信息")
        st.markdown(f"训练时间: {train_info.get('training_time_s', '-')} 秒")
        st.markdown(f"数据集规模: {train_info.get('dataset_size', '-')} 条记录")
        st.markdown(f"特征数量: {train_info.get('feature_count', '-')}")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
col_title, col_status = st.columns([6, 2])
with col_title:
    st.markdown("# SolarCast — 光伏出力预测系统")
    st.markdown(
        "一个完整的时间序列预测系统，采用 LightGBM、LSTM、Seq2Seq 多步预测、概率预测以及基于 SHAP 的可解释性分析。改进版本修复了 LSTM 架构问题，并扩展特征集至 22 维。"
    )
st.markdown("---")

# ---------------------------------------------------------------------------
# Section: Data Overview
# ---------------------------------------------------------------------------
if section == "数据概览":
    df_full = load_features()
    if df_full is None:
        st.warning("Dataset not found. Please run `python src/train.py` first.")
        st.stop()

    st.markdown("## 数据概览")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("总记录数", f"{len(df_full):,}")
    c2.metric("日期范围", f"{df_full['DATE_TIME'].dt.date.min()} 至 {df_full['DATE_TIME'].dt.date.max()}")
    c3.metric("白天记录", f"{df_full['is_daytime'].sum():,}")
    c4.metric("特征数量", str(len(FEATURE_COLS)))

    st.markdown("### AC 功率时间序列")
    dr = date_range if isinstance(date_range, (list, tuple)) and len(date_range) == 2 else None
    st.plotly_chart(fig_time_series(df_full, dr), width='stretch')

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### AC 功率分布")
        fig_hist = px.histogram(
            df_full[df_full["is_daytime"] == 1], x="AC_POWER", nbins=60,
            color_discrete_sequence=[PALETTE["lgbm"]],
            title="AC 功率分布（仅白天）",
        )
        fig_hist.update_layout(**PLOTLY_LAYOUT, height=280)
        st.plotly_chart(fig_hist, width='stretch')

    with col_b:
        st.markdown("### 辐照度与 AC 功率关系")
        fig_scatter = px.scatter(
            df_full[df_full["is_daytime"] == 1].sample(min(1000, len(df_full))),
            x="IRRADIATION", y="AC_POWER",
            color="MODULE_TEMPERATURE",
            color_continuous_scale="RdYlBu_r",
            title="辐照度与 AC 功率（颜色：组件温度）",
        )
        fig_scatter.update_traces(marker=dict(size=4, opacity=0.6))
        fig_scatter.update_layout(**PLOTLY_LAYOUT, height=280)
        st.plotly_chart(fig_scatter, width='stretch')

    # EDA images
    st.markdown("### 探索性数据分析")
    eda_cols = st.columns(3)
    eda_imgs = [
        ("eda_power_distribution.png", "白天 vs 夜间功率分布"),
        ("eda_correlation_heatmap.png", "特征相关性热力图"),
        ("eda_anomaly_detection.png", "异常检测"),
    ]
    for i, (fname, caption) in enumerate(eda_imgs):
        img_path = fig_image(fname)
        if img_path:
            eda_cols[i].image(img_path, caption=caption, use_column_width=True)

    st.markdown("### 数据样本（前 20 条）")
    display_cols = ["DATE_TIME", "AC_POWER", "DC_POWER", "AMBIENT_TEMPERATURE",
                    "MODULE_TEMPERATURE", "IRRADIATION", "is_daytime"]
    st.dataframe(df_full[display_cols].head(20), width='stretch')

# ---------------------------------------------------------------------------
# Section: Model Performance
# ---------------------------------------------------------------------------
elif section == "模型性能":
    st.markdown("## 模型性能")

    metrics = load_metrics()
    if metrics is None:
        st.warning("指标未找到，请先运行 `python src/train.py`。")
        st.stop()

    # --- Metrics table ---
    st.markdown("### 测试集评估指标")

    # Extract main models
    lgbm_m = metrics.get("LightGBM", {})
    lstm_m = metrics.get("LSTM", {})
    mc_lstm_m = metrics.get("MC_Dropout_LSTM", {})

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("LightGBM MAE", f"{lgbm_m.get('MAE', '-'):.3f} kW",
              delta=f"LSTM: {lstm_m.get('MAE', '-'):.3f}")
    c2.metric("LightGBM RMSE", f"{lgbm_m.get('RMSE', '-'):.3f} kW",
              delta=f"LSTM: {lstm_m.get('RMSE', '-'):.3f}")
    c3.metric("LightGBM MAPE", f"{lgbm_m.get('MAPE', '-'):.2f}%",
              delta=f"LSTM: {lstm_m.get('MAPE', '-'):.2f}%")
    c4.metric("LightGBM R²", f"{lgbm_m.get('R2', '-'):.4f}",
              delta=f"LSTM: {lstm_m.get('R2', '-'):.4f}")

    # Build full metrics table
    table_data = []
    for name, m in metrics.items():
        if isinstance(m, dict) and "MAE" in m:
            table_data.append({
                "模型": name,
                "MAE (kW)": f"{m['MAE']:.3f}",
                "RMSE (kW)": f"{m['RMSE']:.3f}",
                "MAPE (%)": f"{m['MAPE']:.2f}",
                "R²": f"{m['R2']:.4f}",
            })
    if table_data:
        st.dataframe(pd.DataFrame(table_data).set_index("模型"), width='stretch')

    # --- Prediction curve ---
    st.markdown(f"### 预测曲线 — {model_choice}")
    df_full = load_features()
    if df_full is not None:
        try:
            pfig = fig_prediction_overlay(df_full, model_choice)
            if pfig:
                st.plotly_chart(pfig, width='stretch')
        except Exception as e:
            st.info(f"交互式图表不可用，加载静态图片。({e})")
            fname = "lgbm_prediction_curve.png" if model_choice == "LightGBM" else "lstm_prediction_curve.png"
            img = fig_image(fname)
            if img:
                st.image(img, use_column_width=True)

    # --- Comparison bar chart ---
    st.markdown("### 对比 — 所有指标")
    img_cmp = fig_image("metrics_comparison.png")
    img_overlay = fig_image("comparison_overlay.png")
    cA, cB = st.columns(2)
    if img_cmp:
        cA.image(img_cmp, use_column_width=True)
    if img_overlay:
        cB.image(img_overlay, use_column_width=True)

    # --- Scatter plots ---
    st.markdown("### 散点图 — 真实值 vs 预测值")
    imgA = fig_image("lgbm_scatter.png")
    imgB = fig_image("lstm_scatter.png")
    sA, sB = st.columns(2)
    if imgA:
        sA.image(imgA, caption="LightGBM", use_column_width=True)
    if imgB:
        sB.image(imgB, caption="LSTM", use_column_width=True)

    # --- LSTM loss curve ---
    st.markdown("### LSTM 训练过程")
    img_loss = fig_image("lstm_loss_curve.png")
    if img_loss:
        st.image(img_loss, use_column_width=True)

    # --- Error analysis ---
    st.markdown("### LSTM 误差分析")
    img_err = fig_image("error_analysis_lstm.png")
    if img_err:
        st.image(img_err, use_column_width=True)

# ---------------------------------------------------------------------------
# Section: Feature Analysis
# ---------------------------------------------------------------------------
elif section == "特征分析":
    st.markdown("## 特征分析与可解释性")

    st.markdown("### SHAP 特征重要性（LightGBM）")
    shap_bar = fig_image("shap_importance.png")
    shap_bee = fig_image("shap_beeswarm.png")

    colA, colB = st.columns(2)
    if shap_bar:
        colA.image(shap_bar, caption="各特征平均 |SHAP| 值", use_column_width=True)
    else:
        colA.info("SHAP 条形图未找到，请运行 train.py 生成。")
    if shap_bee:
        colB.image(shap_bee, caption="SHAP Beeswarm 汇总", use_column_width=True)

    st.markdown("### 敏感性分析 — 边际效应")
    st.markdown(
        "下图展示了在其他特征固定在中位数时，各个环境变量对 AC 功率预测的独立边际效应"
        "（使用 LightGBM 模型的 ceteris paribus 分析）。"
    )

    imgI = fig_image("irradiation_sensitivity.png")
    imgT = fig_image("temperature_sensitivity.png")
    cI, cT = st.columns(2)
    if imgI:
        cI.image(imgI, caption="边际效应 — 辐照度", use_column_width=True)
    if imgT:
        cT.image(imgT, caption="边际效应 — 组件温度", use_column_width=True)

    st.markdown("### 超参数调优")
    hp_cols = st.columns(3)
    hp_imgs = [
        ("hyperparam_num_leaves.png", "num_leaves 影响"),
        ("hyperparam_learning_rate.png", "learning_rate 影响"),
        ("hyperparam_subsample.png", "subsample 影响"),
    ]
    for i, (fname, caption) in enumerate(hp_imgs):
        img = fig_image(fname)
        if img:
            hp_cols[i].image(img, caption=caption, use_column_width=True)

    st.markdown("### 特征消融实验")
    img_ab = fig_image("ablation_study.png")
    if img_ab:
        st.image(img_ab, caption="按特征组分的 R² 分数", use_column_width=True)

    if not any([shap_bar, shap_bee, imgI, imgT]):
        st.warning("未找到分析图表，请先运行 `python src/train.py`。")

# ---------------------------------------------------------------------------
# Section: Advanced Analysis
# ---------------------------------------------------------------------------
elif section == "高级分析":
    st.markdown("## 高级分析")

    st.markdown("### 天气条件分段评估")
    st.markdown("按辐照度变异性划分的天气条件下的模型性能细分。")
    img_w = fig_image("weather_segmentation.png")
    if img_w:
        st.image(img_w, use_column_width=True)
    else:
        st.info("天气分段图未找到，请运行 train.py 生成。")

    st.markdown("### 多步预测（Seq2Seq LSTM）")
    st.markdown("预测未来 1 小时（4 个 15 分钟步长）。")
    img_ms = fig_image("multi_step_prediction.png")
    if img_ms:
        st.image(img_ms, use_column_width=True)
    else:
        st.info("多步预测图未找到，请运行 train.py 生成。")

    st.markdown("### 概率预测")
    st.markdown("用于不确定性量化的预测区间。")

    prob_cols = st.columns(2)
    img_q = fig_image("lgbm_quantile_prediction.png")
    img_mc = fig_image("mc_dropout_prediction.png")
    if img_q:
        prob_cols[0].image(img_q, caption="LightGBM 分位数回归（80% 置信区间）", use_column_width=True)
    else:
        prob_cols[0].info("分位数预测图未找到。")
    if img_mc:
        prob_cols[1].image(img_mc, caption="MC Dropout LSTM（95% 置信区间）", use_column_width=True)
    else:
        prob_cols[1].info("MC Dropout 预测图未找到。")
