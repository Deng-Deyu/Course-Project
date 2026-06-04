# SolarCast — Improved Photovoltaic Power Forecasting System
# Course: 人工智能基础 (Fundamentals of Artificial Intelligence)
# Topic: 选题五 — 光伏出力预测 (Photovoltaic Power Forecasting)
# Dataset: Kaggle Solar Power Generation Data (Plant 1)

__version__ = "2.0.0"

from src.data_processing import FEATURE_COLS, TARGET_COL, build_dataset
from src.models import (
    LightGBMForecaster,
    LSTMForecaster,
    Seq2SeqLSTM,
    MCDropoutLSTM,
    SequenceDataset,
    MultiStepDataset,
    count_parameters,
)
from src.metrics import compute_metrics, compute_interval_metrics
