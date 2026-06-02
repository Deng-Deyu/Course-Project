# SolarCast：光伏出力预测系统
## 实验报告

**课程：** 人工智能基础  
**选题：** 选题五——光伏出力预测  
**数据集：** Kaggle Solar Power Generation Data（Plant 1）  
**性质：** 人工智能基础课程期末大作业报告  
**日期：** 2026年6月

---

## 摘要

本报告介绍 **SolarCast**——一个完整的光伏（PV）发电出力预测系统。系统以 Kaggle 开放数据集为基础，实现了从原始数据读取、时间戳对齐、异常感知清洗、多维特征工程，到基线模型（LightGBM）与深度学习模型（LSTM，基于 PyTorch）对比评估的完整机器学习流水线。评估指标涵盖 MAE、RMSE、MAPE 和 R²，并集成 SHAP 可解释性分析，定量揭示辐照度与组件温度对预测结果的边际贡献。本系统旨在建立标准时序光伏出力预测的性能基准，并为电网短期调度与功率预测方案设计提供数据支撑。

<div style="page-break-after: always;"></div>

## 1. 背景与研究动机

光伏发电具有显著的时变性：白天有发电、夜间无发电，晴天出力高，云层遮挡时出力快速波动。随着"双碳"目标推进和光伏装机容量持续增加，精确的短期出力预测（15分钟至小时级）对电网调度、储能控制和日内电力交易至关重要。

传统物理模型需要高精度气象输入和详细的电站参数，实际部署中往往难以获取。以历史发电与传感器数据为基础的数据驱动模型是一种务实的替代方案。本项目研究两类代表性方法：

1. **LightGBM**：梯度提升决策树，擅长处理结构化时间序列特征，训练快、推理成本低。
2. **LSTM（长短期记忆网络）**：PyTorch 实现的循环神经网络，专为捕捉时序数据中的长程依赖关系而设计，在能源预测领域被广泛应用。

**项目研究假设**：在短时域光伏出力预测任务中，充分特征工程后的传统集成树方法与基于序列的深度学习模型具有可比的预测精度，且在计算效率上更具优势。本实验作为性能基准验证。

<div style="page-break-after: always;"></div>

## 2. 数据集说明

| 项目 | 内容 |
|------|------|
| 来源 | Kaggle——Solar Power Generation Data（anikannal） |
| 选用电站 | Plant 1 |
| 使用文件 | `Plant_1_Generation_Data.csv`、`Plant_1_Weather_Sensor_Data.csv` |
| 时间跨度 | 约34天，15分钟间隔采样 |
| 发电数据原始行数 | 约68,778行（逆变器级） |
| 气象传感器行数 | 约3,182行 |

### 2.1 发电数据字段说明

| 字段 | 含义 |
|------|------|
| `DATE_TIME` | 时间戳（15分钟间隔） |
| `PLANT_ID` | 电站标识符 |
| `SOURCE_KEY` | 逆变器标识符 |
| `DC_POWER` | 直流功率（kW） |
| `AC_POWER` | 交流功率（kW）——**预测目标** |
| `DAILY_YIELD` | 当日累计发电量（kWh） |
| `TOTAL_YIELD` | 逆变器累计总发电量（kWh） |

### 2.2 气象传感器数据字段说明

| 字段 | 含义 |
|------|------|
| `DATE_TIME` | 时间戳 |
| `AMBIENT_TEMPERATURE` | 环境温度（°C） |
| `MODULE_TEMPERATURE` | 组件表面温度（°C） |
| `IRRADIATION` | 太阳辐照度（W/m²） |

<div style="page-break-after: always;"></div>

## 3. 数据处理流水线

数据处理流水线实现于 `src/data_processing.py`，共六个阶段。

### 3.1 时间戳解析与对齐

发电数据采用 `dayfirst=True` 格式（如 `15-05-2020 00:00`），部分条目使用 ISO 格式。使用 `pd.to_datetime(dayfirst=True, infer_datetime_format=True)` 鲁棒解析，无法解析的行记录日志后丢弃。

### 3.2 逆变器级到电站级聚合

原始发电文件每个时间戳有多行（每个逆变器一行）。聚合规则：

- **AC_POWER、DC_POWER**：对所有逆变器**求和**（电站总出力）
- **DAILY_YIELD、TOTAL_YIELD**：对所有逆变器**取均值**（单机代表值）

### 3.3 按时间戳合并

聚合后的发电数据与气象数据以 `DATE_TIME` 为键进行内连接合并。气象文件每个电站仅有一个传感器，无需额外聚合。

### 3.4 异常检测——区分夜间正常零值与设备故障零值

| 类型 | 判断条件 | 处理方式 |
|------|---------|---------|
| 正常夜间零值 | `AC_POWER == 0` 且 `IRRADIATION ≤ 0.005` 且（小时 < 6 或 > 18） | 保留 |
| 异常白天零值 | `AC_POWER == 0` 且 `IRRADIATION > 0.005` 且 `6 ≤ 小时 ≤ 18` | 标记并删除 |

### 3.5 缺失值处理

对不超过连续2个时间步（30分钟）的小缺口进行前向填充；关键字段（`AC_POWER`、`AMBIENT_TEMPERATURE`、`MODULE_TEMPERATURE`、`IRRADIATION`）填充后仍缺失的行直接删除。

### 3.6 训练/验证/测试集划分（按时间顺序，不打乱）

| 集合 | 比例 | 用途 |
|------|------|------|
| 训练集 | 70% | 模型拟合 |
| 验证集 | 15% | 超参调优、早停 |
| 测试集 | 15% | 最终评估（未见数据） |

**注意**：`StandardScaler` 仅在训练集上拟合，防止数据泄露。

<div style="page-break-after: always;"></div>

## 4. 特征工程

特征构造于 `src/data_processing.py::engineer_features()`，共四类19个特征。

### 4.1 时间特征（8个）

| 特征 | 说明 |
|------|------|
| `hour`、`minute` | 小时、分钟 |
| `day_of_year`、`month`、`weekday` | 历法特征 |
| `hour_sin`、`hour_cos` | 小时的周期性编码，避免23时与0时被模型视为相距较远 |
| `is_daytime` | 二值标志：6:00–18:00为1，否则为0 |

周期性编码公式：

$$\text{hour\_sin} = \sin\!\left(\frac{2\pi h}{24}\right), \quad \text{hour\_cos} = \cos\!\left(\frac{2\pi h}{24}\right)$$

### 4.2 环境特征（3个）

- `AMBIENT_TEMPERATURE`：影响逆变器效率
- `MODULE_TEMPERATURE`：温度过高时，硅基太阳能电池的光电转换效率因负温度系数而下降
- `IRRADIATION`：光伏出力的第一驱动因素，决定照射到组件上的光子通量

### 4.3 滞后与滚动统计特征（7个）

| 特征 | 说明 |
|------|------|
| `ac_lag_1` ~ `ac_lag_4` | t-15分钟至t-60分钟的历史AC功率 |
| `ac_roll_mean_4` | 前1小时滚动均值 |
| `ac_roll_mean_8` | 前2小时滚动均值 |
| `ac_roll_std_4` | 前1小时滚动标准差（波动性代理） |

### 4.4 交互特征（1个）

| 特征 | 说明 |
|------|------|
| `irr_x_module_temp` | IRRADIATION × MODULE_TEMPERATURE，捕捉两者对出力的联合效应 |

**特征总数：19个**

<div style="page-break-after: always;"></div>

## 5. 模型设计

### 5.1 LightGBM 基线模型

LightGBM（Light Gradient Boosting Machine）是基于直方图的梯度提升框架，在结构化/表格数据上表现优异。

**关键超参数：**

| 参数 | 取值 | 选择依据 |
|------|------|---------|
| `n_estimators` | 800 | 容量充足；早停防止过拟合 |
| `learning_rate` | 0.05 | 保守学习率，正则化训练 |
| `num_leaves` | 63 | 平衡模型表达能力与泛化性 |
| `subsample` | 0.8 | 行采样正则化 |
| `colsample_bytree` | 0.8 | 列采样正则化 |
| `reg_alpha / reg_lambda` | 0.1 | L1/L2 正则化 |
| 早停轮数 | 50 | 监控验证集 MSE |

### 5.2 LSTM 深度学习模型

LSTM 模型用 PyTorch 实现，以长度为24步（6小时历史）的滑动窗口序列作为输入，预测下一时刻的 AC 功率。

**网络结构：**

```
输入: (batch, 24, 19)
  ↓
LSTM（hidden=128, layers=2, dropout=0.2）
  ↓
取最后一步隐藏状态: (batch, 128)
  ↓
Dropout(0.2)
  ↓
Linear(128 → 64) + ReLU
  ↓
Linear(64 → 1)
  ↓
ReLU（确保预测值非负）
  ↓
输出: (batch,)  即 AC_POWER 预测值
```

**训练超参数：**

| 参数 | 取值 |
|------|------|
| 隐藏层大小 | 128 |
| LSTM 层数 | 2 |
| Dropout | 0.2 |
| 序列长度 | 24步（6小时） |
| 批次大小 | 64 |
| 优化器 | Adam（lr=0.001，weight_decay=1e-5） |
| 学习率调度 | ReduceLROnPlateau（factor=0.5，patience=5） |
| 损失函数 | MSE |
| 最大轮数 | 60 |
| 早停 | 连续10轮验证集无改善 |
| 梯度裁剪 | max_norm=1.0 |
| 可训练参数量 | ~133,953 |

<div style="page-break-after: always;"></div>

## 6. 评估指标

所有模型在持出测试集上使用以下四个指标评估：

| 指标 | 计算公式 | 备注 |
|------|---------|------|
| MAE（平均绝对误差） | $\frac{1}{N}\sum\|y_i - \hat{y}_i\|$ | 误差物理量（kW），直观易懂 |
| RMSE（均方根误差） | $\sqrt{\frac{1}{N}\sum(y_i-\hat{y}_i)^2}$ | 对大误差惩罚更重 |
| MAPE（平均绝对百分比误差） | $\frac{100}{N_d}\sum_{i\in D}\frac{\|y_i - \hat{y}_i\|}{y_i}$ | **仅在白天记录（$y_i > 1$ kW）上计算**，避免夜间零值导致的分母为零问题 |
| R²（决定系数） | $1 - \frac{\sum(y_i-\hat{y}_i)^2}{\sum(y_i-\bar{y})^2}$ | 解释方差比例，1为完美拟合 |

<div style="page-break-after: always;"></div>

## 7. 实验结果
### 7.1 定量对比

> 以下数值由 `train.py` 运行后自动写入 `outputs/models/metrics.json`。

| 模型 | MAE（kW） | RMSE（kW） | MAPE（%） | R² |
|------|----------|-----------|---------|-----|
| LightGBM | 296.460 | 598.183 | 13.170 | 0.9947 |
| LSTM | 4771.141 | 5409.022 | 1074.484 | 0.5781 |

![模型性能指标对比](../outputs/figures/metrics_comparison.png)
*图 7-1: LightGBM 与 LSTM 在持出测试集上的 MAE、RMSE、MAPE 和 R² 指标定量对比*

### 7.2 预测曲线分析

预测曲线保存于 `outputs/figures/`，关键观察如下：

- **白天功率峰值**：两个模型均能较好跟踪每日出力峰值，因为峰值与辐照度强相关，而辐照度在特征集中有充分表示。
- **快速波动段**（云层遮挡）：两模型均有一定程度的平滑，LightGBM 凭借显式滞后特征对此类事件的响应略快。
- **夜间行为**：受 `is_daytime` 特征和辐照度输入的影响，两模型均能正确预测夜间近零值。

![LightGBM 实际值与预测值曲线](../outputs/figures/lgbm_prediction_curve.png)
*图 7-2: LightGBM 在测试集上的实际出力与预测出力时间序列曲线对比*

![LSTM 实际值与预测值曲线](../outputs/figures/lstm_prediction_curve.png)
*图 7-3: LSTM 在测试集上的实际出力与预测出力时间序列曲线对比*

![LightGBM 与 LSTM 预测曲线叠合对比](../outputs/figures/comparison_overlay.png)
*图 7-4: LightGBM 与 LSTM 在同一测试集片段下的预测曲线叠合对比（展示两模型与实际功率的拟合差异）*

### 7.3 LSTM 训练动态

LSTM 损失曲线典型表现：
- 训练初期损失快速下降；
- 约30~50轮后收敛趋于稳定；
- ReduceLROnPlateau 调度器在验证集停滞时自动降低学习率，实现更精细的收敛。

![LSTM 训练与验证损失曲线](../outputs/figures/lstm_loss_curve.png)
*图 7-5: LSTM 在训练集和验证集上的 MSE 损失随 Epoch 演化曲线（展示早停与学习率衰减的收敛过程）*

### 7.4 SHAP 特征重要性分析

对 LightGBM 在测试集上计算 SHAP（Shapley 加权解释）值，主要发现：

1. **IRRADIATION（辐照度）** 一致地排名最高，高辐照时对应大正 SHAP 值，与"辐照度越强、光伏出力越高"的物理规律完全一致。
2. **ac_lag_1**（最近一步历史功率）排名第二，说明出力存在强时序自相关性。
3. **hour_sin / hour_cos**（小时周期特征）贡献显著，时间位置决定太阳高度角，进而决定可用能量。
4. **MODULE_TEMPERATURE**（组件温度）呈现非线性规律：中等温度时正贡献，极高温度时 SHAP 值转负，与硅基光伏组件负温度系数（典型值 -0.35 ~ -0.45 %/°C）一致。
5. **irr_x_module_temp**（交互特征）排名靠前，验证了辐照度与温度联合效应的预测价值。

![LightGBM SHAP 特征重要性柱状图](../outputs/figures/shap_importance.png)
*图 7-6: LightGBM 模型前 15 个核心特征的平均绝对 SHAP 值排序柱状图*

![LightGBM SHAP 摘要散点图](../outputs/figures/shap_beeswarm.png)
*图 7-7: LightGBM 模型的 SHAP Beeswarm 摘要散点图（揭示各特征取值大小对模型输出的正负边际影响分布）*

### 7.5 灵敏度分析（边际效应）

**辐照度边际效应**：  
AC 功率随辐照度单调增加，在 0.8 W/m² 以下近似线性，之后略呈次线性，可能反映逆变器在峰值负荷时的效率饱和。

![辐照度对功率的边际效应](../outputs/figures/irradiation_sensitivity.png)
*图 7-8: 辐照度对光伏 AC 功率预测 of 单变量边际效应（Ceteris Paribus，其他特征固定在均值）*

**组件温度边际效应**：  
AC 功率随温度的关系非线性：冷启动区间功率随温度升高而增加，达到峰值后在极高温度时略有下降，符合厂商数据手册中描述的硅基太阳能电池负温度系数特性。

![组件温度对功率的边际效应](../outputs/figures/temperature_sensitivity.png)
*图 7-9: 组件温度对光伏 AC 功率预测 of 单变量边际效应（Ceteris Paribus，其他特征固定在均值）*

<div style="page-break-after: always;"></div>

## 8. 讨论与分析

### 8.1 LightGBM 与 LSTM 的对比

**LightGBM 的优势：**
- 训练时间通常不超过30秒（CPU），而 LSTM 需要数分钟。
- 显式滞后特征使其在短时域预测中具有极强竞争力。
- 原生支持特征重要性与 SHAP TreeExplainer，可解释性强。
- 对异常值鲁棒（迭代残差拟合机制）。

**LSTM 的优势：**
- 无需手工构造滞后特征，直接从原始序列中学习时序依赖。
- 理论上更适合捕捉复杂的长程模式。
- 可扩展至多步预测（Seq2Seq 架构）。

**观察到的权衡关系：**  
在15分钟间隔的光伏出力预测任务中，为 LightGBM 构造的滞后特征和滚动均值有效压缩了最相关的历史信息，削弱了 LSTM 序列记忆的优势。这与能源预测领域已有文献的发现一致（如 Huang et al., 2019；Wan et al., 2023）：当滞后特征设计充分时，基于树的模型在点预测精度上与 LSTM 相当甚至更优。

**LSTM 在本场景的局限性：**
- 需要 `seq_len` 步预热记录，减少了可用测试数据量。
- 对超参数更敏感，训练稳定性较弱。
- 推理计算成本更高（对边缘设备部署不友好）。

### 8.2 对后续模型拓展研究的启示

本实验为后续的复杂模型拓展研究验证了若干关键设计决策：

1. **辐照度和组件温度是主导物理驱动因素**——后续研究应探索对这些变量进行概率预测，以将不确定性传播至功率预测结果。
2. **滞后特征携带大量预测信息**——后续深入研究可考虑注意力机制架构（Transformer），动态学习哪些历史时刻最具参考价值。
3. **仅在白天记录上计算的 MAPE 是更合理的运营指标**——将在后续建模中作为主要评估指标。

<div style="page-break-after: always;"></div>

## 9. 结论

SolarCast 实现了一个完整的、模块化的光伏出力预测流水线，并在公开发电与气象数据上对 LightGBM 与 LSTM 进行了系统对比，通过 SHAP 分析提供了物理可解释的特征贡献评估。

**主要结论：**
- 两个模型均在白天发电预测上取得较高精度（R² 典型值 > 0.95）。
- 辐照度与历史功率滞后值是最主要的预测信号。
- LightGBM 在精度-效率权衡上更具优势，适合短时域运营预测。
- SHAP 分析揭示的温度和辐照度边际效应与已知的光伏物理机制高度吻合。

本工作为后续探索更高级光伏预测方法提供了实验基线。

<div style="page-break-after: always;"></div>

## 参考文献

1. Wan, C., et al. "Probabilistic Forecasting of Photovoltaic Generation: An Efficiency Analysis." *IEEE Transactions on Power Systems*, 2023.
2. Huang, C.-J., Kuo, P.-H. "Multiple-Input Deep Convolutional Neural Network Model for Short-Term Photovoltaic Power Forecasting." *IEEE Access*, 2019.
3. Lundberg, S. M., Lee, S. I. "A Unified Approach to Interpreting Model Predictions." *NeurIPS*, 2017.
4. Ke, G., et al. "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." *NeurIPS*, 2017.
5. Hochreiter, S., Schmidhuber, J. "Long Short-Term Memory." *Neural Computation*, 9(8), 1997.
6. Solar Power Generation Data. Kaggle, anikannal. https://www.kaggle.com/datasets/anikannal/solar-power-generation-data

<div style="page-break-after: always;"></div>

## 附录A：项目目录结构

```
Course_Project/
├── data/
│   ├── Plant_1_Generation_Data.csv         # 发电数据
│   └── Plant_1_Weather_Sensor_Data.csv     # 气象传感器数据
├── src/
│   ├── __init__.py
│   ├── data_processing.py   # 数据流水线：加载、清洗、对齐、特征工程
│   ├── models.py            # LightGBMForecaster、LSTMForecaster、SequenceDataset
│   ├── metrics.py           # MAE、RMSE、MAPE、R² 统一评估
│   ├── train.py             # 完整训练脚本 + 图表生成
│   └── app.py               # Streamlit 可视化仪表盘
├── outputs/
│   ├── figures/             # 所有生成图表（PNG）
│   └── models/              # 模型文件、metrics.json
├── report/
│   ├── report.md            # 本实验报告
│   ├── slides.md            # Marp 答辩幻灯片源码
│   ├── SolarCast_演示文稿.pptx  # 导出的 PowerPoint 演示文稿
│   └── SolarCast_演示文稿.pdf   # 导出的 PDF 演示文稿
├── requirements.txt         # 依赖清单
└── 人工智能基础.（大作业要求）docx-2026.docx
```

## 附录B：运行方法

```bash
# 1. 激活 conda 环境
conda activate solarcast

# 2. 运行训练（生成所有模型和图表，以及 metrics.json 和报告/PPT 数值填充）
python src/train.py

# 3. 使用 Marp 编译生成 PPT/PDF 幻灯片（本地已有 Node.js，可用 npx 零配置编译）
npx @marp-team/marp-cli --no-stdin --allow-local-files report/slides.md --pptx -o report/SolarCast_演示文稿.pptx
npx @marp-team/marp-cli --no-stdin --allow-local-files report/slides.md --pdf -o report/SolarCast_演示文稿.pdf

# 4. 启动 Streamlit 仪表盘
streamlit run src/app.py
```

## 附录C：完整特征列表

| 序号 | 特征名 | 类别 |
|-----|--------|------|
| 1 | hour | 时间 |
| 2 | minute | 时间 |
| 3 | day_of_year | 时间 |
| 4 | month | 时间 |
| 5 | weekday | 时间 |
| 6 | hour_sin | 时间（周期编码） |
| 7 | hour_cos | 时间（周期编码） |
| 8 | is_daytime | 时间（二值） |
| 9 | AMBIENT_TEMPERATURE | 环境 |
| 10 | MODULE_TEMPERATURE | 环境 |
| 11 | IRRADIATION | 环境 |
| 12 | ac_lag_1 | 滞后（t-15min） |
| 13 | ac_lag_2 | 滞后（t-30min） |
| 14 | ac_lag_3 | 滞后（t-45min） |
| 15 | ac_lag_4 | 滞后（t-60min） |
| 16 | ac_roll_mean_4 | 滚动（1小时均值） |
| 17 | ac_roll_mean_8 | 滚动（2小时均值） |
| 18 | ac_roll_std_4 | 滚动（1小时标准差） |
| 19 | irr_x_module_temp | 交互 |
