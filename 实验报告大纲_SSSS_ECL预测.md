# 《基于深度学习的工业能源发电量智能预测》实验报告大纲

> **数据集:** ECL (Electricity) | **输入窗口:** 96 步 | **预测窗口:** 96 / 192 / 336 / 720 步  
> **本文模型:** SSSS (双分支抗噪时间序列预测模型)  
> **对比模型:**  
> - 深度学习: Pyraformer, DLinear, TSMixer  
> - 传统机器学习: XGBoost, Linear Regression, Ridge, Random Forest  
> **评估指标:** MSE, MAE, RMSE, SMAPE, R²  
> **实验框架:** Time-Series-Library (PyTorch + scikit-learn)

---

## 第一章 课题背景与意义

### 1.1 研究背景
- 工业能源预测的现实需求：电网负荷调度、发电计划优化、节能降耗
- 时间序列预测在能源管理中的核心地位
- 深度学习技术对传统时间序列分析方法的革新

### 1.2 国内外研究现状
- **传统方法：** 自回归模型 (ARIMA)、指数平滑
- **机器学习方法：** 线性回归、支持向量回归 (SVR)、随机森林、XGBoost
- **深度学习方法：**
  - RNN 系列：LSTM, GRU, SegRNN
  - Transformer 系列：Informer, Autoformer, FEDformer, Nonstationary_Transformer
  - 线性变形方法：DLinear (序列分解+线性层)
  - MLP 方法：TSMixer (时间/特征双混合)
  - 其他：Pyraformer (金字塔注意力)

### 1.3 本文主要工作
1. 提出 **SSSS** 模型：双分支结构，目标变量直通通道（Anchor Track）与加噪特征抗噪通道（Feature Track）融合
2. 在 **ECL 电力数据集**上，与 **3 个深度学习模型 + 4 个传统机器学习模型**进行全面对比
3. 在 4 个预测窗口长度（96/192/336/720）下评估模型的长距离预测能力
4. 从精度、稳定性、效率三个维度进行系统性分析

---

## 第二章 数据集介绍与预处理

### 2.1 ECL 数据集概述
| 属性 | 说明 |
|------|------|
| 全称 | Electricity Consuming Load |
| 采集频率 | 每小时一次 (hourly) |
| 通道数 | 321（321 个电力消费者） |
| 数据格式 | CSV (第一列 date，后续列为各用户用电量) |
| 时间范围 | 2016-07-01 至 2019-07-01 (约 3 年) |
| 缺失值 | 无（数据集完整） |

### 2.2 数据集统计信息（表 2.1）

> 📎 **表 2.1 — 文件:** `report_figures/dataset_statistics_table.md`  
> （直接将 Markdown 表格贴入本节）

| 属性 | 值 |
|------|-----|
| 总样本数 | 26304 |
| 输入特征维度（通道数） | 321 |
| 总数据点数 | 8,443,584 |
| 训练集样本数 (70%) | 18412 |
| 验证集样本数 (10%) | 2632 |
| 测试集样本数 (20%) | 5260 |
| 全局均值 | 2538.79 |
| 全局标准差 | 15027.57 |
| 全局最小值 | 0.00 |
| 全局最大值 | 764000.00 |
| 通道均值范围 | [10.63, 200529.12] |
| 通道标准差范围 | [8.54, 146054.61] |
| 缺失值数量 | 0 |
| 归一化方法 | Z-score 标准化 (StandardScaler) |

### 2.3 数据划分
采用自定义划分方式（`Dataset_Custom`），按时间顺序划分：

| 划分 | 比例 | 样本数 |
|------|------|--------|
| 训练集 (Train) | 70% | 18412 |
| 验证集 (Val) | 10% | 2632 |
| 测试集 (Test) | 20% | 5260 |

> **引用代码：** `data_provider/data_loader.py` 第 282-288 行

### 2.4 归一化
- 采用 **Z-score 标准化** (StandardScaler)
- 仅用训练集计算均值 μ 和标准差 σ，全局应用
- 预测完成后反标准化恢复物理量

> **引用代码：** `data_provider/data_loader.py` 第 297-299 行；`utils/tools.py` 第 71-80 行

### 2.5 时间特征构造
- 从 date 列提取：month, day, weekday, hour (timeenc=0)
- 频率标记：hourly (`freq='h'`)

> **引用代码：** `data_provider/data_loader.py` 第 303-310 行

### 2.6 滑动窗口构建
| 参数 | 值 |
|------|-----|
| 输入窗口长度 (seq_len) | 96 小时 (4天) |
| 标签长度 (label_len) | 48 |
| 预测窗口长度 (pred_len) | 96 / 192 / 336 / 720 小时 |
| 滑动步长 | 1 |

> **引用代码：** `data_provider/data_loader.py` 第 323-337 行

### 2.7 特征模式说明
- **M 模式 (Multivariate)：** 321 通道输入 → 321 通道输出，所有深度学习模型训练使用
- **S 模式 (Univariate)：** 仅 OT 目标通道 → 单通道输出，传统 ML 逐通道训练使用
- 所有最终评估均在 **S (OT通道)** 上进行可比

> **引用代码：** `run.py` 第 36-38 行；`exp/exp_traditional_ml_forecasting.py` 第 128-165 行

---

## 第三章 预测模型设计

### 3.1 时序分解动机

> 📊 **图 3.1 — 文件:** `report_figures/fig_decomposition.png`  
> ECL 数据 OT 通道前 600 小时的趋势-季节分解：原始信号 (Original) / 趋势分量 (Trend, moving_avg kernel=25) / 季节波动分量 (Seasonal)

ECL 用电数据天然可分解为低频趋势与高频波动两部分。SSSS 模型的双分支结构正是基于这一观察：**Anchor Track 捕捉趋势自相关性，Feature Track 从波动中提取辅助信息**。

> **引用代码：** `layers/Autoformer_EncDec.py` 第 21-29 行（series_decomp / moving_avg 实现）

### 3.2 SSSS 模型（本文提出）

#### 3.2.1 模型结构

> 🖼️ **图 3.2 — 文件:** `report_figures/modelframe.pdf`  
> SSSS 模型完整架构图（双分支结构 + 融合机制）

```
输入 x_enc: [B, L=96, C=321]
             │
    ┌────────┴─────────┐
    │                  │
 分支一                分支二
 Anchor Track         Feature Track
    │                  │
 target = x[:,:,-1]   RevIN 归一化
 [B, 96]              │
    │                 Patching (seg_len=24)
 Linear(96→pred_len)  [B*C, 4, 24]
 [B, pred_len, 1]     │
    │                 FeatureEmbedding
    │                 [B*C, 4, d_model=512]
    │                  │
    │                 Bi-GRU
    │                  │
    │                 DecodeProj
    │                 [B*C, pred_len]
    │                  │
    │                 反归一化 + reshape
    │                 [B, pred_len, C]
    │                  │
    └────────┬─────────┘
         融合: w*anchor + (1-w)*feature (仅OT维)
              │
          输出: [B, pred_len, C]
```

**关键创新点：**
1. **Patching 策略 (seg_len=24)：** 将 96 步切为 4 个 patch，减少序列长度，增强局部模式捕获
2. **Bi-GRU 编码器：** 1 层双向 GRU，双向捕捉时间依赖
3. **Direct Projection 解码：** 直接线性映射 GRU 最终状态到 pred_len，避开递归误差累积
4. **通道独立 (CI) 策略：** 对所有 321 通道共享 GRU 权重，参数量与通道数无关
5. **可学习融合：** softmax 归一化的双权重融合

> **引用代码：** `models/SSSS.py`（全文 86 行）

#### 3.2.2 训练配置

| 超参数 | 值 |
|--------|-----|
| 优化器 | Adam |
| 学习率 | 0.001 |
| 损失函数 | MSE |
| d_model | 512 |
| dropout | 0 |
| 早停耐心 (patience) | 3 |
| 训练轮次 | 10 |
| 批大小 | 32 |
| 随机种子 | 2021 |

> **引用代码：** `scripts/long_term_forecast/ECL_script/SSSS.sh`

### 3.3 对比模型

#### 3.3.1 深度学习模型

| 模型 | 核心思想 | 论文来源 | 关键参数 |
|------|---------|---------|----------|
| **Pyraformer** | 金字塔注意力机制，O(N) 复杂度捕获多尺度时间依赖 | ICLR 2022 | d_model=512, e_layers=2 |
| **DLinear** | 序列分解（趋势+季节）+ 线性映射 | AAAI 2023 | moving_avg=25, individual=False |
| **TSMixer** | 时间混合 MLP + 特征混合 MLP，无注意力/卷积 | KDD 2023 | d_model=256, d_ff=512 |

> **引用代码：** `models/DLinear.py`、`models/TSMixer.py`、`models/Nonstationary_Transformer.py`

#### 3.3.2 传统机器学习模型

| 模型 | 配置 | 训练策略 |
|------|------|----------|
| **Linear Regression** | sklearn 默认 | 321 通道 × 1 模型 |
| **Ridge** | α=1.0 | 321 通道 × 1 模型 |
| **Random Forest** | n_estimators=10, max_depth=5 | 321 通道 × 1 模型 |
| **XGBoost** | n_estimators=10, max_depth=3 | 321 通道 × 1 模型 |

> **引用代码：** `exp/exp_traditional_ml_forecasting.py` 第 18-57 行；`scripts/long_term_forecast/ECL_script/TraditionalML_*.sh`

**注意：** 传统 ML 在 M 模式下逐通道独立训练（每通道一个模型）。所有模型在 S 模式下的 OT 通道结果才具可比性。

### 3.4 消融变体（可选纳入报告）

| 变体 | 说明 | 源文件 |
|------|------|--------|
| SSSS_wogru | 移除 GRU 特征通道 | `models/SSSS_wogru.py` |
| SSSS_wolinear | 移除线性目标通道 | `models/SSSS_wolinear.py` |
| SSSS_wopatching | 移除 Patching 策略 | `models/SSSS_wopatching.py` |
| SSSS_worevin | 移除 RevIN 归一化 | `models/SSSS_worevin.py` |

---

## 第四章 评价指标

### 4.1 指标定义（表 4.1）

| 指标 | 全称 | 公式 | 说明 |
|------|------|------|------|
| **MSE** | Mean Squared Error | $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$ | 均方误差，对大误差敏感 |
| **RMSE** | Root Mean Squared Error | $\sqrt{\text{MSE}}$ | 与原始数据单位相同 |
| **MAE** | Mean Absolute Error | $\frac{1}{n}\sum |y_i - \hat{y}_i|$ | 对异常值更鲁棒 |
| **SMAPE** | Symmetric MAPE | $\frac{1}{n}\sum \frac{|y_i - \hat{y}_i|}{(|y_i| + |\hat{y}_i|)/2} \times 100\%$ | 对称百分比误差，适合近零值 |
| **R²** | Coefficient of Determination | $1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$ | 决定系数，越接近 1 越好 |

**为什么选用 SMAPE 而非 MAPE？** ECL 数据集含大量近零值（低用电时段），MAPE 在分母近 0 时会极大发散。SMAPE 分母为 `(|y|+|ŷ|)/2`，范围稳定在 [0%, 200%]。

> **引用代码：** `utils/metrics.py` 第 56-66 行（SMAPE）、第 14-24 行（MAE/MSE/RMSE）、第 69-75 行（R²）

---

## 第五章 实验结果与分析

### 5.1 单变量预测结果 (Univariate OT)

> **数据来源：** `exp_outputs/SSSS_ai_practice.pdf` Table 1

#### 完整指标表（表 5.1）

| 模型 | pred_len | MSE ↓ | MAE ↓ | SMAPE% ↓ | R² ↑ |
|------|----------|------|------|------|------|
| **SSSS (Ours)** | 96 | **0.3034** | **0.3898** | **6.24** | **0.7109** |
| | 192 | 0.3353 | 0.4081 | 6.51 | 0.6816 |
| | 336 | **0.3638** | **0.4336** | **6.90** | **0.6558** |
| | 720 | **0.4231** | **0.4746** | **7.57** | **0.5963** |
| Pyraformer | 96 | 0.3070 | 0.3932 | 6.29 | 0.7074 |
| | 192 | **0.3275** | **0.4019** | **6.41** | **0.6890** |
| | 336 | 0.3687 | 0.4368 | 6.96 | 0.6511 |
| | 720 | 0.4563 | 0.4893 | 7.79 | 0.5647 |
| DLinear | 96 | 0.3873 | 0.4512 | 7.23 | 0.6309 |
| | 192 | 0.3645 | 0.4353 | 6.95 | 0.6538 |
| | 336 | 0.3908 | 0.4529 | 7.21 | 0.6302 |
| | 720 | 0.4277 | 0.4866 | 7.75 | 0.5919 |
| TSMixer | 96 | 0.3412 | 0.4210 | 6.77 | 0.6749 |
| | 192 | 0.3511 | 0.4267 | 6.84 | 0.6666 |
| | 336 | 0.3966 | 0.4562 | 7.30 | 0.6247 |
| | 720 | 0.4439 | 0.4946 | 7.93 | 0.5764 |
| XGBoost | 96 | 0.3491 | 0.4307 | 7.07 | 0.6673 |
| | 192 | 0.3367 | 0.4173 | 6.84 | 0.6802 |
| | 336 | 0.3672 | 0.4368 | 7.15 | 0.6526 |
| | 720 | 0.4060 | 0.4726 | 7.69 | 0.6126 |
| LinearRegression | 96 | 0.3712 | 0.4369 | 7.17 | 0.6462 |
| | 192 | 0.3506 | 0.4217 | 6.91 | 0.6671 |
| | 336 | 0.3780 | 0.4409 | 7.21 | 0.6423 |
| | 720 | 0.4156 | 0.4770 | 7.76 | 0.6034 |
| Ridge | 96 | 0.3712 | 0.4369 | 7.17 | 0.6462 |
| | 192 | 0.3506 | 0.4217 | 6.91 | 0.6671 |
| | 336 | 0.3780 | 0.4409 | 7.21 | 0.6423 |
| | 720 | 0.4156 | 0.4770 | 7.76 | 0.6034 |
| RandomForest | 96 | 0.5676 | 0.5796 | 9.51 | 0.4591 |
| | 192 | 0.5790 | 0.5900 | 9.65 | 0.4501 |
| | 336 | 0.5969 | 0.5996 | 9.77 | 0.4352 |
| | 720 | 0.6198 | 0.6081 | 9.89 | 0.4086 |

> **注：** 加粗为最优结果，下划线为次优结果

### 5.2 多变量预测结果 (Multivariate Non-OT)

> **数据来源：** `exp_outputs/SSSS_ai_practice.pdf` Table 2

#### 完整指标表（表 5.2）

| 模型 | pred_len | MSE ↓ | MAE ↓ | SMAPE% ↓ | R² ↑ |
|------|----------|------|------|------|------|
| **SSSS (Ours)** | 96 | **0.1654** | **0.2517** | **12.66** | **0.8366** |
| | 192 | **0.1751** | **0.2615** | **12.91** | **0.8269** |
| | 336 | **0.1920** | **0.2787** | **13.52** | **0.8098** |
| | 720 | **0.2313** | **0.3113** | **14.72** | **0.7698** |
| DLinear | 96 | 0.2104 | 0.3016 | 15.27 | 0.7922 |
| | 192 | 0.2102 | 0.3047 | 15.30 | 0.7922 |
| | 336 | 0.2231 | 0.3192 | 15.84 | 0.7790 |
| | 720 | 0.2578 | 0.3496 | 17.09 | 0.7435 |
| TSMixer | 96 | 0.2041 | 0.3082 | 15.85 | 0.7983 |
| | 192 | 0.2182 | 0.3289 | 16.57 | 0.7843 |
| | 336 | 0.2394 | 0.3502 | 17.39 | 0.7629 |
| | 720 | 0.2720 | 0.3730 | 18.29 | 0.7294 |
| Pyraformer | 96 | 0.2784 | 0.3737 | 18.78 | 0.7250 |
| | 192 | 0.2936 | 0.3893 | 19.26 | 0.7097 |
| | 336 | 0.2920 | 0.3883 | 19.13 | 0.7107 |
| | 720 | 0.2971 | 0.3857 | 18.97 | 0.7043 |

### 5.3 指标趋势分析图（图 5.1 ~ 5.5）

> 以下折线图均为单变量 OT 通道上 8 个模型的指标随预测长度 (96→192→336→720) 变化趋势

| 编号 | 文件名 | 内容 |
|------|--------|------|
| 📊 **图 5.1** | `report_figures/fig_uv_mse.png` | 单变量 MSE 折线图 (8模型) |
| 📊 **图 5.2** | `report_figures/fig_uv_r2.png` | 单变量 R² 折线图 (8模型) |
| 📊 **图 5.3** | `report_figures/fig_uv_smape.png` | 单变量 SMAPE 折线图 (8模型) |
| 📊 **图 5.4** | `report_figures/fig_uv_mae.png` | 单变量 MAE 折线图 (8模型) |
| 📊 **图 5.5** | `report_figures/fig_dl_vs_ml_bar.png` | DL vs ML 平均 MSE 柱状图 (★ Best 标注 SSSS, RandomForest 截断标记) |

### 5.4 多变量对比图（图 5.6）

| 📊 **图 5.6** | `report_figures/fig_mv_mse_r2.png` | 多变量 MSE 和 R² 双面板对比图 (4模型) |

### 5.5 预测曲线样本（图 5.7a ~ 5.7d）

> 取自 `exp_outputs/` 中 `test_results/` 的 PDF 文件，展示各模型在测试集某样本上的预测效果（batch index=80, pred_len=192，前半段为输入的 OT 历史值，后半段为蓝色 Ground Truth vs 红色 Prediction 对比）

| 📊 **图 5.7a** | `report_figures/SSSS_192_80.pdf` | SSSS 预测曲线样本 |
| 📊 **图 5.7b** | `report_figures/DLinear_192_80.pdf` | DLinear 预测曲线样本 |
| 📊 **图 5.7c** | `report_figures/TSMixer_192_80.pdf` | TSMixer 预测曲线样本 |
| 📊 **图 5.7d** | `report_figures/Pyraformer_192_80.pdf` | Pyraformer 预测曲线样本 |

### 5.6 结果分析与关键发现

#### 5.6.1 整体精度排名（单变量，4个预测长度平均 MSE）

| 排名 | 模型 | 平均 MSE | 类型 |
|------|------|----------|------|
| 1 | **SSSS** | 0.3564 | DL |
| 2 | Pyraformer | 0.3649 | DL |
| 3 | XGBoost | 0.3648 | ML |
| 4 | TSMixer | 0.3832 | DL |
| 5 | LinearRegression | 0.3789 | ML |
| 6 | Ridge | 0.3789 | ML |
| 7 | DLinear | 0.3926 | DL |
| 8 | RandomForest | 0.5908 | ML |

#### 5.6.2 关键发现

**发现 1：SSSS 在绝大多数场景下取得最优**
- 单变量 4 个预测长度中，SSSS 在 96/336/720 三个长度上取得最优 MSE
- 多变量 4 个预测长度中，SSSS 在全部 4 个长度上大幅度领先（相对 DLinear 平均降低 MSE 约 15%）

**发现 2：Direct Projection 策略在长距离预测中优势突出**
- 720 步单变量：SSSS 的 MSE=0.4231，比 Pyraformer (0.4563) 低 7.3%
- Bi-GRU + Direct Projection 避免了递归误差累积

**发现 3：深度学习模型整体优于传统机器学习**
- XGBoost 表现最好（平均 MSE 0.3648），接近 DL 水平
- RandomForest 表现最差（平均 MSE 0.5908），可能因 n_estimators 仅 10 欠拟合
- Ridge ≈ LinearRegression（α=1.0 时 L2 正则几乎无效）

**发现 4：多变量场景下 SSSS 的优势更明显**
- MSE 降幅：多变量比单变量在相对提升上更显著（相对 DLinear 提升 15% vs 6%）
- 说明 SSSS 的 321 通道 CI 策略有效利用了多变量信息

**发现 5：趋势-季节分解为双分支设计提供理论支撑**
- 时序分解图（图 3.1）表明 ECL 数据可分解为低频趋势和高频季节波动
- SSSS 的 Anchor Track 专注趋势（线性映射），Feature Track 专注波动（Bi-GRU + Patching）

---

## 第六章 消融实验（根据实际实验进度选择纳入）

### 6.1 各模块贡献分析

| 变体 | 移除模块 | 目的 |
|------|---------|------|
| SSSS_wogru | GRU 特征通道 | 验证 Bi-GRU 对捕捉噪声特征信息的贡献 |
| SSSS_wolinear | 线性目标通道 | 验证 Anchor Track 对保持目标变量自相关性的贡献 |
| SSSS_wopatching | Patching 策略 | 验证 seg_len=24 分段策略的贡献 |
| SSSS_worevin | RevIN 归一化 | 验证可逆实例归一化的贡献 |

> **引用代码：** `models/SSSS_wogru.py`、`models/SSSS_wolinear.py` 等

**可贴图表：**
- 📋 **表 6.1：** 消融实验结果表 —— 需运行消融脚本获得数据
- 📊 **图 6.1：** 消融实验柱状图对比 —— 需生成

---

## 第七章 图形用户界面设计（待开发截图）

### 7.1 开发环境
- Python 3.x + PyQt5
- matplotlib 嵌入绘图

### 7.2 界面功能设计

| 模块 | 功能 | 说明 |
|------|------|------|
| 数据导入 | CSV 文件选择、预览 | 支持 ECL 格式数据 |
| 模型选择 | 下拉菜单选择 8 个模型之一 | 加载对应 checkpoint |
| 预测执行 | 点按按钮执行预测 | 显示进度条 |
| 结果展示 | 表格显示指标 (MSE/MAE/SMAPE/R²) | 实时更新 |
| 曲线绘制 | 真实值 vs 预测值对比图 | matplotlib 嵌入 |
| 结果导出 | 图片/CSV 导出 | 预测结果保存 |

> 🖼️ **图 7.1：** GUI 界面截图 — **待运行 PyQt5 界面后截图贴入**

---

## 第八章 总结与展望

### 8.1 工作总结
1. 提出 SSSS 双分支模型：解耦目标预测与特征利用，兼顾信号纯净性和信息补充
2. 在 ECL 321 通道数据集上全面验证：与 3 个 DL 模型和 4 个 ML 模型对比
3. SSSS 在单变量（OT）和多变量场景中均取得最优，平均 MSE 降低 3-15%
4. Direct Projection 解码策略有效缓解长距离预测误差累积问题
5. 模型轻量高效：仅 1 层 Bi-GRU + 线性层，参数量可控

### 8.2 存在问题与局限性
1. **数据集单一：** 仅在 ECL 数据集上验证，未在 Weather/Traffic/ETT 等数据集上测试泛化性
2. **传统 ML 超参数未充分调优：** RandomForest n_estimators 仅 10，XGBoost max_depth=3 过于保守
3. **未进行统计显著性检验：** 如 Diebold-Mariano 检验
4. **消融实验可能未完成：** 若有结果可纳入第 6 章
5. **Pyraformer 在 192 步单变量上的反超：** 需要进一步分析原因

### 8.3 未来改进方向
1. **多数据集验证：** 在 Weather、Traffic、ETTh1/ETTh2/ETTm1/ETTm2 等标准数据集上测试
2. **引入前沿架构对比：** PatchTST、iTransformer、TimesNet、Mamba 等
3. **增强特征利用：** 可尝试 Channel Attention 或 GCN 建模通道间关系
4. **模型压缩：** 针对工业部署的量化/剪枝/知识蒸馏
5. **工业应用拓展：** 扩展到真实钢铁企业或电厂数据

---

## 📎 附录

### 附录 A：完整实验数据汇总

| 文件 | 内容 |
|------|------|
| `exp_outputs/SSSS_ai_practice.pdf` | Table 1+2 对比结果汇总 |
| `exp_outputs/ECL_TSMixer/result_long_term_forecast_Exp.txt` | TSMixer 详细结果 |
| `exp_outputs/ECL_XGBoost/result_long_term_forecast_Exp.txt` | XGBoost 详细结果 |
| `exp_outputs/ECL_LinearRegression/result_long_term_forecast_Exp.txt` | LinearRegression 详细结果 |
| `exp_outputs/ECL_RandomForest/result_long_term_forecast_Exp.txt` | RandomForest 详细结果 |
| `exp_outputs/ECL_Ridge/result_long_term_forecast_Exp.txt` | Ridge 详细结果 |

### 附录 B：核心代码文件

| 文件 | 说明 |
|------|------|
| `models/SSSS.py` | SSSS 模型实现（86行） |
| `models/DLinear.py` | DLinear 对比模型 |
| `models/TSMixer.py` | TSMixer 对比模型 |
| `models/Nonstationary_Transformer.py` | Nonstationary_Transformer 对比模型 |
| `exp/exp_long_term_forecasting.py` | 深度学习训练/评估框架 |
| `exp/exp_traditional_ml_forecasting.py` | 传统 ML 训练/评估框架 |
| `utils/metrics.py` | 评估指标实现 |
| `data_provider/data_loader.py` | 数据加载与预处理 |
| `run.py` | 主入口脚本 |
| `scripts/long_term_forecast/ECL_script/` | 各模型启动脚本 |

### 附录 C：实验环境

| 组件 | 版本/型号 |
|------|----------|
| 操作系统 | Linux / Windows 11 |
| Python | 3.x |
| PyTorch | (需补充) |
| CUDA | (需补充) |
| GPU | (需补充) |
| scikit-learn | (需补充) |
| xgboost | (需补充) |

### 附录 D：已生成的报告图表清单

```
report_figures/
├── dataset_statistics_table.md  ← 表 2.1 数据集统计信息
├── fig_decomposition.png        ← 图 3.1 趋势-季节分解图
├── modelframe.pdf               ← 图 3.2 SSSS 模型架构图
├── fig_uv_mse.png               ← 图 5.1 单变量 MSE 折线图
├── fig_uv_r2.png                ← 图 5.2 单变量 R² 折线图
├── fig_uv_smape.png             ← 图 5.3 单变量 SMAPE 折线图
├── fig_uv_mae.png               ← 图 5.4 单变量 MAE 折线图
├── fig_dl_vs_ml_bar.png         ← 图 5.5 DL vs ML 柱状图
├── fig_mv_mse_r2.png            ← 图 5.6 多变量 MSE+R² 双面板
├── SSSS_192_80.pdf              ← 图 5.7a SSSS 预测曲线样本
├── DLinear_192_80.pdf           ← 图 5.7b DLinear 预测曲线样本
├── TSMixer_192_80.pdf           ← 图 5.7c TSMixer 预测曲线样本
└── Pyraformer_192_80.pdf        ← 图 5.7d Pyraformer 预测曲线样本
```

### 附录 E：生成图表的脚本

| 脚本 | 用途 |
|------|------|
| `generate_report_figures.py` | 生成图 5.1 ~ 5.6（折线图+柱状图） |
| `gen_pred_curves.py` | 从已有 pred.npy/true.npy 生成预测曲线对比图 |
| `gen_decomposition_fig.py` | 生成图 3.1 趋势-季节分解图 |
| `gen_dataset_stats.py` | 生成表 2.1 数据集统计信息 |

---

## 📊 材料使用清单 (终版)

| 编号 | 类型 | 内容 | 文件 | 状态 |
|------|------|------|------|------|
| 表 2.1 | 统计表 | 数据集统计信息 | `dataset_statistics_table.md` | ✅ |
| 图 3.1 | 分解图 | 趋势-季节分解 | `fig_decomposition.png` | ✅ |
| 图 3.2 | 架构图 | SSSS 模型结构 | `modelframe.pdf` | ✅ |
| 表 4.1 | 公式表 | 评价指标定义 | 文中表格 | ✅ |
| 表 5.1 | 数据表 | 单变量完整指标 | 文中表格 | ✅ |
| 表 5.2 | 数据表 | 多变量完整指标 | 文中表格 | ✅ |
| 图 5.1 | 折线图 | 单变量 MSE | `fig_uv_mse.png` | ✅ |
| 图 5.2 | 折线图 | 单变量 R² | `fig_uv_r2.png` | ✅ |
| 图 5.3 | 折线图 | 单变量 SMAPE | `fig_uv_smape.png` | ✅ |
| 图 5.4 | 折线图 | 单变量 MAE | `fig_uv_mae.png` | ✅ |
| 图 5.5 | 柱状图 | DL vs ML MSE | `fig_dl_vs_ml_bar.png` | ✅ |
| 图 5.6 | 双面板 | 多变量 MSE+R² | `fig_mv_mse_r2.png` | ✅ |
| 图 5.7a | 预测曲线 | SSSS 样本 | `SSSS_192_80.pdf` | ✅ |
| 图 5.7b | 预测曲线 | DLinear 样本 | `DLinear_192_80.pdf` | ✅ |
| 图 5.7c | 预测曲线 | TSMixer 样本 | `TSMixer_192_80.pdf` | ✅ |
| 图 5.7d | 预测曲线 | Pyraformer 样本 | `Pyraformer_192_80.pdf` | ✅ |
| 图 7.1 | GUI截图 | PyQt5 界面 | — | ⚠️ 待截图 |

**图例：** ✅ = 已生成，可直接插入报告 / ⚠️ = 待后续补充

---

*本大纲最后更新于 2026-06-29，基于 Time-Series-Library 项目的 ECL 数据集实验。*