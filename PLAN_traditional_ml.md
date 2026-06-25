# 在 Time-Series-Library 中集成传统机器学习模型（Electricity 对比实验）

## 背景与目标
- 目标：将传统机器学习模型（scikit-learn / 统计模型）接入现有框架，使其能与 TimesNet、DLinear 等深度模型在同一套 Electricity 长期预测任务上生成可对比的指标。
- 约束：尽量复用现有数据加载、指标计算、结果保存逻辑；对已有深度模型训练流程做最小改动。

## 现状分析
1. **模型注册**：所有模型放在 `models/*.py`，需在 `models/__init__.py` import，并在 `exp/exp_basic.py` 的 `model_dict` 中注册。
2. **训练流程**：`exp/exp_long_term_forecasting.py` 默认使用 PyTorch 训练循环（`loss.backward()`、`optim.Adam`），无法直接训练 sklearn 模型。
3. **数据格式**：Electricity 通过 `--data custom` 使用 `Dataset_Custom`，按 70/10/20 切分；batch 为 `(batch_x, batch_y, batch_x_mark, batch_y_mark)`，形状 `[B, L, C]`。
4. **指标保存**：`test()` 将 `metrics.npy`、`pred.npy`、`true.npy` 写入 `results/<setting>/`，并追加到 `result_long_term_forecast_<des>.txt`。
5. **已有依赖**：`requirements.txt` 已包含 `scikit-learn==1.7.2`，无需额外安装。

## 推荐方案：方案 A（与框架最一致）

### 思路
新增一个实验类 `exp/exp_traditional_ml_forecasting.py`，对应新的 `--task_name traditional_ml_forecast`。该任务不再做 PyTorch 反向传播，而是：
1. 从 `train_loader` 收集所有样本；
2. 对每个变量独立训练一个 sklearn 模型；
3. 在 `test_loader` 上做滚动或直接多步预测；
4. 复用现有 `utils.metrics.metric()` 与结果保存逻辑。

### 具体改动

#### 1. 新增模型包装器 `models/TraditionalML.py`
- 继承 `nn.Module`，仅作为统一接口占位：
  - `forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None)` 在 `traditional_ml_forecast` 任务下返回一个占位张量。
  - 真实训练/推理逻辑放在 `exp_traditional_ml_forecasting.py` 中，避免把 sklearn 拟合逻辑硬塞进 `nn.Module`。
- 该文件也需要在 `models/__init__.py` 与 `exp/exp_basic.py` 注册。

#### 2. 新增实验类 `exp/exp_traditional_ml_forecasting.py`
关键方法设计：
- `_build_model()`：实例化 `TraditionalML.Model(args)`。
- `_select_optimizer()`：返回一个空优化器或基于一个 dummy Parameter 的 Adam，满足 `Exp_Basic` 初始化时不报错即可。
- `train(setting)`：
  - 加载 `train_data, train_loader`。
  - 遍历 `train_loader`，拼接所有 `batch_x` 与 `batch_y` 为完整 numpy 数组。
  - 对每个通道 `c in range(C)`：
    - 输入 `X = train_x[:, :, c]`，形状 `[N, seq_len]`；
    - 输出 `Y = train_y[:, -pred_len:, c]`，形状 `[N, pred_len]`；
    - 训练一个 sklearn 模型（如 `MultiOutputRegressor(Ridge())`）。
  - 将拟合好的模型列表保存到 `checkpoints/<setting>/ml_models.pkl`。
- `test(setting, test=0)`：
  - 加载 `test_loader` 与已保存的 sklearn 模型。
  - 对每个 batch，每个通道用对应模型预测未来 `pred_len` 步。
  - 拼接 `preds/trues`，调用 `utils.metrics.metric()`，保存 `metrics.npy`、`pred.npy`、`true.npy`，并追加到 `result_long_term_forecast_<des>.txt`。

#### 3. 修改 `run.py`
- import `Exp_Traditional_ML_Forecast`。
- 在 `task_name` 路由中增加：
  ```python
  elif args.task_name == 'traditional_ml_forecast':
      Exp = Exp_Traditional_ML_Forecast
  ```
- （可选）增加 argparse 参数 `--ml_model` 用于选择 sklearn 模型类型。

#### 4. 新增运行脚本
在 `scripts/long_term_forecast/ECL_script/` 下新增：
- `Ridge.sh`
- `RandomForest.sh`
- `LinearRegression.sh`

示例命令：
```bash
python -u run.py \
  --task_name traditional_ml_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_96 \
  --model TraditionalML \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --itr 1
```

### 多步预测策略
对每个变量，`seq_len` 历史 → `pred_len` 未来。sklearn 原生支持两种方式：
1. **直接多输出**：`MultiOutputRegressor(Ridge())`，一次性预测 `pred_len` 步。
2. **递归多步**：训练单步模型，每步用预测值作为下一步输入。

建议默认使用 **直接多输出**，实现简单且与深度学习模型直接多步预测对齐。可在代码中通过参数切换。

### 模型列表（第一阶段）
建议优先实现以下 sklearn 模型：
- `LinearRegression`
- `Ridge`
- `Lasso`
- `RandomForestRegressor`
- `GradientBoostingRegressor`
- （可选）`XGBRegressor`（若已安装 xgboost）

## 备选方案：方案 B（侵入最小）

### 思路
不修改 `run.py` 和 `exp/`，新增一个独立脚本 `run_traditional_ml.py`：
1. 复用 `data_provider.data_factory.data_provider()` 加载数据；
2. 用 sklearn 训练并预测；
3. 将 `metrics.npy`、`pred.npy`、`true.npy` 保存到与现有框架一致的 `results/<setting>/` 目录；
4. 指标同样追加到 `result_long_term_forecast_<des>.txt`。

### 优点
- 不影响现有训练流程；
- 实现最快，调试方便。

### 缺点
- 不完全通过 `run.py` 运行，无法复用 `--task_name` 路由与 `Exp_Basic` 的设备管理。

## 推荐选择
**推荐方案 A**，理由：
1. 与项目现有的 `run.py + exp/` 架构保持一致；
2. 可复用命令行参数、指标计算、结果保存、脚本模板；
3. 便于后续扩展更多传统模型或统计模型（如 ARIMA）。

## 实施步骤
1. 新增 `models/TraditionalML.py` 并在 `models/__init__.py`、`exp/exp_basic.py` 注册。
2. 新增 `exp/exp_traditional_ml_forecasting.py`。
3. 修改 `run.py` 增加 `traditional_ml_forecast` 任务路由。
4. 新增 `scripts/long_term_forecast/ECL_script/TraditionalML_*.sh` 脚本。
5. 在 Electricity 上运行验证，对比指标输出格式与深度学习模型一致。
6. （可选）将指标汇总到统一表格脚本。

## 风险与注意点
1. **内存**：Electricity 训练集约 18k 行 × 321 变量，收集全部样本后内存占用较大；可分批拟合或逐变量处理。
2. **训练时间**：RandomForest/GBDT 在 321 变量上训练较慢，建议先做小规模子集测试。
3. **输出维度**：M 模式下 `c_out=321`，MS/S 模式下需要单独处理 `f_dim` 切片。
4. **标准化**：训练/测试数据已通过 `Dataset_Custom` 做 StandardScaler，sklearn 模型在该标准化数据上训练即可。
