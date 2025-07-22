# PI-Lab step4_ptt_bp_analysis README

## 概述
`step4_ptt_bp_analysis.py` 是一个专为分析脉搏传导时间（PTT）与血压及其他心血管生理指标相关性设计的 Python 脚本，针对清华大学实验室的 HUB 设备数据与 Biopac 生理数据，旨在为血压预测模型提供数据支持和分析结果。核心特性包括：

- **数据同步**：将窗口化的 PTT 数据与 Biopac 生理数据进行时间同步，确保分析精度。
- **多指标分析**：支持 9 种生理指标（收缩压、舒张压、平均动脉压、心输出量、心指数、心率、系统血管阻力、呼吸率等）与 6 种 PTT 传感器组合的相关性分析。
- **相关性分析**：计算 PTT 与生理指标的皮尔逊相关系数，生成热图（含显著性标记，p<0.05）。
- **回归建模**：构建基于 PTT 的线性回归模型，预测主要生理指标（如收缩压、舒张压），并评估模型性能（R² 和 MAE）。
  - 回归建模：构建线性回归模型，预测生理指标，支持三种拟合方式：
    - 池化拟合：合并所有实验数据建模。
    - 单独实验拟合：为每个实验单独建模。
    - 实验×传感器对拟合：为每个实验的每个传感器对单独建模。
- **实验间比较**：支持单个实验分析和多实验对比，识别跨实验的模式和差异。
- **可视化与对比**：生成相关性热图、回归散点图和性能对比图，突出最佳传感器对。
- **结果保存**：输出 CSV 文件和 PNG 图像。

## 安装与依赖
### 依赖库
确保 Python 环境中已安装以下库：
- `numpy`：用于数值计算。
- `pandas`：用于数据处理和 CSV 文件操作。
- `scipy`：用于统计分析（皮尔逊相关系数）。
- `matplotlib`：用于生成可视化图像。
- `seaborn`：用于绘制相关性热图。
- `scikit-learn`：用于线性回归建模和评估。
- `pickle`：用于保存模型（可选）。

### 安装步骤
1. 安装核心依赖：
   ```bash
   pip install numpy pandas scipy matplotlib seaborn scikit-learn
   ```
2. 确保系统中支持中文显示（用于格式化标签）：
   - Windows：确保安装了 `SimHei` 字体或等效字体。
   - macOS/Linux：默认支持 `Arial Unicode MS` 或 `DejaVu Sans`。

## 使用方法
### 前提条件
- **输入数据**：
  - **PTT 数据**：需要 `step3_ir_ptt_peak_detector2.py` 生成的窗口化 PTT 数据，位于 `PI_Lab/ptt_output2/exp_X/`，包括：
    - `window_validation_exp_X.csv`：窗口验证信息。
    - `ptt_windowed_exp_X.csv`：PTT 时间序列数据。
  - **生理数据**：需要 Biopac 数据，位于 `PI_Lab/output/csv_output/`，文件名格式为 `X_biopac_aligned.csv`，包含：
    - `timestamp`（第一列，浮点数，单位秒）。
    - 生理指标列（如 `systolic_bp`, `diastolic_bp`, `mean_bp` 等）。
- **输出目录**：确保 `PI_Lab/ptt_bp_analysis/`（或自定义路径）有读写权限，用于保存结果文件。
- **环境要求**：Python 3.6+，支持 `matplotlib` 和 `seaborn` 的图形环境（若在非交互式环境中运行，图像将自动保存为 PNG 文件）。

### 运行脚本
1. 修改脚本中的 `output_dir` 和 `ptt_output_dir` 变量（默认分别为 `PI_Lab/ptt_bp_analysis` 和 `PI_Lab/ptt_output2`），确保指向正确路径：
   ```python
   analyzer = PTTBloodPressureAnalyzer(output_dir='/your/output/path', ptt_output_dir='/your/ptt/path')
   ```
2. 执行脚本：
   ```bash
   python ptt_bp_analysis.py
   ```
3. 脚本运行流程：
   - 扫描 `ptt_output_dir` 目录，加载实验 1-11 的 PTT 数据（`window_validation_exp_X.csv` 和 `ptt_windowed_exp_X.csv`）。
   - 加载对应的 Biopac 生理数据（`X_biopac_aligned.csv`）。
   - 对每个实验进行时间同步，将 PTT 数据与生理数据按窗口对齐。
   - 计算 PTT 与生理指标（均值、标准差、最小值、最大值）的皮尔逊相关系数。
   - 构建线性回归模型，预测主要生理指标（收缩压、舒张压、平均动脉压等）。
   - 进行实验间比较，分析跨实验的相关性和模型性能。
   - 保存 4 个标准化 CSV 文件和 3 类可视化图像到 `PI_Lab/ptt_bp_analysis/`。
   - 输出详细日志，包括数据加载、同步、相关性、回归模型性能和文件路径。

### 参数说明
- **`output_dir`**：结果存储目录（默认 `PI_Lab/ptt_bp_analysis`）。
- **`ptt_output_dir`**：PTT 数据目录（默认 `PI_Lab/ptt_output2`）。
- **`physiological_indicators`**：支持的生理指标（9 种，包含收缩压、舒张压、心率等）。
- **`ptt_combinations_en`**：PTT 传感器组合（6 种，如 `Nose→Finger`）。
- **时间同步**：基于窗口中心时间（`window_center`），匹配 Biopac 数据的时间戳。
- **相关性阈值**：p-value < 0.05 视为显著相关。
- **回归模型**：使用标准化后的 PTT 数据（`StandardScaler`）进行线性回归，至少需要 10 个样本。

## 输出说明
### 日志输出
脚本在终端显示详细处理过程，包括：
- **实验列表**：显示分析的实验编号（默认 `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]`）。
- **数据加载**：显示每个实验的 PTT 数据（有效窗口数、PTT 数据点）和生理数据（记录数、可用指标）。
- **同步结果**：显示每个实验的同步窗口数量。
- **相关性分析**：列出显著相关性（p<0.05），包括传感器对、生理指标、相关系数、p-value 和样本数。
- **回归模型**：显示每个模型的 R² 分数、均绝对误差（MAE）和样本数。
- **实验间比较**：总结各实验的显著相关性和最强相关性。
- **文件保存**：列出所有输出文件的路径。

### 文件输出
- **存储路径**：`PI_Lab/ptt_bp_analysis/`（所有实验的整体和单独分析结果）。
- **文件格式**：4 个标准化 CSV 文件和 3 类可视化图像（PNG 格式）。
  1. **`synchronized_ptt_cardiovascular_data.csv`**：
     - 所有实验的同步数据，包含 PTT 和生理指标。
     - 列：`exp_id`, `window_id`, `start_time`, `end_time`, `window_center`, `sensor_pair`, `ptt_ms`, `[indicator]_mean`, `[indicator]_std`, `[indicator]_min`, `[indicator]_max`, `[indicator]_count`（针对每个生理指标）。
     - 示例：`exp_id=1`, `sensor_pair=sensor2-sensor3`, `ptt_ms=45.2`, `systolic_bp_mean=120.5`。
  2. **`ptt_cardiovascular_correlations.csv`**：
     - 整体相关性分析结果。
     - 列：`sensor_pair`, `sensor_combination`, `physiological_parameter`, `parameter_label`, `correlation_coefficient`, `p_value`, `n_samples`, `statistically_significant`。
     - 示例：`Nose→Finger` 与 `Systolic BP`，`correlation_coefficient=-0.65`, `p_value=0.001`。
  3. **`ptt_cardiovascular_model_evaluation.csv`**：
     - 回归模型评估结果。
     - 列：`physiological_parameter`, `parameter_label`, `r2_score`, `mae`, `n_samples`。
     - 示例：`Systolic BP`，`r2_score=0.72`, `mae=5.2`。
  4. **`experiment_comparison.csv`**：
     - 实验间比较结果，记录显著相关性。
     - 列：`experiment`, `sensor_pair`, `parameter`, `correlation`, `n_samples`。
     - 示例：`experiment=1`, `sensor_pair=sensor2-sensor3`, `parameter=systolic_bp_mean`, `correlation=-0.65`。
  5. **`ptt_cardiovascular_correlation_heatmap_(整体分析).png`**：
     - 整体相关性热图，显示所有 PTT 组合与生理指标的相关系数（显著性标记 * 表示 p<0.05）。
  6. **`ptt_cardiovascular_correlation_focused_整体分析_聚焦.png`**：
     - 聚焦热图，仅显示关键生理指标（收缩压、舒张压、平均动脉压等）。
  7. **`ptt_cardiovascular_correlation_focused_实验X.png`**：
     - 每个实验的聚焦热图，显示关键指标的相关性。
  8. **`ptt_cardiovascular_regression_analysis.png`**：
     - 回归模型预测散点图，显示实际值与预测值的对比，标注 R² 和 MAE。

### 示例输出
```
🩺 PTT-Cardiovascular Parameters Correlation Analysis
============================================================

🔬 开始PTT与生理参数综合分析
📋 分析实验列表: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

=== 第一部分：整体分析 ===
🔍 分析实验 1
✅ 加载生理数据: 10000条记录
📊 可用生理指标: ['systolic_bp', 'diastolic_bp', 'mean_bp', 'hr']
📊 实验1: 总窗口40, 有效窗口30, 有效PTT数据650
📊 同步数据: 650个时间窗口
📈 计算整体相关性...
📊 计算相关性：6个传感器对 × 16个生理指标
💾 保存相关性热图: ptt_bp_analysis/ptt_cardiovascular_correlation_heatmap_(整体分析).png
🎯 构建整体回归模型...
📈 Systolic BP模型: R²=0.720, MAE=5.20, N=650
📈 Diastolic BP模型: R²=0.650, MAE=4.80, N=650
💾 保存回归分析图: ptt_bp_analysis/ptt_cardiovascular_regression_analysis.png
💾 保存同步数据: ptt_bp_analysis/synchronized_ptt_cardiovascular_data.csv
💾 保存相关性分析: ptt_bp_analysis/ptt_cardiovascular_correlations.csv
💾 保存模型评估: ptt_bp_analysis/ptt_cardiovascular_model_evaluation.csv

=== 第二部分：单个实验分析 ===
🔍 单独分析实验 1
📊 同步数据: 650个时间窗口
💾 保存实验1相关性: ptt_bp_analysis/ptt_cardiovascular_correlations_exp_1.csv
💾 保存聚焦热图: ptt_bp_analysis/ptt_cardiovascular_correlation_focused_实验1.png

=== 第三部分：实验间比较 ===
💾 保存实验比较: ptt_bp_analysis/experiment_comparison.csv
📋 实验间比较总结:
   实验1: 12个显著相关性
     最强: Nose→Finger-systolic_bp_mean (r=-0.650)
   实验2: 10个显著相关性
     最强: Nose→Ear-systolic_bp_mean (r=-0.620)

=== 第四部分：创建聚焦热图 ===
💾 保存聚焦热图: ptt_bp_analysis/ptt_cardiovascular_correlation_focused_整体分析_聚焦.png

📋 Analysis Summary:
   • Total samples: 7150
   • PTT combinations: 6
   • Physiological parameters: 16
   • Regression models: 5
   • Individual experiments analyzed: 5
🏆 Top Significant Correlations (Overall Analysis):
   1. Nose→Finger ←→ Systolic BP
       r=-0.650 ↓, p=1.20e-05, N=650
   2. Nose→Ear ←→ Systolic BP
       r=-0.620 ↓, p=2.50e-05, N=650
📈 Best Prediction Models:
   1. Systolic BP: R²=0.720, MAE=5.20
   2. Diastolic BP: R²=0.650, MAE=4.80
📊 Individual Experiment Summary:
   Experiment 1: 650 samples
   Experiment 2: 600 samples
✅ Enhanced cardiovascular analysis completed!
📁 Results saved in: ptt_bp_analysis
```

## 注意事项
1. **依赖库**：
   - 确保安装 `seaborn` 以生成高质量热图。
   - 如果中文标签显示异常，检查系统中是否安装 `SimHei` 或等效字体。
2. **输入文件**：
   - 确保 `ptt_output_dir` 包含 `step3_ir_ptt_peak_detector2.py` 生成的 `window_validation_exp_X.csv` 和 `ptt_windowed_exp_X.csv`。
   - 确保 `output/csv_output/` 包含 `X_biopac_aligned.csv`，且时间戳单位为秒。
3. **时间同步**：
   - 脚本假设 Biopac 数据的时间戳是绝对时间，PTT 数据的时间戳是相对时间（从实验开始）。
   - 如果时间戳单位不一致（如毫秒），可能导致同步失败，需检查 `preprocess_step2.py` 的输出。
4. **数据质量**：
   - 相关性分析要求每个传感器对和生理指标组合至少有 10 个有效样本。
   - 回归模型要求至少 20 个样本，确保模型稳定性。
   - 如果有效窗口比例低（< 50%），检查 `step3_ir_ptt_peak_detector2.py` 的窗口验证结果。
5. **性能优化**：
   - 对于大规模数据（多实验或长时段数据），建议增加内存资源或限制实验列表：
     ```python
     exp_list = [1, 2, 3]  # 仅分析实验 1-3
     ```
6. **可视化**：
   - 在非交互式环境（如服务器）运行时，图像将自动保存为 PNG 文件。
   - 检查 `ptt_cardiovascular_correlation_focused_实验X.png` 和 `ptt_cardiovascular_regression_analysis.png` 以验证相关性和模型性能。
7. **错误处理**：
   - **文件不存在**：检查 `X_biopac_aligned.csv` 或 `ptt_output2/exp_X/` 文件是否缺失。
   - **数据列不足**：确保 Biopac 数据包含至少一种生理指标（如 `systolic_bp`）。
   - **相关性计算失败**：检查 PTT 或生理数据是否存在大量缺失值（NaN）。
   - **回归模型失败**：确保每个模型有足够样本（> 20），否则检查 PTT 数据质量。
8. **后续步骤**：
   - 使用 `ptt_cardiovascular_correlations.csv` 识别强相关性（如 |r| > 0.6, p < 0.05）进行进一步分析。
   - 使用 `ptt_cardiovascular_model_evaluation.csv` 选择性能最佳的模型（高 R²，低 MAE）进行优化。
   - 检查 `experiment_comparison.csv` 分析跨实验的稳定性。

## 下一步建议
1. **数据验证**：
   - 检查 `ptt_cardiovascular_correlations.csv`，重点关注显著相关性（p < 0.05），验证是否与预期生理规律一致（如 PTT 与收缩压负相关）。
   - 使用 `ptt_cardiovascular_correlation_focused_整体分析_聚焦.png` 检查关键指标的相关性，确认主要传感器对（如 Nose→Finger）的表现。
   - 使用 `ptt_cardiovascular_regression_analysis.png` 验证模型预测精度，检查散点是否接近理想线。
2. **模型优化**：
   - 若回归模型性能不佳（R² < 0.7），尝试非线性模型（如随机森林或支持向量机）：
     ```python
     from sklearn.ensemble import RandomForestRegressor
     model = RandomForestRegressor(n_estimators=100)
     ```
   - 增加特征（如 IBI、窗口内信号质量）以提高模型预测能力。
3. **实验扩展**：
   - 收集更多实验数据（扩展到实验 12+），重新运行 `preprocess_step1.py`、`preprocess_step2.py` 和 `step3_ir_ptt_peak_detector2.py`。
   - 验证跨实验的 PTT-血压相关性稳定性，增强模型泛化能力。
4. **学术报告**：
   - 使用 `ptt_cardiovascular_correlation_heatmap_(整体分析).png` 和 `ptt_cardiovascular_regression_analysis.png` 撰写分析报告，突出 PTT 与血压的强相关性和模型性能。
   - 投稿至生物医学工程期刊（如 *IEEE Transactions on Biomedical Engineering* 或 *Physiological Measurement*）。
5. **模型部署**：
   - 保存最佳回归模型（`models[physio_col]['model']`）为 pickle 文件，用于实时血压预测：
     ```python
     import pickle
     with open('best_bp_model.pkl', 'wb') as f:
         pickle.dump(models['systolic_bp_mean'], f)
     ```

- **时间**：2025年7月14日，21:52 KST



好的！你的问题是想深入理解 `step4_ptt_bp_analysis.py` 中涉及的**拟合与回归分析**相关函数，以及它们的功能、实现内容和输出结果。由于代码内容较多，我会以清晰、系统的方式，结合你的清华大学实验室 PTT 血压预测项目的背景，逐步讲解代码中与拟合和回归分析相关的函数，梳理每个函数的实现内容、作用、输出，以及它们在整个分析流程中的意义。我还会特别关注你的“师兄建议：使用合理区间的 PTT 数据分析与血压的相关性”，并解释三种拟合方式（综合分析、单独实验拟合、实验×传感器对拟合）的差异和适用场景。

---

## 1. 拟合与回归分析的整体背景
在你的项目中，目标是通过脉搏传导时间（PTT，基于 HUB 设备采集的 PPG 信号）预测血压等心血管生理指标（如收缩压、舒张压）。代码提供了三种拟合方式，每种方式对应不同的数据处理和建模策略：

1. **综合分析（池化拟合，选项 1）**：
   - 将所有实验（1-11）的 PTT 和生理数据合并，构建统一的回归模型。
   - 适合探索全局趋势，但可能忽略实验间或传感器对的差异。
   - 对应函数：`run_comprehensive_analysis` 和 `build_regression_models`。

2. **单独实验拟合（选项 2）**：
   - 为每个实验单独构建回归模型，分析实验间的差异。
   - 适合评估实验一致性，但不细分传感器对。
   - 对应函数：`run_individual_regression_analysis` 和 `build_regression_models`。

3. **实验×传感器对拟合（选项 3，推荐）**：
   - 为每个实验的每个传感器对（如 `Nose→Finger`）单独构建回归模型。
   - 细化分析，符合师兄建议，突出不同传感器对在合理 PTT 区间内的表现。
   - 对应函数：`run_individual_experiment_sensor_regression_analysis`。

4. **完整对比分析（选项 4）**：
   - 运行所有三种拟合方式，并对比性能。
   - 适合全面评估，生成对比图和最佳传感器对排名。
   - 对应函数：综合调用上述函数。

所有拟合方式都基于线性回归（`LinearRegression`），使用标准化后的 PTT 作为特征，预测生理指标（如 `systolic_bp_mean`）。代码通过计算 R² 和 MAE（均值绝对误差）评估模型性能，并生成可视化结果（如散点图、热图）。

---

## 2. 回归分析相关函数详解
以下是与拟合和回归分析直接相关的核心函数，我将逐一讲解其功能、实现内容、输入输出，以及在项目中的作用。

### 2.1 `build_regression_models(self, sync_df, exp_id=None)`
#### 功能
- 为每个传感器对（如 `Nose→Finger`）和生理指标（如 `systolic_bp_mean`）构建线性回归模型，预测生理指标。
- 支持两种场景：
  - **池化模式**（`exp_id=None`）：合并所有实验数据，构建全局模型。
  - **单实验模式**（`exp_id` 指定）：为特定实验构建模型。
- 数据标准化（`StandardScaler`）后训练，生成拟合直线图，保存模型性能。

#### 实现内容
1. **数据准备**：
   - 筛选主要生理指标（如 `systolic_bp_mean`, `diastolic_bp_mean`）。
   - 按 `exp_id` 和 `window_id` 聚合数据，生成 `ptt_pivot`（PTT 数据）和 `physio_agg`（生理指标均值）。
   - 合并数据为 `model_data`，确保每行对应一个窗口。

2. **模型训练**：
   - 对每个传感器对和生理指标：
     - 提取有效数据（去除 NaN，样本数 ≥ 5）。
     - 标准化 PTT（`ptt_ms`）和生理指标。
     - 训练线性回归模型，预测 `y = physio_col`（如收缩压）。
     - 计算 R² 和 MAE。

3. **拟合直线图**：
   - 绘制散点图（PTT vs 生理指标）和拟合直线。
   - 显示方程（如 `y = -0.123·x + 120.456`）、R² 和 MAE。
   - 保存到 `expX_{physio}_vs_{pair}_fit.png`。

4. **结果保存**：
   - 模型性能（R², MAE, 样本数，斜率，截距）保存到 `all_experiments_regression_metrics.csv`（单实验模式）。

#### 输入
- `sync_df`：同步后的 `DataFrame`，包含 `exp_id`, `window_id`, `sensor_pair`, `ptt_ms`, 生理指标（如 `systolic_bp_mean`）。
- `exp_id`：实验编号（可选，用于单实验模式）。

#### 输出
- **返回值**：
  - `all_models`：字典，键为 `sensor_pair→physio_col`，值包含模型、标准化器、性能指标。
  - `all_model_data`：字典，存储每个模型的训练数据。
- **文件**：
  - 拟合直线图：`expX_{physio}_vs_{pair}_fit.png`（单实验模式）或 `{physio}_vs_{pair}_fit.png`（池化模式）。
  - 性能 CSV：`all_experiments_regression_metrics.csv`（单实验模式）。

#### 项目中的作用
- **池化模式**：用于综合分析（选项 1），探索 PTT 与血压的全局关系。
- **单实验模式**：用于单独实验分析（选项 2），评估特定实验的模型性能。
- **与师兄建议的关联**：通过筛选有效窗口（`hr_diff_bpm ≤ 5`）和时间同步，确保使用合理区间的 PTT 数据。

#### 改进建议
- **多特征回归**：添加心率（`hr_mean`）等特征：
  ```python
  X = model_data.loc[mask, [ptt_col, 'hr_mean']].values
  ```
- **非线性模型**：如果 R² 较低，尝试随机森林：
  ```python
  from sklearn.ensemble import RandomForestRegressor
  model = RandomForestRegressor(n_estimators=100)
  ```

---

### 2.2 `run_comprehensive_analysis(self)`
#### 功能
- 执行综合分析（选项 1，池化拟合），合并所有实验（1-11）的数据，分析 PTT 与生理指标的相关性和回归模型。
- 调用 `build_regression_models`（池化模式），生成全局模型和可视化。

#### 实现内容
1. **数据收集**：
   - 循环调用 `analyze_experiment` 处理实验 1-11，获取每个实验的同步数据（`sync_df`）。
   - 合并所有实验数据为 `combined_df`。

2. **相关性分析**：
   - 调用 `calculate_correlations` 计算 PTT 与生理指标的皮尔逊相关系数。
   - 生成热图（`create_correlation_heatmap` 和 `create_focused_correlation_heatmap`）。

3. **回归建模**：
   - 调用 `build_regression_models(sync_df=combined_df, exp_id=None)`，为每个传感器对和生理指标构建全局模型。
   - 生成回归散点图（`create_regression_plots`）。

4. **结果保存**：
   - 保存同步数据、相关性结果和模型性能到 CSV 文件。
   - 保存热图和散点图。

#### 输入
- 无直接输入，内部调用 `analyze_experiment` 获取数据。

#### 输出
- **返回值**：
  - 字典，包含：
    - `combined_data`：合并的同步数据。
    - `correlations`：相关性结果。
    - `models`：回归模型和性能。
- **文件**：
  - `synchronized_ptt_cardiovascular_data.csv`：同步数据。
  - `ptt_cardiovascular_correlations.csv`：相关性结果。
  - `ptt_cardiovascular_model_evaluation.csv`：模型性能。
  - `ptt_cardiovascular_correlation_heatmap_整体分析.png`：全局热图。
  - `ptt_cardiovascular_correlation_focused_整体分析_聚焦.png`：聚焦热图。
  - `ptt_cardiovascular_regression_analysis.png`：回归散点图。
  - `{physio}_vs_{pair}_fit.png`：拟合直线图。

#### 项目中的作用
- 提供全局视角，评估 PTT 与血压的总体相关性和预测能力。
- 适合初步分析，识别最佳传感器对（如 `Nose→Finger`）。
- **与师兄建议的关联**：通过时间同步和有效窗口筛选，确保分析基于合理区间的 PTT 数据。

#### 改进建议
- **数据均衡**：如果某些实验数据量差异大，考虑加权合并：
  ```python
  weights = sync_df.groupby('exp_id').size() / len(sync_df)
  ```
- **交叉验证**：添加 k 折交叉验证评估模型稳定性：
  ```python
  from sklearn.model_selection import cross_val_score
  scores = cross_val_score(model, X_scaled, y_scaled, cv=5, scoring='r2')
  ```

---

### 2.3 `run_individual_regression_analysis(self)`
#### 功能
- 执行单独实验拟合（选项 2），为每个实验（1-11）单独构建回归模型，分析实验间差异。
- 调用 `analyze_experiment` 获取单实验数据，再调用 `build_regression_models`（单实验模式）。

#### 实现内容
1. **单实验处理**：
   - 循环处理实验 1-11，调用 `analyze_experiment` 获取同步数据和模型。
   - 检查数据量（`len(sync_data) ≥ 20`），跳过数据不足的实验。

2. **模型性能汇总**：
   - 从 `analyze_experiment` 的模型结果中提取性能（R², MAE, 样本数）。
   - 为每个实验和生理指标选择最佳模型（按 R²）。

3. **可视化**：
   - 调用 `create_individual_model_comparison`，生成实验间的 R² 和 MAE 热图。

4. **结果保存**：
   - 保存模型性能到 `individual_experiment_models.csv`。
   - 保存对比图到 `individual_model_performance_comparison.png`。

#### 输入
- 无直接输入，内部调用 `analyze_experiment`。

#### 输出
- **返回值**：
  - `individual_models`：字典，键为 `exp_X`，值包含各实验的模型。
- **文件**：
  - `individual_experiment_models.csv`：各实验的模型性能（实验、指标、R²、MAE、样本数、传感器对）。
  - `individual_model_performance_comparison.png`：实验间 R² 和 MAE 热图。
  - `ptt_cardiovascular_correlations_exp_X.csv`：单实验相关性结果（来自 `analyze_experiment`）。
  - `ptt_cardiovascular_correlation_focused_实验X.png`：单实验聚焦热图。
  - `expX_{physio}_vs_{pair}_fit.png`：单实验拟合直线图。

#### 项目中的作用
- 分析每个实验的独立性能，适合发现实验间的差异（例如，受试者或实验条件的影响）。
- **与师兄建议的关联**：通过单实验分析，确保每个实验的 PTT 数据都在合理区间（`hr_diff_bpm ≤ 5`）。

#### 改进建议
- **实验筛选**：如果某些实验数据量少，调整筛选条件：
  ```python
  if not exp_data or len(exp_data['sync_data']) < 10:  # 从 20 降低到 10
  ```
- **模型比较**：添加统计检验（如 t 检验）比较实验间模型性能：
  ```python
  from scipy.stats import ttest_ind
  r2_exp1 = model_df[model_df['experiment'] == 1]['r2_score']
  r2_exp2 = model_df[model_df['experiment'] == 2]['r2_score']
  t_stat, p_val = ttest_ind(r2_exp1, r2_exp2)
  ```

---

### 2.4 `run_individual_experiment_sensor_regression_analysis(self)`
#### 功能
- 执行实验×传感器对拟合（选项 3，推荐），为每个实验的每个传感器对单独构建回归模型。
- 细化分析，评估不同传感器对（如 `Nose→Finger`）在各实验中的表现。

#### 实现内容
1. **数据处理**：
   - 循环处理实验 1-11，调用 `analyze_experiment` 获取同步数据。
   - 检查数据量（`len(sync_data) ≥ 10`），跳过数据不足的实验。

2. **模型训练**：
   - 对每个实验的每个传感器对和生理指标：
     - 筛选有效数据（样本数 ≥ 5）。
     - 标准化 PTT 和生理指标，训练线性回归模型。
     - 计算 R² 和 MAE。

3. **性能汇总**：
   - 收集模型性能（实验、传感器对、指标、R²、MAE、样本数）。
   - 保存到 `experiment_sensor_models.csv`。

4. **可视化与排名**：
   - 调用 `create_experiment_sensor_comparison`，生成 2×2 热图：
     - 实验×（传感器对×指标）的 R² 和 MAE。
     - 跨实验平均 R² 和 MAE。
   - 保存最佳传感器对到 `best_sensors_across_experiments.csv`。

5. **统计总结**：
   - 打印总模型数、成功模型数（R² > 0）、成功率。
   - 显示 Top 5 最佳模型（按 R² 排序）。

#### 输入
- 无直接输入，内部调用 `analyze_experiment`。

#### 输出
- **返回值**：
  - `all_models`：字典，键为 `exp_X` 和 `sensor_pair`，值包含模型和性能。
- **文件**：
  - `experiment_sensor_models.csv`：模型性能（实验、传感器对、指标、R²、MAE、样本数）。
  - `experiment_sensor_performance_comparison.png`：2×2 热图。
  - `best_sensors_across_experiments.csv`：跨实验最佳传感器对（指标、最佳传感器、平均 R²、MAE）。
  - `ptt_cardiovascular_correlations_exp_X.csv`：单实验相关性结果（来自 `analyze_experiment`）。
  - `ptt_cardiovascular_correlation_focused_实验X.png`：单实验聚焦热图。
  - `expX_{physio}_vs_{pair}_fit.png`：单实验拟合直线图。

#### 项目中的作用
- **细化分析**：为每个实验和传感器对单独建模，揭示特定传感器对（如 `Nose→Finger`）在不同实验中的性能差异。
- **推荐方式**：符合师兄建议，专注于合理区间的 PTT 数据（通过 `load_ptt_data` 的筛选），并突出最佳传感器对。
- **设备优化**：结果可指导 HUB 设备选择最优传感器组合。

#### 改进建议
- **特征扩展**：添加更多特征（如心率、脉搏波形态）：
  ```python
  X = valid_data[['ptt_ms', 'hr_mean']].values
  ```
- **模型多样性**：尝试支持向量回归（SVR）：
  ```python
  from sklearn.svm import SVR
  model = SVR(kernel='rbf')
  ```

---

### 2.5 `create_regression_plots(self, models)`
#### 功能
- 生成回归模型的散点图，展示实际值与预测值的对比。
- 用于综合分析（选项 1），显示池化模型的预测性能。

#### 实现内容
1. **散点图绘制**：
   - 对每个生理指标，绘制实际值（`y_true`）与预测值（`y_pred`）的散点图。
   - 添加理想预测线（y=x）。

2. **标注性能**：
   - 显示 R² 和 MAE。

3. **保存图像**：
   - 保存到 `ptt_cardiovascular_regression_analysis.png`。

#### 输入
- `models`：`build_regression_models` 返回的模型字典。

#### 输出
- **文件**：
  - `ptt_cardiovascular_regression_analysis.png`：散点图，显示实际值与预测值的分布。

#### 项目中的作用
- 直观展示池化模型的预测性能，适合学术报告。
- **与师兄建议的关联**：基于合理区间的 PTT 数据，验证预测效果。

#### 改进建议
- **添加拟合线**：在散点图中显示回归线：
  ```python
  ax.plot(y_true, model_data['model'].predict(X_scaled), 'g-', label='Fit Line')
  ```

---

### 2.6 `create_individual_model_comparison(self, model_df)`
#### 功能
- 生成单独实验模型的性能对比图（选项 2）。
- 绘制实验×生理指标的 R² 和 MAE 热图。

#### 实现内容
1. **数据透视**：
   - 创建 `pivot_mae` 和 `pivot_r2`，按实验和指标组织 MAE 和 R²。

2. **热图绘制**：
   - 使用 `imshow` 绘制热图，标注数值。
   - R² 用蓝色（`Blues`），MAE 用红色（`Reds`）。

3. **保存图像**：
   - 保存到 `individual_model_performance_comparison.png`。

#### 输入
- `model_df`：`run_individual_regression_analysis` 生成的模型性能 `DataFrame`。

#### 输出
- **文件**：
  - `individual_model_performance_comparison.png`：实验间 R² 和 MAE 热图。

#### 项目中的作用
- 比较各实验的模型性能，识别数据质量或实验条件的差异。

---

### 2.7 `create_experiment_sensor_comparison(self, model_df)`
#### 功能
- 生成实验×传感器对模型的性能对比图（选项 3）。
- 绘制 2×2 热图：
  - 实验×（传感器对×指标）的 R² 和 MAE。
  - 跨实验平均 R² 和 MAE。
- 保存最佳传感器对排名。

#### 实现内容
1. **数据透视**：
   - 创建 `exp_r2_pivot`, `exp_mae_pivot`：实验×（传感器对×指标）。
   - 创建 `sensor_r2_pivot`, `sensor_mae_pivot`：传感器对×指标的平均值。

2. **热图绘制**：
   - 使用 `seaborn.heatmap`，R² 用蓝色，MAE 用红色。
   - 标注数值，显示性能差异。

3. **最佳传感器对**：
   - 按指标选择 R² 最高的传感器对，保存到 `best_sensors_across_experiments.csv`。

#### 输入
- `model_df`：`run_individual_experiment_sensor_regression_analysis` 生成的模型性能 `DataFrame`。

#### 输出
- **文件**：
  - `experiment_sensor_performance_comparison.png`：2×2 热图。
  - `best_sensors_across_experiments.csv`：最佳传感器对（指标、最佳传感器、平均 R²、MAE）。

#### 项目中的作用
- 细化性能对比，识别最佳传感器对（如 `Nose→Finger`），指导设备优化。
- **与师兄建议的关联**：通过细化到传感器对，确保分析基于高质量 PTT 数据。

---

## 3. 三种拟合方式的对比
| **方式** | **描述** | **优点** | **缺点** | **适用场景** |
|----------|----------|----------|----------|--------------|
| **综合分析（选项 1）** | 合并所有实验数据，构建全局模型 | 数据量大，模型更稳健；适合全局趋势分析 | 忽略实验间和传感器对差异 | 初步探索 PTT 与血压关系 |
| **单独实验拟合（选项 2）** | 每个实验单独建模 | 揭示实验间差异（如受试者差异） | 数据量可能不足；不细分传感器对 | 分析实验一致性 |
| **实验×传感器对拟合（选项 3）** | 每个实验的每个传感器对单独建模 | 细化分析，突出最佳传感器对；符合师兄建议 | 计算复杂度高；数据量要求高 | 优化传感器选择，设备开发 |

---

## 4. 运行与验证
### 运行步骤
1. **准备环境**：
   ```bash
   pip install numpy pandas scipy matplotlib seaborn scikit-learn
   ```
2. **运行脚本**：
   ```bash
   python step4_ptt_bp_analysis.py
   ```
3. **选择方式**：
   - 输入 `3`（推荐）：运行实验×传感器对拟合，生成细化结果。
   - 输入 `4`：运行所有方式，全面对比。

### 验证结果
1. **检查输出文件**：
   - `best_sensors_across_experiments.csv`：确认 `Nose→Finger` 是否为最佳传感器对（通常 R² > 0.7）。
   - `experiment_sensor_models.csv`：检查 R² 和 MAE，评估模型性能。
2. **查看图像**：
   - `experiment_sensor_performance_comparison.png`：观察传感器对和实验的性能分布。
   - `expX_systolic_bp_mean_vs_sensor2-sensor3_fit.png`：检查拟合直线是否合理。
3. **调试问题**：
   - 如果 `experiment_sensor_models.csv` 为空，检查数据量：
     ```python
     print(f"同步数据量: {len(sync_df)}")
     ```

---

## 5. 下一步建议
1. **优化模型**：
   - 如果 R² < 0.7，尝试非线性模型（如随机森林）：
     ```python
     from sklearn.ensemble import RandomForestRegressor
     model = RandomForestRegressor(n_estimators=100)
     ```
2. **特征工程**：
   - 添加心率、脉搏波形态参数（如上升时间）：
     ```python
     X = valid_data[['ptt_ms', 'hr_mean', 'rise_time']].values
     ```
3. **实时预测**：
   - 保存最佳模型：
     ```python
     import pickle
     with open('best_model.pkl', 'wb') as f:
         pickle.dump(all_models['exp_1']['sensor2-sensor3']['systolic_bp_mean'], f)
     ```
4. **学术报告**：
   - 整理 `best_sensors_across_experiments.csv` 和热图，撰写论文，强调 `Nose→Finger` 的优越性。

---

## 6. 总结
- **综合分析（选项 1）**：全局视角，适合初步分析，调用 `run_comprehensive_analysis`。
- **单独实验拟合（选项 2）**：实验间差异分析，调用 `run_individual_regression_analysis`。
- **实验×传感器对拟合（选项 3）**：细化分析，推荐方式，调用 `run_individual_experiment_sensor_regression_analysis`。
- **可视化与对比**：`create_regression_plots`, `create_individual_model_comparison`, `create_experiment_sensor_comparison` 生成直观结果。

如果你有具体数据或运行结果（如 `experiment_sensor_models.csv` 的片段），可以提供，我会进一步分析模型性能或调试问题！