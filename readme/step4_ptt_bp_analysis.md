# PI-Lab step4_ptt_bp_analysis README

## 概述
`ptt_bp_analysis.py` 是一个专为分析脉搏传导时间（PTT）与血压及其他心血管生理指标相关性设计的 Python 脚本，针对清华大学实验室的 HUB 设备数据与 Biopac 生理数据，旨在为血压预测模型提供数据支持和分析结果。本脚本基于师兄的建议，专注于使用合理区间的 PTT 数据（来自 `step3_ir_ptt_peak_detector2.py` 的有效窗口）进行相关性分析和回归建模，生成专业化的英文标签输出，适用于学术研究和报告。核心特性包括：

- **数据同步**：将窗口化的 PTT 数据与 Biopac 生理数据进行时间同步，确保分析精度。
- **多指标分析**：支持 9 种生理指标（收缩压、舒张压、平均动脉压、心输出量、心指数、心率、系统血管阻力、呼吸率等）与 6 种 PTT 传感器组合的相关性分析。
- **相关性分析**：计算 PTT 与生理指标的皮尔逊相关系数，生成热图（含显著性标记，p<0.05）。
- **回归建模**：构建基于 PTT 的线性回归模型，预测主要生理指标（如收缩压、舒张压），并评估模型性能（R² 和 MAE）。
- **实验间比较**：支持单个实验分析和多实验对比，识别跨实验的模式和差异。
- **专业可视化**：生成英文标签的热图和回归散点图，适合学术报告。
- **批量处理**：自动处理所有实验数据（实验 1-11），输出标准化 CSV 文件和可视化图像。

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