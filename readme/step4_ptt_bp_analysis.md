# PI-Lab step4_ptt_bp_analysis README

`ptt_bp_analysis.py` 是一个专为 PTT（脉搏传导时间）与血压及相关生理指标相关性分析设计的 Python 脚本，基于师兄建议使用合理区间的 PTT 数据进行分析，旨在为新生儿血压预测模型提供数据支持。本脚本通过窗口化验证的 PTT 数据（来自 `ptt_output2` 目录）与 Biopac 生理数据同步，计算相关性并构建回归模型。核心特性包括：

- **数据同步**：将 PTT 数据与 Biopac 生理数据（包括收缩压、舒张压等）进行时间窗口同步。
- **相关性分析**：计算 PTT 与多种生理指标（如血压、心率、呼吸率）的皮尔逊相关系数。
- **回归建模**：基于线性回归，预测生理指标并评估模型性能（R²、MAE）。
- **多实验支持**：支持单实验分析、跨实验分析及综合分析，生成热图和性能对比图。
- **可视化**：生成相关性热图、拟合曲线图及模型性能对比图，便于学术报告。
- **优化设计**：聚焦重要生理指标（如收缩压、舒张压），支持中英文标签切换。

## 安装与依赖
### 依赖库
确保 Python 环境中已安装以下库：
- `numpy`：用于数值计算。
- `pandas`：用于数据处理和 CSV 文件操作。
- `matplotlib`：用于生成相关性热图和拟合曲线图。
- `seaborn`：用于增强热图美观性。
- `scipy`：用于统计分析（如皮尔逊相关系数）。
- `sklearn`：用于线性回归和数据标准化。

### 安装步骤
1. 安装核心依赖：
   ```bash
   pip install numpy pandas matplotlib seaborn scipy scikit-learn
   ```
2. 确保 `matplotlib` 支持中文字体（如 SimHei），可手动安装字体包。

## 使用方法
### 前提条件
- **输入数据**：
  - PTT 数据：`ptt_output2/exp_X/ptt_windowed_exp_X.csv`（由 `step3_ir_ptt_peak_detector2.py` 生成）。
  - 生理数据：`output/csv_output/X_biopac_aligned.csv`（包含时间戳和生理指标，如 `systolic_bp`）。
- **数据格式**：
  - PTT CSV 文件需包含 `window_id`、`sensor_pair`、`ptt_ms` 等列。
  - Biopac CSV 文件需包含 `timestamp` 和生理指标列（如 `systolic_bp`）。
- **输出目录**：确保 `ptt_bp_analysis/` 有读写权限，用于保存分析结果。
- **环境要求**：Python 3.6+，支持 `matplotlib` 的图形环境。

### 运行脚本
1. 修改脚本中的 `output_dir` 变量（默认 `ptt_bp_analysis`），确保指向结果保存目录。
2. 执行脚本：
   ```bash
   python ptt_bp_analysis.py
   ```
3. 选择分析模式：
   - 输入 `1`：运行综合分析（单实验 + 跨实验）。
   - 输入 `2`：运行单实验回归分析。
   - 输入 `3`：运行跨实验回归分析。
   - 默认选择 `1`。

### 参数说明
- **`output_dir`**：结果存储目录（默认 `ptt_bp_analysis`）。
- **`ptt_output_dir`**：PTT 数据目录（默认 `ptt_output2`）。
- **生理指标**：支持 `systolic_bp`、`diastolic_bp` 等，自动识别可用指标。
- **传感器组合**：6 种 PTT 组合（如 `nose→finger`）。

## 输出说明
### 日志输出
脚本在终端显示详细处理过程，包括：
- **实验列表**：显示待处理的实验编号（如 `[1, 2, ...]`）。
- **数据加载**：显示 PTT 和生理数据的加载状态。
- **同步结果**：显示有效窗口数量。
- **相关性分析**：显示 PTT 与生理指标的相关系数和显著性。
- **模型评估**：显示 R²、MAE 和样本数。
- **文件保存**：列出所有输出文件的路径。

### 文件输出
- **存储路径**：`ptt_bp_analysis/`。
- **文件格式**：CSV 文件和 PNG 图像。
  1. **`synchronized_ptt_cardiovascular_data.csv`**：同步数据，包含 PTT 和生理指标。
  2. **`ptt_cardiovascular_correlations.csv`**：相关性分析结果。
  3. **`overall_regression_metrics.csv`**：整体回归模型评估。
  4. **`individual_experiment_models.csv`**：单实验模型评估。
  5. **`ptt_cardiovascular_correlation_heatmap_overall.png`**：整体相关性热图。
  6. **`ptt_cardiovascular_correlation_focused_overall.png`**：聚焦重要指标的热图。
  7. **`individual_model_performance_comparison.png`**：模型性能对比图。
  8. **`*_vs_*_fit.png`**：每个传感器对与生理指标的拟合曲线图。

### 示例输出
```
🩺 PTT-Cardiovascular Parameters Correlation Analysis
============================================================

🔬 Enhanced PTT-Cardiovascular Parameters Correlation Analyzer
📁 Results will be saved to: ptt_bp_analysis
📊 Analyzing 9 physiological indicators
🎯 Using 6 PTT sensor combinations

📋 请选择分析方式:
1. 综合分析 (单实验+跨实验)
2. 单实验分析
3. 跨实验分析

🔬 运行综合分析...

 运行综合分析...
🔬 开始PTT与生理参数综合分析
📋 分析实验列表: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

=== 第一部分：单个实验分析 ===
🔬 开始单个实验分析...

🔍 单独分析实验 1

🔍 分析实验 1
✅ 加载生理数据: 66669条记录
📊 可用生理指标: ['bp', 'cardiac_index', 'cardiac_output', 'diastolic_bp', 'hr', 'mean_bp', 'rsp', 'systemic_vascular_resistance', 'systolic_bp']
📊 实验1: 总窗口468, 有效窗口140, 有效PTT数据2774
📊 同步完成: 2774个有效窗口
📊 同步数据: 2774个时间窗口

📊 计算相关性：6个传感器对 × 36个生理指标
🔍 发现传感器对: ['sensor2-sensor3' 'sensor2-sensor4' 'sensor2-sensor5' 'sensor3-sensor4'
 'sensor3-sensor5' 'sensor4-sensor5']

🔧 处理传感器对: sensor2-sensor3
📊 PTT列 ptt_sensor2-sensor3 NaN比例: 0.00%
📈 sensor2-sensor3→systolic_bp_mean模型: R²=0.344, MAE=1.32, N=18
   📊 PTT相关性: r=0.157, p=1.37e-03
💾 保存特征拟合图: ptt_bp_analysis\exp1_systolic_bp_mean_vs_sensor2-sensor3_fit.png
📈 sensor2-sensor3→diastolic_bp_mean模型: R²=0.200, MAE=1.16, N=18
   📊 PTT相关性: r=-0.112, p=2.27e-02
💾 保存特征拟合图: ptt_bp_analysis\exp1_diastolic_bp_mean_vs_sensor2-sensor3_fit.png
📈 sensor2-sensor3→mean_bp_mean模型: R²=0.034, MAE=1.02, N=18
   📊 PTT相关性: r=-0.042, p=3.96e-01
💾 保存特征拟合图: ptt_bp_analysis\exp1_mean_bp_mean_vs_sensor2-sensor3_fit.png
📈 sensor2-sensor3→cardiac_output_mean模型: R²=0.024, MAE=0.20, N=18
   📊 PTT相关性: r=-0.040, p=4.16e-01
💾 保存特征拟合图: ptt_bp_analysis\exp1_cardiac_output_mean_vs_sensor2-sensor3_fit.png
📈 sensor2-sensor3→cardiac_index_mean模型: R²=0.025, MAE=0.02, N=18
   📊 PTT相关性: r=-0.041, p=4.02e-01
💾 保存特征拟合图: ptt_bp_analysis\exp1_cardiac_index_mean_vs_sensor2-sensor3_fit.png

🔧 处理传感器对: sensor2-sensor4
......

=== 第二部分：整体分析 ===

📈 计算整体相关性...

📊 计算相关性：6个传感器对 × 36个生理指标
💾 保存相关性热图: ptt_bp_analysis/ptt_cardiovascular_correlation_heatmap_overall.png

🎯 构建整体回归模型...
🔍 发现传感器对: ['sensor2-sensor3' 'sensor2-sensor4' 'sensor2-sensor5' 'sensor3-sensor4'
 'sensor3-sensor5' 'sensor4-sensor5']

🔧 处理传感器对: sensor2-sensor3
📊 PTT列 ptt_sensor2-sensor3 NaN比例: 0.00%
📈 sensor2-sensor3→systolic_bp_mean模型: R²=0.430, MAE=1.37, N=19
   📊 PTT相关性: r=0.146, p=2.77e-03
💾 保存特征拟合图: ptt_bp_analysis\systolic_bp_mean_vs_sensor2-sensor3_fit.png
📈 sensor2-sensor3→diastolic_bp_mean模型: R²=0.067, MAE=1.22, N=19
   📊 PTT相关性: r=-0.088, p=7.02e-02
💾 保存特征拟合图: ptt_bp_analysis\diastolic_bp_mean_vs_sensor2-sensor3_fit.png
📈 sensor2-sensor3→mean_bp_mean模型: R²=0.031, MAE=1.16, N=19
   📊 PTT相关性: r=-0.010, p=8.31e-01
💾 保存特征拟合图: ptt_bp_analysis\mean_bp_mean_vs_sensor2-sensor3_fit.png
📈 sensor2-sensor3→cardiac_output_mean模型: R²=0.054, MAE=0.19, N=19
   📊 PTT相关性: r=-0.041, p=4.04e-01
💾 保存特征拟合图: ptt_bp_analysis\cardiac_output_mean_vs_sensor2-sensor3_fit.png
📈 sensor2-sensor3→cardiac_index_mean模型: R²=0.055, MAE=0.02, N=19
   📊 PTT相关性: r=-0.042, p=3.94e-01
💾 保存特征拟合图: ptt_bp_analysis\cardiac_index_mean_vs_sensor2-sensor3_fit.png
......
💾 保存同步数据: ptt_bp_analysis/synchronized_ptt_cardiovascular_data.csv
💾 保存相关性数据: ptt_bp_analysis/ptt_cardiovascular_correlations.csv

=== 第三部分：创建聚焦热图（只显示重要指标）===
💾 保存聚焦热图: ptt_bp_analysis/ptt_cardiovascular_correlation_focused_overall_focus.png

🏆 Top Significant Correlations (Overall Analysis):
    1. Finger→Ear ←→ Respiration Rate (breaths/min)
       r=+0.424 ↑, p=1.18e-75, N=1714
    2. Finger→Ear ←→ Respiration Rate (breaths/min) (Max)
       r=+0.393 ↑, p=1.95e-64, N=1714
    3. Finger→Ear ←→ Respiration Rate (breaths/min) (Min)
       r=+0.383 ↑, p=3.90e-61, N=1714
    4. Finger→Ear ←→ Systolic BP (mmHg) (Min)
       r=+0.375 ↑, p=1.99e-58, N=1714
    5. Nose→Wrist ←→ Cardiac Index (L/min/m²)
       r=-0.363 ↓, p=9.28e-12, N=331
    6. Nose→Wrist ←→ Cardiac Output (L/min)
       r=-0.363 ↓, p=9.72e-12, N=331
    7. Finger→Ear ←→ Systolic BP (mmHg)
       r=+0.356 ↑, p=1.84e-52, N=1714
    8. Finger→Ear ←→ Cardiac Output (L/min) (Min)
       r=+0.356 ↑, p=2.31e-52, N=1714
    9. Finger→Ear ←→ Cardiac Index (L/min/m²) (Min)
       r=+0.356 ↑, p=2.91e-52, N=1714
   10. Nose→Wrist ←→ Diastolic BP (mmHg) (Min)
       r=-0.333 ↓, p=5.38e-10, N=331

✅ 分析完成!
📁 所有结果保存在: ptt_bp_analysis
```

## 注意事项
1. **依赖库**：
   - 确保 `matplotlib` 支持中文字体，避免标签显示为方框。
2. **输入文件**：
   - 检查 PTT 和 Biopac 数据路径是否正确，时间戳单位需一致（秒）。
3. **数据质量**：
   - 若相关性低，检查窗口有效性或调整滤波参数。
4. **性能优化**：
   - 跨实验分析可能耗时，建议分批处理。
5. **错误处理**：
   - 文件不存在或数据不足时，脚本会跳过相应实验并记录日志。

## 下一步建议
1. **模型优化**：
   - 使用 `overall_regression_metrics.csv` 选择最佳传感器对，尝试非线性模型（如随机森林）。

- **时间**：2025年7月22日