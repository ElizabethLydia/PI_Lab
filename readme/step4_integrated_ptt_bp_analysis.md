# PI-Lab step4_integrated_ptt_bp_analysis README

## 概述
`step4_integrated_ptt_bp_analysis.py` 是一个整合分析脚本，用于处理step3_ptt_bp_analysis.py的输出结果，进行跨受试者和跨实验的综合分析，生成相关性柱状图、分布图、线性拟合散点图等可视化结果。核心特性包括：
- **跨受试者整合**：加载每个受试者的step3结果，进行整体分析。
- **相关性可视化**：生成柱状图和分布图，显示PTT与生理指标的相关性分布。
- **线性拟合**：不跨实验和跨实验的PTT-生理指标拟合，生成散点图和统计指标。
- **多模式分析**：支持单实验、整体、融合等多种分析方式。
- **输出**：保存CSV文件和PNG图像到全局目录。

## 安装与依赖
### 依赖库
确保 Python 环境中已安装以下库：
- `numpy`：用于数值计算。
- `pandas`：用于数据处理和CSV操作。
- `matplotlib`：用于生成图表。
- `seaborn`：用于增强可视化效果。
- `scipy`：用于统计分析。
- `sklearn`：用于线性回归和评估。

### 安装步骤
```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn
```

## 使用方法
### 前提条件
- **输入数据**：每个受试者的 `/root/autodl-tmp/{subject_id}/ptt_bp_analysis/` 目录下step3生成的CSV文件（如 `ptt_cardiovascular_correlations.csv`、`synchronized_ptt_cardiovascular_data.csv`）。
- **输出目录**：确保 `/root/autodl-tmp/integrated_analysis` 有写权限。
- **环境要求**：Python 3.6+，支持 `matplotlib` 的图形环境。

### 运行脚本
1. **修改 subject_list**（可选）：
   编辑 `step4_integrated_ptt_bp_analysis.py`，设置目标受试者：
   ```python
   subject_list = ['00112', '00113']  # 示例：处理 00112 和 00113
   ```
   - 这将只处理指定的受试者文件夹。
   - **批量处理所有受试者**：使用默认代码：
     ```python
     subject_list = sorted([d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d)) and d.startswith('00')])
     ```
     这将处理根目录下的所有 '00xxx' 格式受试者。修改后保存文件并重新运行。

2. **执行脚本**：
   ```bash
   python step4_integrated_ptt_bp_analysis.py
   ```
3. **选择分析模式**：
   - 输入 `1`：综合分析（所有）。
   - 输入 `2`：单实验相关性分析。
   - 输入 `3`：综合实验相关性分析。
   - 输入 `4`：不跨实验线性拟合。
   - 输入 `5`：跨实验线性拟合。
   - 默认 `1`。

## 参数说明
- `root_path = '/root/autodl-tmp/'`：数据根目录。
- `output_dir = 'integrated_analysis'`：结果存储目录。
- 生理指标：支持 `systolic_bp`、`diastolic_bp` 等，自动识别。
- 传感器组合：6 种 PTT 组合（如 `Nose→Finger`）。
- `subject_list`：指定受试者文件夹列表；默认处理所有 '00xxx' 格式文件夹（如上所述）。

## 输出说明
### 日志输出
脚本显示处理进度，包括加载文件、生成图表等。

### 文件输出
- **存储路径**：`/root/autodl-tmp/integrated_analysis/` 及其子目录。
- **文件格式**：CSV 和 PNG。
  1. `individual_experiments_correlations/exp_{exp_id}/exp_{exp_id}_correlations.csv`：单实验相关性数据。
  2. `overall_correlations/overall_correlations.csv`：整体相关性数据。
  3. `integrated_experiments/integrated_exp_{exp_id}.csv`：整合实验数据。
  4. `comprehensive_integrated.csv`：综合整合数据。
  5. `*_bar_*.png`：相关性柱状图。
  6. `r_distribution_*.png`：r 值分布图。
  7. `scatter_fit_*.png`：线性拟合散点图。

## 文件层级结构
#### 输入数据结构
- 数据根目录：`/root/autodl-tmp/{subject_id}/`
  - step3 输出文件夹：`ptt_bp_analysis/` 包含分析文件
    - `ptt_cardiovascular_correlations.csv`：整体相关性
    - `synchronized_ptt_cardiovascular_data.csv`：同步数据
    - `ptt_cardiovascular_correlations_exp_{exp_id}.csv`：单实验相关性

#### 输出数据结构
- 输出根目录：`/root/autodl-tmp/integrated_analysis/`
  - 单实验相关性：`individual_experiments_correlations/exp_{exp_id}/`
    - `exp_{exp_id}_correlations.csv`
    - `correlation_bar_*_multi.png`
    - `correlation_bar_*_{pair}.png`
    - `r_distribution_*.png`
  - 整体相关性：`overall_correlations/`
    - `overall_correlations.csv`
    - `correlation_bar_*_multi.png`
    - `correlation_bar_*_{pair}.png`
    - `r_distribution_*.png`
  - 整合数据：`integrated_experiments/`
    - `integrated_exp_{exp_id}.csv`
  - 综合数据：`comprehensive_integrated.csv`
  - per-exp 拟合：`per_experiment_fits/`
    - `per_exp_{exp_id}_cleaned.csv`
    - `scatter_fit_*_*.png`
  - cross-exp 拟合：`cross_experiment_fits/`
    - `cross_experiments_cleaned.csv`
    - `scatter_fit_*_*.png`

## 示例输出
```
🔬 Integrated PTT-Cardiovascular Parameters Correlation Analyzer
📁 Results will be saved to: /root/autodl-tmp/integrated_analysis
📂 Loading from each subject's ptt_bp_analysis/
📋 发现 10 个受试者

📋 请选择分析方式:
1. 综合分析 (所有)
2. 单实验相关性分析 (每个实验的柱状图)
3. 综合实验相关性分析 (所有实验的柱状图)
4. 不跨实验的线性拟合
5. 跨实验的线性拟合

=== 单实验分析 ===
📊 生成实验 1 的柱状图
💾 保存 experiment 1 的 correlations CSV: integrated_analysis/individual_experiments_correlations/exp_1/exp_1_correlations.csv
💾 保存 multi-pair 柱状图: integrated_analysis/individual_experiments_correlations/exp_1/correlation_bar_systolic_bp_multi_exp1.png
💾 保存 per-pair 柱状图: integrated_analysis/individual_experiments_correlations/exp_1/correlation_bar_systolic_bp_Nose-Finger_exp1.png
...

✅ 分析完成！
📁 结果保存在: /root/autodl-tmp/integrated_analysis
```

## 注意事项
1. **输入文件**：确保每个受试者的 ptt_bp_analysis/ 目录存在step3生成的CSV文件。
2. **数据质量**：若某些受试者缺少数据，脚本会跳过并记录。
3. **可视化**：生成大量PNG文件，建议检查输出目录。
4. **性能**：对于大量受试者，分析可能耗时。

## 下一步建议
1. **结果验证**：检查柱状图和分布图，识别高相关性组合。
2. **模型构建**：使用整合CSV数据训练血压预测模型。
3. **扩展分析**：添加更多统计测试或非线性拟合。

- **时间**：2025年7月23日 