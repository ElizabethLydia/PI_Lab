# PI-Lab step3_ir_ptt_peak_detector README

## 概述
`step3_ir_ptt_peak_detector.py` 是一个专为脉搏传输时间（PTT）峰值检测设计的脚本，针对HUB 设备的红外（IR）通道数据进行处理，旨在为后续血压预测模型提供高质量的 PTT 数据。脚本基于以下优化特性：

- **专注 IR 通道**：IR 信号质量最佳，优先用于峰值检测。
- **稳健峰值检测**：支持 `neurokit2`、`heartpy` 或改进的 `scipy` 方法，自动选择最佳方案。
- **IBI 验证**：通过心跳间期（IBI）质量控制确保峰值可靠性（心率范围 50-200 BPM）。
- **心跳匹配**：跨传感器（nose、finger、wrist、ear）匹配同一心跳的峰值，生成 6 种 PTT 组合。
- **批量处理**：自动处理所有实验数据，输出标准化 CSV 文件，存储到实验子文件夹。
- **可视化**：生成峰值检测的可视化图像，便于验证结果。

## 安装与依赖
### 依赖库
确保您的 Python 环境已安装以下库：
- `numpy`
- `pandas`
- `scipy`
- `tqdm`
- `matplotlib`（用于可视化）
- `neurokit2`（可选，推荐用于专业峰值检测）
- `heartpy`（可选，备选峰值检测库）

### 安装步骤
1. 安装依赖：
   ```bash
   pip install numpy pandas scipy tqdm matplotlib
   ```
2. 安装可选库（推荐）：
   ```bash
   pip install neurokit2 heartpy
   ```
   如果无法安装 `neurokit2` 或 `heartpy`，脚本将自动使用 `scipy` 备选方案。

## 使用方法
### 前提条件
- **输入数据**：需要 `preprocess_step2.py` 生成的 HUB 数据 CSV 文件，位于 `PI_Lab/output/csv_output/`（或自定义路径），文件名格式如 `X_hub_sensor{i}_aligned.csv`（i=2,3,4,5）。
- **数据格式**：每个 CSV 文件需包含 `timestamp` 列（第一列，浮点数，单位秒）和 IR 通道列（第三列）。
- **输出目录**：确保 `PI_Lab/ptt_output/`（或自定义路径）有读写权限，用于保存结果文件。
- **时间戳要求**：时间戳为浮点数（秒），与 `preprocess_step1.py` 和 `preprocess_step2.py` 输出一致。

### 运行脚本
1. 修改脚本中的 `data_path` 变量（默认 `PI_Lab/output/csv_output`），确保指向 HUB 数据 CSV 文件目录：
   ```python
   data_path = '/your/csv/path'
   ```
2. 执行脚本：
   ```bash
   python step3_ir_ptt_peak_detector.py
   ```
3. 脚本将：
   - 扫描指定目录，识别所有实验（如 `1`, `2`, ...）的 HUB 数据文件。
   - 对每个实验的 IR 通道数据进行带通滤波（0.5-3.0 Hz）和峰值检测。
   - 验证峰值质量（基于 IBI 范围 300-1200 ms，50-200 BPM）。
   - 跨传感器匹配同一心跳的峰值，计算 6 种 PTT 组合（如 nose→finger）。
   - 保存 5 个标准化 CSV 文件到 `PI_Lab/ptt_output/exp_X/` 子文件夹。
   - 生成峰值检测可视化图像（PNG 格式）。

### 参数说明
- `data_path`：HUB 数据 CSV 文件目录（默认 `PI_Lab/output/csv_output`）。
- `output_dir`：结果存储目录（默认 `PI_Lab/ptt_output`）。
- `sensors`：处理的传感器列表（默认 `['sensor2', 'sensor3', 'sensor4', 'sensor5']`，对应 nose、finger、wrist、ear）。
- `fs = 100`：采样率（100 Hz，与前处理一致）。
- `min_hr = 50`, `max_hr = 200`：心率范围（BPM），用于 IBI 验证。
- `filter_lowcut = 0.5`, `filter_highcut = 3.0`：带通滤波频率范围（Hz）。
- `method = 'auto'`：峰值检测方法，自动选择 `neurokit2`（优先）、`heartpy` 或 `scipy_advanced`。

## 输出说明
### 日志输出
脚本在终端显示详细处理过程，包括：
- 实验列表和处理进度（使用 `tqdm` 进度条）。
- 每个传感器的峰值检测结果（峰值数量、心率均值/标准差、IBI 范围、信号质量）。
- 跨传感器心跳匹配结果（有效传感器、完整心跳数量）。
- PTT 分析统计（每对传感器的均值、标准差、最小/最大值等）。
- 保存的文件路径和可视化图像路径。

### 文件输出
- **存储路径**：`PI_Lab/ptt_output/exp_X/`（每个实验一个子文件夹）。
- **文件格式**：5 个标准化 CSV 文件和 1 个可视化图像。
  1. **`sensor_summary_exp_X.csv`**：
     - 传感器质量汇总（峰值数量、信号质量、心率/IBI 统计、HRV 指标如 RMSSD 和 pNN50）。
     - 列：`sensor`, `sensor_name`, `peak_count`, `quality`, `valid_ibi_ratio`, `hr_mean_bpm`, `hr_std_bpm`, `ibi_mean_ms`, `ibi_std_ms`, `rmssd_ms`, `pnn50_percent`, `signal_duration_s`。
  2. **`all_peaks_exp_X.csv`**：
     - 所有传感器峰值详细信息。
     - 列：`sensor`, `sensor_name`, `peak_number`, `peak_index`, `peak_time_s`, `quality`。
  3. **`all_ibi_exp_X.csv`**：
     - 所有传感器的 IBI 详细信息。
     - 列：`sensor`, `sensor_name`, `ibi_number`, `ibi_ms`, `hr_bpm`, `is_valid`, `quality`。
  4. **`matched_heartbeats_exp_X.csv`**：
     - 跨传感器匹配的心跳数据，包含每个心跳的峰值时间。
     - 列：`heartbeat_id`, `sensor2_peak_time_s`, `sensor3_peak_time_s`, `sensor4_peak_time_s`, `sensor5_peak_time_s`。
  5. **`ptt_matrix_exp_X.csv`**：
     - PTT 统计矩阵，汇总 6 种传感器组合的 PTT 统计（nose→finger, nose→wrist, 等）。
     - 列：`sensor_pair`, `sensor_names`, `valid_heartbeats`, `mean_ptt_ms`, `std_ptt_ms`, `min_ptt_ms`, `max_ptt_ms`, `median_ptt_ms`, `q25_ptt_ms`, `q75_ptt_ms`。
  6. **`ptt_timeseries_exp_X.csv`**：
     - PTT 时间序列数据，用于后续血压建模。
     - 列：`heartbeat_id`, `sensor_pair`, `sensor_names`, `ptt_ms`, `sensorX_time_s`, `sensorY_time_s`。
  7. **`ir_peaks_exp_X.png`**：
     - 每个实验的 IR 通道信号和检测到的峰值可视化（每个传感器一个子图，标记峰值编号）。

### 示例输出
```
🩺 IR通道专门的PTT峰值检测器（批量处理版）
============================================================
📖 优化特性:
   • 专注IR通道峰值检测
   • 稳健的IBI计算和质量控制
   • 智能心跳匹配
   • 标准化CSV输出便于建模
   • 批量处理所有实验，存储到expX子文件夹
============================================================

🔬 开始IR通道PTT峰值检测分析（批量处理）
📋 实验列表: ['1', '2', '3']
🎯 检测策略:
   - 专注IR通道（信号质量最佳）
   - 稳健峰值检测 + IBI质量控制
   - 心率范围: 50-200 BPM
   - 滤波范围: 0.5-3.0 Hz
   - 输出5个标准CSV文件，按expX子文件夹存储

处理实验: 100%|██████████| 3/3 [00:15<00:00, 5.00s/exp]
🔍 开始处理实验 1
  🟢 sensor2(nose): 750峰值, HR=72.5±5.2BPM, IBI=750-900ms, 质量=excellent(95%)
  🟡 sensor3(finger): 740峰值, HR=73.1±4.8BPM, IBI=760-910ms, 质量=good(90%)
  🟠 sensor4(wrist): 700峰值, HR=70.2±6.1BPM, IBI=780-950ms, 质量=fair(85%)
  🟢 sensor5(ear): 755峰值, HR=72.8±5.0BPM, IBI=745-890ms, 质量=excellent(96%)
📍 有效传感器: ['sensor2', 'sensor3', 'sensor5']
📍 参考传感器: sensor2 (质量: excellent)
📊 完整心跳数量: 730/750
📊 PTT分析 (3个传感器组合):
  📊 nose→finger: 45.2±3.5ms (730心跳)
  📊 nose→ear: 38.7±3.1ms (730心跳)
  📊 finger→ear: -6.5±2.8ms (730心跳)
💾 保存传感器汇总: PI_Lab/ptt_output/exp_1/sensor_summary_exp_1.csv
💾 保存所有峰值: PI_Lab/ptt_output/exp_1/all_peaks_exp_1.csv
💾 保存所有IBI: PI_Lab/ptt_output/exp_1/all_ibi_exp_1.csv
💾 保存匹配心跳: PI_Lab/ptt_output/exp_1/matched_heartbeats_exp_1.csv
💾 保存PTT矩阵: PI_Lab/ptt_output/exp_1/ptt_matrix_exp_1.csv
💾 保存PTT时间序列: PI_Lab/ptt_output/exp_1/ptt_timeseries_exp_1.csv
   📈 共2190个PTT数据点，可用于血压建模
📊 保存可视化: PI_Lab/ptt_output/exp_1/ir_peaks_exp_1.png

✅ IR通道PTT峰值检测完成！
📁 结果保存在: PI_Lab/ptt_output/exp_X
📊 输出文件说明:
   1. sensor_summary_exp_X.csv - 传感器质量汇总
   2. all_peaks_exp_X.csv - 所有峰值详细信息
   3. all_ibi_exp_X.csv - 所有IBI详细信息
   4. ptt_matrix_exp_X.csv - PTT矩阵汇总
   5. ptt_timeseries_exp_X.csv - PTT时间序列（用于建模）
🎯 下一步: 使用ptt_timeseries_exp_X.csv进行血压建模
```

## 注意事项
1. **依赖库**：推荐安装 `neurokit2` 以获得最佳峰值检测效果。若未安装，脚本将自动回退到 `heartpy` 或 `scipy` 方法。
2. **输入文件**：确保 `data_path` 指向正确的 HUB 数据 CSV 文件目录，且文件格式与 `preprocess_step2.py` 输出一致。
3. **信号质量**：脚本通过 IBI 验证（300-1200 ms）评估信号质量，低质量传感器（`poor` 或 `error`）可能导致 PTT 计算失败。
4. **心跳匹配**：至少需要 2 个高质量传感器（`excellent` 或 `good`）才能计算 PTT。若有效传感器不足，脚本会尝试放宽标准（包括 `fair` 质量）。
5. **性能优化**：对于大规模数据，建议分批处理或增加内存资源（尤其在处理多个实验时）。
6. **可视化**：需要在支持 `matplotlib` 的环境中运行（如 Jupyter 或本地 Python）。若在非交互式环境中，图像将保存为 PNG 文件。
7. **错误处理**：
   - 若出现 `文件不存在` 或 `数据列不足`，检查输入 CSV 文件路径和格式。
   - 若出现 `峰值检测失败` 或 `PTT分析失败`，检查 IR 信号是否包含足够有效数据（至少 5 个峰值）。
8. **后续步骤**：使用 `ptt_timeseries_exp_X.csv` 进行血压建模，建议验证 PTT 与血压的相关性（如线性模型 `a*PTT + b`）。
- **时间**：2025年7月12日