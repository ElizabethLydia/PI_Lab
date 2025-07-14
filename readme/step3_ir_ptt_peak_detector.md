# PI-Lab step3_ir_ptt_peak_detector README

# PI-Lab step3_ir_ptt_peak_detector README

## 概述
`step3_ir_ptt_peak_detector.py` 是一个专为脉搏传导时间（PTT）峰值检测设计的 Python 脚本，针对清华大学实验室的 HUB 设备红外（IR）通道数据进行处理，旨在为后续血压预测模型提供高质量的 PTT 数据。该脚本基于师兄的建议进行了优化，专注于 IR 通道信号的峰值检测，并结合傅里叶心率分析和严格的质量控制，生成标准化的输出文件，适用于血压预测研究。核心特性包括：

- **专注 IR 通道**：IR 信号质量最佳，优先用于峰值检测。
- **多方法峰值检测**：支持 `neurokit2`（优先）、`heartpy` 或改进的 `scipy` 方法，自动选择最佳方案。
- **IBI 质量控制**：通过心跳间期（IBI，300-1200 ms）验证峰值可靠性，确保心率在 50-200 BPM 范围内。
- **跨传感器心跳匹配**：对 4 个传感器（nose、finger、wrist、ear）生成 6 种 PTT 组合（如 nose→finger）。
- **傅里叶心率验证**：通过功率谱密度（PSD）分析验证心率一致性，与 `data_processor.py` 的方法一致。
- **批量处理**：自动处理所有实验数据，输出标准化 CSV 文件，存储到 `exp_X` 子文件夹。
- **可视化**：生成 IR 信号峰值图和 PSD 图，便于结果验证和学术报告。

## 安装与依赖
### 依赖库
确保 Python 环境中已安装以下库：
- `numpy`：用于数值计算。
- `pandas`：用于数据处理和 CSV 文件操作。
- `scipy`：用于信号滤波、峰值检测和傅里叶分析。
- `tqdm`：用于显示处理进度。
- `matplotlib`：用于生成峰值检测和 PSD 可视化。
- `neurokit2`（可选，推荐用于专业峰值检测）。
- `heartpy`（可选，备选峰值检测库）。

### 安装步骤
1. 安装核心依赖：
   ```bash
   pip install numpy pandas scipy tqdm matplotlib
   ```
2. 安装可选库（推荐以获得最佳峰值检测效果）：
   ```bash
   pip install neurokit2 heartpy
   ```
   若无法安装 `neurokit2` 或 `heartpy`，脚本将自动回退到改进的 `scipy` 方法。

## 使用方法
### 前提条件
- **输入数据**：需要 `preprocess_step2.py` 生成的 HUB 数据 CSV 文件，位于 `PI_Lab/output/csv_output/`（或自定义路径），文件名格式为 `X_hub_sensor{i}_aligned.csv`（i=2,3,4,5，分别对应 nose、finger、wrist、ear）。
- **数据格式**：
  - 每列 CSV 文件需包含：
    - `timestamp`（第一列，浮点数，单位秒，实验开始后的相对时间）。
    - IR 通道信号（第三列，浮点数）。
  - 时间戳应与 `preprocess_step1.py` 和 `preprocess_step2.py` 的输出一致。
- **输出目录**：确保 `PI_Lab/ptt_output/`（或自定义路径）有读写权限，用于保存结果文件（每个实验存储在 `exp_X` 子文件夹中）。
- **环境要求**：Python 3.6+，支持 `matplotlib` 的图形环境（若在非交互式环境中运行，图像将自动保存为 PNG 文件）。

### 运行脚本
1. 修改脚本中的 `data_path` 变量（默认 `PI_Lab/output/csv_output`），确保指向 HUB 数据 CSV 文件目录：
   ```python
   detector = IRBasedPTTPeakDetector(data_path='/your/csv/path')
   ```
2. 执行脚本：
   ```bash
   python step3_ir_ptt_peak_detector.py
   ```
3. 脚本运行流程：
   - 扫描 `data_path` 目录，识别所有实验（如 `1`, `2`, ...）的 HUB 数据文件（基于 `X_hub_sensor2_aligned.csv`）。
   - 对每个实验的 IR 通道数据进行带通滤波（0.5-3.0 Hz）。
   - 使用指定方法（`neurokit2`、`heartpy` 或 `scipy_advanced`）检测峰值，计算 IBI 和心率。
   - 通过傅里叶分析（PSD）验证心率一致性。
   - 跨传感器匹配同一心跳的峰值，生成 6 种 PTT 组合（nose→finger, nose→wrist, nose→ear, finger→wrist, finger→ear, wrist→ear）。
   - 保存 5 个标准化 CSV 文件和 2 个可视化图像到 `PI_Lab/ptt_output/exp_X/`。
   - 输出详细日志，包括传感器质量、峰值数量、心率统计、PTT 分析和文件路径。

### 参数说明
- **`data_path`**：HUB 数据 CSV 文件目录（默认 `PI_Lab/output/csv_output`）。
- **`output_dir`**：结果存储目录（默认 `PI_Lab/ptt_output`）。
- **`sensors`**：处理的传感器列表（默认 `['sensor2', 'sensor3', 'sensor4', 'sensor5']`，对应 nose、finger、wrist、ear）。
- **`fs`**：采样率（动态计算，基于时间戳差值，默认为 100 Hz）。
- **`min_hr = 50`, `max_hr = 200`**：心率范围（BPM），用于 IBI 验证（300-1200 ms）。
- **`filter_lowcut = 0.5`, `filter_highcut = 3.0`**：带通滤波频率范围（Hz）。
- **`refractory_period = 0.3`**：峰值检测的最小间隔（秒，相当于 200 BPM）。
- **`method = 'auto'`**：峰值检测方法，自动选择 `neurokit2`（优先）、`heartpy` 或 `scipy_advanced`。
- **`ibi_tolerance = 0.15`**：IBI 容差，用于心跳匹配。

## 输出说明
### 日志输出
脚本在终端显示详细处理过程，包括：
- **实验列表**：显示待处理的实验编号（如 `['1', '2', '3']`）。
- **处理进度**：使用 `tqdm` 进度条显示批量处理进度。
- **传感器结果**：
  - 每个传感器的峰值数量、心率均值/标准差（BPM）、IBI 范围（ms）、傅里叶心率（BPM）、信号质量（`excellent`, `good`, `fair`, `poor`, `error`）。
  - 示例：`🟢 sensor2(nose): 750峰值, HR=72.5±5.2BPM, FFT HR=73.0BPM, IBI=750-900ms, 质量=excellent(95%)`。
- **心跳匹配**：显示有效传感器、参考传感器、完整心跳数量。
- **PTT 分析**：每对传感器组合的 PTT 统计（均值、标准差、有效心跳数）。
- **文件保存**：列出所有输出文件的路径。
- **总结**：说明输出文件用途和下一步建议（如血压建模）。

### 文件输出
- **存储路径**：`PI_Lab/ptt_output/exp_X/`（每个实验一个子文件夹，例如 `exp_1`）。
- **文件格式**：5 个标准化 CSV 文件和 2 个可视化图像（PNG 格式）。
  1. **`sensor_summary_exp_X.csv`**：
     - 传感器质量和统计汇总。
     - 列：`sensor`, `sensor_name`, `peak_count`, `quality`, `valid_ibi_ratio`, `hr_mean_bpm`, `hr_std_bpm`, `ibi_mean_ms`, `ibi_std_ms`, `rmssd_ms`, `pnn50_percent`, `signal_duration_s`, `fft_hr_bpm`, `fft_peak_freq_hz`。
     - 示例：记录 nose 传感器的 750 个峰值，HR=72.5±5.2 BPM，傅里叶心率 73.0 BPM。
  2. **`all_peaks_exp_X.csv`**：
     - 所有传感器的峰值详细信息。
     - 列：`sensor`, `sensor_name`, `peak_number`, `peak_index`, `peak_time_s`, `quality`。
     - 示例：nose 传感器的第 1 个峰值，时间 2.34 秒，质量 `excellent`。
  3. **`all_ibi_exp_X.csv`**：
     - 所有传感器的 IBI 详细信息。
     - 列：`sensor`, `sensor_name`, `ibi_number`, `ibi_ms`, `hr_bpm`, `is_valid`, `quality`。
     - 示例：finger 传感器的第 1 个 IBI，值 820 ms，心率 73.2 BPM，`is_valid=True`。
  4. **`matched_heartbeats_exp_X.csv`**：
     - 跨传感器匹配的心跳数据，记录每个心跳的峰值时间。
     - 列：`heartbeat_id`, `sensor2_peak_time_s`, `sensor3_peak_time_s`, `sensor4_peak_time_s`, `sensor5_peak_time_s`。
     - 示例：心跳 ID 1，nose 峰值时间 2.34 秒，finger 峰值时间 2.38 秒。
  5. **`ptt_matrix_exp_X.csv`**：
     - PTT 统计矩阵，汇总 6 种传感器组合的 PTT 统计信息。
     - 列：`sensor_pair`, `sensor_names`, `valid_heartbeats`, `mean_ptt_ms`, `std_ptt_ms`, `min_ptt_ms`, `max_ptt_ms`, `median_ptt_ms`, `q25_ptt_ms`, `q75_ptt_ms`。
     - 示例：`nose→finger`，均值 45.2 ms，标准差 3.5 ms，730 个有效心跳。
  6. **`ptt_timeseries_exp_X.csv`**：
     - PTT 时间序列数据，直接用于后续血压建模。
     - 列：`heartbeat_id`, `sensor_pair`, `sensor_names`, `ptt_ms`, `sensorX_time_s`, `sensorY_time_s`。
     - 示例：`nose→finger`，PTT=45.2 ms，nose 时间 2.34 秒，finger 时间 2.38 秒。
  7. **`ir_peaks_exp_X.png`**：
     - 每个实验的 IR 通道信号和检测到的峰值可视化（每个传感器一个子图，红点标记峰值，每 10 个峰值标注编号）。
  8. **`psd_exp_X.png`**：
     - 功率谱密度（PSD）分析图，显示每个传感器的 IR 通道频率分布，标注峰值心率（BPM）。

### 示例输出
```
🩺 IR通道专门的PTT峰值检测器（批量处理版，含傅里叶分析）
============================================================
📖 优化特性:
   • 专注IR通道峰值检测
   • 稳健的IBI计算和质量控制
   • 傅里叶心率分析验证（照抄data_processor.py）
   • 智能心跳匹配
   • 标准化CSV输出便于建模
   • 批量处理所有实验，存储到expX子文件夹
============================================================

🔬 开始IR通道PTT峰值检测分析（批量处理）
📋 实验列表: ['1', '2', '3']
🎯 检测策略:
   - 专注IR通道（信号质量最佳）
   - 稳健峰值检测 + IBI质量控制
   - 傅里叶心率分析验证（照抄data_processor.py）
   - 心率范围: 50-200 BPM
   - 滤波范围: 0.5-3.0 Hz
   - 输出5个标准CSV文件 + PSD图，按expX子文件夹存储

处理实验: 100%|██████████| 3/3 [00:15<00:00, 5.00s/exp]
🔍 开始处理实验 1
  🟢 sensor2(nose): 750峰值, HR=72.5±5.2BPM, FFT HR=73.0BPM, IBI=750-900ms, 质量=excellent(95%)
  🟡 sensor3(finger): 740峰值, HR=73.1±4.8BPM, FFT HR=72.8BPM, IBI=760-910ms, 质量=good(90%)
  🟠 sensor4(wrist): 700峰值, HR=70.2±6.1BPM, FFT HR=70.5BPM, IBI=780-950ms, 质量=fair(85%)
  🟢 sensor5(ear): 755峰值, HR=72.8±5.0BPM, FFT HR=73.2BPM, IBI=745-890ms, 质量=excellent(96%)
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
📊 保存IR信号图: PI_Lab/ptt_output/exp_1/ir_peaks_exp_1.png
📊 保存PSD图: PI_Lab/ptt_output/exp_1/psd_exp_1.png

✅ IR通道PTT峰值检测完成！
📁 结果保存在: PI_Lab/ptt_output/exp_X
📊 输出文件说明:
   1. sensor_summary_exp_X.csv - 传感器质量汇总（含傅里叶心率）
   2. all_peaks_exp_X.csv - 所有峰值详细信息
   3. all_ibi_exp_X.csv - 所有IBI详细信息
   4. ptt_matrix_exp_X.csv - PTT矩阵汇总
   5. ptt_timeseries_exp_X.csv - PTT时间序列（用于建模）
   6. psd_exp_X.png - 各传感器IR通道PSD图（与data_processor.py一致）
🎯 下一步: 使用ptt_timeseries_exp_X.csv进行血压建模，检查fft_hr_bpm验证心率一致性
```

## 注意事项
1. **依赖库**：
   - 推荐安装 `neurokit2` 以获得最佳峰值检测性能。若未安装，脚本将回退到 `heartpy` 或 `scipy_advanced`，但精度可能略低。
   - 确保 `matplotlib` 可用以生成可视化图像。
2. **输入文件**：
   - 确保 `data_path` 指向正确的 HUB 数据 CSV 文件目录，文件格式与 `preprocess_step2.py` 输出一致（包含 `timestamp` 和 IR 通道列）。
   - 检查时间戳是否为浮点数（秒），避免单位错误（如毫秒）。
3. **信号质量**：
   - 脚本通过 IBI 验证（300-1200 ms）评估信号质量，低质量传感器（`poor` 或 `error`）可能导致 PTT 计算失败。
   - 如果某个传感器信号质量差，检查原始数据是否存在噪声或缺失。
4. **心跳匹配**：
   - 需要至少 2 个高质量传感器（`excellent` 或 `good`，峰值数 ≥ 5，IBI 有效率 ≥ 50%）才能计算 PTT。
   - 若有效传感器不足，脚本会尝试放宽标准（包括 `fair` 质量，峰值数 ≥ 3），但可能降低 PTT 精度。
5. **时间戳问题**：
   - 如果时间戳异常（负值、零值或跳跃），脚本会使用默认采样率 100 Hz，可能导致峰值检测偏差。
   - 建议检查 `preprocess_step2.py` 的时间戳对齐逻辑。
6. **性能优化**：
   - 对于大规模数据（例如上万秒的信号），建议增加内存资源或分批处理（指定 `experiment_list`）。
   - 示例：`detector.run_analysis(experiment_list=['1', '2'])`。
7. **可视化**：
   - 在非交互式环境（如服务器）运行时，图像将自动保存为 PNG 文件（`ir_peaks_exp_X.png` 和 `psd_exp_X.png`）。
   - 建议在本地环境中检查图像，验证峰值检测和 PSD 分析结果。
8. **错误处理**：
   - **文件不存在**：检查 `X_hub_sensor{i}_aligned.csv` 是否存在，确认 `preprocess_step2.py` 已正确运行。
   - **数据列不足**：确保 CSV 文件包含至少 3 列（`timestamp`, 其他列，IR 通道）。
   - **峰值检测失败**：检查 IR 信号是否包含足够有效数据（至少 5 个峰值），必要时调整滤波参数（`filter_lowcut`, `filter_highcut`）或峰值检测阈值。
   - **PTT 分析失败**：检查传感器数据是否包含足够高质量峰值（`excellent` 或 `good`）。
9. **后续步骤**：
   - 使用 `ptt_timeseries_exp_X.csv` 进行血压建模，推荐线性模型（`a*PTT + b`）或非线性模型（如随机森林）。
   - 检查 `sensor_summary_exp_X.csv` 中的 `fft_hr_bpm` 和 `hr_mean_bpm`，验证心率一致性。
   - 使用 `ir_peaks_exp_X.png` 和 `psd_exp_X.png` 验证峰值检测和傅里叶分析结果，适合学术报告。
- **时间**：2025年7月13日