

# PI-Lab step3_ir_ptt_peak_detector2 README

## 概述
`step3_ir_ptt_peak_detector2.py` 是一个专为脉搏传导时间（PTT）峰值检测设计的 Python 脚本，针对 HUB 设备红外（IR）通道数据进行处理，旨在为血压预测模型提供高质量的 PTT 数据。本脚本基于师兄的建议进行了优化，采用**窗口化时频域验证**策略，通过密集滑动窗口 **（20秒窗口，5秒步长）** 确保**时域峰值检测与频域傅里叶心率分析的一致性**，仅在验证通过的窗口内计算 PTT，从而提高数据可靠性。核心特性包括：

- **专注 IR 通道**：IR 信号质量最佳，优先用于峰值检测。
- ==**窗口化分析**：使用 20 秒滑动窗口（5 秒步长），进行密集时频域验证。==
- ==**时频域一致性**：时域心率（峰值检测）与频域心率（FFT）差异 < 5 BPM 的窗口才视为有效。==
- **多方法峰值检测**：支持 `neurokit2`（优先）、`heartpy` 或改进的 `scipy` 方法，自动选择最佳方案。
- **IBI 质量控制**：通过心跳间期（IBI，300-1200 ms）验证峰值可靠性，确保心率在 50-200 BPM 范围内。
- **跨传感器心跳匹配**：对 4 个传感器（nose、finger、wrist、ear）生成 6 种 PTT 组合（如 nose→finger）。
- **批量处理**：自动处理所有实验数据，输出标准化 CSV 文件，存储到 `exp_X` 子文件夹。
- **可视化**：生成窗口验证状态图和时频域心率对比图，便于结果验证和学术报告。

## 安装与依赖
### 依赖库
确保 Python 环境中已安装以下库：
- `numpy`：用于数值计算。
- `pandas`：用于数据处理和 CSV 文件操作。
- `scipy`：用于信号滤波、峰值检测和傅里叶分析。
- `tqdm`：用于显示处理进度。
- `matplotlib`：用于生成窗口验证和心率对比可视化。
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
- **输出目录**：确保 `PI_Lab/ptt_output2/`（或自定义路径）有读写权限，用于保存结果文件（每个实验存储在 `exp_X` 子文件夹中）。
- **环境要求**：Python 3.6+，支持 `matplotlib` 的图形环境（若在非交互式环境中运行，图像将自动保存为 PNG 文件）。

### 运行脚本
1. 修改脚本中的 `data_path` 变量（默认 `PI_Lab/output/csv_output`），确保指向 HUB 数据 CSV 文件目录：
   ```python
   detector = IRWindowedPTTPeakDetector(data_path='/your/csv/path')
   ```
2. 执行脚本：
   ```bash
   python step3_ir_ptt_peak_detector2.py
   ```
3. 脚本运行流程：
   - 扫描 `data_path` 目录，识别所有实验（如 `1`, `2`, ...）的 HUB 数据文件（基于 `X_hub_sensor2_aligned.csv`）。
   - 对每个实验的 IR 通道数据进行带通滤波（0.5-3.0 Hz）。
   - 使用 20 秒滑动窗口（5 秒步长）进行密集分析，计算时域心率（峰值检测）和频域心率（FFT）。
   - 验证窗口有效性：仅当时域心率与频域心率差异 < 5 BPM 且至少检测到 3 个峰值时，窗口才有效。
   - 在有效窗口内跨传感器匹配同一心跳的峰值，生成 6 种 PTT 组合（nose→finger, nose→wrist, nose→ear, finger→wrist, finger→ear, wrist→ear）。
   - 保存 5 个标准化 CSV 文件和 2 个可视化图像到 `PI_Lab/ptt_output2/exp_X/`。
   - 输出详细日志，包括窗口验证状态、心率对比、PTT 统计和文件路径。

### 参数说明
- **`data_path`**：HUB 数据 CSV 文件目录（默认 `PI_Lab/output/csv_output`）。
- **`output_dir`**：结果存储目录（默认 `PI_Lab/ptt_output2`）。
- **`sensors`**：处理的传感器列表（默认 `['sensor2', 'sensor3', 'sensor4', 'sensor5']`，对应 nose、finger、wrist、ear）。
- **`fs`**：采样率（动态计算，基于时间戳差值，默认为 100 Hz）。
- **`min_hr = 50`, `max_hr = 200`**：心率范围（BPM），用于 IBI 验证（300-1200 ms）。
- **`filter_lowcut = 0.5`, `filter_highcut = 3.0`**：带通滤波频率范围（Hz）。
- **`refractory_period = 0.3`**：峰值检测的最小间隔（秒，相当于 200 BPM）。
- **`window_duration = 20`**：窗口大小（秒）。
- **`window_step = 5`**：滑动窗口步长（秒，密集滑窗）。
- **`hr_tolerance_bpm = 5`**：时频域心率差异容忍度（BPM）。
- **`method = 'auto'`**：峰值检测方法，自动选择 `neurokit2`（优先）、`heartpy` 或 `scipy_advanced`。
- **`ibi_tolerance = 0.15`**：IBI 容差，用于心跳匹配。

## 输出说明
### 日志输出
脚本在终端显示详细处理过程，包括：
- **实验列表**：显示待处理的实验编号（如 `['1', '2', '3']`）。
- **处理进度**：使用 `tqdm` 进度条显示批量处理进度。
- **窗口分析**：
  - 每个传感器的窗口数量、有效窗口比例、时域心率（`peak_hr_bpm`）、频域心率（`fft_hr_bpm`）和心率差异（`hr_diff_bpm`）。
  - 示例：`✅ 有效窗口: 30/40 (75.0%)`。
- **心跳匹配**：显示多传感器有效窗口数量和匹配的心跳数量。
- **PTT 分析**：每对传感器组合的 PTT 统计（均值、标准差、有效心跳数、窗口数）。
- **文件保存**：列出所有输出文件的路径。
- **总结**：说明输出文件用途和下一步建议（如血压建模）。

### 文件输出
- **存储路径**：`PI_Lab/ptt_output2/exp_X/`（每个实验一个子文件夹，例如 `exp_1`）。
- **文件格式**：5 个标准化 CSV 文件和 2 个可视化图像（PNG 格式）。
  1. **`window_validation_exp_X.csv`**：
     - 窗口验证详情，记录每个窗口的时频域心率和有效性。
     - 列：`exp_id`, `sensor`, `sensor_name`, `window_id`, `start_time_s`, `end_time_s`, `duration_s`, `peak_count`, `peak_hr_bpm`, `fft_hr_bpm`, `hr_diff_bpm`, `is_valid`, `validation_status`。
     - 示例：nose 传感器的窗口 ID 1，`peak_hr_bpm=72.5`, `fft_hr_bpm=73.0`, `is_valid=True`。
  2. **`valid_peaks_exp_X.csv`**：
     - 有效窗口的峰值详细信息。
     - 列：`exp_id`, `sensor`, `sensor_name`, `window_id`, `peak_number_in_window`, `global_peak_time_s`, `global_peak_index`, `window_peak_hr_bpm`, `window_fft_hr_bpm`, `window_hr_diff_bpm`。
     - 示例：nose 传感器的窗口 ID 1，第 1 个峰值，时间 2.34 秒。
  3. **`matched_heartbeats_windowed_exp_X.csv`**：
     - 跨传感器匹配的心跳数据，记录每个心跳的峰值时间和窗口信息。
     - 列：`heartbeat_id`, `window_id`, `reference_sensor`, `window_start_s`, `window_end_s`, `sensor2_peak_time_s`, `sensor2_peak_idx`, `sensor3_peak_time_s`, `sensor3_peak_idx`, ...。
     - 示例：心跳 ID 1，窗口 ID 1，nose 峰值时间 2.34 秒，finger 峰值时间 2.38 秒。
  4. **`ptt_windowed_exp_X.csv`**：
     - 窗口化 PTT 时间序列数据，直接用于后续血压建模。
     - 列：`heartbeat_id`, `window_id`, `sensor_pair`, `sensor_names`, `ptt_ms`, `sensorX_time_s`, `sensorY_time_s`, `window_start_s`, `window_end_s`。
     - 示例：`nose→finger`，PTT=45.2 ms，nose 时间 2.34 秒，finger 时间 2.38 秒。
  5. **`ptt_summary_windowed_exp_X.csv`**：
     - PTT 统计汇总，汇总 6 种传感器组合的 PTT 统计信息。
     - 列：`sensor_pair`, `count`, `mean_ptt_ms`, `std_ptt_ms`, `min_ptt_ms`, `max_ptt_ms`, `median_ptt_ms`, `num_windows`。
     - 示例：`nose→finger`，均值 45.2 ms，标准差 3.5 ms，730 个有效心跳，30 个窗口。
  6. **`windowed_validation_exp_X.png`**：
     - 窗口验证状态图，显示每个传感器的 IR 信号、有效窗口（绿色背景）、无效窗口（红色背景）和检测到的峰值（红点）。
  7. **`hr_validation_exp_X.png`**：
     - 时频域心率对比图，显示每个传感器的有效窗口（绿色点）和无效窗口（红色点），标注 ±5 BPM 容忍带。

### 示例输出
```
🩺 密集滑窗时频域验证PTT峰值检测器（优化版）
======================================================================
📖 密集滑窗优化实现:
   • 20秒窗口，5秒密集滑窗步长
   • 时域峰值检测 vs 频域FFT心率验证
   • 心率差异<5BPM才认为窗口有效（放宽容忍度）
   • 只在有效窗口内计算PTT
   • 更多窗口，更细粒度的质量控制
======================================================================

🔬 开始窗口化时频域验证PTT分析（密集滑窗版）
📋 实验列表: ['1', '2', '3']
🎯 验证策略:
   - 20s窗口, 5s滑窗步长（密集覆盖）
   - 时域峰值检测 vs 频域FFT心率
   - 心率差异<5BPM才认为窗口有效（放宽容忍度）
   - 只在有效窗口内计算PTT
   - 详细的窗口质量报告

处理实验: 100%|██████████| 3/3 [00:18<00:00, 6.00s/exp]
🔍 开始处理实验 1
  📊 sensor2 计算采样率: 100.0Hz
  📊 nose: 创建了40个窗口
    ✅ 有效窗口: 30/40 (75.0%)
  📊 sensor3 计算采样率: 100.0Hz
  📊 finger: 创建了40个窗口
    ✅ 有效窗口: 28/40 (70.0%)
  📊 sensor4 计算采样率: 100.0Hz
  📊 wrist: 创建了40个窗口
    ✅ 有效窗口: 25/40 (62.5%)
  📊 sensor5 计算采样率: 100.0Hz
  📊 ear: 创建了40个窗口
    ✅ 有效窗口: 32/40 (80.0%)
📊 多传感器有效窗口: 30
💓 匹配的心跳数量: 650
💾 保存窗口验证汇总: PI_Lab/ptt_output2/exp_1/window_validation_exp_1.csv
💾 保存有效峰值: PI_Lab/ptt_output2/exp_1/valid_peaks_exp_1.csv
💾 保存窗口化匹配心跳: PI_Lab/ptt_output2/exp_1/matched_heartbeats_windowed_exp_1.csv
💾 保存窗口化PTT: PI_Lab/ptt_output2/exp_1/ptt_windowed_exp_1.csv
💾 保存PTT统计汇总: PI_Lab/ptt_output2/exp_1/ptt_summary_windowed_exp_1.csv
📊 窗口化PTT统计:
  sensor2-sensor3: 45.2±3.5ms (650心跳, 30窗口)
  sensor2-sensor5: 38.7±3.1ms (650心跳, 30窗口)
  sensor3-sensor5: -6.5±2.8ms (650心跳, 30窗口)
📊 保存窗口验证图: PI_Lab/ptt_output2/exp_1/windowed_validation_exp_1.png
📊 保存心率对比图: PI_Lab/ptt_output2/exp_1/hr_validation_exp_1.png

✅ 密集滑窗时频域验证PTT分析完成！
📁 结果保存在: PI_Lab/ptt_output2/exp_X
📊 输出文件说明:
   1. window_validation_exp_X.csv - 窗口验证详情
   2. valid_peaks_exp_X.csv - 有效窗口的峰值
   3. matched_heartbeats_windowed_exp_X.csv - 窗口化匹配心跳
   4. ptt_windowed_exp_X.csv - 窗口化PTT时间序列
   5. ptt_summary_windowed_exp_X.csv - PTT统计汇总
   6. windowed_validation_exp_X.png - 窗口验证状态图
   7. hr_validation_exp_X.png - 时频域心率对比图
🎯 密集滑窗验证完成！更多窗口，更高精度的PTT质量控制！
```

## 注意事项
1. **依赖库**：
   - 推荐安装 `neurokit2` 以获得最佳峰值检测性能。若未安装，脚本将回退到 `heartpy` 或 `scipy_advanced`，但精度可能略低。
   - 确保 `matplotlib` 可用以生成可视化图像。
2. **输入文件**：
   - 确保 `data_path` 指向正确的 HUB 数据 CSV 文件目录，文件格式与 `preprocess_step2.py` 输出一致（包含 `timestamp` 和 IR 通道列）。
   - 检查时间戳是否为浮点数（秒），避免单位错误（如毫秒）。
3. **窗口验证**：
   - 窗口有效性基于时域心率与频域心率差异 < 5 BPM 且至少检测到 3 个峰值。
   - 若有效窗口比例低（< 50%），检查信号质量或调整滤波参数（`filter_lowcut`, `filter_highcut`）。
4. **心跳匹配**：
   - 需要至少 2 个传感器在同一窗口内有效才能计算 PTT。
   - 峰值匹配时间窗为 ±0.2 秒，确保匹配的峰值属于同一心跳。
5. **时间戳问题**：
   - 如果时间戳异常（负值、零值或跳跃），脚本会使用默认采样率 100 Hz，可能导致峰值检测偏差。
   - 建议检查 `preprocess_step2.py` 的时间戳对齐逻辑。
6. **性能优化**：
   - 密集滑窗（20 秒窗口，5 秒步长）会增加计算量，对于大规模数据，建议增加内存资源或分批处理（指定 `experiment_list`）。
   - 示例：`detector.run_windowed_analysis(experiment_list=['1', '2'])`。
7. **可视化**：
   - 在非交互式环境（如服务器）运行时，图像将自动保存为 PNG 文件（`windowed_validation_exp_X.png` 和 `hr_validation_exp_X.png`）。
   - 建议在本地环境中检查图像，验证窗口有效性和时频域心率一致性。
8. **错误处理**：
   - **文件不存在**：检查 `X_hub_sensor{i}_aligned.csv` 是否存在，确认 `preprocess_step2.py` 已正确运行。
   - **数据列不足**：确保 CSV 文件包含至少 3 列（`timestamp`, 其他列，IR 通道）。
   - **峰值检测失败**：检查窗口信号是否包含足够有效数据（至少 3 个峰值），必要时调整滤波参数或峰值检测阈值：
     ```python
     thresholds = [
         (signal_mean + 0.15 * signal_std, 0.075 * signal_std),
         (signal_mean + 0.075 * signal_std, 0.0375 * signal_std)
     ]
     ```
   - **PTT 计算失败**：检查是否存在足够的多传感器有效窗口（至少 2 个传感器）。
9. **后续步骤**：
   - 使用 `ptt_windowed_exp_X.csv` 进行血压建模，推荐线性模型（`a*PTT + b`）或非线性模型（如随机森林）。
   - 检查 `window_validation_exp_X.csv` 和 `hr_validation_exp_X.png`，确保窗口有效性和心率一致性。
   - 使用 `windowed_validation_exp_X.png` 验证窗口划分和峰值检测结果，适合学术报告。

## 下一步建议
1. **数据验证**：
   - 检查 `window_validation_exp_X.csv`，确保有效窗口比例合理（> 50%），验证 `peak_hr_bpm` 和 `fft_hr_bpm` 的一致性。
   - 使用 `hr_validation_exp_X.png` 确认时域和频域心率点是否集中在 ±5 BPM 容忍带内。
   - 使用 `windowed_validation_exp_X.png` 验证有效窗口（绿色背景）是否覆盖主要信号区域，峰值（红点）是否准确。
2. **血压建模**：
   - 使用 `ptt_windowed_exp_X.csv` 与 Biopac 数据（`X_biopac_aligned.csv`）进行时间同步，计算 PTT 与血压的相关性。
   - 推荐使用 `ptt_bp_analysis.py` 进行相关性分析和回归建模。
3. **优化窗口参数**：
   - 若有效窗口比例低，尝试调整窗口大小（`window_duration=30`）或步长（`window_step=10`）：
     ```python
     self.window_duration = 30
     self.window_step = 10
     ```
   - 若心率差异较大，调整容忍度（`hr_tolerance_bpm=7`）或滤波参数（`filter_lowcut=0.3`, `filter_highcut=4.0`）。
4. **扩展实验**：
   - 收集更多实验数据（如 `exp_3`, `exp_4`），重新运行 `preprocess_step1.py` 和 `preprocess_step2.py`，生成缺失的 HUB 数据。
   - 验证多实验的 PTT 数据一致性，增强模型泛化能力。
5. **学术报告**：
   - 使用 `ptt_summary_windowed_exp_X.csv` 和可视化图像（`windowed_validation_exp_X.png`, `hr_validation_exp_X.png`）撰写分析报告，突出窗口化验证的高精度和可靠性。
   - 投稿至生物医学工程期刊（如 *IEEE Transactions on Biomedical Engineering* 或 *Physiological Measurement*）。

- **时间**：2025年7月14日

