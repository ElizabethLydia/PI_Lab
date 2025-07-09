### README

#### 项目概述
此代码是一个综合生理信号处理脚本，旨在处理清华大学实验室中的Biopac和HUB设备数据，用于PTT（脉搏传输时间）预测血压的项目。该脚本由清华大学实验室编写，整合了数据同步、可视化、对比分析和心率计算等功能，适用于处理大规模生理信号数据（如PPG、血压、心率等）。

#### 功能模块
1. **Biopac同步模块**：
   - 解析Biopac导出的CSV文件头信息，提取时间戳、采样间隔和通道信息。
   - 将数据转换为标准CSV格式，并根据时间段自动切分到对应段文件夹。
   - 支持自动移动Biopac分段文件到指定目录。

2. **对比分析模块**：
   - 比较Segment 1和Segment 7的数据，针对指定通道（默认`ir`）进行频率与平均血压的相关性分析。
   - 使用带通滤波（0.5-3Hz）和波峰检测，生成趋势图。

3. **可视化模块**：
   - 生成多种图表，包括分布分析（Biopac和Oximeter）、信号叠加图、子图、传感器通道网格图、滤波叠加图和PSD（功率谱密度）分析。
   - 支持自动时间窗口筛选（基于`sensor2.csv`）和数据滤波（0.5-3Hz）。

4. **心率分析模块**：
   - 自动计算所有段中`hr.csv`文件的平均心率，并提供汇总统计（总体平均、最高、最低）。

#### 依赖环境
- Python 3.x
- 依赖库：
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `scipy`
  - `argparse`
  - `csv`
  - `os`
  - `glob`
  - `datetime`
  - `collections`

安装依赖：
```bash
pip install pandas numpy matplotlib scipy
```

#### 文件结构
- **输入**：Biopac导出的CSV文件（包含头信息和数据）。
- **输出目录**（默认`output`）：
  - 分段CSV文件（`mean_bp.csv`, `sensor*.csv`等）。
  - `visualization`文件夹：包含所有图表。
- **目录结构示例**：
  ```
  /root/PI_Lab/
  ├── 1/
  │   ├── Biopac/
  │   ├── HUB/
  │   ├── Oximeter/
  ├── 2/
  │   ├── Biopac/
  │   ├── HUB/
  │   ├── Oximeter/
  ├── output/
  │   ├── mean_bp.csv
  │   ├── sensor1.csv
  │   ├── visualization/
  ```

#### 使用方法
1. **准备数据**：
   - 确保输入文件为Biopac导出的CSV，包含头信息（如时间戳、通道）。
   - 数据目录下有编号文件夹（1-11），每个文件夹包含`Camera1/timestamps.csv`用于时间段划分。

2. **运行脚本**：
   - 使用命令行执行：
     ```bash
     python data_processor.py input_file.csv -o output -t 8
     ```
   - 参数说明：
     - `input_file`：Biopac CSV文件路径。
     - `-o`/`--output`：输出目录（默认`output`）。
     - `-t`/`--timezone`：UTC时差（默认8小时，适用于中国时间）。

3. **输出结果**：
   - 控制台日志显示处理进度。
   - 输出目录中生成分段CSV文件。
   - `visualization`文件夹中保存图表（如`biopac_distribution_analysis_seg1.png`）。

#### 注意事项
- 确保输入文件编码为UTF-8。
- 时间戳格式需与脚本解析规则匹配（`YYYY-MM-DD HH:MM:SS.mmm`）。
- 若无`sensor2.csv`或时间戳异常，可手动指定`start_ts`和`end_ts`（暂未实现）。
- 大数据量处理可能较慢，考虑优化或分段运行。

#### 联系方式
如有问题，请联系清华大学实验室相关负责人或开发者。

---

### 代码理解

#### 总体设计
该脚本是一个综合性工具，分为多个模块，旨在处理清华大学实验室的生理信号数据（Biopac和HUB设备）。代码结构清晰，采用模块化设计，通过`argparse`支持命令行参数，适合自动化批量处理。以下是对主要部分的理解：

1. **全局配置**：
   - `sensor_mapping`：定义传感器位置映射（如`sensor2`对应`nose`），便于数据标识。
   - 忽略警告（`warnings.filterwarnings('ignore')`），避免日志干扰。

2. **Biopac同步模块**：
   - **`parse_biopac_header`**：
     - 解析文件头，提取记录时间（转换为UTC时间戳）、采样间隔和通道信息。
     - 支持多语言头信息（中文/英文），灵活性高。
   - **`convert_biopac_to_csv`**：
     - 读取Biopac数据，生成独立CSV文件（按通道分）。
     - 自动切分数据到段文件夹（基于`Camera1/timestamps.csv`）。
   - **时间处理**：将本地时间转换为UTC时间，考虑时差（默认8小时）。

3. **对比分析模块**：
   - **`load_comparison_data`**：加载Segment中的HUB和Biopac数据，仅处理`mean_bp.csv`。
   - **`get_comp_peak_stats`**：通过滤波和波峰检测计算频率。
   - **`find_data_for_comparison`**：在10秒窗口内寻找一致的频率数据，生成对比样本。
   - **`plot_comparison`**：绘制Segment 1与7的频率-血压散点图，含趋势线。

4. **可视化模块**：
   - **滤波**：`bandpass_filter`（0.5-3Hz）去除噪声，适配心率信号。
   - **分布分析**：`plot_biopac_distribution_analysis`和`plot_oximeter_distribution_analysis`生成柱状图，含统计信息。
   - **信号图**：`plot_combined`和`plot_subplots`显示真值和血氧信号。
   - **传感器图**：`plot_channels_grid`和`plot_channels_separately`展示HUB传感器数据。
   - **叠加图**：`plot_all_channels_overlay_filtered`叠加滤波后信号。
   - **PSD分析**：`plot_psd_analysis`计算功率谱密度，提取心率特征。

5. **心率分析**：
   - **`calculate_average_hr`**：读取`hr.csv`，计算每个段的平均心率并汇总。

6. **主程序**：
   - **`main`**：按步骤执行数据同步、可视化、心率分析和对比分析。
   - 支持中断处理和异常捕获，日志详细。

#### 代码特点
- **鲁棒性**：异常处理完善（如空文件、时间戳错误）。
- **自动化**：基于时间戳自动切分和窗口选择。
- **可扩展性**：模块化设计，便于添加新功能（如更多滤波或分析）。
- **可视化丰富**：生成多种图表，覆盖分布、时间序列和频谱。

#### 改进方向
- **性能优化**：大文件可加入并行处理或降采样。
- **时间窗口灵活性**：支持手动指定`start_ts`和`end_ts`。
- **重复值处理**：当前未优化，建议参考之前降采样方案。
- **文档**：添加函数级注释和参数说明。

此代码是为清华实验室定制，针对特定数据格式和项目需求优化，适合PTT预测血压的预处理和分析流程。