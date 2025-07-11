# PI-Lab Preprocess(step1/2) README

## 概述
此项目包含两个主要预处理脚本，用于处理Biopac高频数据和HUB低频数据，为PTT（脉搏传输时间）预测血压项目提供支持。两个脚本分别实现以下功能：

- **`preprocess_step1.py`**：  
  - 对Biopac数据进行降采样（目标频率100Hz）并处理重复时间戳。  
  - 对HUB数据进行插值去重，保持数据精度。  
  - 基于参考信号（优先使用HUB的`sensor2`）进行插值对齐。  
  - 以`.pkl`和`.npy`两种格式保存对齐后的数据。

- **`preprocess_step2.py`**：  
  - 从`preprocess_step1.py`生成的`.pkl`或`.npy`文件中加载对齐数据。  
  - 整合Biopac数据为单个CSV文件（例如`00017_biopac_aligned.csv`）。  
  - 将HUB数据按传感器分开保存为独立的CSV文件（例如`00017_hub_sensor{i}_aligned.csv`）。  
  - 提供数据可视化和统计分析功能，用于检查数据对齐情况和数据质量。

## 安装与依赖
### 依赖库
确保您的Python环境已安装以下库：
- `numpy`
- `pandas`
- `scipy`
- `tqdm`
- `pickle`（Python内置）
- `matplotlib`

### 安装步骤
1. 克隆或下载脚本到本地：
   ```bash
   git clone <https://github.com/ElizabethLydia/PI_Lab>  # 如果有版本控制
   cd <script-directory>
   ```
2. 创建虚拟环境（可选但推荐）：
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```
3. 安装依赖：
   ```bash
   pip install numpy pandas scipy tqdm matplotlib
   ```

我将帮您把 `preprocess_step1.py` 和 `preprocess_step2.py` 的功能和使用说明补充到现有的 `preprocess_readme.md` 中，保持文档结构清晰，并确保内容涵盖两个脚本的核心功能。以下是更新后的 README 文件，整合了两个脚本的说明，遵循原始 README 的风格和结构。



# PI-Lab 数据预处理脚本 README

## 概述
此项目包含两个主要预处理脚本，用于处理Biopac高频数据和HUB低频数据，为PTT（脉搏传输时间）预测血压项目提供支持。两个脚本分别实现以下功能：

- **`preprocess_step1.py`**：  
  - 对Biopac数据进行降采样（目标频率100Hz）并处理重复时间戳。  
  - 对HUB数据进行插值去重，保持数据精度。  
  - 基于参考信号（优先使用HUB的`sensor2`）进行插值对齐。  
  - 以`.pkl`和`.npy`两种格式保存对齐后的数据。

- **`preprocess_step2.py`**：  
  - 从`preprocess_step1.py`生成的`.pkl`或`.npy`文件中加载对齐数据。  
  - 整合Biopac数据为单个CSV文件（例如`00017_biopac_aligned.csv`）。  
  - 将HUB数据按传感器分开保存为独立的CSV文件（例如`00017_hub_sensor{i}_aligned.csv`）。  
  - 提供数据可视化和统计分析功能，用于检查数据对齐情况和数据质量。

## 安装与依赖
### 依赖库
确保您的Python环境已安装以下库：
- `numpy`
- `pandas`
- `scipy`
- `tqdm`
- `pickle`（Python内置）
- `matplotlib`（用于`preprocess_step2.py`中的数据可视化）

### 安装步骤
1. 克隆或下载脚本到本地：
   ```bash
   git clone <https://github.com/ElizabethLydia/PI_Lab>  # 如果有版本控制
   cd <script-directory>
   ```
2. 创建虚拟环境（可选但推荐）：
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```
3. 安装依赖：
   ```bash
   pip install numpy pandas scipy tqdm matplotlib
   ```

## 使用方法
### 前提条件
- **数据目录结构**：数据应位于`/root/PI_Lab/00017`（或自定义路径）下，包含数字命名的实验文件夹（如`1`, `2`），每个文件夹下有`Biopac`和`HUB`子目录，包含`.csv`文件。
- **时间戳要求**：每个`.csv`文件需包含`timestamp`列，时间戳为浮点数格式（秒）。
- **输出目录**：确保`/root/PI_Lab/output`（或自定义路径）有读写权限，用于保存`.pkl`、`.npy`和`.csv`文件。

### 运行脚本
#### 步骤 1：运行 `preprocess_step1.py`
1. 修改脚本中的`pi_lab_folder`变量以指定数据路径（默认`/root/PI_Lab/00017`）：
   ```python
   pi_lab_folder = '/your/data/path'
   ```
2. 执行脚本：
   ```bash
   python preprocess_step1.py
   ```
3. 脚本将：
   - 扫描数据目录，处理所有实验文件夹。
   - 对Biopac数据降采样（目标频率100Hz）并处理重复时间戳。
   - 对HUB数据进行插值去重。
   - 基于参考信号（优先HUB的`sensor2`）进行插值对齐。
   - 保存对齐结果到`/root/PI_Lab/output`下的`.pkl`和`.npy`文件。

#### 步骤 2：运行 `preprocess_step2.py`
1. 确保`preprocess_step1.py`已生成`.pkl`或`.npy`文件，位于`/root/PI_Lab/output`。
2. 修改脚本中的`output_dir`变量（默认`D:\code\Python\PI_Lab\output`）以匹配`.pkl`/`.npy`文件路径：
   ```python
   output_dir = '/your/output/path'
   ```
3. 执行脚本：
   ```bash
   python preprocess_step2.py
   ```
4. 脚本将：
   - 加载`.pkl`或`.npy`文件，检查数据完整性并打印统计信息（行数、列名、非空值、均值、标准差等）。
   - 可视化Biopac和HUB数据（`bp`和`sensor2`的`red`信号）以检查对齐情况。
   - 整合Biopac数据为单个CSV文件（例如`00017_biopac_aligned.csv`）。
   - 将HUB数据按传感器保存为独立CSV文件（例如`00017_hub_sensor2_aligned.csv`）。
   - 输出文件保存至`output_dir/csv_output`目录。

### 参数说明
#### `preprocess_step1.py`
- `TARGET_FREQ = 100`：目标采样频率（Hz），用于Biopac数据降采样。
- `MAX_EXPERIMENTS = None`：限制处理的实验数量，设为整数（如5）限制处理前5个实验，`None`表示处理全部。
- `pi_lab_folder`：数据根目录，需包含实验文件夹。

#### `preprocess_step2.py`
- `output_dir`：`.pkl`/`.npy`文件所在目录。
- `csv_dir`：输出CSV文件的存储目录（自动创建，默认为`output_dir/csv_output`）。

## 输出说明
### 日志输出
#### `preprocess_step1.py`
- 显示每个实验的加载和对齐进度。
- 提供Biopac文件的降采样信息（原始频率、目标频率、降采样步长、压缩比）。
- 显示HUB文件的重复时间戳修复情况。
- 输出最终统计（处理时间、文件数量等）。

#### `preprocess_step2.py`
- 显示加载的`.pkl`/`.npy`文件列表。
- 打印每个实验的数据统计（文件数、行数、列名、非空值、均值、标准差）。
- 显示保存的CSV文件路径。

### 文件输出
#### `preprocess_step1.py`
- **存储路径**：`/root/PI_Lab/output/`（或自定义`output_dir`）。
- **文件格式**：
  - `.pkl`：序列化保存对齐后的数据字典，文件名如`experiment_1_aligned.pkl`。
  - `.npy`：以NumPy格式保存对齐数据，文件名如`experiment_1_aligned.npy`。
- **文件内容**：包含一个字典，键为实验名，值为包含`biopac`和`hub`数据的嵌套字典。
- **文件大小**：以MB为单位显示，取决于数据量。

#### `preprocess_step2.py`
- **存储路径**：`output_dir/csv_output/`（自动创建）。
- **文件格式**：
  - Biopac数据：整合为单个CSV文件，文件名如`00017_biopac_aligned.csv`。
  - HUB数据：按传感器保存为独立CSV文件，文件名如`00017_hub_sensor{i}_aligned.csv`。
- **文件内容**：
  - Biopac CSV：包含`timestamp`列和所有Biopac信号列，缺失值使用前后填充。
  - HUB CSV：包含`timestamp`列（第一列）和传感器特定列。
- **可视化输出**：生成Biopac和HUB数据的对齐检查图（`bp`和`sensor2`的`red`信号）。

### 示例输出
#### `preprocess_step1.py`
```
🚀 PI-Lab智能数据预处理
============================================================
数据路径: /root/PI_Lab/00017
目标频率: 100Hz
策略: Biopac降采样 + HUB插值 + 插值对齐 + 双格式保存
============================================================
发现实验: ['1', '2', '3']
总共处理: 3 个实验

[1/3] 开始处理实验 1
==================================================
处理实验 1
==================================================
Biopac文件 (9 个) - 降采样+插值策略:
    Biopac mean_bp-11.csv (240,059 行)
      估算频率: 2000.0Hz -> 目标: 100Hz
      降采样: 240,059 -> 12,003 行 (步长: 20)
      插值处理重复时间戳...
      最终: 12,003 行 (压缩比: 20.0:1)
  ...
HUB文件 (4 个) - 使用插值策略:
    HUB sensor2.csv (66,669 行)
      插值处理重复时间戳...
      修复了 10,456 个重复时间戳
      最终: 56,213 行
  ...

插值对齐阶段
==================================================
对齐实验 1...
  使用 sensor2 作为参考 (56,213 行)
  ...
  保存PKL: /root/PI_Lab/output/experiment_1_aligned.pkl
  保存NPY (单文件): /root/PI_Lab/output/experiment_1_aligned.npy, 大小: 2.50 MB
```

#### `preprocess_step2.py`
```
找到 6 个文件: ['experiment_1_aligned.pkl', 'experiment_1_aligned.npy', ...]

加载实验: 1
  Biopac文件数: 9, HUB文件数: 4
  bp: 56,213 行, 列: ['timestamp', 'bp']
    非空值: {'timestamp': 56213, 'bp': 56213}
    均值: {'timestamp': 123456.78, 'bp': 120.45}
    标准差: {'timestamp': 1000.12, 'bp': 15.67}
  sensor2: 56,213 行, 列: ['timestamp', 'red', 'ir']
    非空值: {'timestamp': 56213, 'red': 56213, 'ir': 56213}
    均值: {'timestamp': 123456.78, 'red': 0.85, 'ir': 0.92}
    标准差: {'timestamp': 1000.12, 'red': 0.05, 'ir': 0.06}
  保存整合Biopac CSV: D:\code\Python\PI_Lab\output\csv_output\1_biopac_aligned.csv
  保存HUB CSV: D:\code\Python\PI_Lab\output\csv_output\1_hub_sensor2_aligned.csv
✅ 数据查看和保存完成！
```

## 注意事项
1. **数据路径**：
   - `preprocess_step1.py`：确保`pi_lab_folder`指向正确的数据目录（如`/root/PI_Lab/00017`）。
   - `preprocess_step2.py`：确保`output_dir`指向`preprocess_step1.py`生成的`.pkl`/`.npy`文件目录。
2. **时间戳格式**：时间戳需为浮点数（秒）。若为字符串格式，需预处理CSV文件。
3. **性能优化**：对于数百万行数据，建议分批处理或增加内存资源（尤其是`preprocess_step1.py`）。
4. **错误处理**：
   - 若出现`插值错误`或`警告: 无有效参考数据`，检查输入CSV文件是否包含`timestamp`列或数据是否完整。
   - 若`preprocess_step2.py`无法加载`.pkl`/`.npy`文件，检查文件是否存在或路径是否正确。
5. **可视化**：`preprocess_step2.py`需要`matplotlib`支持。若在非交互式环境中运行，需配置`%matplotlib inline`或保存图像到文件。
6. **版本更新**：脚本可能根据需求调整，建议定期检查更新。
- **时间**：2025年7月11日