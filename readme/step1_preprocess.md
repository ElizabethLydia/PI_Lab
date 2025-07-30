# PI-Lab 数据预处理脚本 (step1_preprocess.py) README

## 概述

`step1_preprocess.py` 是 PTT（脉搏传输时间）预测血压项目 (`PI_Lab`) 的数据预处理脚本，用于处理 PhysioNet2025 数据集中的 Biopac 高频数据和 HUB 低频数据，为后续 PTT 分析（如 ECG/PPG 信号处理）提供支持。该脚本运行于 Python 3.10 环境，支持 GPU 加速（CUDA 12.2，Tesla V100），并生成对齐后的数据，保存为 `.pkl` 和 `.npy` 格式。

### 功能
- **Biopac 数据处理**（原本2000hz左右）：
  - 降采样到目标频率（默认 100Hz），减少高频数据量。
  - 使用插值处理重复时间戳，确保时间序列唯一性。
- **HUB 数据处理**（原本111hz左右）：
  - 通过插值去重，保持低频数据精度。
- **数据对齐**：
  - 以 HUB 的 `sensor2`（若可用）或最短时间序列为参考，使用线性插值对齐所有数据。
- **输出格式**：
  - 保存对齐数据为 `.pkl`（序列化字典）和 `.npy`（NumPy 格式）文件。
- **优化**：
  - 支持指定受试者 ID（如 `[112]`），串行处理以提高稳定性。
  - 强制重新处理数据（无输出目录检查）。
  - 使用清华大学镜像加速依赖安装。

## 安装与依赖

### 依赖库
确保 Python 3.10 环境已安装以下库：
- `numpy`
- `pandas`
- `scipy`
- `tqdm`
- `matplotlib`（可选，用于后续可视化）
- `neurokit2`（用于后续 PTT/ECG/PPG 分析）
- `pytorch`（支持 CUDA 12.2，深度学习模块如 `causal_conv1d_cuda`）
- `pickle`（Python 内置）

### 安装步骤
1. **克隆项目**：
   ```bash
   git clone https://github.com/ElizabethLydia/PI_Lab
   cd PI_Lab
   ```

2. **创建 Conda 环境**：
   ```bash
   conda create -n ptt_bp python=3.10 -y
   conda activate ptt_bp
   ```

3. **配置清华大学镜像**：
   ```bash
   conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
   conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
   conda config --set show_channel_urls yes
   pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
   ```

4. **安装依赖**：
   ```bash
   conda install -y numpy pandas scipy scikit-learn matplotlib seaborn tqdm pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia
   pip install neurokit2
   ```

5. **验证安装**：
   ```bash
   python -c "import numpy, pandas, scipy, tqdm, neurokit2, torch; print('依赖已安装'); print('PyTorch CUDA:', torch.cuda.is_available())"
   ```
   预期输出：
   ```
   依赖已安装
   PyTorch CUDA: True
   ```

## 使用方法

### 前提条件
- **数据目录结构**：
  - 数据位于 `/root/shared/PhysioNet2025/<date_folder>/<subject_id>`（如 `2025-01-01/00112`）。
  - 每个受试者包含数字命名的实验文件夹（如 `1`, `2`）。
  - 实验文件夹包含 `Biopac` 和 `HUB` 子目录，存储 `.csv` 文件。
- **时间戳要求**：
  - 每个 `.csv` 文件需包含 `timestamp` 列（浮点数，单位秒）。
- **输出目录**：
  - 确保 `/root/autodl-tmp/<subject_id>/output` 有写权限，用于保存 `.pkl` 和 `.npy` 文件。

### 运行脚本
1. **修改 `subject_ids`**：
   编辑 `step1_preprocess.py`，设置目标受试者 ID：
   ```python
   subject_ids = [112, 113]  # 示例：处理 00112 和 00113
   ```
   - 这会生成 target_subjects 如 ['00112', '00113']，并过滤只处理这些受试者。
   - **批量处理所有受试者**：注释掉以下行：
     ```python
     # subject_ids = [112]
     # target_subjects = [f'00{num:03d}' for num in subject_ids]
     # subject_folders = [s for s in all_subject_folders if s in target_subjects]
     ```
     然后添加或修改为：
     ```python
     subject_folders = sorted(all_subject_folders)
     ```
     这将处理日期文件夹下的所有 '00xxx' 格式受试者。修改后保存文件并重新运行。

2. **执行脚本**：
   ```bash
   conda activate ptt_bp
   cd /root/projects/PI_Lab
   python step1_preprocess.py
   ```
   - 若需后台运行（防止中断）：
     ```bash
     nohup python step1_preprocess.py &
     tail -f nohup.out
     ```

3. **脚本功能**：
   - 扫描 `/root/shared/PhysioNet2025/<date_folder>/<subject_id>`，处理指定受试者的实验。
   - Biopac 数据：降采样到 100Hz，插值处理重复时间戳。
   - HUB 数据：插值去重，保持精度。
   - 插值对齐：以 HUB `sensor2`（或最短时间序列）为参考。
   - 保存结果到 `/root/autodl-tmp/<subject_id>/output`。

### 参数说明
- `dataset_root = '/root/shared/PhysioNet2025/'`：数据根目录。
- `TARGET_FREQ = 100`：Biopac 降采样目标频率（Hz）。
- `MAX_EXPERIMENTS = None`：限制实验数量，`None` 表示处理全部。
- `subject_ids = [112]`：指定受试者 ID 列表；为空或注释掉相关逻辑可处理所有受试者（如上所述）。

## 输出说明

### 日志输出
- 显示实验加载和对齐进度。
- Biopac：降采样信息（原始频率、目标频率、压缩比）。
- HUB：重复时间戳修复数量。
- 统计：处理时间、实验数量、文件数量。

**示例**：
```
============================================================
处理subject: 00112 在 2025-01-01
发现实验: ['1', '2']
处理实验 1
==================================================
Biopac文件 (9 个) - 降采样+插值策略:
    Biopac mean_bp-11.csv (240,059 行)
      估算频率: 2000.0Hz -> 目标: 100Hz
      降采样: 240,059 -> 12,003 行 (步长: 20)
      最终: 12,003 行 (压缩比: 20.0:1)
HUB文件 (4 个) - 使用插值策略:
    HUB sensor2.csv (66,669 行)
      修复了 10,456 个重复时间戳
      最终: 56,213 行
插值对齐阶段
==================================================
对齐实验 1...
  使用 sensor2 作为参考 (56,213 行)
  保存PKL: /root/autodl-tmp/00112/output/experiment_1_aligned.pkl
  保存NPY: /root/autodl-tmp/00112/output/experiment_1_aligned.npy, 大小: 2.50 MB
============================================================
处理完成统计
============================================================
成功处理实验: 2
数据加载耗时: 10.5 秒
数据对齐耗时: 5.2 秒
总处理耗时: 15.7 秒
平均每个实验: 7.8 秒
```

### 文件输出
- **路径**：`/root/autodl-tmp/<subject_id>/output/`（如 `/root/autodl-tmp/00112/output`）。
- **格式**：
  - `.pkl`：序列化数据字典，文件名如 `experiment_1_aligned.pkl`。
  - `.npy`：NumPy 格式，文件名如 `experiment_1_aligned.npy`。
- **内容**：字典结构，键为实验 ID（如 `1`），值为 `biopac` 和 `hub` 数据（DataFrame）。
- **文件大小**：以 MB 为单位显示，取决于数据量。

### 输出说明

- **CSV 文件**：
  - 路径：`/root/autodl-tmp/<subject_id>/csv_output/`（如 `/root/autodl-tmp/00112/csv_output`）。
  - Biopac：整合文件如 `<subject>_<experiment>_biopac_aligned.csv`，包含所有Biopac通道的数据，对齐到参考时间戳，并使用前后填充处理NaN。
  - HUB：独立文件如 `<subject>_<experiment>_hub_<key>_aligned.csv`，每个HUB传感器一个文件，包含对齐后的数据。

### 文件层级结构

#### 输入数据结构
- 数据根目录：`/root/shared/PhysioNet2025/`
  - 日期文件夹：`<date_folder>`（如 `2025-01-01`）
    - 受试者文件夹：`<subject_id>`（如 `00112`）
      - 实验文件夹：`<experiment>`（如 `1`, `2`）
        - Biopac 子目录：`Biopac/` 包含多个 `.csv` 文件（如 `mean_bp-11.csv`）
        - HUB 子目录：`HUB/` 包含多个 `.csv` 文件（如 `sensor2.csv`）

#### 输出数据结构
- 输出根目录：`/root/autodl-tmp/<subject_id>/`
  - 输出文件夹：`output/` 包含对齐数据文件
    - `.pkl` 文件：`experiment_<experiment>_aligned.pkl`（序列化字典，包含对齐后的Biopac和HUB数据）
    - `.npy` 文件：`experiment_<experiment>_aligned.npy`（NumPy格式，包含相同数据）
  - CSV 输出文件夹：`csv_output/` 包含CSV格式文件
    - Biopac CSV：`<subject>_<experiment>_biopac_aligned.csv`（整合所有Biopac通道）
    - HUB CSV：`<subject>_<experiment>_hub_<key>_aligned.csv`（每个HUB传感器独立文件）

## 示例运行

```bash
conda activate ptt_bp
cd /root/projects/PI_Lab
python step1_preprocess.py
```

**预期结果**：
- `/root/autodl-tmp/00112/output/`：包含 `.pkl` 和 `.npy` 文件。
- `/root/autodl-tmp/00112/csv_output/`：包含对应的`csv`文件。
- 日志（终端或 `nohup.out`）记录处理进度和统计。
