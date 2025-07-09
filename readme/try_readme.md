# README

## 概述

通过 IPython 笔记本，逐步完成数据预处理前的分析和处理后的可视化检查。

---

## 验证环境

### 依赖库及版本检查

在开始之前，验证环境配置，确保所有依赖库正确安装。以下是必要的库及版本信息：

- **Pandas**: 用于数据处理和分析。
- **NumPy**: 提供数值计算支持。
- **SciPy**: 提供科学计算功能。
- **NeuroKit2**: 用于生理信号处理（如 PPG 和心率分析）。
- **Scikit-learn**: 提供机器学习相关工具。

运行以下代码以验证环境：

```bash
import pandas as pd
import numpy as np
import scipy as sp
import neurokit2 as nk
from sklearn import __version__ as sklearn_version

print(f"Pandas: {pd.__version__}")
print(f"NumPy: {np.__version__}")
print(f"SciPy: {sp.__version__}")
print(f"NeuroKit2: {nk.__version__}")
print(f"Scikit-learn: {sklearn_version}")
```

**安装依赖**（如需）：
```bash
pip install pandas numpy scipy neurokit2 scikit-learn
```

---

## 了解数据集结构

### 数据分析目标

本笔记本用于读取并分析 HUB 文件夹中的 PPG 数据和 Biopac 文件夹中的血压数据，检查每列的统计信息，为后续 PTT 预测血压奠定基础。

### 数据路径

- **HUB 数据路径**: `/root/PI_Lab/00017/1/HUB`（请根据实际路径修改）。
- **Biopac 数据路径**: `/root/PI_Lab/00017/1/Biopac`（请根据实际路径修改）。

(数据说明)[/root/PI_Lab/数据说明.txt]

```txt
HUB 文件: sensor4.csv
列名: ['timestamp', 'red', 'ir', 'green', 'ax', 'ay', 'az', 'rx', 'ry', 'rz', 'mx', 'my', 'mz', 'temp', 'time']
前2行数据预览:
      timestamp      red        ir   green    ax    ay    az    rx   ry    rz    mx     my    mz   temp      time
0  1.750838e+09  72802.0  124688.0  3427.0 -0.59  9.52  1.85 -3.35 -1.1 -3.72 -7.35  47.55 -4.95  34.48  218419.0
1  1.750838e+09  72802.0  124688.0  3427.0 -0.59  9.52  1.85 -3.35 -1.1 -3.72 -7.35  47.55 -4.95  34.48  218428.0

HUB 文件: sensor3.csv
列名: ['timestamp', 'red', 'ir', 'green', 'ax', 'ay', 'az', 'rx', 'ry', 'rz', 'mx', 'my', 'mz', 'temp', 'time']
前2行数据预览:
      timestamp       red        ir   green    ax    ay   az   rx     ry    rz    mx   my     mz   temp      time
0  1.750838e+09  119489.0  137421.0  5585.0 -9.48  3.28  0.2 -7.8 -21.65  0.06  31.5  7.5 -18.75  34.55  218419.0
1  1.750838e+09  119489.0  137421.0  5585.0 -9.48  3.28  0.2 -7.8 -21.65  0.06  31.5  7.5 -18.75  34.55  218428.0

HUB 文件: sensor5.csv
列名: ['timestamp', 'red', 'ir', 'green', 'ax', 'ay', 'az', 'rx', 'ry', 'rz', 'mx', 'my', 'mz', 'temp', 'time']
前2行数据预览:
      timestamp      red       ir   green    ax    ay    az    rx    ry    rz     mx     my    mz  temp      time
0  1.750838e+09  55395.0  60087.0  1867.0 -8.36 -5.75  0.85 -5.61  2.38  0.43  34.05 -13.05 -5.85  33.3  218420.0
1  1.750838e+09  55379.0  60101.0  1874.0 -8.37 -5.66  0.65 -6.22  2.38  0.43  34.05 -13.05 -5.85  33.3  218429.0

HUB 文件: sensor2.csv
列名: ['timestamp', 'red', 'ir', 'green', 'ax', 'ay', 'az', 'rx', 'ry', 'rz', 'mx', 'my', 'mz', 'temp', 'time']
前2行数据预览:
      timestamp      red       ir   green    ax    ay    az    rx    ry    rz     mx     my     mz  temp      time
0  1.750838e+09  58018.0  77928.0  3221.0  9.58  1.61  1.79 -4.51  0.24 -1.77  66.00  27.15  46.05  32.7  218427.0
1  1.750838e+09  58016.0  77966.0  3221.0  9.60  1.51  1.94 -5.24  0.67 -1.34  66.75  27.15  46.05  32.7  218436.0

BIOPAC 文件: diastolic_bp-1.csv
列名: ['timestamp', 'diastolic_bp']
前2行数据预览:
      timestamp  diastolic_bp
0  1.750838e+09       61.6516
1  1.750838e+09       61.6516

BIOPAC 文件: rsp-1.csv
列名: ['timestamp', 'rsp']
前2行数据预览:
      timestamp      rsp
0  1.750838e+09  1.58661
1  1.750838e+09  1.58569

BIOPAC 文件: systemic_vascular_resistance-1.csv
列名: ['timestamp', 'systemic_vascular_resistance']
前2行数据预览:
      timestamp  systemic_vascular_resistance
0  1.750838e+09                       870.514
1  1.750838e+09                       869.598

BIOPAC 文件: hr-1.csv
列名: ['timestamp', 'hr']
前2行数据预览:
      timestamp       hr
0  1.750838e+09  70.4639
1  1.750838e+09  70.4639

BIOPAC 文件: cardiac_output-1.csv
列名: ['timestamp', 'cardiac_output']
前2行数据预览:
      timestamp  cardiac_output
0  1.750838e+09         7.21130
1  1.750838e+09         7.21802

BIOPAC 文件: bp-1.csv
列名: ['timestamp', 'bp']
前2行数据预览:
      timestamp       bp
0  1.750838e+09  88.0188
1  1.750838e+09  88.2019

BIOPAC 文件: systolic_bp-1.csv
列名: ['timestamp', 'systolic_bp']
前2行数据预览:
      timestamp  systolic_bp
0  1.750838e+09      115.604
1  1.750838e+09      115.604

BIOPAC 文件: mean_bp-1.csv
列名: ['timestamp', 'mean_bp']
前2行数据预览:
      timestamp  mean_bp
0  1.750838e+09  82.8614
1  1.750838e+09  82.8614

BIOPAC 文件: cardiac_index-1.csv
列名: ['timestamp', 'cardiac_index']
前2行数据预览:
      timestamp  cardiac_index
0  1.750838e+09        4.79279
1  1.750838e+09        4.79095
```
---

## 数据大小与特征分析

### 分析目标

分析 PI-Lab 数据的大小、采样率、重复时间戳比例等特征，为后续预处理（如降采样或去重）提供依据。

```txt
=== PI-Lab数据分析 ===
分析路径: /root/PI_Lab/00017
发现实验: ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
分析前 3 个实验...

=== 分析实验: 1 ===

--- Biopac文件 ---

文件: diastolic_bp-1.csv
大小: 31.9 MB
列名: ['timestamp', 'diastolic_bp']
样本行数: 1000
估计采样率: 2000.0 Hz
平均时间间隔: 0.000500 秒
重复时间戳: 0/1000 (0.0%)
估计总行数: 1,000

文件: rsp-1.csv
大小: 28.2 MB
列名: ['timestamp', 'rsp']
样本行数: 1000
估计采样率: 2000.0 Hz
平均时间间隔: 0.000500 秒
重复时间戳: 0/1000 (0.0%)
估计总行数: 1,000

文件: systemic_vascular_resistance-1.csv
大小: 31.9 MB
列名: ['timestamp', 'systemic_vascular_resistance']
样本行数: 1000
估计采样率: 2000.0 Hz
平均时间间隔: 0.000500 秒
重复时间戳: 0/1000 (0.0%)
估计总行数: 1,000

文件: hr-1.csv
大小: 31.9 MB
列名: ['timestamp', 'hr']
样本行数: 1000
估计采样率: 2000.0 Hz
平均时间间隔: 0.000500 秒
重复时间戳: 0/1000 (0.0%)
估计总行数: 1,000

文件: cardiac_output-1.csv
大小: 31.9 MB
列名: ['timestamp', 'cardiac_output']
样本行数: 1000
估计采样率: 2000.0 Hz
平均时间间隔: 0.000500 秒
重复时间戳: 0/1000 (0.0%)
估计总行数: 1,000

文件: bp-1.csv
大小: 31.9 MB
列名: ['timestamp', 'bp']
样本行数: 1000
估计采样率: 2000.0 Hz
平均时间间隔: 0.000500 秒
重复时间戳: 0/1000 (0.0%)
估计总行数: 1,000

文件: systolic_bp-1.csv
大小: 31.9 MB
列名: ['timestamp', 'systolic_bp']
样本行数: 1000
估计采样率: 2000.0 Hz
平均时间间隔: 0.000500 秒
重复时间戳: 0/1000 (0.0%)
估计总行数: 1,000

文件: mean_bp-1.csv
大小: 31.9 MB
列名: ['timestamp', 'mean_bp']
样本行数: 1000
估计采样率: 2000.0 Hz
平均时间间隔: 0.000500 秒
重复时间戳: 0/1000 (0.0%)
估计总行数: 1,000

文件: cardiac_index-1.csv
大小: 31.9 MB
列名: ['timestamp', 'cardiac_index']
样本行数: 1000
估计采样率: 2000.0 Hz
平均时间间隔: 0.000500 秒
重复时间戳: 0/1000 (0.0%)
估计总行数: 1,000

--- HUB文件 ---

文件: sensor4.csv
大小: 7.2 MB
列名: ['timestamp', 'red', 'ir', 'green', 'ax', 'ay', 'az', 'rx', 'ry', 'rz', 'mx', 'my', 'mz', 'temp', 'time']
样本行数: 1000
估计采样率: 111.2 Hz
平均时间间隔: 0.008991 秒
重复时间戳: 170/1000 (17.0%)
总行数: 66,671
总重复时间戳: 10,455/66,671 (15.7%)

文件: sensor3.csv
大小: 7.5 MB
列名: ['timestamp', 'red', 'ir', 'green', 'ax', 'ay', 'az', 'rx', 'ry', 'rz', 'mx', 'my', 'mz', 'temp', 'time']
样本行数: 1000
估计采样率: 111.1 Hz
平均时间间隔: 0.009000 秒
重复时间戳: 170/1000 (17.0%)
总行数: 66,670
总重复时间戳: 10,456/66,670 (15.7%)

文件: sensor5.csv
大小: 7.3 MB
列名: ['timestamp', 'red', 'ir', 'green', 'ax', 'ay', 'az', 'rx', 'ry', 'rz', 'mx', 'my', 'mz', 'temp', 'time']
样本行数: 1000
估计采样率: 111.1 Hz
平均时间间隔: 0.009001 秒
重复时间戳: 0/1000 (0.0%)
总行数: 66,670
总重复时间戳: 0/66,670 (0.0%)

文件: sensor2.csv
大小: 7.1 MB
列名: ['timestamp', 'red', 'ir', 'green', 'ax', 'ay', 'az', 'rx', 'ry', 'rz', 'mx', 'my', 'mz', 'temp', 'time']
样本行数: 1000
估计采样率: 111.1 Hz
平均时间间隔: 0.009001 秒
重复时间戳: 170/1000 (17.0%)
总行数: 66,669
总重复时间戳: 10,456/66,669 (15.7%)
```

---

## pkl 数据查看

### 目标

查看预处理后的 `.pkl` 文件内容，检查数据对齐情况，并保存为 CSV 格式（整合 Biopac 数据为单文件，HUB 数据独立保存，时间戳第一列）。

### 实现步骤

1. **文件加载与检查**：
   - 扫描 `/root/PI_Lab/output` 目录，加载 `.pkl` 和 `.npy` 文件。
   - 打印实验名称、Biopac 和 HUB 文件数及各列统计信息。

2. **数据可视化**：
   - 绘制 Biopac 的 `bp` 和 HUB 的 `sensor2_red` 数据对齐图。

3. **CSV 保存**：
   - **Biopac**: 整合为单文件（如 `experiment_{name}_biopac_aligned.csv`），时间戳第一列，后接所有信号列。
   - **HUB**: 每个传感器保存为独立文件（如 `experiment_{name}_hub_sensor4_aligned.csv`），时间戳第一列。

```txt
  保存整合Biopac CSV: /root/PI_Lab/output/csv_output/2_biopac_aligned.csv
  保存HUB CSV: /root/PI_Lab/output/csv_output/2_hub_sensor4_aligned.csv
  保存HUB CSV: /root/PI_Lab/output/csv_output/2_hub_sensor3_aligned.csv
  保存HUB CSV: /root/PI_Lab/output/csv_output/2_hub_sensor5_aligned.csv
  保存HUB CSV: /root/PI_Lab/output/csv_output/2_hub_sensor2_aligned.csv
```

---

## 使用方法

1. **环境配置**：
   - 确保安装所有依赖库，运行环境验证代码检查版本。
   - 修改 `hub_folder` 和 `Biopac_folder` 路径为实际数据路径。

2. **运行分析**：
   - 执行数据结构检查代码，查看文件列表和列名。
   - 运行数据大小与特征分析代码，获取统计信息和建议。

3. **查看 pkl 数据**：
   - 运行 pkl 查看代码，生成可视化图和 CSV 文件。
   - 检查 `/root/PI_Lab/output/csv_output/` 下的输出文件。
