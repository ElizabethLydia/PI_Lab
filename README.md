# 🩺 PI Lab - 基于PPG的血压预测研究

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Research](https://img.shields.io/badge/Research-HCI-orange.svg)]()
[![Status](https://img.shields.io/badge/Status-Completed-green.svg)]()
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC_BY--NC--SA_4.0-lightgrey.svg)](LICENSE)

[English Version](README_EN.md)

> 清华大学人机交互实验室实习项目：基于脉搏波传导时间（PTT）的连续无创血压监测方法研究（仅限非商业学术研究使用）

## 📖 项目简介

本项目属于**普适健康监测**研究方向，探索使用多传感器光电容积描记法（Photoplethysmography, PPG）测量脉搏波传导时间（Pulse Transit Time, PTT），并建立PTT与血压之间的映射关系，实现可穿戴设备的连续无创血压监测。

### 核心研究内容

- **多传感器PTT计算**：利用4个不同位置的PPG传感器，构建多点PTT测量网络
- **物理建模方法**：基于Moens-Korteweg方程建立PTT与血压的关系模型
- **机器学习建模**：使用回归模型和深度学习方法（LSTM、CNN-LSTM）进行血压预测
- **个体化校准**：针对不同生理状态和个体差异的自适应建模

### 研究背景

传统血压测量方法（如袖带式血压计）无法实现连续监测。本研究探索基于PTT的方法：

```
血压升高 → 血管弹性↓ → 脉搏波速度(PWV)↑ → PTT↓
```

通过建立PTT与血压的反比关系模型，实现实时血压预测。

## 🗂️ 项目结构

```
PI_Lab/
├── step1_preprocess.py              # 数据预处理：滤波、质量检查
├── step2_ir_ptt_peak_detector.py    # 峰值检测与PTT计算
├── step3_ptt_bp_analysis.py         # PTT与血压关系分析
├── step4_integrated_ptt_bp_analysis.py  # 综合分析与建模
├── data_processor.py                # 数据处理核心模块
├── check_results.py                 # 结果验证工具
├── results_summary.py               # 结果汇总分析
│
├── readme/                          # 各模块详细文档
│   ├── step1_preprocess.md
│   ├── step2_ir_ptt_peak_detector.md
│   ├── step3_ptt_bp_analysis.md
│   └── step4_integrated_ptt_bp_analysis.md
│
├── blood_pressure_reconstruction/   # 血压重建算法
├── personal_analysis/               # 个体化分析
├── step*_calibrated_check_results/  # 校准结果检查
│
├── 研究方案详解.md                   # 完整研究方案文档
├── 数据说明.txt                      # 数据集说明
└── *.pdf                            # 相关文献资料
```

## 🔬 数据处理流程

本项目采用四步处理pipeline，从原始生理信号到血压预测模型：

### Step 1: 数据预处理 (`step1_preprocess.py`)

**目标**：对齐和标准化多源异构生理信号数据

- **Biopac数据处理**（高频2000Hz）：
  - 降采样至100Hz，减少计算开销
  - 插值处理重复时间戳
- **HUB传感器数据处理**（低频111Hz）：
  - 插值去重，保持数据精度
  - 处理4个传感器位置：nose(sensor2)、finger(sensor3)、wrist(sensor4)、ear(sensor5)
- **时间对齐**：
  - 以HUB sensor2为参考基准
  - 线性插值对齐所有传感器数据
- **输出**：`.pkl`和`.npy`格式的对齐数据，CSV格式供后续分析

### Step 2: PTT峰值检测 (`step2_ir_ptt_peak_detector.py`)

**目标**：精确检测心跳峰值并计算多传感器PTT

- **窗口化时频域验证策略**：
  - 20秒滑动窗口，5秒步长（密集覆盖）
  - 时域心率（峰值检测） vs 频域心率（FFT）
  - 仅当心率差异 < 5 BPM时窗口有效
- **多方法峰值检测**：
  - 优先使用`neurokit2`专业库
  - 备选`heartpy`或改进的`scipy`方法
- **心跳间期（IBI）质量控制**：
  - IBI范围：300-1200ms（心率50-200 BPM）
  - 自动过滤异常峰值
- **跨传感器PTT计算**：
  - 生成6种传感器组合的PTT：
    - nose→finger, nose→wrist, nose→ear
    - finger→wrist, finger→ear, wrist→ear
  - 峰值匹配时间窗：±0.2秒
- **输出**：窗口验证详情、有效峰值、匹配心跳、PTT时间序列、统计汇总

### Step 3: PTT-血压关系分析 (`step3_ptt_bp_analysis.py`)

**目标**：建立PTT与血压的量化关系模型

- **数据同步**：
  - 将PTT数据与参考血压（ABP）时间对齐
  - 处理不同生理状态（静息、运动）数据
- **相关性分析**：
  - 计算PTT与收缩压/舒张压的Pearson/Spearman相关系数
  - 分析不同传感器组合的预测能力
- **回归建模**：
  - 线性回归：经典PTT-BP关系
  - 非线性模型：适应复杂生理状态
- **输出**：相关性矩阵、散点图、回归模型参数

### Step 4: 综合建模与预测 (`step4_integrated_ptt_bp_analysis.py`)

**目标**：集成多特征构建血压预测系统

- **多模态特征融合**：
  - PTT时间序列特征
  - 心率变异性（HRV）特征
  - 运动状态标签
- **机器学习建模**：
  - Random Forest、XGBoost等集成方法
  - 时序神经网络（LSTM/CNN-LSTM）
- **个体化校准**：
  - 在线学习适应个体差异
  - 迁移学习从群体到个体
- **临床验证**：
  - AAMI标准：MAE < 5mmHg, SD < 8mmHg
  - BHS标准：Grade A (MAE < 5mmHg)
  - ESH标准：≥85%测量误差在±10mmHg内
- **输出**：血压预测模型、性能评估报告、可视化结果

## 🚀 快速开始

### 环境要求

```bash
Python >= 3.8
numpy
scipy
pandas
matplotlib
scikit-learn
tensorflow / pytorch  # 用于深度学习模型
wfdb  # 用于生理信号处理
```

### 数据处理流程

```bash
# Step 1: 数据预处理
python step1_preprocess.py

# Step 2: PTT计算与峰值检测
python step2_ir_ptt_peak_detector.py

# Step 3: PTT-血压关系分析
python step3_ptt_bp_analysis.py

# Step 4: 综合建模与预测
python step4_integrated_ptt_bp_analysis.py
```

### 批处理运行

```bash
# 使用多进程加速
python step2_ir_ptt_peak_detector_mulPro.py
python step3_ptt_bp_analysis_mulPro.py
```

## 📊 数据集

本项目使用的数据包含：
- 多个受试者在不同生理状态下的PPG信号
- 同步记录的参考血压（ABP）信号
- 实验条件包括：静息状态、不同运动强度等

详细数据说明参见：[数据说明.txt](数据说明.txt)

## 📈 研究成果

### 关键发现

1. **多传感器PTT优于单点测量**：6种PTT组合提供互补信息
2. **个体差异显著**：需要个体化校准策略
3. **运动状态影响**：需要针对不同生理状态建立自适应模型

### 局限性

- 物理模型在复杂生理状态下的准确性受限
- 数据集规模对深度学习模型性能的制约
- 个体差异导致泛化能力挑战

## 📚 相关文献

项目参考的主要文献（已包含在仓库中）：
- Camera Wavelength Selection for Multi-wavelength PTT based BP Monitoring.pdf
- Camera-Based Neonatal Blood Pressure Estimation.pdf
- Can Photoplethysmography Replace Arterial Blood.pdf

更多文献汇报见：[PI—Lab 文献汇报1.pdf](PI—Lab%20文献汇报1.pdf)

## 🔧 工具说明

### 核心模块

- **data_processor.py**：数据加载、预处理、质量控制
- **check_results.py**：结果验证与可视化
- **results_summary.py**：批量结果统计分析
- **COLOR_SCHEME_README.md**：可视化配色方案说明

### 辅助脚本

- **cleanup_old_moved_folders.py**：清理临时文件
- **complete_subjects.txt / incomplete_subjects.txt**：受试者处理状态跟踪

## 🎯 项目意义

本项目虽然由于物理方法在数据集上的局限性未能达到发表论文的水平，但展示了：

✅ 完整的生理信号处理流程  
✅ 从物理建模到机器学习的多方法探索  
✅ 严格的临床验证标准评估  
✅ 工程化的代码实现与文档  

这是一次宝贵的科研实践经历，积累了生理信号处理、机器学习建模和实验数据分析的经验。

## 👤 作者

**Elizabeth Lydia**
- 清华大学人机交互实验室实习生
- 研究方向：普适健康监测、生理信号处理
- GitHub: [@ElizabethLydia](https://github.com/ElizabethLydia)

## 📄 许可证

- 许可协议：Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
- 允许行为：署名后自由分享、修改，但仅限非商业用途，并需保持相同许可方式再分发。
- 禁止行为：任何商业化使用、闭源再发布或未经授权的专利主张。

完整条款见 [LICENSE](LICENSE)。

## 🙏 致谢

感谢清华大学人机交互实验室提供的研究平台、数据支持和技术指导。

