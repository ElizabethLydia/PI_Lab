# PPG-ABP信号分析工具

这是一个专门用于分析PPG（光电容积脉搏波）和ABP（动脉血压）信号相似性的Python工具包。

## 🎯 主要功能

### 1. 信号预处理
- **Butterworth滤波**: 带通滤波，可调节频带范围和阶数
- **小波去噪**: 使用db4小波进行软阈值去噪
- **形态学滤波**: 开运算和闭运算，去除基线漂移
- **运动伪影去除**: 基于加速度数据的运动伪影检测和去除
- **组合滤波**: 多种方法的组合使用

### 2. 相关性分析
- **Pearson相关系数**: 线性相关性
- **Spearman相关系数**: 单调相关性
- **互信息**: 非线性依赖性
- **频率域相关性**: FFT幅值相关性
- **相干性**: 频率域线性相关性
- **信噪比**: 信号质量评估

### 3. 可视化
- **信号对比图**: 原始信号vs预处理后信号的对比
- **相关性热力图**: 不同预处理方法的相关性指标对比
- **实验比较图**: 多个实验的结果比较

## 📁 文件结构

```
PI_Lab/blood_pressure_reconstruction/
├── ppg_abp_analysis_00017.py          # 基础分析脚本
├── ppg_abp_advanced_analysis.py       # 高级分析脚本
├── example_usage.py                    # 使用示例
├── README.md                           # 说明文档
└── step1.1_integrate_biopac_with_abp.py  # 数据整合脚本
```

## 🚀 快速开始

### 1. 基础分析

```python
from ppg_abp_advanced_analysis import PPGABPAnalyzer

# 创建分析器
analyzer = PPGABPAnalyzer(subject_id="00017", experiment="1")

# 运行完整分析
analyzer.run_complete_analysis(
    segment_length=1000,  # 绘图段长度
    start_idx=None        # 自动选择开始位置
)
```

### 2. 自定义参数分析

```python
# 自定义预处理参数
analyzer.butterworth_params = (0.3, 12.0, 100, 6)  # 更宽的频带，更高阶数
analyzer.wavelet_params = ('db6', 6)                 # 不同的小波和分解层数
analyzer.morphological_params = (7,)                 # 更大的结构元素

# 运行分析
analyzer.run_complete_analysis(
    segment_length=1500,
    start_idx=10000
)
```

### 3. 多实验比较

```python
from ppg_abp_advanced_analysis import AdvancedPPGABPAnalyzer

# 创建高级分析器
advanced_analyzer = AdvancedPPGABPAnalyzer(subject_id="00017")

# 比较多个实验
comparison_df = advanced_analyzer.compare_experiments(
    experiments=['1', '2', '3'],
    segment_length=1000,
    start_idx=None
)

# 绘制比较图
advanced_analyzer.plot_experiment_comparison(comparison_df)
```

## 📊 输出结果

### 1. 文件输出
- **信号对比图**: `{subject}_{experiment}_signals_comparison.png`
- **相关性热力图**: `{subject}_{experiment}_correlation_heatmap.png`
- **分析报告**: `{subject}_{experiment}_analysis_report.txt`
- **实验比较结果**: `{subject}_experiments_comparison.csv`

### 2. 分析报告内容
- 数据概览（长度、时间范围、采样频率）
- 各种预处理方法的相关性指标
- 最佳预处理方法推荐
- 信号质量评估

## ⚙️ 参数配置

### Butterworth滤波参数
```python
analyzer.butterworth_params = (lowcut, highcut, fs, order)
# lowcut: 低截止频率 (Hz)
# highcut: 高截止频率 (Hz)
# fs: 采样频率 (Hz)
# order: 滤波器阶数
```

### 小波去噪参数
```python
analyzer.wavelet_params = (wavelet, level)
# wavelet: 小波类型 ('db4', 'db6', 'haar', 'sym4'等)
# level: 分解层数
```

### 形态学滤波参数
```python
analyzer.morphological_params = (size,)
# size: 结构元素大小
```

## 🔬 使用示例

### 运行特定示例
```bash
# 运行示例1: 基础分析
python example_usage.py 1

# 运行示例2: 自定义参数
python example_usage.py 2

# 运行示例3: 多实验比较
python example_usage.py 3
```

### 运行所有示例
```bash
python example_usage.py
```

## 📈 分析结果解读

### 相关性强度评价
- **r > 0.9**: 相关性极强
- **r > 0.7**: 相关性强
- **r > 0.5**: 相关性中等
- **r < 0.5**: 相关性较弱

### 信噪比评价
- **SNR > 20 dB**: 信号质量很好
- **SNR > 10 dB**: 信号质量良好
- **SNR < 10 dB**: 信号质量较差

## 🛠️ 自定义扩展

### 添加新的预处理方法
```python
def custom_filter(self, signal_data):
    """自定义滤波方法"""
    # 实现你的滤波逻辑
    filtered_signal = your_filter_function(signal_data)
    return filtered_signal

# 在apply_preprocessing_methods中添加
self.processed_signals['自定义滤波'] = {
    'ppg': self.custom_filter(ppg_ir),
    'abp': abp
}
```

### 添加新的相关性指标
```python
def calculate_custom_metric(self, ppg_valid, abp_valid):
    """计算自定义相关性指标"""
    # 实现你的指标计算逻辑
    return custom_metric_value

# 在calculate_correlation_metrics中添加
metrics['custom_metric'] = self.calculate_custom_metric(ppg_valid, abp_valid)
```

## ⚠️ 注意事项

1. **数据格式**: 确保PPG和ABP数据文件存在且格式正确
2. **内存使用**: 处理长信号时注意内存使用情况
3. **参数调优**: 根据具体数据特点调整预处理参数
4. **结果验证**: 相关性结果需要结合临床意义进行解释

## 🔍 故障排除

### 常见问题
1. **数据加载失败**: 检查文件路径和格式
2. **预处理失败**: 调整参数或检查数据质量
3. **相关性计算失败**: 检查数据是否包含NaN值
4. **图表显示问题**: 检查matplotlib后端设置

### 调试建议
- 使用较小的数据段进行测试
- 逐步增加预处理复杂度
- 检查中间结果和日志输出

## 📚 参考文献

- "Can Photoplethysmography Replace Arterial Blood Pressure in the Assessment of Blood Pressure?"
- 相关信号处理和生物医学信号分析文献

## 🤝 贡献

欢迎提交问题报告和改进建议！

## �� 许可证

本项目仅供学术研究使用。
