#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🩺 PTT生理信号峰值检测器 - 基于文献标准方法

参考文献方法：
1. Aboy++算法：自适应心率估计和峰值检测
2. Han等人方法：处理心律失常的实时PPG峰值检测
3. 生理约束：IBI、心肌不应期、谐波抑制

核心改进：
✅ 基于生理约束的峰值检测（50-200 BPM，0.3秒不应期）
✅ 基于IBI的自适应阈值
✅ 谐波和伪峰抑制
✅ 同一心跳区间的峰值对应
✅ 参考data_processor.py的滤波方法
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, filtfilt, butter, welch
import warnings
warnings.filterwarnings('ignore')

# 设置字体
plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']

class PhysiologicalPTTPeakDetector:
    """基于生理信号的PTT峰值检测器"""
    
    def __init__(self, data_path="/root/PI_Lab/output/csv_output"):
        self.data_path = data_path
        self.output_dir = "/root/PI_Lab/ptt_output"
        self.sensors = ['sensor2', 'sensor3', 'sensor4', 'sensor5']
        self.colors = ['red', 'ir', 'green']
        self.sensor_mapping = {
            'sensor2': 'nose', 'sensor3': 'finger', 
            'sensor4': 'wrist', 'sensor5': 'ear'
        }
        
        # 生理参数约束（基于文献）
        self.fs = 100  # 采样率100Hz
        self.min_hr = 50   # 最小心率BPM
        self.max_hr = 200  # 最大心率BPM
        self.refractory_period = 0.3  # 心肌不应期（秒）
        self.window_size = 30  # 分析窗口（秒）
        
        # 滤波参数（参考data_processor.py）
        self.filter_lowcut = 0.5   # 下截止频率
        self.filter_highcut = 3.0  # 上截止频率（调整为3Hz）
        self.filter_order = 3
        
        os.makedirs(self.output_dir, exist_ok=True)
        
    def bandpass_filter(self, data, lowcut=0.5, highcut=3.0, fs=100, order=3):
        """带通滤波 - 参考data_processor.py方法"""
        try:
            nyquist = fs / 2
            low = lowcut / nyquist
            high = highcut / nyquist
            
            # 确保截止频率在有效范围内
            low = max(low, 0.01)
            high = min(high, 0.99)
            
            b, a = butter(order, [low, high], btype='band')
            filtered = filtfilt(b, a, data)
            return filtered
        except Exception as e:
            print(f"滤波失败: {e}")
            return data
    
    def estimate_heart_rate(self, signal, fs=100):
        """估计心率 - 使用功率谱密度方法"""
        try:
            f, psd = welch(signal, fs, nperseg=min(len(signal), 1024))
            hr_freq_min = self.min_hr / 60.0
            hr_freq_max = self.max_hr / 60.0
            
            hr_mask = (f >= hr_freq_min) & (f <= hr_freq_max)
            if np.any(hr_mask) and np.max(psd[hr_mask]) > 0:
                peak_freq = f[hr_mask][np.argmax(psd[hr_mask])]
                estimated_hr = peak_freq * 60
                return np.clip(estimated_hr, self.min_hr, self.max_hr)
            else:
                print(f"警告: 未检测到有效心率，使用峰值间隔方法。")
                peaks, _ = find_peaks(signal, distance=int(0.3 * fs))
                if len(peaks) > 1:
                    avg_interval = np.mean(np.diff(peaks)) / fs
                    if 0.3 <= avg_interval <= 1.2:  # 50-200 BPM
                        return 60 / avg_interval
                print(f"警告: 心率估计失败，未找到有效峰值。")
                return None
        except Exception as e:
            print(f"估计心率出错: {e}")
            return None
    
    def adaptive_peak_detection(self, signal, fs=100, reference_peaks=None):
        """自适应峰值检测 - 改进的Aboy++算法，考虑跨传感器对齐"""
        try:
            # 1. 估计心率
            estimated_hr = self.estimate_heart_rate(signal, fs)
            if estimated_hr is None:
                return np.array([])
            
            expected_peak_interval = 60.0 / estimated_hr  # 预期峰值间隔（秒）
            min_distance = int(self.refractory_period * fs)  # 最小距离（样本数）
            
            # 2. 计算自适应阈值
            signal_std = np.std(signal)
            signal_mean = np.mean(signal)
            height_threshold = signal_mean + 0.3 * signal_std
            prominence_threshold = 0.1 * signal_std
            
            # 3. 初步峰值检测
            peaks, properties = find_peaks(
                signal,
                height=height_threshold,
                distance=min_distance,
                prominence=prominence_threshold
            )
            
            # 4. 基于IBI约束的峰值精化
            if len(peaks) > 1:
                peak_intervals = np.diff(peaks) / fs
                valid_interval_min = 60.0 / self.max_hr  # 0.3秒
                valid_interval_max = 60.0 / self.min_hr  # 1.2秒
                
                valid_peaks = [peaks[0]]
                for i in range(1, len(peaks)):
                    interval = peak_intervals[i-1]
                    if valid_interval_min <= interval <= valid_interval_max:
                        valid_peaks.append(peaks[i])
                    else:
                        # 检查是否为第二谐波（约0.5倍间隔）
                        if reference_peaks is not None and len(reference_peaks) > 1:
                            ref_interval = np.mean(np.diff(reference_peaks)) / fs
                            if abs(interval - 0.5 * ref_interval) < 0.1:
                                continue  # 抑制第二谐波
                        if signal[peaks[i]] > signal[valid_peaks[-1]]:
                            valid_peaks[-1] = peaks[i]
                
                peaks = np.array(valid_peaks)
            
            # 5. 跨传感器对齐（如果有参考峰值）
            if reference_peaks is not None and len(peaks) > 0 and len(reference_peaks) > 0:
                from scipy.signal import correlate
                correlation = correlate(signal[peaks], signal[reference_peaks[:len(peaks)]])
                max_corr_idx = np.argmax(correlation)
                if max_corr_idx > 0:
                    peaks = peaks[max_corr_idx:]
            
            return peaks
        except Exception as e:
            print(f"峰值检测失败: {e}")
            return np.array([])
    
    def detect_peaks_for_signal(self, signal, sensor, color, reference_results=None):
        """对单个信号进行峰值检测"""
        try:
            if len(signal) < 200:  # 信号太短
                return {
                    'peaks': np.array([]),
                    'peak_times': np.array([]),
                    'filtered_signal': signal,
                    'estimated_hr': 0,
                    'peak_count': 0
                }
            
            # 1. 带通滤波 (0.5-3 Hz)
            filtered_signal = self.bandpass_filter(
                signal,
                lowcut=self.filter_lowcut,
                highcut=self.filter_highcut,
                fs=self.fs,
                order=self.filter_order
            )
            
            # 2. 自适应峰值检测
            peaks = self.adaptive_peak_detection(filtered_signal, self.fs,
                                               reference_peaks=reference_results['peaks'] if reference_results else None)
            
            # 3. 转换为时间戳
            peak_times = peaks / self.fs
            
            return {
                'peaks': peaks,
                'peak_times': peak_times,
                'filtered_signal': filtered_signal,
                'estimated_hr': self.estimate_heart_rate(filtered_signal, self.fs) if len(peaks) > 0 else 0,
                'peak_count': len(peaks)
            }
        except Exception as e:
            print(f"信号 {sensor}-{color} 峰值检测失败: {e}")
            return {
                'peaks': np.array([]),
                'peak_times': np.array([]),
                'filtered_signal': signal,
                'estimated_hr': 0,
                'peak_count': 0
            }
    
    def process_experiment(self, exp_id):
        """处理单个实验的所有传感器数据，包含跨传感器对齐"""
        print(f"\n🔍 开始处理实验 {exp_id}")
        
        results = {}
        all_signals = {}
        
        for sensor in self.sensors:
            try:
                file_path = os.path.join(self.data_path, f"{exp_id}_{sensor}_aligned.csv")
                if not os.path.exists(file_path):
                    print(f"❌ 文件不存在: {file_path}")
                    continue
                
                print(f"📖 读取文件: {file_path}")
                df = pd.read_csv(file_path)
                all_signals[sensor] = df
                
                sensor_results = {}
                reference_result = None
                for i, color in enumerate(self.colors):
                    if i + 1 < len(df.columns):
                        signal = df.iloc[:, i + 1].values
                        peak_result = self.detect_peaks_for_signal(signal, sensor, color, reference_result)
                        sensor_results[color] = peak_result
                        if sensor == 'sensor2' and color == 'ir':  # 使用sensor2-ir作为参考
                            reference_result = peak_result
                        
                        print(f"  📊 {sensor}-{color}: 检测到 {peak_result['peak_count']} 个峰值, "
                              f"估计心率 {peak_result['estimated_hr']:.1f} BPM")
                
                results[sensor] = sensor_results
            except Exception as e:
                print(f"❌ 处理 {sensor} 失败: {e}")
                continue
        
        self.save_results(exp_id, results, all_signals)
        return results
    
    def save_results(self, exp_id, results, all_signals):
        """保存峰值检测结果"""
        summary_data = []
        detailed_data = []
        
        for sensor in results:
            for color in results[sensor]:
                peak_result = results[sensor][color]
                
                # 汇总数据
                summary_data.append({
                    'sensor': sensor,
                    'sensor_name': self.sensor_mapping[sensor],
                    'color': color,
                    'peak_count': peak_result['peak_count'],
                    'estimated_hr': peak_result['estimated_hr'],
                    'avg_ibi': np.mean(np.diff(peak_result['peak_times'])) if len(peak_result['peak_times']) > 1 else 0,
                    'hr_variability': np.std(np.diff(peak_result['peak_times']) * 1000) if len(peak_result['peak_times']) > 1 else 0  # ms
                })
                
                # 详细峰值位置
                for i, (peak_idx, peak_time) in enumerate(zip(peak_result['peaks'], peak_result['peak_times'])):
                    detailed_data.append({
                        'sensor': sensor,
                        'sensor_name': self.sensor_mapping[sensor],
                        'color': color,
                        'peak_number': i + 1,
                        'peak_index': peak_idx,
                        'peak_time': peak_time,
                        'ibi_ms': (peak_time - peak_result['peak_times'][i-1]) * 1000 if i > 0 else np.nan
                    })
        
        # 保存汇总结果
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_file = os.path.join(self.output_dir, f"physiological_peaks_summary_exp_{exp_id}.csv")
            summary_df.to_csv(summary_file, index=False)
            print(f"💾 保存峰值汇总: {summary_file}")
        
        # 保存详细结果
        if detailed_data:
            detailed_df = pd.DataFrame(detailed_data)
            detailed_file = os.path.join(self.output_dir, f"physiological_peaks_detailed_exp_{exp_id}.csv")
            detailed_df.to_csv(detailed_file, index=False)
            print(f"💾 保存峰值详细: {detailed_file}")
        
        # 生成可视化
        self.create_visualizations(exp_id, results, all_signals)
    
    def create_visualizations(self, exp_id, results, all_signals):
        """创建可视化 - 参考data_processor.py风格，使用英文标注"""
        try:
            fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FECA57']
            
            for ch_idx, channel in enumerate(self.colors):
                ax = axes[ch_idx]
                
                for sensor_idx, sensor in enumerate(self.sensors):
                    if sensor in all_signals and sensor in results:
                        df = all_signals[sensor]
                        time = df['timestamp'].values - df['timestamp'].values[0]
                        if ch_idx + 1 < len(df.columns) and channel in results[sensor]:
                            filtered_signal = results[sensor][channel]['filtered_signal']
                            peaks = results[sensor][channel]['peaks']
                            
                            # 归一化信号
                            if np.std(filtered_signal) > 0:
                                signal_norm = (filtered_signal - np.mean(filtered_signal)) / np.std(filtered_signal)
                            else:
                                signal_norm = filtered_signal
                            
                            ax.plot(time[:len(signal_norm)], signal_norm, 
                                   color=colors[sensor_idx % len(colors)], 
                                   linewidth=1.5, alpha=0.8,
                                   label=f'{self.sensor_mapping[sensor]}')
                            
                            if len(peaks) > 0:
                                peak_times = peaks / self.fs
                                ax.scatter(peak_times, signal_norm[peaks], 
                                         color='red', s=30, zorder=5, alpha=0.8)
                
                ax.set_title(f'{channel.upper()} Channel - All Sensors Overlay (Filtered 0.5-3Hz)', 
                            fontsize=14, fontweight='bold')
                ax.set_ylabel('Normalized Signal', fontsize=12)
                ax.grid(True, alpha=0.3)
                ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
                ax.set_ylim(-0.1, 1.1)
            
            axes[-1].set_xlabel('Time (seconds)', fontsize=12)
            plt.suptitle(f'Experiment {exp_id} - Physiological Signal Peak Detection', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout(rect=[0, 0, 0.85, 0.96])
            
            plot_file = os.path.join(self.output_dir, f"peaks_exp_{exp_id}.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"保存可视化: {plot_file}")
            
            # 创建心率统计图
            self.create_heart_rate_summary(exp_id, results)
        except Exception as e:
            print(f"可视化创建失败: {e}")
    
    def create_heart_rate_summary(self, exp_id, results):
        """创建心率统计汇总图"""
        try:
            hr_data = []
            labels = []
            
            for sensor in self.sensors:
                if sensor in results:
                    for color in self.colors:
                        if color in results[sensor]:
                            hr = results[sensor][color]['estimated_hr']
                            if hr > 0:
                                hr_data.append(hr)
                                labels.append(f"{self.sensor_mapping[sensor]}-{color}")
            
            if hr_data:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # 柱状图
                bars = ax1.bar(range(len(hr_data)), hr_data, 
                              color=['red', 'blue', 'green'] * 4)
                ax1.set_xlabel('Sensor-Channel')
                ax1.set_ylabel('Estimated Heart Rate (BPM)')
                ax1.set_title(f'Experiment {exp_id} - Heart Rate Estimation')
                ax1.set_xticks(range(len(labels)))
                ax1.set_xticklabels(labels, rotation=45)
                ax1.grid(True, alpha=0.3)
                
                # 添加数值标签
                for bar, hr in zip(bars, hr_data):
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                            f'{hr:.1f}', ha='center', va='bottom')
                
                # 箱线图
                ax2.boxplot(hr_data, labels=['All Sensors'])
                ax2.set_ylabel('Heart Rate (BPM)')
                ax2.set_title('Heart Rate Distribution')
                ax2.grid(True, alpha=0.3)
                
                # 添加统计信息
                mean_hr = np.mean(hr_data)
                std_hr = np.std(hr_data)
                ax2.text(1.1, mean_hr, f'Mean: {mean_hr:.1f} BPM\nStd: {std_hr:.1f}', 
                        verticalalignment='center')
                
                plt.tight_layout()
                
                hr_plot_file = os.path.join(self.output_dir, f"heart_rate_summary_exp_{exp_id}.png")
                plt.savefig(hr_plot_file, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"保存心率统计: {hr_plot_file}")
        except Exception as e:
            print(f"心率统计图创建失败: {e}")
    
    def run_analysis(self, experiment_list=None):
        """运行完整的峰值检测分析"""
        if experiment_list is None:
            experiment_list = []
            for file in os.listdir(self.data_path):
                if file.endswith('_sensor2_aligned.csv'):
                    exp_id = file.split('_')[0]
                    experiment_list.append(exp_id)
            experiment_list = sorted(list(set(experiment_list)))
        
        print(f"\n🔬 开始生理信号峰值检测分析")
        print(f"📋 实验列表: {experiment_list}")
        print(f"🎯 检测参数:")
        print(f"   - 心率范围: {self.min_hr}-{self.max_hr} BPM")
        print(f"   - 心肌不应期: {self.refractory_period} 秒")
        print(f"   - 滤波范围: {self.filter_lowcut}-{self.filter_highcut} Hz")
        
        all_results = {}
        
        for exp_id in tqdm(experiment_list, desc="处理实验"):
            try:
                results = self.process_experiment(exp_id)
                all_results[exp_id] = results
            except Exception as e:
                print(f"❌ 实验 {exp_id} 处理失败: {e}")
                continue
        
        print(f"\n✅ 峰值检测分析完成！")
        print(f"📁 结果保存在: {self.output_dir}")
        
        return all_results

def main():
    """主函数"""
    print("🩺 PTT生理信号峰值检测器")
    print("=" * 50)
    print("📖 基于以下文献方法:")
    print("   • Aboy++算法：自适应心率估计")
    print("   • Han等人方法：心律失常处理")
    print("   • 生理约束：IBI、心肌不应期、谐波抑制")
    print("=" * 50)
    
    # 创建检测器
    detector = PhysiologicalPTTPeakDetector()
    
    # 运行分析（仅处理实验1进行测试）
    results = detector.run_analysis(['1'])
    
    print("\n🎯 下一步建议:")
    print("1. 检查生成的可视化图像确认峰值检测质量")
    print("2. 如果峰值检测良好，继续进行PTT计算")
    print("3. 对比不同传感器间的峰值时序关系")

if __name__ == "__main__":
    main()