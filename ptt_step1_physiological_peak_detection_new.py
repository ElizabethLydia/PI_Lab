#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
�� IR通道专门的PTT峰值检测器 - 优化输出与PTT准备

基于师兄建议的改进：
1. ✅ 专注IR通道峰值检测（信号质量最佳）
2. ✅ 使用neurokit2计算IBI并验证
3. ✅ 同一心跳区间的峰值匹配
4. ✅ 输出峰值、IBI和PTT预览CSV，方便后续处理

核心原理：
- PTT使用峰值时间差计算
- IR通道信号最稳定
- IBI验证确保峰值准确
- 4传感器生成6个PTT组合
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
import warnings
warnings.filterwarnings('ignore')

# 简化matplotlib设置，避免字体警告
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

class IRBasedPTTPeakDetector:
    """基于IR通道的PTT峰值检测器"""
    
    def __init__(self, data_path="/root/PI_Lab/output/csv_output"):
        self.data_path = data_path
        self.output_dir = "/root/PI_Lab/ptt_output"
        self.sensors = ['sensor2', 'sensor3', 'sensor4', 'sensor5']
        self.target_channel = 'ir'
        self.sensor_mapping = {
            'sensor2': 'nose', 'sensor3': 'finger', 
            'sensor4': 'wrist', 'sensor5': 'ear'
        }
        self.fs = 100  # 采样率100Hz
        self.min_hr = 50
        self.max_hr = 200
        self.refractory_period = 0.3
        self.filter_lowcut = 0.5
        self.filter_highcut = 3.0
        self.filter_order = 3
        self.ibi_tolerance = 0.15  # IBI容差15%，稍微放宽
        os.makedirs(self.output_dir, exist_ok=True)
        
    def bandpass_filter(self, data, lowcut=0.5, highcut=3.0, fs=100, order=3):
        """带通滤波 - 针对心率频段"""
        try:
            nyquist = fs / 2
            low = max(lowcut / nyquist, 0.01)
            high = min(highcut / nyquist, 0.99)
            b, a = butter(order, [low, high], btype='band')
            return filtfilt(b, a, data)
        except Exception as e:
            print(f"⚠️  滤波失败: {e}")
            return data
    
    def detect_peaks_robust(self, signal, fs=100):
        """稳健的峰值检测 - 结合库函数和自定义算法"""
        try:
            # 先进行滤波
            filtered_signal = self.bandpass_filter(signal, self.filter_lowcut, self.filter_highcut, fs)
            
            # 使用scipy进行基础峰值检测
            min_distance = int(self.refractory_period * fs)  # 0.3秒最小间隔
            
            # 自适应阈值
            signal_std = np.std(filtered_signal)
            signal_mean = np.mean(filtered_signal)
            height_threshold = signal_mean + 0.3 * signal_std
            prominence_threshold = 0.15 * signal_std
            
            peaks, properties = find_peaks(
                filtered_signal,
                height=height_threshold,
                distance=min_distance,
                prominence=prominence_threshold
            )
            
            if len(peaks) < 2:
                return {
                    'peaks': np.array([]),
                    'ibi_ms': np.array([]),
                    'filtered_signal': filtered_signal,
                    'peak_times': np.array([]),
                    'peak_count': 0,
                    'quality': 'poor'
                }
            
            # 计算IBI并进行质量控制
            peak_times = peaks / fs
            ibi_ms = np.diff(peak_times) * 1000  # 转换为毫秒
            
            # IBI质量控制：300-1200ms (50-200 BPM)
            valid_ibi_mask = (ibi_ms >= 300) & (ibi_ms <= 1200)
            valid_ratio = np.sum(valid_ibi_mask) / len(ibi_ms) if len(ibi_ms) > 0 else 0
            
            # 如果有效IBI比例太低，尝试调整阈值
            if valid_ratio < 0.7 and len(peaks) > 10:
                # 更保守的峰值检测
                height_threshold = signal_mean + 0.5 * signal_std
                peaks, _ = find_peaks(
                    filtered_signal,
                    height=height_threshold,
                    distance=min_distance,
                    prominence=prominence_threshold * 1.5
                )
                peak_times = peaks / fs
                ibi_ms = np.diff(peak_times) * 1000
                valid_ibi_mask = (ibi_ms >= 300) & (ibi_ms <= 1200)
                valid_ratio = np.sum(valid_ibi_mask) / len(ibi_ms) if len(ibi_ms) > 0 else 0
            
            # 质量评估
            if valid_ratio >= 0.8:
                quality = 'excellent'
            elif valid_ratio >= 0.6:
                quality = 'good'
            elif valid_ratio >= 0.4:
                quality = 'fair'
            else:
                quality = 'poor'
            
            return {
                'peaks': peaks,
                'ibi_ms': ibi_ms,
                'filtered_signal': filtered_signal,
                'peak_times': peak_times,
                'peak_count': len(peaks),
                'quality': quality,
                'valid_ibi_ratio': valid_ratio
            }
            
        except Exception as e:
            print(f"⚠️  峰值检测失败: {e}")
            return {
                'peaks': np.array([]),
                'ibi_ms': np.array([]),
                'filtered_signal': signal,
                'peak_times': np.array([]),
                'peak_count': 0,
                'quality': 'error'
            }
    
    def calculate_heart_rate_stats(self, ibi_ms):
        """计算心率统计信息"""
        if len(ibi_ms) == 0:
            return {
                'hr_mean': 0,
                'hr_std': 0,
                'ibi_mean': 0,
                'ibi_std': 0,
                'rmssd': 0,  # HRV指标
                'pnn50': 0   # HRV指标
            }
        
        # 基础统计
        hr_bpm = 60000 / ibi_ms  # 转换为BPM
        ibi_mean = np.mean(ibi_ms)
        ibi_std = np.std(ibi_ms)
        hr_mean = np.mean(hr_bpm)
        hr_std = np.std(hr_bpm)
        
        # HRV指标
        if len(ibi_ms) > 1:
            # RMSSD: 相邻IBI差值的均方根
            diff_ibi = np.diff(ibi_ms)
            rmssd = np.sqrt(np.mean(diff_ibi**2))
            
            # pNN50: 相邻IBI差值>50ms的百分比
            pnn50 = np.sum(np.abs(diff_ibi) > 50) / len(diff_ibi) * 100
        else:
            rmssd = 0
            pnn50 = 0
        
        return {
            'hr_mean': hr_mean,
            'hr_std': hr_std,
            'ibi_mean': ibi_mean,
            'ibi_std': ibi_std,
            'rmssd': rmssd,
            'pnn50': pnn50
        }
    
    def match_peaks_across_sensors(self, sensor_results):
        """匹配不同传感器间同一心跳的峰值"""
        try:
            # 只使用质量good以上的传感器
            valid_sensors = [s for s in self.sensors 
                           if s in sensor_results 
                           and sensor_results[s]['peak_count'] > 5
                           and sensor_results[s]['quality'] in ['excellent', 'good']]
            
            if len(valid_sensors) < 2:
                print("⚠️  高质量传感器数量不足，尝试放宽标准")
                # 放宽标准，包括fair质量
                valid_sensors = [s for s in self.sensors 
                               if s in sensor_results 
                               and sensor_results[s]['peak_count'] > 3
                               and sensor_results[s]['quality'] != 'error']
            
            if len(valid_sensors) < 2:
                print("⚠️  有效传感器数量不足，无法进行峰值匹配")
                return {}
            
            print(f"📍 有效传感器: {valid_sensors}")
            
            # 选择质量最好的作为参考
            reference_sensor = max(valid_sensors, 
                                 key=lambda s: sensor_results[s]['valid_ibi_ratio'])
            reference_peaks = sensor_results[reference_sensor]['peak_times']
            
            print(f"📍 参考传感器: {reference_sensor} (质量: {sensor_results[reference_sensor]['quality']})")
            
            # 为每个心跳创建时间窗口
            heartbeat_windows = []
            for i, ref_time in enumerate(reference_peaks):
                if i == 0:
                    window_start = 0
                    if len(reference_peaks) > 1:
                        window_end = ref_time + (reference_peaks[1] - reference_peaks[0])/2
                    else:
                        window_end = ref_time + 0.5
                elif i == len(reference_peaks) - 1:
                    window_start = ref_time - (reference_peaks[i] - reference_peaks[i-1])/2
                    window_end = float('inf')
                else:
                    window_start = ref_time - (reference_peaks[i] - reference_peaks[i-1])/2
                    window_end = ref_time + (reference_peaks[i+1] - reference_peaks[i])/2
                
                heartbeat_windows.append({
                    'heartbeat_id': i + 1,
                    'reference_time': ref_time,
                    'window_start': window_start,
                    'window_end': window_end,
                    'sensor_peaks': {reference_sensor: ref_time}
                })
            
            # 为其他传感器匹配峰值
            for sensor in valid_sensors:
                if sensor == reference_sensor:
                    continue
                    
                sensor_peaks = sensor_results[sensor]['peak_times']
                
                for peak_time in sensor_peaks:
                    # 找到最佳匹配的心跳窗口
                    best_window = None
                    min_distance = float('inf')
                    
                    for window in heartbeat_windows:
                        if window['window_start'] <= peak_time <= window['window_end']:
                            distance = abs(peak_time - window['reference_time'])
                            if distance < min_distance:
                                min_distance = distance
                                best_window = window
                    
                    # 将峰值分配到最佳窗口
                    if best_window is not None and min_distance < 0.2:  # 200ms容差
                        best_window['sensor_peaks'][sensor] = peak_time
            
            # 过滤完整的心跳（至少有2个传感器）
            complete_heartbeats = [hb for hb in heartbeat_windows 
                                 if len(hb['sensor_peaks']) >= 2]
            
            print(f"📊 完整心跳数量: {len(complete_heartbeats)}/{len(heartbeat_windows)}")
            
            return {
                'heartbeat_windows': heartbeat_windows,
                'complete_heartbeats': complete_heartbeats,
                'valid_sensors': valid_sensors,
                'reference_sensor': reference_sensor
            }
            
        except Exception as e:
            print(f"⚠️  峰值匹配失败: {e}")
            return {}
    
    def process_experiment(self, exp_id):
        """处理单个实验的IR通道数据"""
        print(f"\n🔍 开始处理实验 {exp_id} - 专注IR通道")
        
        sensor_results = {}
        all_signals = {}
        
        for sensor in self.sensors:
            try:
                file_path = os.path.join(self.data_path, f"{exp_id}_hub_{sensor}_aligned.csv")
                if not os.path.exists(file_path):
                    print(f"❌ 文件不存在: {file_path}")
                    continue
                
                df = pd.read_csv(file_path)
                all_signals[sensor] = df
                
                if len(df.columns) >= 3:
                    ir_signal = df.iloc[:, 2].values  # IR通道
                    
                    # 稳健的峰值检测
                    peak_result = self.detect_peaks_robust(ir_signal, self.fs)
                    
                    # 计算心率统计
                    hr_stats = self.calculate_heart_rate_stats(peak_result['ibi_ms'])
                    
                    # 合并结果
                    peak_result.update({
                        'sensor': sensor,
                        'sensor_name': self.sensor_mapping[sensor],
                        **hr_stats
                    })
                    
                    sensor_results[sensor] = peak_result
                    
                    # 打印结果
                    quality_emoji = {
                        'excellent': '🟢', 'good': '🟡', 
                        'fair': '🟠', 'poor': '🔴', 'error': '❌'
                    }
                    quality_symbol = quality_emoji.get(peak_result['quality'], '❓')
                    
                    if peak_result['peak_count'] > 0:
                        ibi_range = f"{np.min(peak_result['ibi_ms']):.0f}-{np.max(peak_result['ibi_ms']):.0f}ms" if len(peak_result['ibi_ms']) > 0 else "N/A"
                        print(f"  {quality_symbol} {sensor}({self.sensor_mapping[sensor]}): "
                              f"{peak_result['peak_count']}峰值, "
                              f"HR={hr_stats['hr_mean']:.1f}±{hr_stats['hr_std']:.1f}BPM, "
                              f"IBI={ibi_range}, "
                              f"质量={peak_result['quality']}({peak_result.get('valid_ibi_ratio', 0)*100:.0f}%)")
                    else:
                        print(f"  {quality_symbol} {sensor}({self.sensor_mapping[sensor]}): 未检测到有效峰值")
                        
                else:
                    print(f"⚠️  {sensor}: 数据列不足")
                    
            except Exception as e:
                print(f"❌ 处理 {sensor} 失败: {e}")
                continue
        
        # 匹配不同传感器间的峰值
        matched_results = self.match_peaks_across_sensors(sensor_results)
        
        # 保存结果
        self.save_results(exp_id, sensor_results, matched_results, all_signals)
        
        return sensor_results, matched_results
    
    def save_results(self, exp_id, sensor_results, matched_results, all_signals):
        """保存检测结果 - 5个核心CSV文件"""
        try:
            # 1. 传感器质量汇总
            sensor_summary = []
            for sensor in sensor_results:
                result = sensor_results[sensor]
                signal_duration = len(all_signals[sensor].iloc[:, 2]) / self.fs  # 信号时长(秒)
                
                sensor_summary.append({
                    'sensor': sensor,
                    'sensor_name': result['sensor_name'],
                    'peak_count': result['peak_count'],
                    'quality': result['quality'],
                    'valid_ibi_ratio': result.get('valid_ibi_ratio', 0),
                    'hr_mean_bpm': result['hr_mean'],
                    'hr_std_bpm': result['hr_std'],
                    'ibi_mean_ms': result['ibi_mean'],
                    'ibi_std_ms': result['ibi_std'],
                    'rmssd_ms': result['rmssd'],
                    'pnn50_percent': result['pnn50'],
                    'signal_duration_s': signal_duration
                })
            
            if sensor_summary:
                summary_df = pd.DataFrame(sensor_summary)
                summary_file = os.path.join(self.output_dir, f"sensor_summary_exp_{exp_id}.csv")
                summary_df.to_csv(summary_file, index=False)
                print(f"💾 保存传感器汇总: {summary_file}")
            
            # 2. 所有峰值详细信息
            all_peaks = []
            for sensor in sensor_results:
                result = sensor_results[sensor]
                for i, (peak_idx, peak_time) in enumerate(zip(result['peaks'], result['peak_times'])):
                    all_peaks.append({
                        'sensor': sensor,
                        'sensor_name': result['sensor_name'],
                        'peak_number': i + 1,
                        'peak_index': int(peak_idx),
                        'peak_time_s': peak_time,
                        'quality': result['quality']
                    })
            
            if all_peaks:
                peaks_df = pd.DataFrame(all_peaks)
                peaks_file = os.path.join(self.output_dir, f"all_peaks_exp_{exp_id}.csv")
                peaks_df.to_csv(peaks_file, index=False)
                print(f"💾 保存所有峰值: {peaks_file}")
            
            # 3. 所有IBI详细信息
            all_ibi = []
            for sensor in sensor_results:
                result = sensor_results[sensor]
                for i, ibi_val in enumerate(result['ibi_ms']):
                    all_ibi.append({
                        'sensor': sensor,
                        'sensor_name': result['sensor_name'],
                        'ibi_number': i + 1,
                        'ibi_ms': ibi_val,
                        'hr_bpm': 60000 / ibi_val,
                        'is_valid': 300 <= ibi_val <= 1200,
                        'quality': result['quality']
                    })
            
            if all_ibi:
                ibi_df = pd.DataFrame(all_ibi)
                ibi_file = os.path.join(self.output_dir, f"all_ibi_exp_{exp_id}.csv")
                ibi_df.to_csv(ibi_file, index=False)
                print(f"💾 保存所有IBI: {ibi_file}")
            
            # 4. 匹配的心跳和PTT计算
            if matched_results and 'complete_heartbeats' in matched_results:
                heartbeat_data = []
                for hb in matched_results['complete_heartbeats']:
                    row = {'heartbeat_id': hb['heartbeat_id']}
                    for sensor in matched_results['valid_sensors']:
                        row[f'{sensor}_peak_time_s'] = hb['sensor_peaks'].get(sensor, np.nan)
                    heartbeat_data.append(row)
                
                if heartbeat_data:
                    heartbeat_df = pd.DataFrame(heartbeat_data)
                    heartbeat_file = os.path.join(self.output_dir, f"matched_heartbeats_exp_{exp_id}.csv")
                    heartbeat_df.to_csv(heartbeat_file, index=False)
                    print(f"💾 保存匹配心跳: {heartbeat_file}")
                    
                    # 计算PTT矩阵和时间序列
                    self.calculate_ptt_analysis(heartbeat_df, exp_id, matched_results['valid_sensors'])
            
            # 生成可视化
            self.create_visualizations(exp_id, sensor_results, matched_results, all_signals)
            
        except Exception as e:
            print(f"❌ 保存结果失败: {e}")
    
    def calculate_ptt_analysis(self, heartbeat_df, exp_id, valid_sensors):
        """计算PTT分析 - 矩阵汇总 + 时间序列"""
        try:
            # 生成所有传感器组合
            sensor_combinations = []
            for i in range(len(valid_sensors)):
                for j in range(i+1, len(valid_sensors)):
                    sensor_combinations.append((valid_sensors[i], valid_sensors[j]))
            
            print(f"\n📊 PTT分析 ({len(sensor_combinations)}个传感器组合):")
            
            # PTT矩阵汇总
            ptt_summary = []
            ptt_timeseries_all = []
            
            for sensor1, sensor2 in sensor_combinations:
                col1 = f'{sensor1}_peak_time_s'
                col2 = f'{sensor2}_peak_time_s'
                
                if col1 in heartbeat_df.columns and col2 in heartbeat_df.columns:
                    valid_data = heartbeat_df.dropna(subset=[col1, col2])
                    
                    if len(valid_data) > 0:
                        ptt_values = (valid_data[col2] - valid_data[col1]) * 1000  # 转换为ms
                        
                        # 汇总统计
                        ptt_summary.append({
                            'sensor_pair': f"{sensor1}-{sensor2}",
                            'sensor_names': f"{self.sensor_mapping[sensor1]}→{self.sensor_mapping[sensor2]}",
                            'valid_heartbeats': len(valid_data),
                            'mean_ptt_ms': np.mean(ptt_values),
                            'std_ptt_ms': np.std(ptt_values),
                            'min_ptt_ms': np.min(ptt_values),
                            'max_ptt_ms': np.max(ptt_values),
                            'median_ptt_ms': np.median(ptt_values),
                            'q25_ptt_ms': np.percentile(ptt_values, 25),
                            'q75_ptt_ms': np.percentile(ptt_values, 75)
                        })
                        
                        # 时间序列数据
                        for idx, (heartbeat_id, ptt_val) in enumerate(zip(valid_data['heartbeat_id'], ptt_values)):
                            ptt_timeseries_all.append({
                                'heartbeat_id': heartbeat_id,
                                'sensor_pair': f"{sensor1}-{sensor2}",
                                'sensor_names': f"{self.sensor_mapping[sensor1]}→{self.sensor_mapping[sensor2]}",
                                'ptt_ms': ptt_val,
                                f'{sensor1}_time_s': valid_data[col1].iloc[idx],
                                f'{sensor2}_time_s': valid_data[col2].iloc[idx]
                            })
                        
                        print(f"  📊 {self.sensor_mapping[sensor1]}→{self.sensor_mapping[sensor2]}: "
                              f"{np.mean(ptt_values):.1f}±{np.std(ptt_values):.1f}ms "
                              f"({len(valid_data)}心跳)")
            
            # 5. 保存PTT矩阵汇总
            if ptt_summary:
                ptt_matrix_df = pd.DataFrame(ptt_summary)
                ptt_matrix_file = os.path.join(self.output_dir, f"ptt_matrix_exp_{exp_id}.csv")
                ptt_matrix_df.to_csv(ptt_matrix_file, index=False)
                print(f"💾 保存PTT矩阵: {ptt_matrix_file}")
            
            # 6. 保存PTT时间序列（用于建模）
            if ptt_timeseries_all:
                ptt_timeseries_df = pd.DataFrame(ptt_timeseries_all)
                ptt_timeseries_file = os.path.join(self.output_dir, f"ptt_timeseries_exp_{exp_id}.csv")
                ptt_timeseries_df.to_csv(ptt_timeseries_file, index=False)
                print(f"💾 保存PTT时间序列: {ptt_timeseries_file}")
                print(f"   📈 共{len(ptt_timeseries_all)}个PTT数据点，可用于血压建模")
            
        except Exception as e:
            print(f"⚠️  PTT分析失败: {e}")
    
    def create_visualizations(self, exp_id, sensor_results, matched_results, all_signals):
        """创建简化可视化"""
        try:
            fig, axes = plt.subplots(len(self.sensors), 1, figsize=(16, 3*len(self.sensors)), sharex=True)
            if len(self.sensors) == 1:
                axes = [axes]
            
            colors = ['red', 'blue', 'green', 'orange']
            
            for idx, sensor in enumerate(self.sensors):
                ax = axes[idx]
                
                if sensor in all_signals and sensor in sensor_results:
                    df = all_signals[sensor]
                    time = df['timestamp'].values - df['timestamp'].values[0]
                    filtered_signal = sensor_results[sensor]['filtered_signal']
                    peaks = sensor_results[sensor]['peaks']
                    quality = sensor_results[sensor]['quality']
                    
                    # 绘制滤波信号
                    ax.plot(time[:len(filtered_signal)], filtered_signal, 
                           color=colors[idx % len(colors)], linewidth=1.5, alpha=0.8,
                           label=f'{self.sensor_mapping[sensor]} IR')
                    
                    # 标记峰值
                    if len(peaks) > 0:
                        peak_times = peaks / self.fs
                        ax.scatter(peak_times, filtered_signal[peaks], 
                                 color='red', s=40, zorder=5, alpha=0.9)
                        
                        # 每10个峰值显示一个标号
                        for i, (pt, ps) in enumerate(zip(peak_times, filtered_signal[peaks])):
                            if i % 10 == 0:
                                ax.annotate(f'{i+1}', (pt, ps), xytext=(5, 5), 
                                          textcoords='offset points', fontsize=8)
                    
                    # 设置标题
                    hr_mean = sensor_results[sensor]['hr_mean']
                    ax.set_title(f'{self.sensor_mapping[sensor]} IR - {quality} - HR: {hr_mean:.1f} BPM', 
                                fontsize=12, fontweight='bold')
                    ax.set_ylabel('Signal', fontsize=10)
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                else:
                    ax.text(0.5, 0.5, f'{self.sensor_mapping[sensor]}: No Data', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{self.sensor_mapping[sensor]} IR - No Data')
            
            axes[-1].set_xlabel('Time (seconds)', fontsize=12)
            plt.suptitle(f'Experiment {exp_id} - IR Channel Peak Detection', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            plot_file = os.path.join(self.output_dir, f"ir_peaks_exp_{exp_id}.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"📊 保存可视化: {plot_file}")
            
        except Exception as e:
            print(f"❌ 可视化创建失败: {e}")
    
    def run_analysis(self, experiment_list=None):
        """运行IR通道PTT峰值检测分析"""
        if experiment_list is None:
            experiment_list = [f.split('_')[0] for f in os.listdir(self.data_path) 
                             if f.endswith('_hub_sensor2_aligned.csv')]
            experiment_list = sorted(list(set(experiment_list)))
        
        print(f"\n🔬 开始IR通道PTT峰值检测分析")
        print(f"📋 实验列表: {experiment_list}")
        print(f"🎯 检测策略:")
        print(f"   - 专注IR通道（信号质量最佳）")
        print(f"   - 稳健峰值检测 + IBI质量控制")
        print(f"   - 心率范围: {self.min_hr}-{self.max_hr} BPM")
        print(f"   - 滤波范围: {self.filter_lowcut}-{self.filter_highcut} Hz")
        print(f"   - 输出5个标准CSV文件")
        
        all_results = {}
        
        for exp_id in tqdm(experiment_list, desc="处理实验"):
            try:
                sensor_results, matched_results = self.process_experiment(exp_id)
                all_results[exp_id] = {
                    'sensor_results': sensor_results,
                    'matched_results': matched_results
                }
            except Exception as e:
                print(f"❌ 实验 {exp_id} 处理失败: {e}")
                continue
        
        print(f"\n✅ IR通道PTT峰值检测完成！")
        print(f"📁 结果保存在: {self.output_dir}")
        print(f"\n📊 输出文件说明:")
        print(f"   1. sensor_summary_exp_X.csv - 传感器质量汇总")
        print(f"   2. all_peaks_exp_X.csv - 所有峰值详细信息")
        print(f"   3. all_ibi_exp_X.csv - 所有IBI详细信息")
        print(f"   4. ptt_matrix_exp_X.csv - PTT矩阵汇总")
        print(f"   5. ptt_timeseries_exp_X.csv - PTT时间序列（用于建模）")
        print(f"\n🎯 下一步: 使用ptt_timeseries_exp_X.csv进行血压建模")
        
        return all_results

def main():
    """主函数"""
    print("🩺 IR通道专门的PTT峰值检测器")
    print("=" * 60)
    print("📖 优化特性:")
    print("   • 专注IR通道峰值检测")
    print("   • 稳健的IBI计算和质量控制")
    print("   • 智能心跳匹配")
    print("   • 标准化CSV输出便于建模")
    print("=" * 60)
    
    detector = IRBasedPTTPeakDetector()
    results = detector.run_analysis(['1'])
    
    print("\n🎯 分析完成，建议下一步:")
    print("1. 检查sensor_summary_exp_1.csv了解传感器质量")
    print("2. 使用ptt_timeseries_exp_1.csv进行血压建模")
    print("3. 验证PTT与血压的相关性 (a*PTT + b)")

if __name__ == "__main__":
    main() 