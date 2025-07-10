#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🩺 PTT生理信号峰值检测器 - 专注IR通道和IBI验证

基于师兄建议的改进：
1. ✅ 专注IR通道峰值检测（信号质量最佳）
2. ✅ IBI验证机制：确保峰值数量与心率计算一致
3. ✅ 同一心跳区间的峰值匹配
4. ✅ 为PTT矩阵计算做准备（4传感器→6组合）

核心原理：
- PTT使用峰值计算
- IR通道信号最稳定
- 峰值检测数量 = 傅里叶变换心率计算
- 同一心跳的峰值需要时序对应
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, filtfilt, butter, welch
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class IRBasedPTTPeakDetector:
    """基于IR通道的PTT峰值检测器"""
    
    def __init__(self, data_path="/root/PI_Lab/output/csv_output"):
        self.data_path = data_path
        self.output_dir = "/root/PI_Lab/ptt_output"
        self.sensors = ['sensor2', 'sensor3', 'sensor4', 'sensor5']  # 4个传感器
        self.target_channel = 'ir'  # 专注IR通道
        self.sensor_mapping = {
            'sensor2': 'nose', 'sensor3': 'finger', 
            'sensor4': 'wrist', 'sensor5': 'ear'
        }
        
        # 生理参数约束
        self.fs = 100  # 采样率100Hz
        self.min_hr = 50   # 最小心率BPM
        self.max_hr = 200  # 最大心率BPM
        self.refractory_period = 0.3  # 心肌不应期（秒）
        
        # 滤波参数：0.5-3Hz（心率频率范围）
        self.filter_lowcut = 0.5   
        self.filter_highcut = 3.0  
        self.filter_order = 3
        
        # IBI验证参数
        self.ibi_tolerance = 0.1  # IBI容差（10%）
        
        os.makedirs(self.output_dir, exist_ok=True)
        
    def bandpass_filter(self, data, lowcut=0.5, highcut=3.0, fs=100, order=3):
        """带通滤波 - 专门针对心率频段"""
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
            print(f"⚠️  滤波失败: {e}")
            return data
    
    def estimate_heart_rate_fft(self, signal, fs=100):
        """使用FFT估计心率 - 作为验证基准"""
        try:
            # 计算功率谱密度
            welch_result = welch(signal, fs, nperseg=min(len(signal), 1024))
            f, psd = welch_result[0], welch_result[1]
            
            # 心率频率范围: 50-200 BPM = 0.83-3.33 Hz
            hr_freq_min = self.min_hr / 60.0
            hr_freq_max = self.max_hr / 60.0
            
            # 找到心率频率范围内的主频率
            hr_mask = (f >= hr_freq_min) & (f <= hr_freq_max)
            if np.any(hr_mask) and np.max(psd[hr_mask]) > 0:
                peak_freq = f[hr_mask][np.argmax(psd[hr_mask])]
                estimated_hr = peak_freq * 60  # 转换为BPM
                return np.clip(estimated_hr, self.min_hr, self.max_hr)
            else:
                return None
        except Exception as e:
            print(f"⚠️  FFT心率估计失败: {e}")
            return None
    
    def estimate_heart_rate_peaks(self, peaks, fs=100):
        """基于峰值估计心率"""
        if len(peaks) < 2:
            return None
        
        # 计算平均IBI
        intervals = np.diff(peaks) / fs  # 转换为秒
        valid_intervals = intervals[(intervals >= 0.3) & (intervals <= 1.2)]  # 50-200 BPM
        
        if len(valid_intervals) == 0:
            return None
        
        avg_interval = np.mean(valid_intervals)
        return 60.0 / avg_interval  # 转换为BPM
    
    def validate_peaks_with_ibi(self, peaks, signal, fs=100):
        """使用IBI验证峰值检测的准确性"""
        if len(peaks) < 2:
            return False, "峰值数量不足"
        
        # 1. 基于峰值的心率估计
        hr_peaks = self.estimate_heart_rate_peaks(peaks, fs)
        if hr_peaks is None:
            return False, "峰值心率估计失败"
        
        # 2. 基于FFT的心率估计
        hr_fft = self.estimate_heart_rate_fft(signal, fs)
        if hr_fft is None:
            return False, "FFT心率估计失败"
        
        # 3. 验证两种方法的一致性
        hr_diff = abs(hr_peaks - hr_fft) / hr_fft
        if hr_diff > self.ibi_tolerance:
            return False, f"心率不一致: 峰值{hr_peaks:.1f} vs FFT{hr_fft:.1f} BPM (差异{hr_diff*100:.1f}%)"
        
        # 4. 验证预期峰值数量
        signal_duration = len(signal) / fs  # 信号持续时间（秒）
        expected_peaks = int(hr_fft * signal_duration / 60)  # 预期峰值数量
        peak_count_diff = abs(len(peaks) - expected_peaks) / expected_peaks
        
        if peak_count_diff > 0.2:  # 允许20%误差
            return False, f"峰值数量不匹配: 检测{len(peaks)} vs 预期{expected_peaks} (差异{peak_count_diff*100:.1f}%)"
        
        return True, f"验证通过: HR={hr_peaks:.1f}BPM, 峰值数量={len(peaks)}"
    
    def adaptive_peak_detection_ir(self, signal, fs=100):
        """针对IR通道的自适应峰值检测"""
        try:
            # 1. 估计心率作为先验知识
            estimated_hr = self.estimate_heart_rate_fft(signal, fs)
            if estimated_hr is None:
                print("⚠️  无法估计心率，使用默认参数")
                estimated_hr = 75  # 默认心率
            
            # 2. 计算自适应参数
            min_distance = int(self.refractory_period * fs)  # 最小间隔
            expected_interval = 60.0 / estimated_hr * fs  # 预期间隔（样本）
            
            # 3. 自适应阈值
            signal_std = np.std(signal)
            signal_mean = np.mean(signal)
            
            # 基于信号质量调整阈值
            snr_estimate = signal_std / np.mean(np.abs(signal - signal_mean))
            if snr_estimate > 0.1:  # 高质量信号
                height_threshold = signal_mean + 0.2 * signal_std
                prominence_threshold = 0.05 * signal_std
            else:  # 低质量信号
                height_threshold = signal_mean + 0.4 * signal_std
                prominence_threshold = 0.15 * signal_std
            
            # 4. 初步峰值检测
            peaks, properties = find_peaks(
                signal,
                height=height_threshold,
                distance=min_distance,
                prominence=prominence_threshold
            )
            
            # 5. IBI约束精化
            if len(peaks) > 1:
                peak_intervals = np.diff(peaks) / fs
                valid_interval_min = 60.0 / self.max_hr  # 0.3秒
                valid_interval_max = 60.0 / self.min_hr  # 1.2秒
                
                refined_peaks = [peaks[0]]  # 保留第一个峰值
                
                for i in range(1, len(peaks)):
                    interval = peak_intervals[i-1]
                    
                    if valid_interval_min <= interval <= valid_interval_max:
                        refined_peaks.append(peaks[i])
                    elif interval < valid_interval_min:
                        # 间隔太短，选择更高的峰值
                        if signal[peaks[i]] > signal[refined_peaks[-1]]:
                            refined_peaks[-1] = peaks[i]
                
                peaks = np.array(refined_peaks)
            
            # 6. IBI验证
            is_valid, message = self.validate_peaks_with_ibi(peaks, signal, fs)
            
            return {
                'peaks': peaks,
                'is_valid': is_valid,
                'validation_message': message,
                'estimated_hr_fft': estimated_hr,
                'estimated_hr_peaks': self.estimate_heart_rate_peaks(peaks, fs)
            }
            
        except Exception as e:
            print(f"⚠️  峰值检测失败: {e}")
            return {
                'peaks': np.array([]),
                'is_valid': False,
                'validation_message': f"检测失败: {e}",
                'estimated_hr_fft': None,
                'estimated_hr_peaks': None
            }
    
    def match_peaks_across_sensors(self, sensor_results):
        """匹配不同传感器间同一心跳的峰值"""
        try:
            # 1. 找到有效的传感器结果
            valid_sensors = []
            for sensor in self.sensors:
                if (sensor in sensor_results and 
                    sensor_results[sensor]['is_valid'] and 
                    len(sensor_results[sensor]['peaks']) > 0):
                    valid_sensors.append(sensor)
            
            if len(valid_sensors) < 2:
                print("⚠️  有效传感器数量不足，无法进行峰值匹配")
                return {}
            
            print(f"📍 有效传感器: {valid_sensors}")
            
            # 2. 使用第一个有效传感器作为参考
            reference_sensor = valid_sensors[0]
            reference_peaks = sensor_results[reference_sensor]['peaks'] / self.fs  # 转换为时间（秒）
            
            # 3. 为每个心跳创建时间窗口
            heartbeat_windows = []
            for i, ref_time in enumerate(reference_peaks):
                if i == 0:
                    # 第一个心跳：从开始到中点
                    window_start = 0
                    window_end = ref_time + (reference_peaks[1] - reference_peaks[0])/2 if len(reference_peaks) > 1 else ref_time + 0.5
                elif i == len(reference_peaks) - 1:
                    # 最后一个心跳：从中点到结束
                    window_start = ref_time - (reference_peaks[i] - reference_peaks[i-1])/2
                    window_end = float('inf')
                else:
                    # 中间心跳：前后中点之间
                    window_start = ref_time - (reference_peaks[i] - reference_peaks[i-1])/2
                    window_end = ref_time + (reference_peaks[i+1] - reference_peaks[i])/2
                
                heartbeat_windows.append({
                    'heartbeat_id': i + 1,
                    'reference_time': ref_time,
                    'window_start': window_start,
                    'window_end': window_end,
                    'sensor_peaks': {reference_sensor: ref_time}
                })
            
            # 4. 为其他传感器匹配峰值
            for sensor in valid_sensors[1:]:
                sensor_peaks = sensor_results[sensor]['peaks'] / self.fs  # 转换为时间
                
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
                    if best_window is not None:
                        best_window['sensor_peaks'][sensor] = peak_time
            
            # 5. 过滤完整的心跳（所有传感器都有峰值）
            complete_heartbeats = []
            for window in heartbeat_windows:
                if len(window['sensor_peaks']) >= len(valid_sensors):  # 所有传感器都有峰值
                    complete_heartbeats.append(window)
            
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
        
        # 1. 处理每个传感器的IR通道
        for sensor in self.sensors:
            try:
                file_path = os.path.join(self.data_path, f"{exp_id}_hub_{sensor}_aligned.csv")
                if not os.path.exists(file_path):
                    print(f"❌ 文件不存在: {file_path}")
                    continue
                
                df = pd.read_csv(file_path)
                all_signals[sensor] = df
                
                # 获取IR通道信号（假设是第2列：red, ir, green）
                if len(df.columns) >= 3:  # timestamp + 3个颜色通道
                    ir_signal = df.iloc[:, 2].values  # IR通道
                    
                    # 滤波
                    filtered_signal = self.bandpass_filter(
                        ir_signal, self.filter_lowcut, self.filter_highcut, self.fs
                    )
                    
                    # 峰值检测和验证
                    peak_result = self.adaptive_peak_detection_ir(filtered_signal, self.fs)
                    
                    # 添加额外信息
                    peak_result.update({
                        'sensor': sensor,
                        'sensor_name': self.sensor_mapping[sensor],
                        'filtered_signal': filtered_signal,
                        'peak_times': peak_result['peaks'] / self.fs,
                        'peak_count': len(peak_result['peaks'])
                    })
                    
                    sensor_results[sensor] = peak_result
                    
                    # 打印结果
                    status = "✅" if peak_result['is_valid'] else "❌"
                    print(f"  {status} {sensor}({self.sensor_mapping[sensor]}): "
                          f"{peak_result['peak_count']}峰值, "
                          f"HR={peak_result['estimated_hr_peaks']:.1f}BPM - "
                          f"{peak_result['validation_message']}")
                else:
                    print(f"⚠️  {sensor}: 数据列不足")
                    
            except Exception as e:
                print(f"❌ 处理 {sensor} 失败: {e}")
                continue
        
        # 2. 匹配不同传感器间的峰值
        matched_results = self.match_peaks_across_sensors(sensor_results)
        
        # 3. 保存结果
        self.save_results(exp_id, sensor_results, matched_results, all_signals)
        
        return sensor_results, matched_results
    
    def save_results(self, exp_id, sensor_results, matched_results, all_signals):
        """保存检测结果"""
        try:
            # 1. 保存传感器级别结果
            sensor_summary = []
            for sensor in sensor_results:
                result = sensor_results[sensor]
                sensor_summary.append({
                    'sensor': sensor,
                    'sensor_name': result['sensor_name'],
                    'peak_count': result['peak_count'],
                    'is_valid': result['is_valid'],
                    'hr_fft': result['estimated_hr_fft'],
                    'hr_peaks': result['estimated_hr_peaks'],
                    'validation_message': result['validation_message']
                })
            
            if sensor_summary:
                summary_df = pd.DataFrame(sensor_summary)
                summary_file = os.path.join(self.output_dir, f"ir_peaks_summary_exp_{exp_id}.csv")
                summary_df.to_csv(summary_file, index=False)
                print(f"💾 保存传感器汇总: {summary_file}")
            
            # 2. 保存匹配的心跳数据
            if matched_results and 'complete_heartbeats' in matched_results:
                heartbeat_data = []
                for hb in matched_results['complete_heartbeats']:
                    row = {
                        'heartbeat_id': hb['heartbeat_id'],
                        'reference_time': hb['reference_time']
                    }
                    # 添加每个传感器的峰值时间
                    for sensor in matched_results['valid_sensors']:
                        if sensor in hb['sensor_peaks']:
                            row[f'{sensor}_peak_time'] = hb['sensor_peaks'][sensor]
                        else:
                            row[f'{sensor}_peak_time'] = np.nan
                    
                    heartbeat_data.append(row)
                
                if heartbeat_data:
                    heartbeat_df = pd.DataFrame(heartbeat_data)
                    heartbeat_file = os.path.join(self.output_dir, f"matched_heartbeats_exp_{exp_id}.csv")
                    heartbeat_df.to_csv(heartbeat_file, index=False)
                    print(f"💾 保存匹配心跳: {heartbeat_file}")
                    
                    # 计算PTT矩阵预览
                    self.calculate_ptt_preview(heartbeat_df, exp_id)
            
            # 3. 生成可视化
            self.create_visualizations(exp_id, sensor_results, matched_results, all_signals)
            
        except Exception as e:
            print(f"❌ 保存结果失败: {e}")
    
    def calculate_ptt_preview(self, heartbeat_df, exp_id):
        """计算PTT矩阵预览（4传感器→6组合）"""
        try:
            sensors = ['sensor2', 'sensor3', 'sensor4', 'sensor5']
            sensor_combinations = []
            
            # 生成所有传感器组合
            for i in range(len(sensors)):
                for j in range(i+1, len(sensors)):
                    sensor_combinations.append((sensors[i], sensors[j]))
            
            print(f"\n📊 PTT矩阵预览 (6个传感器组合):")
            ptt_summary = []
            
            for sensor1, sensor2 in sensor_combinations:
                col1 = f'{sensor1}_peak_time'
                col2 = f'{sensor2}_peak_time'
                
                if col1 in heartbeat_df.columns and col2 in heartbeat_df.columns:
                    # 计算PTT (传播方向：近心→远心)
                    valid_data = heartbeat_df.dropna(subset=[col1, col2])
                    if len(valid_data) > 0:
                        ptt_values = (valid_data[col2] - valid_data[col1]) * 1000  # 转换为毫秒
                        
                        ptt_summary.append({
                            'sensor_pair': f"{sensor1}-{sensor2}",
                            'sensor_names': f"{self.sensor_mapping[sensor1]}-{self.sensor_mapping[sensor2]}",
                            'valid_heartbeats': len(valid_data),
                            'mean_ptt_ms': np.mean(ptt_values),
                            'std_ptt_ms': np.std(ptt_values),
                            'min_ptt_ms': np.min(ptt_values),
                            'max_ptt_ms': np.max(ptt_values)
                        })
                        
                        print(f"  {self.sensor_mapping[sensor1]}→{self.sensor_mapping[sensor2]}: "
                              f"{np.mean(ptt_values):.1f}±{np.std(ptt_values):.1f}ms "
                              f"({len(valid_data)}心跳)")
            
            # 保存PTT预览
            if ptt_summary:
                ptt_df = pd.DataFrame(ptt_summary)
                ptt_file = os.path.join(self.output_dir, f"ptt_matrix_preview_exp_{exp_id}.csv")
                ptt_df.to_csv(ptt_file, index=False)
                print(f"💾 保存PTT预览: {ptt_file}")
            
        except Exception as e:
            print(f"⚠️  PTT预览计算失败: {e}")
    
    def create_visualizations(self, exp_id, sensor_results, matched_results, all_signals):
        """创建IR通道专门的可视化"""
        try:
            # 1. IR通道信号和峰值可视化
            fig, subplot_axes = plt.subplots(len(self.sensors), 1, figsize=(16, 3*len(self.sensors)), sharex=True)
            if len(self.sensors) == 1:
                subplot_axes = [subplot_axes]
            
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FECA57']
            
            for idx, sensor in enumerate(self.sensors):
                ax = subplot_axes[idx]
                
                if sensor in all_signals and sensor in sensor_results:
                    # 获取时间轴
                    df = all_signals[sensor]
                    time = df['timestamp'].values - df['timestamp'].values[0]
                    
                    # 绘制滤波后的IR信号
                    filtered_signal = sensor_results[sensor]['filtered_signal']
                    peaks = sensor_results[sensor]['peaks']
                    
                    ax.plot(time[:len(filtered_signal)], filtered_signal, 
                           color=colors[idx], linewidth=1.5, alpha=0.8,
                           label=f'{self.sensor_mapping[sensor]} IR')
                    
                    # 标记峰值
                    if len(peaks) > 0:
                        peak_times = peaks / self.fs
                        ax.scatter(peak_times, filtered_signal[peaks], 
                                 color='red', s=40, zorder=5, alpha=0.9, marker='o')
                        
                        # 添加峰值编号
                        for i, (pt, ps) in enumerate(zip(peak_times, filtered_signal[peaks])):
                            if i % 3 == 0:  # 每3个显示一个编号，避免拥挤
                                ax.annotate(f'{i+1}', (pt, ps), xytext=(5, 5), 
                                          textcoords='offset points', fontsize=8, alpha=0.7)
                    
                    # 设置标题和标签
                    status = "✅ Valid" if sensor_results[sensor]['is_valid'] else "❌ Invalid"
                    hr = sensor_results[sensor]['estimated_hr_peaks']
                    ax.set_title(f'{self.sensor_mapping[sensor]} IR Channel - {status} - HR: {hr:.1f} BPM', 
                                fontsize=12, fontweight='bold')
                    ax.set_ylabel('Filtered Signal', fontsize=10)
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                
                else:
                    ax.text(0.5, 0.5, f'{self.sensor_mapping[sensor]}: No Data', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=12)
                    ax.set_title(f'{self.sensor_mapping[sensor]} IR Channel - No Data')
            
            subplot_axes[-1].set_xlabel('Time (seconds)', fontsize=12)
            plt.suptitle(f'Experiment {exp_id} - IR Channel Peak Detection Results', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # 保存图像
            plot_file = os.path.join(self.output_dir, f"ir_peaks_exp_{exp_id}.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"📊 保存IR峰值图: {plot_file}")
            
            # 2. 心跳匹配可视化
            if matched_results and 'complete_heartbeats' in matched_results:
                self.create_heartbeat_matching_plot(exp_id, matched_results, sensor_results)
            
        except Exception as e:
            print(f"❌ 可视化创建失败: {e}")
    
    def create_heartbeat_matching_plot(self, exp_id, matched_results, sensor_results):
        """创建心跳匹配可视化"""
        try:
            complete_heartbeats = matched_results['complete_heartbeats']
            valid_sensors = matched_results['valid_sensors']
            
            if len(complete_heartbeats) == 0:
                print("⚠️  没有完整心跳数据，跳过匹配可视化")
                return
            
            # 准备数据
            heartbeat_ids = [hb['heartbeat_id'] for hb in complete_heartbeats]
            sensor_times = {sensor: [] for sensor in valid_sensors}
            
            for hb in complete_heartbeats:
                for sensor in valid_sensors:
                    if sensor in hb['sensor_peaks']:
                        sensor_times[sensor].append(hb['sensor_peaks'][sensor])
                    else:
                        sensor_times[sensor].append(np.nan)
            
            # 创建图表
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FECA57']
            
            # 上图：心跳时间序列
            for idx, sensor in enumerate(valid_sensors):
                times = sensor_times[sensor]
                ax1.plot(heartbeat_ids, times, 'o-', color=colors[idx % len(colors)], 
                        linewidth=2, markersize=6, alpha=0.8,
                        label=f'{self.sensor_mapping[sensor]}')
            
            ax1.set_xlabel('Heartbeat ID')
            ax1.set_ylabel('Peak Time (seconds)')
            ax1.set_title('Matched Heartbeat Timing Across Sensors')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 下图：PTT热力图（如果有足够数据）
            if len(valid_sensors) >= 2:
                ptt_matrix = np.full((len(valid_sensors), len(valid_sensors)), np.nan)
                
                for i, sensor1 in enumerate(valid_sensors):
                    for j, sensor2 in enumerate(valid_sensors):
                        if i != j:
                            times1 = np.array(sensor_times[sensor1])
                            times2 = np.array(sensor_times[sensor2])
                            
                            # 计算平均PTT
                            valid_mask = ~(np.isnan(times1) | np.isnan(times2))
                            if np.sum(valid_mask) > 0:
                                ptt_values = (times2[valid_mask] - times1[valid_mask]) * 1000  # ms
                                ptt_matrix[i, j] = np.mean(ptt_values)
                
                # 绘制热力图
                im = ax2.imshow(ptt_matrix, cmap='RdYlBu_r', aspect='auto')
                ax2.set_xticks(range(len(valid_sensors)))
                ax2.set_yticks(range(len(valid_sensors)))
                ax2.set_xticklabels([self.sensor_mapping[s] for s in valid_sensors])
                ax2.set_yticklabels([self.sensor_mapping[s] for s in valid_sensors])
                ax2.set_xlabel('To Sensor')
                ax2.set_ylabel('From Sensor')
                ax2.set_title('Average PTT Matrix (ms)')
                
                # 添加数值标注
                for i in range(len(valid_sensors)):
                    for j in range(len(valid_sensors)):
                        if not np.isnan(ptt_matrix[i, j]):
                            text = ax2.text(j, i, f'{ptt_matrix[i, j]:.1f}',
                                          ha="center", va="center", color="black", fontweight='bold')
                
                plt.colorbar(im, ax=ax2, label='PTT (ms)')
            
            plt.tight_layout()
            
            # 保存图像
            match_plot_file = os.path.join(self.output_dir, f"heartbeat_matching_exp_{exp_id}.png")
            plt.savefig(match_plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"📊 保存心跳匹配图: {match_plot_file}")
            
        except Exception as e:
            print(f"❌ 心跳匹配图创建失败: {e}")
    
    def run_analysis(self, experiment_list=None):
        """运行IR通道专门的PTT峰值检测分析"""
        if experiment_list is None:
            # 自动检测可用实验
            experiment_list = []
            for file in os.listdir(self.data_path):
                if file.endswith('_hub_sensor2_aligned.csv'):
                    exp_id = file.split('_')[0]
                    experiment_list.append(exp_id)
            experiment_list = sorted(list(set(experiment_list)))
        
        print(f"\n🔬 开始IR通道PTT峰值检测分析")
        print(f"📋 实验列表: {experiment_list}")
        print(f"🎯 检测策略:")
        print(f"   - 专注IR通道（信号质量最佳）")
        print(f"   - IBI验证确保峰值数量与心率一致")
        print(f"   - 心率范围: {self.min_hr}-{self.max_hr} BPM")
        print(f"   - 滤波范围: {self.filter_lowcut}-{self.filter_highcut} Hz")
        print(f"   - 4传感器 → 6个PTT组合")
        
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
        print(f"🎯 下一步: 使用PTT矩阵进行血压建模 (a*PTT + b)")
        
        return all_results

def main():
    """主函数"""
    print("🩺 IR通道专门的PTT峰值检测器")
    print("=" * 60)
    print("📖 基于师兄建议的改进:")
    print("   • 专注IR通道峰值检测（信号质量最佳）")
    print("   • IBI验证机制：峰值数量与心率一致")
    print("   • 同一心跳区间的峰值匹配")
    print("   • 4传感器 → 6个PTT组合用于建模")
    print("=" * 60)
    
    # 创建检测器
    detector = IRBasedPTTPeakDetector()
    
    # 运行分析（测试实验1）
    results = detector.run_analysis(['1'])
    
    print("\n🎯 分析完成，建议下一步:")
    print("1. 检查IR峰值检测质量和IBI验证结果")
    print("2. 确认心跳匹配的准确性")
    print("3. 使用PTT矩阵进行血压建模 (a*PTT + b)")
    print("4. 验证PTT与血压的相关性")

if __name__ == "__main__":
    main() 