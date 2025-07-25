#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🩺 IR通道PTT峰值检测器 - 窗口化时频域验证版本（师兄建议版）

师兄的核心建议：
1. ✅ 专注IR通道峰值检测（信号质量最佳）
2. ✅ 使用库函数转化为IBI，避免误识别
3. ✅ 分成20-30s片段，每个窗口内验证时频域一致性
4. ✅ FFT心率和峰值检测心率差异<5BPM才认为有效
5. ✅ 只在有效窗口内计算PTT，提高准确性
6. ✅ 输出包含窗口验证信息的详细结果

核心改进：
- 30s滑动窗口分析
- 每个窗口的时域（峰值）vs频域（FFT）心率验证
- 质量控制：只在验证通过的窗口内计算PTT
- 详细的窗口质量报告
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks, welch
import warnings

# 尝试导入专业库
try:
    import neurokit2 as nk
    NEUROKIT_AVAILABLE = True
except ImportError:
    NEUROKIT_AVAILABLE = False
    print("⚠️  NeuroKit2未安装，将使用scipy备选方案")

try:
    import heartpy as hp
    HEARTPY_AVAILABLE = True
except ImportError:
    HEARTPY_AVAILABLE = False
    print("⚠️  HeartPy未安装，将使用scipy备选方案")

warnings.filterwarnings('ignore')

class IRWindowedPTTPeakDetector:
    """窗口化时频域验证的IR通道PTT峰值检测器"""
    
    def __init__(self, data_path="output/csv_output", method="auto"):
        self.data_path = data_path
        self.output_dir = "ptt_output2"
        self.sensors = ['sensor2', 'sensor3', 'sensor4', 'sensor5']
        self.target_channel = 'ir'
        self.sensor_mapping = {
            'sensor2': 'nose', 'sensor3': 'finger', 
            'sensor4': 'wrist', 'sensor5': 'ear'
        }
        # 动态计算采样率，而不是固定100Hz
        self.fs = None  # 将在数据加载时动态计算
        self.default_fs = 100  # 默认采样率作为后备
        self.min_hr = 50
        self.max_hr = 200
        self.refractory_period = 0.3
        self.filter_lowcut = 0.5
        self.filter_highcut = 3.0
        self.filter_order = 3
        self.ibi_tolerance = 0.15
        
        # 窗口化参数（师兄建议） - 密集滑窗版本
        self.window_duration = 20  # 20秒窗口
        self.window_step = 5       # 5秒滑窗步长（更密集）
        self.hr_tolerance_bpm = 5  # 时频域心率差异容忍度（放宽）
        
        # 选择峰值检测方法
        self.detection_method = self._select_method(method)
        print(f"🔧 峰值检测方法: {self.detection_method}")
        print(f"🪟 窗口参数: {self.window_duration}s窗口, {self.window_step}s滑窗步长（密集滑窗）")
        print(f"🎯 验证标准: 时频域心率差异<{self.hr_tolerance_bpm}BPM")
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
    def calculate_sampling_rate(self, timestamps):
        """动态计算采样率，基于时间戳差值"""
        if len(timestamps) < 2:
            return self.default_fs
        
        # 计算时间戳差值
        time_diff = np.diff(timestamps)
        # 过滤掉负值和零值
        valid_diffs = time_diff[time_diff > 0]
        if len(valid_diffs) == 0:
            return self.default_fs
        
        # 计算采样率
        mean_interval = np.mean(valid_diffs)
        sampling_rate = 1 / mean_interval
        
        # 合理性检查：采样率应该在合理范围内
        if 50 <= sampling_rate <= 2500:
            return sampling_rate
        else:
            print(f"⚠️ 计算出的采样率 {sampling_rate:.1f}Hz 超出合理范围，使用默认值 {self.default_fs}Hz")
            return self.default_fs
    
    def _select_method(self, method):
        """智能选择可用的峰值检测方法"""
        if method == "auto":
            if NEUROKIT_AVAILABLE:
                return "neurokit2"
            elif HEARTPY_AVAILABLE:
                return "heartpy"
            else:
                return "scipy_advanced"
        elif method == "neurokit2" and NEUROKIT_AVAILABLE:
            return "neurokit2"
        elif method == "heartpy" and HEARTPY_AVAILABLE:
            return "heartpy"
        else:
            return "scipy_advanced"

    def bandpass_filter(self, data, lowcut=0.5, highcut=3.0, fs=100, order=3):
        """带通滤波 - 增强版，处理NaN值"""
        try:
            if len(data) == 0:
                return data
                
            # 处理NaN值
            data_array = np.array(data, dtype=float)
            nan_count = np.isnan(data_array).sum()
            if nan_count > 0:
                data_series = pd.Series(data_array)
                data_interpolated = data_series.interpolate(method='linear')
                data_clean = data_interpolated.fillna(method='bfill').fillna(method='ffill').values
            else:
                data_clean = data_array.copy()
            
            # 检查数据变异性
            data_std = np.std(data_clean)
            if data_std < 1e-10:
                return data_clean
            
            nyquist = fs / 2
            low = max(lowcut / nyquist, 0.01)
            high = min(highcut / nyquist, 0.99)
            
            if low >= high:
                return data_clean
            
            b, a = butter(order, [low, high], btype='band')
            filtered_data = filtfilt(b, a, data_clean)
            
            if np.isnan(filtered_data).sum() > 0:
                return data_clean
            
            return filtered_data
            
        except Exception as e:
            print(f"⚠️  滤波失败: {e}")
            try:
                data_array = np.array(data, dtype=float)
                data_series = pd.Series(data_array)
                data_interpolated = data_series.interpolate(method='linear')
                return data_interpolated.fillna(method='bfill').fillna(method='ffill').values
            except:
                return np.array(data, dtype=float)

    def get_fft_hr(self, signal, fs=100, min_hr=50, max_hr=200):
        """计算FFT心率（照抄data_processor.py的get_hr）"""
        try:
            p, q = welch(signal, fs, nfft=int(1e5/fs), nperseg=np.min((len(signal)-1, 256)))
            freq_range = (p > min_hr/60) & (p < max_hr/60)
            if np.any(freq_range):
                peak_freq = p[freq_range][np.argmax(q[freq_range])]
                return peak_freq * 60
            else:
                return 0.0
        except Exception as e:
            print(f"⚠️  FFT心率计算失败: {e}")
            return 0.0
    
    def detect_peaks_in_window(self, signal, fs=100):
        """在单个窗口内进行峰值检测"""
        try:
            filtered_signal = self.bandpass_filter(signal, self.filter_lowcut, self.filter_highcut, fs)
            
            if self.detection_method == "neurokit2":
                peaks_dict, _ = nk.ppg_peaks(filtered_signal, sampling_rate=fs, method="elgendi")
                peak_indices = np.where(peaks_dict['PPG_Peaks'] == 1)[0]
            elif self.detection_method == "heartpy":
                working_data, _ = hp.process(filtered_signal, sample_rate=fs)
                peak_indices = working_data['peaklist']
            else:
                # scipy高级方法
                min_distance = int(self.refractory_period * fs)
                signal_std = np.std(filtered_signal)
                signal_mean = np.mean(filtered_signal)
                
                thresholds = [
                    (signal_mean + 0.2 * signal_std, 0.1 * signal_std),
                    (signal_mean + 0.1 * signal_std, 0.05 * signal_std),
                    (signal_mean, 0.02 * signal_std)
                ]
                
                peak_indices = np.array([])
                for height_threshold, prominence_threshold in thresholds:
                    peak_indices, _ = find_peaks(
                        filtered_signal,
                        height=height_threshold,
                        distance=min_distance,
                        prominence=prominence_threshold
                    )
                    if len(peak_indices) >= 3:
                        break
            
            if len(peak_indices) < 2:
                return {
                    'peaks': np.array([]),
                    'peak_times': np.array([]),
                    'ibi_ms': np.array([]),
                    'peak_hr_bpm': 0,
                    'filtered_signal': filtered_signal
                }
            
            peak_times = peak_indices / fs
            ibi_ms = np.diff(peak_times) * 1000
            
            # 计算时域心率（基于IBI）
            valid_ibi = ibi_ms[(ibi_ms >= 300) & (ibi_ms <= 1200)]
            if len(valid_ibi) > 0:
                peak_hr_bpm = np.mean(60000 / valid_ibi)
            else:
                peak_hr_bpm = 0
            
            return {
                'peaks': peak_indices,
                'peak_times': peak_times,
                'ibi_ms': ibi_ms,
                'peak_hr_bpm': peak_hr_bpm,
                'filtered_signal': filtered_signal
            }
            
        except Exception as e:
            print(f"⚠️  窗口峰值检测失败: {e}")
            return {
                'peaks': np.array([]),
                'peak_times': np.array([]),
                'ibi_ms': np.array([]),
                'peak_hr_bpm': 0,
                'filtered_signal': signal
            }
    
    def create_windows(self, signal_length, fs):
        """创建密集滑窗"""
        window_samples = int(self.window_duration * fs)
        step_samples = int(self.window_step * fs)
        
        windows = []
        start = 0
        window_id = 0
        
        while start + window_samples <= signal_length:
            end = start + window_samples
            windows.append({
                'window_id': window_id,
                'start_sample': start,
                'end_sample': end,
                'start_time_s': start / fs,
                'end_time_s': end / fs,
                'duration_s': self.window_duration
            })
            start += step_samples
            window_id += 1
        
        return windows
    
    def analyze_sensor_windowed(self, signal, sensor_name, fs):
        """对单个传感器进行窗口化分析"""
        windows = self.create_windows(len(signal), fs)
        window_results = []
        
        print(f"  📊 {sensor_name}: 创建了{len(windows)}个窗口")
        
        for window in windows:
            start_idx = window['start_sample']
            end_idx = window['end_sample']
            window_signal = signal[start_idx:end_idx]
            
            # 时域峰值检测
            peak_result = self.detect_peaks_in_window(window_signal, fs)
            
            # 频域FFT心率
            fft_hr = self.get_fft_hr(window_signal, fs, self.min_hr, self.max_hr)
            
            # 时频域一致性验证
            hr_diff = abs(peak_result['peak_hr_bpm'] - fft_hr)
            is_valid = (
                peak_result['peak_hr_bpm'] > 0 and 
                fft_hr > 0 and 
                hr_diff <= self.hr_tolerance_bpm and
                len(peak_result['peaks']) >= 3
            )
            
            # 调整峰值时间到全局时间
            global_peak_times = peak_result['peak_times'] + window['start_time_s']
            global_peak_indices = peak_result['peaks'] + start_idx
            
            window_result = {
                **window,
                'sensor': sensor_name,
                'peak_count': len(peak_result['peaks']),
                'peak_hr_bpm': peak_result['peak_hr_bpm'],
                'fft_hr_bpm': fft_hr,
                'hr_diff_bpm': hr_diff,
                'is_valid': is_valid,
                'global_peak_times': global_peak_times,
                'global_peak_indices': global_peak_indices,
                'ibi_ms': peak_result['ibi_ms'],
                'validation_status': 'valid' if is_valid else 'invalid'
            }
            
            window_results.append(window_result)
        
        # 统计有效窗口
        valid_windows = [w for w in window_results if w['is_valid']]
        valid_ratio = len(valid_windows) / len(window_results) if window_results else 0
        
        print(f"    ✅ 有效窗口: {len(valid_windows)}/{len(window_results)} ({valid_ratio*100:.1f}%)")
        
        return window_results, valid_windows
    
    def match_peaks_across_sensors_windowed(self, sensor_window_results):
        """在有效窗口内匹配不同传感器的峰值 - 改进：独立匹配每对传感器"""
        try:
            # 找到所有传感器
            all_sensors = list(sensor_window_results.keys())
            if len(all_sensors) < 2:
                return {}
            
            # 按窗口ID组织数据
            window_sensor_data = {}
            for sensor, windows in sensor_window_results.items():
                for window in windows:
                    if window['is_valid']:
                        window_id = window['window_id']
                        if window_id not in window_sensor_data:
                            window_sensor_data[window_id] = {}
                        window_sensor_data[window_id][sensor] = window
            
            # 找到至少有2个传感器有效的窗口
            valid_multi_sensor_windows = {
                wid: data for wid, data in window_sensor_data.items() 
                if len(data) >= 2
            }
            
            print(f"📊 多传感器有效窗口: {len(valid_multi_sensor_windows)}")
            
            ptt_data = []  # 直接计算PTT
            ptt_id = 0
            
            for window_id, sensor_data in valid_multi_sensor_windows.items():
                # 获取当前窗口的所有传感器
                current_sensors = list(sensor_data.keys())
                
                # 为每对传感器独立匹配
                for i in range(len(current_sensors)):
                    for j in range(i+1, len(current_sensors)):
                        sensor1 = current_sensors[i]
                        sensor2 = current_sensors[j]
                        peaks1 = sensor_data[sensor1]['global_peak_times']
                        peaks2 = sensor_data[sensor2]['global_peak_times']
                        
                        if len(peaks1) == 0 or len(peaks2) == 0:
                            continue
                        
                        # 为peaks1的每个峰值找peaks2中最近的匹配
                        for t1 in peaks1:
                            time_diffs = np.abs(peaks2 - t1)
                            closest_idx = np.argmin(time_diffs)
                            if time_diffs[closest_idx] <= 0.2:  # 200ms窗口
                                t2 = peaks2[closest_idx]
                                ptt_ms = (t2 - t1) * 1000
                                ptt_data.append({
                                    'ptt_id': ptt_id,
                                    'window_id': window_id,
                                    'sensor_pair': f"{sensor1}-{sensor2}",
                                    'sensor_names': f"{self.sensor_mapping[sensor1]}→{self.sensor_mapping[sensor2]}",
                                    'ptt_ms': ptt_ms,
                                    f'{sensor1}_time_s': t1,
                                    f'{sensor2}_time_s': t2,
                                    'window_start_s': sensor_data[sensor1]['start_time_s'],
                                    'window_end_s': sensor_data[sensor1]['end_time_s']
                                })
                                ptt_id += 1
            
            print(f"💓 计算的PTT数量: {len(ptt_data)}")
            
            return {
                'ptt_data': ptt_data,
                'valid_windows': valid_multi_sensor_windows,
                'total_valid_windows': len(valid_multi_sensor_windows)
            }
            
        except Exception as e:
            print(f"⚠️  窗口化峰值匹配失败: {e}")
            return {}
    
    def process_experiment(self, exp_id):
        """处理单个实验的窗口化分析"""
        print(f"\n🔍 开始处理实验 {exp_id} - 窗口化时频域验证")
        
        exp_output_dir = os.path.join(self.output_dir, f"exp_{exp_id}")
        os.makedirs(exp_output_dir, exist_ok=True)
        self.current_exp_output_dir = exp_output_dir
        
        sensor_signals = {}
        sensor_window_results = {}
        all_valid_windows = {}
        
        # 读取和分析每个传感器
        for sensor in self.sensors:
            try:
                file_path = os.path.join(self.data_path, f"{exp_id}_hub_{sensor}_aligned.csv")
                if not os.path.exists(file_path):
                    print(f"❌ 文件不存在: {file_path}")
                    continue
                
                df = pd.read_csv(file_path)
                if len(df.columns) < 3:
                    print(f"⚠️  {sensor}: 数据列不足")
                    continue
                
                ir_signal = df.iloc[:, 2].values  # IR通道
                
                # 动态计算当前传感器的采样率
                if 'timestamp' in df.columns:
                    current_fs = self.calculate_sampling_rate(df['timestamp'].values)
                    print(f"📊 {sensor} 计算采样率: {current_fs:.1f}Hz")
                else:
                    current_fs = self.default_fs
                    print(f"⚠️ {sensor} 缺少时间戳信息，使用默认采样率: {current_fs}Hz")
                
                sensor_signals[sensor] = {
                    'signal': ir_signal,
                    'dataframe': df,
                    'duration_s': len(ir_signal) / current_fs,
                    'fs': current_fs
                }
                
                # 窗口化分析
                window_results, valid_windows = self.analyze_sensor_windowed(
                    ir_signal, self.sensor_mapping[sensor], current_fs
                )
                
                sensor_window_results[sensor] = window_results
                all_valid_windows[sensor] = valid_windows
                
                print(f"  📊 {sensor}({self.sensor_mapping[sensor]}): "
                      f"信号长度{len(ir_signal)/current_fs:.1f}s, "
                      f"窗口{len(window_results)}个, "
                      f"有效{len(valid_windows)}个")
                
            except Exception as e:
                print(f"❌ 处理 {sensor} 失败: {e}")
                continue
        
        # 跨传感器峰值匹配
        matched_results = self.match_peaks_across_sensors_windowed(all_valid_windows)
        
        # 保存结果
        self.save_windowed_results(exp_id, sensor_window_results, matched_results, sensor_signals)
        
        return sensor_window_results, matched_results
    
    def save_windowed_results(self, exp_id, sensor_window_results, matched_results, sensor_signals):
        """保存窗口化分析结果"""
        try:
            # 1. 窗口验证汇总
            window_summary = []
            for sensor, windows in sensor_window_results.items():
                for window in windows:
                    mean_ibi = np.mean(window['ibi_ms']) if len(window['ibi_ms']) > 0 else np.nan  # 新增：计算窗口平均IBI
                    window_summary.append({
                        'exp_id': exp_id,
                        'sensor': sensor,
                        'sensor_name': self.sensor_mapping[sensor],
                        'window_id': window['window_id'],
                        'start_time_s': window['start_time_s'],
                        'end_time_s': window['end_time_s'],
                        'duration_s': window['duration_s'],
                        'peak_count': window['peak_count'],
                        'peak_hr_bpm': window['peak_hr_bpm'],
                        'fft_hr_bpm': window['fft_hr_bpm'],
                        'hr_diff_bpm': window['hr_diff_bpm'],
                        'is_valid': window['is_valid'],
                        'validation_status': window['validation_status'],
                        'mean_ibi_ms': mean_ibi  # 新增
                    })
            
            if window_summary:
                summary_df = pd.DataFrame(window_summary)
                summary_file = os.path.join(self.current_exp_output_dir, f"window_validation_exp_{exp_id}.csv")
                summary_df.to_csv(summary_file, index=False)
                print(f"💾 保存窗口验证汇总: {summary_file}")
            
            # 2. 有效窗口的峰值详细信息
            valid_peaks = []
            for sensor, windows in sensor_window_results.items():
                for window in windows:
                    if window['is_valid']:
                        mean_ibi = np.mean(window['ibi_ms']) if len(window['ibi_ms']) > 0 else np.nan  # 新增
                        for i, (peak_time, peak_idx) in enumerate(zip(
                            window['global_peak_times'], window['global_peak_indices']
                        )):
                            valid_peaks.append({
                                'exp_id': exp_id,
                                'sensor': sensor,
                                'sensor_name': self.sensor_mapping[sensor],
                                'window_id': window['window_id'],
                                'peak_number_in_window': i + 1,
                                'global_peak_time_s': peak_time,
                                'global_peak_index': int(peak_idx),
                                'window_peak_hr_bpm': window['peak_hr_bpm'],
                                'window_fft_hr_bpm': window['fft_hr_bpm'],
                                'window_hr_diff_bpm': window['hr_diff_bpm'],
                                'mean_ibi_ms': mean_ibi  # 新增：窗口平均IBI
                            })
            
            if valid_peaks:
                peaks_df = pd.DataFrame(valid_peaks)
                peaks_file = os.path.join(self.current_exp_output_dir, f"valid_peaks_exp_{exp_id}.csv")
                peaks_df.to_csv(peaks_file, index=False)
                print(f"💾 保存有效峰值: {peaks_file}")
            
            # 3. 匹配的心跳和PTT计算
            if matched_results and 'ptt_data' in matched_results:
                ptt_data = matched_results['ptt_data']
                
                if ptt_data:
                    ptt_df = pd.DataFrame(ptt_data)
                    ptt_file = os.path.join(self.current_exp_output_dir, f"ptt_windowed_exp_{exp_id}.csv")
                    ptt_df.to_csv(ptt_file, index=False)
                    print(f"💾 保存窗口化PTT: {ptt_file}")
                    
                    # PTT统计汇总
                    ptt_summary = ptt_df.groupby('sensor_pair').agg({
                        'ptt_ms': ['count', 'mean', 'std', 'min', 'max', 'median'],
                        'window_id': 'nunique'
                    }).round(2)
                    ptt_summary.columns = ['count', 'mean_ptt_ms', 'std_ptt_ms', 'min_ptt_ms', 'max_ptt_ms', 'median_ptt_ms', 'num_windows']
                    ptt_summary = ptt_summary.reset_index()
                    
                    ptt_summary_file = os.path.join(self.current_exp_output_dir, f"ptt_summary_windowed_exp_{exp_id}.csv")
                    ptt_summary.to_csv(ptt_summary_file, index=False)
                    print(f"💾 保存PTT统计汇总: {ptt_summary_file}")
                    print(f"📊 窗口化PTT统计:")
                    for _, row in ptt_summary.iterrows():
                        print(f"  {row['sensor_pair']}: {row['mean_ptt_ms']:.1f}±{row['std_ptt_ms']:.1f}ms "
                              f"({row['count']}心跳, {row['num_windows']}窗口)")
            
            # 4. 创建可视化
            self.create_windowed_visualizations(exp_id, sensor_window_results, sensor_signals)
            
        except Exception as e:
            print(f"❌ 保存窗口化结果失败: {e}")
    
    def calculate_windowed_ptt(self, heartbeat_df, exp_id):
        """计算窗口化PTT分析 - 简化：当前版本无需此函数，或直接保存ptt_data"""
        try:
            # 由于匹配中已计算PTT，直接保存
            if 'ptt_data' in matched_results:
                ptt_data = matched_results['ptt_data']
                if ptt_data:
                    ptt_df = pd.DataFrame(ptt_data)
                    ptt_file = os.path.join(self.current_exp_output_dir, f"ptt_windowed_exp_{exp_id}.csv")
                    ptt_df.to_csv(ptt_file, index=False)
                    print(f"💾 保存窗口化PTT: {ptt_file}")
                    
                    # PTT统计汇总
                    ptt_summary = ptt_df.groupby('sensor_pair').agg({
                        'ptt_ms': ['count', 'mean', 'std', 'min', 'max', 'median'],
                        'window_id': 'nunique'
                    }).round(2)
                    ptt_summary.columns = ['count', 'mean_ptt_ms', 'std_ptt_ms', 'min_ptt_ms', 'max_ptt_ms', 'median_ptt_ms', 'num_windows']
                    ptt_summary = ptt_summary.reset_index()
                    
                    ptt_summary_file = os.path.join(self.current_exp_output_dir, f"ptt_summary_windowed_exp_{exp_id}.csv")
                    ptt_summary.to_csv(ptt_summary_file, index=False)
                    print(f"💾 保存PTT统计汇总: {ptt_summary_file}")
                    print(f"📊 窗口化PTT统计:")
                    for _, row in ptt_summary.iterrows():
                        print(f"  {row['sensor_pair']}: {row['mean_ptt_ms']:.1f}±{row['std_ptt_ms']:.1f}ms "
                              f"({row['count']}心跳, {row['num_windows']}窗口)")
            
        except Exception as e:
            print(f"⚠️  窗口化PTT计算失败: {e}")
    
    def create_windowed_visualizations(self, exp_id, sensor_window_results, sensor_signals):
        """创建窗口化可视化"""
        try:
            # 1. 窗口验证状态图
            fig, axes = plt.subplots(len(self.sensors), 1, figsize=(20, 4*len(self.sensors)), sharex=True)
            if len(self.sensors) == 1:
                axes = [axes]
            
            colors = ['red', 'blue', 'green', 'orange']
            
            for idx, sensor in enumerate(self.sensors):
                ax = axes[idx]
                
                if sensor in sensor_signals and sensor in sensor_window_results:
                    signal_data = sensor_signals[sensor]
                    time = np.arange(len(signal_data['signal'])) / signal_data['fs']
                    
                    # 绘制信号
                    filtered_signal = self.bandpass_filter(signal_data['signal'], fs=signal_data['fs'])
                    ax.plot(time, filtered_signal, color=colors[idx % len(colors)], 
                           linewidth=0.8, alpha=0.6, label=f'{self.sensor_mapping[sensor]} IR')
                    
                    # 绘制窗口状态
                    windows = sensor_window_results[sensor]
                    for window in windows:
                        start_time = window['start_time_s']
                        end_time = window['end_time_s']
                        
                        if window['is_valid']:
                            # 有效窗口 - 绿色背景
                            ax.axvspan(start_time, end_time, alpha=0.2, color='green')
                            
                            # 标记峰值
                            if len(window['global_peak_times']) > 0:
                                peak_values = []
                                for peak_idx in window['global_peak_indices']:
                                    if 0 <= peak_idx < len(filtered_signal):
                                        peak_values.append(filtered_signal[peak_idx])
                                
                                if peak_values:
                                    ax.scatter(window['global_peak_times'], peak_values, 
                                             color='red', s=30, zorder=5)
                        else:
                            # 无效窗口 - 红色背景
                            ax.axvspan(start_time, end_time, alpha=0.1, color='red')
                    
                    ax.set_title(f'{self.sensor_mapping[sensor]} IR - windowed_validation '
                               f'(green=valid, red=invalid)', fontsize=12, fontweight='bold')
                    ax.set_ylabel('Signal', fontsize=10)
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                else:
                    ax.text(0.5, 0.5, f'{self.sensor_mapping[sensor]}: No Data', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{self.sensor_mapping[sensor]} IR - No Data')
            
            axes[-1].set_xlabel('Time (seconds)', fontsize=12)
            plt.suptitle(f'Experiment {exp_id} - windowed_validation_result', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            plot_file = os.path.join(self.current_exp_output_dir, f"windowed_validation_exp_{exp_id}.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"📊 保存窗口验证图: {plot_file}")
            
            # 2. 时频域心率对比图
            self.create_hr_comparison_plot(exp_id, sensor_window_results)
            
        except Exception as e:
            print(f"❌ 窗口化可视化创建失败: {e}")
    
    def create_hr_comparison_plot(self, exp_id, sensor_window_results):
        """创建时频域心率对比图"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()
            
            for idx, sensor in enumerate(self.sensors[:4]):
                ax = axes[idx]
                
                if sensor in sensor_window_results:
                    windows = sensor_window_results[sensor]
                    valid_windows = [w for w in windows if w['is_valid']]
                    invalid_windows = [w for w in windows if not w['is_valid']]
                    
                    # 绘制有效窗口
                    if valid_windows:
                        peak_hrs = [w['peak_hr_bpm'] for w in valid_windows]
                        fft_hrs = [w['fft_hr_bpm'] for w in valid_windows]
                        ax.scatter(peak_hrs, fft_hrs, color='green', alpha=0.7, s=50, 
                                 label=f'Valid ({len(valid_windows)})')
                    
                    # 绘制无效窗口
                    if invalid_windows:
                        peak_hrs_invalid = [w['peak_hr_bpm'] for w in invalid_windows if w['peak_hr_bpm'] > 0]
                        fft_hrs_invalid = [w['fft_hr_bpm'] for w in invalid_windows if w['fft_hr_bpm'] > 0]
                        if peak_hrs_invalid and fft_hrs_invalid:
                            ax.scatter(peak_hrs_invalid, fft_hrs_invalid, color='red', alpha=0.5, s=30,
                                     label=f'Invalid ({len(invalid_windows)})')
                    
                    # 绘制理想线和容忍带
                    hr_range = [self.min_hr, self.max_hr]
                    ax.plot(hr_range, hr_range, 'k--', alpha=0.5, label='Perfect Match')
                    
                    # ±xBPM容忍带
                    ax.fill_between(hr_range, 
                                   [h - self.hr_tolerance_bpm for h in hr_range],
                                   [h + self.hr_tolerance_bpm for h in hr_range],
                                   alpha=0.2, color='gray', label='±5BPM Tolerance')
                    
                    ax.set_xlabel('Peak Detection HR (BPM)')
                    ax.set_ylabel('FFT HR (BPM)')
                    ax.set_title(f'{self.sensor_mapping[sensor]} - hr_validation_result')
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    ax.set_xlim(self.min_hr, self.max_hr)
                    ax.set_ylim(self.min_hr, self.max_hr)
                else:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{self.sensor_mapping[sensor]} - No Data')
            
            plt.suptitle(f'Experiment {exp_id} - hr_validation_result', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            hr_plot_file = os.path.join(self.current_exp_output_dir, f"hr_validation_exp_{exp_id}.png")
            plt.savefig(hr_plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"📊 保存心率对比图: {hr_plot_file}")
            
        except Exception as e:
            print(f"❌ 心率对比图创建失败: {e}")
    
    def run_windowed_analysis(self, experiment_list=None):
        """运行窗口化时频域验证分析"""
        if experiment_list is None:
            experiment_list = [f.split('_')[0] for f in os.listdir(self.data_path) 
                             if f.endswith('_hub_sensor2_aligned.csv')]
            experiment_list = sorted(list(set(experiment_list)))
        
        print(f"\n🔬 开始窗口化时频域验证PTT分析（密集滑窗版）")
        print(f"📋 实验列表: {experiment_list}")
        print(f"🎯 验证策略:")
        print(f"   - {self.window_duration}s窗口, {self.window_step}s滑窗步长（密集覆盖）")
        print(f"   - 时域峰值检测 vs 频域FFT心率")
        print(f"   - 心率差异<{self.hr_tolerance_bpm}BPM才认为窗口有效（放宽容忍度）")
        print(f"   - 只在有效窗口内计算PTT")
        print(f"   - 详细的窗口质量报告")
        
        all_results = {}
        
        for exp_id in tqdm(experiment_list, desc="处理实验"):
            try:
                sensor_results, matched_results = self.process_experiment(exp_id)
                all_results[exp_id] = {
                    'sensor_window_results': sensor_results,
                    'matched_results': matched_results
                }
            except Exception as e:
                print(f"❌ 实验 {exp_id} 处理失败: {e}")
                continue
        
        print(f"\n✅ 密集滑窗时频域验证PTT分析完成！")
        print(f"📁 结果保存在: {self.output_dir}/exp_X")
        print(f"\n📊 输出文件说明:")
        print(f"   1. window_validation_exp_X.csv - 窗口验证详情")
        print(f"   2. valid_peaks_exp_X.csv - 有效窗口的峰值")
        print(f"   3. matched_heartbeats_windowed_exp_X.csv - 窗口化匹配心跳")
        print(f"   4. ptt_windowed_exp_X.csv - 窗口化PTT时间序列")
        print(f"   5. ptt_summary_windowed_exp_X.csv - PTT统计汇总")
        print(f"   6. windowed_validation_exp_X.png - 窗口验证状态图")
        print(f"   7. hr_validation_exp_X.png - 时频域心率对比图")
        print(f"\n🎯 密集滑窗验证完成！更多窗口，更高精度的PTT质量控制！")
        
        return all_results

def main():
    """主函数"""
    print("🩺 密集滑窗时频域验证PTT峰值检测器（优化版）")
    print("=" * 70)
    print("📖 密集滑窗优化实现:")
    print("   • 30秒窗口，5秒密集滑窗步长")
    print("   • 时域峰值检测 vs 频域FFT心率验证")
    print("   • 心率差异<5BPM才认为窗口有效（放宽容忍度）")
    print("   • 只在有效窗口内计算PTT")
    print("   • 更多窗口，更细粒度的质量控制")
    print("=" * 70)
    
    detector = IRWindowedPTTPeakDetector()
    results = detector.run_windowed_analysis()
    
    print("\n🎯 分析完成，师兄建议已实现:")
    print("1. 检查window_validation_exp_X.csv了解每个窗口的验证状态")
    print("2. 查看hr_validation_exp_X.png确认时频域心率一致性")
    print("3. 使用ptt_windowed_exp_X.csv进行高质量PTT建模")
    print("4. 只有通过验证的窗口才参与PTT计算，确保准确性！")

if __name__ == "__main__":
    main() 