#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🩺 IR通道专门的PTT峰值检测器 - 优化输出与PTT准备（批量处理版，含傅里叶心率分析）

基于师兄建议的改进：
1. ✅ 专注IR通道峰值检测（信号质量最佳）
2. ✅ 使用neurokit2计算IBI并验证
3. ✅ 同一心跳区间的峰值匹配
4. ✅ 输出峰值、IBI和PTT预览CSV，方便后续处理
5. ✅ 批量处理所有实验，存储到expX子文件夹
6. ✅ 新增傅里叶心率分析，严格照抄data_processor.py的get_hr和plot_psd_analysis

核心原理：
- PTT使用峰值时间差计算
- IR通道信号最稳定
- IBI验证确保峰值准确
- 傅里叶分析验证心率一致性（与data_processor.py一致）
- 4传感器生成6个PTT组合
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks, welch
import warnings

# 尝试导入专业库，如果没有安装就使用备选方案
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

class IRBasedPTTPeakDetector:
    """基于IR通道的PTT峰值检测器 - 支持多种专业库及傅里叶分析"""
    
    def __init__(self, data_path="output/csv_output", method="auto"):
        self.data_path = data_path
        self.output_dir = "ptt_output"
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
        
        # 选择峰值检测方法
        self.detection_method = self._select_method(method)
        print(f"🔧 峰值检测方法: {self.detection_method}")
        
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
            # 检查输入数据
            if len(data) == 0:
                return data
                
            # 处理NaN值
            data_array = np.array(data, dtype=float)
            nan_count = np.isnan(data_array).sum()
            if nan_count > 0:
                # 使用线性插值填充NaN值
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
            
            # 检查滤波结果
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

    def get_hr(self, y, sr=100, min=50, max=200):
        """计算心率（直接照抄data_processor.py的get_hr）"""
        try:
            p, q = welch(y, sr, nfft=1e5/sr, nperseg=np.min((len(y)-1, 256)))
            return p[(p>min/60)&(p<max/60)][np.argmax(q[(p>min/60)&(p<max/60)])]*60
        except Exception as e:
            print(f"⚠️  心率计算失败: {e}")
            return 0.0
    
    def detect_peaks_neurokit2(self, signal, fs=100):
        """使用NeuroKit2进行专业峰值检测"""
        try:
            filtered_signal = self.bandpass_filter(signal, self.filter_lowcut, self.filter_highcut, fs)
            peaks_dict, info_dict = nk.ppg_peaks(filtered_signal, sampling_rate=fs, method="elgendi")
            peak_indices = np.where(peaks_dict['PPG_Peaks'] == 1)[0]
            
            if len(peak_indices) < 2:
                return self._empty_result(filtered_signal, signal)
            
            peak_times = peak_indices / fs
            ibi_ms = np.diff(peak_times) * 1000
            
            try:
                hrv_dict = nk.hrv_time(ibi_ms, sampling_rate=1000, show=False)
                hrv_metrics = hrv_dict.to_dict('records')[0] if not hrv_dict.empty else {}
            except:
                hrv_metrics = {}
            
            return self._process_peak_results(peak_indices, peak_times, ibi_ms, filtered_signal, signal, hrv_metrics)
            
        except Exception as e:
            print(f"⚠️  NeuroKit2峰值检测失败: {e}")
            return self._empty_result(signal, signal, 'error')
    
    def detect_peaks_heartpy(self, signal, fs=100):
        """使用HeartPy进行峰值检测"""
        try:
            filtered_signal = self.bandpass_filter(signal, self.filter_lowcut, self.filter_highcut, fs)
            working_data, measures = hp.process(filtered_signal, sample_rate=fs)
            peak_indices = working_data['peaklist']
            
            if len(peak_indices) < 2:
                return self._empty_result(filtered_signal, signal)
            
            peak_times = np.array(peak_indices) / fs
            ibi_ms = np.diff(peak_times) * 1000
            
            hrv_metrics = {
                'rmssd': measures.get('rmssd', 0),
                'pnn50': measures.get('pnn50', 0),
                'mean_hr': measures.get('bpm', 0)
            }
            
            return self._process_peak_results(peak_indices, peak_times, ibi_ms, filtered_signal, signal, hrv_metrics)
            
        except Exception as e:
            print(f"⚠️  HeartPy峰值检测失败: {e}")
            return self._empty_result(signal, signal, 'error')
    
    def detect_peaks_scipy_advanced(self, signal, fs=100):
        """改进的scipy峰值检测"""
        try:
            filtered_signal = self.bandpass_filter(signal, self.filter_lowcut, self.filter_highcut, fs)
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
                if len(peak_indices) >= 5:
                    break
            
            if len(peak_indices) < 2:
                return self._empty_result(filtered_signal, signal)
            
            peak_times = peak_indices / fs
            ibi_ms = np.diff(peak_times) * 1000
            hrv_metrics = self._calculate_hrv_metrics(ibi_ms)
            
            return self._process_peak_results(peak_indices, peak_times, ibi_ms, filtered_signal, signal, hrv_metrics)
            
        except Exception as e:
            print(f"⚠️  Scipy峰值检测失败: {e}")
            return self._empty_result(signal, signal, 'error')
    
    def _calculate_hrv_metrics(self, ibi_ms):
        """计算HRV指标"""
        if len(ibi_ms) < 2:
            return {}
        
        try:
            diff_ibi = np.diff(ibi_ms)
            rmssd = np.sqrt(np.mean(diff_ibi**2))
            pnn50 = np.sum(np.abs(diff_ibi) > 50) / len(diff_ibi) * 100
            sdnn = np.std(ibi_ms)
            
            return {
                'rmssd': rmssd,
                'pnn50': pnn50,
                'sdnn': sdnn
            }
        except:
            return {}
    
    def _process_peak_results(self, peak_indices, peak_times, ibi_ms, filtered_signal, original_signal, hrv_metrics=None):
        """处理峰值检测结果"""
        valid_ibi_mask = (ibi_ms >= 300) & (ibi_ms <= 1200)
        valid_ratio = np.sum(valid_ibi_mask) / len(ibi_ms) if len(ibi_ms) > 0 else 0
        
        if valid_ratio >= 0.7:
            quality = 'excellent'
        elif valid_ratio >= 0.5:
            quality = 'good'
        elif valid_ratio >= 0.3:
            quality = 'fair'
        else:
            quality = 'poor'
        
        return {
            'peaks': peak_indices,
            'ibi_ms': ibi_ms,
            'filtered_signal': filtered_signal,
            'original_signal': original_signal,
            'peak_times': peak_times,
            'peak_count': len(peak_indices),
            'quality': quality,
            'valid_ibi_ratio': valid_ratio,
            'hrv_metrics': hrv_metrics or {}
        }
    
    def _empty_result(self, filtered_signal, original_signal, quality='poor'):
        """返回空结果"""
        return {
            'peaks': np.array([]),
            'ibi_ms': np.array([]),
            'filtered_signal': filtered_signal,
            'original_signal': original_signal,
            'peak_times': np.array([]),
            'peak_count': 0,
            'quality': quality,
            'valid_ibi_ratio': 0,
            'hrv_metrics': {}
        }
    
    def detect_peaks_robust(self, signal, fs=100):
        """统一的峰值检测入口，自动选择最佳方法"""
        if self.detection_method == "neurokit2":
            return self.detect_peaks_neurokit2(signal, fs)
        elif self.detection_method == "heartpy":
            return self.detect_peaks_heartpy(signal, fs)
        else:
            return self.detect_peaks_scipy_advanced(signal, fs)
    
    def calculate_heart_rate_stats(self, ibi_ms):
        """计算心率统计信息"""
        if len(ibi_ms) == 0:
            return {
                'hr_mean': 0,
                'hr_std': 0,
                'ibi_mean': 0,
                'ibi_std': 0,
                'rmssd': 0,
                'pnn50': 0
            }
        
        hr_bpm = 60000 / ibi_ms
        ibi_mean = np.mean(ibi_ms)
        ibi_std = np.std(ibi_ms)
        hr_mean = np.mean(hr_bpm)
        hr_std = np.std(hr_bpm)
        
        if len(ibi_ms) > 1:
            diff_ibi = np.diff(ibi_ms)
            rmssd = np.sqrt(np.mean(diff_ibi**2))
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
            valid_sensors = [s for s in self.sensors 
                           if s in sensor_results 
                           and sensor_results[s]['peak_count'] > 5
                           and sensor_results[s]['quality'] in ['excellent', 'good']]
            
            if len(valid_sensors) < 2:
                print("⚠️  高质量传感器数量不足，尝试放宽标准")
                valid_sensors = [s for s in self.sensors 
                               if s in sensor_results 
                               and sensor_results[s]['peak_count'] > 3
                               and sensor_results[s]['quality'] != 'error']
            
            if len(valid_sensors) < 2:
                print("⚠️  有效传感器数量不足，无法进行峰值匹配")
                return {}
            
            print(f"📍 有效传感器: {valid_sensors}")
            
            reference_sensor = max(valid_sensors, 
                                 key=lambda s: sensor_results[s]['valid_ibi_ratio'])
            reference_peaks = sensor_results[reference_sensor]['peak_times']
            
            print(f"📍 参考传感器: {reference_sensor} (质量: {sensor_results[reference_sensor]['quality']})")
            
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
            
            for sensor in valid_sensors:
                if sensor == reference_sensor:
                    continue
                    
                sensor_peaks = sensor_results[sensor]['peak_times']
                
                for peak_time in sensor_peaks:
                    best_window = None
                    min_distance = float('inf')
                    
                    for window in heartbeat_windows:
                        if window['window_start'] <= peak_time <= window['window_end']:
                            distance = abs(peak_time - window['reference_time'])
                            if distance < min_distance:
                                min_distance = distance
                                best_window = window
                    
                    if best_window is not None and min_distance < 0.2:
                        best_window['sensor_peaks'][sensor] = peak_time
            
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
        
        exp_output_dir = os.path.join(self.output_dir, f"exp_{exp_id}")
        os.makedirs(exp_output_dir, exist_ok=True)
        self.current_exp_output_dir = exp_output_dir
        
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
                    
                    # 动态计算当前传感器的采样率
                    if 'timestamp' in df.columns:
                        current_fs = self.calculate_sampling_rate(df['timestamp'].values)
                        print(f"📊 {sensor} 计算采样率: {current_fs:.1f}Hz")
                    else:
                        current_fs = self.default_fs
                        print(f"⚠️ {sensor} 缺少时间戳信息，使用默认采样率: {current_fs}Hz")
                    
                    # 稳健的峰值检测
                    peak_result = self.detect_peaks_robust(ir_signal, current_fs)
                    
                    # 计算心率统计
                    hr_stats = self.calculate_heart_rate_stats(peak_result['ibi_ms'])
                    
                    # 计算傅里叶心率（照抄get_hr）
                    fft_hr = self.get_hr(ir_signal, sr=current_fs, min=50, max=200)
                    fft_freq = fft_hr / 60.0  # 转换为Hz
                    
                    # 合并结果
                    peak_result.update({
                        'sensor': sensor,
                        'sensor_name': self.sensor_mapping[sensor],
                        **hr_stats,
                        'fft_hr_bpm': fft_hr,
                        'fft_peak_freq_hz': fft_freq
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
                              f"FFT HR={fft_hr:.1f}BPM, "
                              f"IBI={ibi_range}, "
                              f"质量={peak_result['quality']}({peak_result.get('valid_ibi_ratio', 0)*100:.0f}%)")
                    else:
                        print(f"  {quality_symbol} {sensor}({self.sensor_mapping[sensor]}): 未检测到有效峰值")
                        
                else:
                    print(f"⚠️  {sensor}: 数据列不足")
                    
            except Exception as e:
                print(f"❌ 处理 {sensor} 失败: {e}")
                continue
        
        matched_results = self.match_peaks_across_sensors(sensor_results)
        self.save_results(exp_id, sensor_results, matched_results, all_signals)
        
        return sensor_results, matched_results
    
    def save_results(self, exp_id, sensor_results, matched_results, all_signals):
        """保存检测结果 - 5个核心CSV文件，包含傅里叶心率"""
        try:
            # 1. 传感器质量汇总
            sensor_summary = []
            for sensor in sensor_results:
                result = sensor_results[sensor]
                
                # 动态计算当前传感器的采样率
                if 'timestamp' in all_signals[sensor].columns:
                    current_fs = self.calculate_sampling_rate(all_signals[sensor]['timestamp'].values)
                else:
                    current_fs = self.default_fs
                
                signal_duration = len(all_signals[sensor].iloc[:, 2]) / current_fs
                
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
                    'signal_duration_s': signal_duration,
                    'fft_hr_bpm': result['fft_hr_bpm'],
                    'fft_peak_freq_hz': result['fft_peak_freq_hz']
                })
            
            if sensor_summary:
                summary_df = pd.DataFrame(sensor_summary)
                summary_file = os.path.join(self.current_exp_output_dir, f"sensor_summary_exp_{exp_id}.csv")
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
                peaks_file = os.path.join(self.current_exp_output_dir, f"all_peaks_exp_{exp_id}.csv")
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
                ibi_file = os.path.join(self.current_exp_output_dir, f"all_ibi_exp_{exp_id}.csv")
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
                    heartbeat_file = os.path.join(self.current_exp_output_dir, f"matched_heartbeats_exp_{exp_id}.csv")
                    heartbeat_df.to_csv(heartbeat_file, index=False)
                    print(f"💾 保存匹配心跳: {heartbeat_file}")
                    
                    self.calculate_ptt_analysis(heartbeat_df, exp_id, matched_results['valid_sensors'])
            
            # 生成可视化
            self.create_visualizations(exp_id, sensor_results, matched_results, all_signals)
            
        except Exception as e:
            print(f"❌ 保存结果失败: {e}")
    
    def calculate_ptt_analysis(self, heartbeat_df, exp_id, valid_sensors):
        """计算PTT分析 - 矩阵汇总 + 时间序列"""
        try:
            sensor_combinations = []
            for i in range(len(valid_sensors)):
                for j in range(i+1, len(valid_sensors)):
                    sensor_combinations.append((valid_sensors[i], valid_sensors[j]))
            
            print(f"\n📊 PTT分析 ({len(sensor_combinations)}个传感器组合):")
            
            ptt_summary = []
            ptt_timeseries_all = []
            
            for sensor1, sensor2 in sensor_combinations:
                col1 = f'{sensor1}_peak_time_s'
                col2 = f'{sensor2}_peak_time_s'
                
                if col1 in heartbeat_df.columns and col2 in heartbeat_df.columns:
                    valid_data = heartbeat_df.dropna(subset=[col1, col2])
                    
                    if len(valid_data) > 0:
                        ptt_values = (valid_data[col2] - valid_data[col1]) * 1000
                        
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
            
            if ptt_summary:
                ptt_matrix_df = pd.DataFrame(ptt_summary)
                ptt_matrix_file = os.path.join(self.current_exp_output_dir, f"ptt_matrix_exp_{exp_id}.csv")
                ptt_matrix_df.to_csv(ptt_matrix_file, index=False)
                print(f"💾 保存PTT矩阵: {ptt_matrix_file}")
            
            if ptt_timeseries_all:
                ptt_timeseries_df = pd.DataFrame(ptt_timeseries_all)
                ptt_timeseries_file = os.path.join(self.current_exp_output_dir, f"ptt_timeseries_exp_{exp_id}.csv")
                ptt_timeseries_df.to_csv(ptt_timeseries_file, index=False)
                print(f"💾 保存PTT时间序列: {ptt_timeseries_file}")
                print(f"   📈 共{len(ptt_timeseries_all)}个PTT数据点，可用于血压建模")
            
        except Exception as e:
            print(f"⚠️  PTT分析失败: {e}")
    
    def create_visualizations(self, exp_id, sensor_results, matched_results, all_signals):
        """创建可视化 - IR信号峰值图 + PSD图（照抄plot_psd_analysis）"""
        try:
            # 1. IR信号和峰值图
            fig, axes = plt.subplots(len(self.sensors), 1, figsize=(16, 3*len(self.sensors)), sharex=True)
            if len(self.sensors) == 1:
                axes = [axes]
            
            colors = ['red', 'blue', 'green', 'orange']
            
            for idx, sensor in enumerate(self.sensors):
                ax = axes[idx]
                
                if sensor in all_signals and sensor in sensor_results:
                    result = sensor_results[sensor]
                    filtered_signal = result['filtered_signal']
                    peaks = result['peaks']
                    quality = result['quality']
                    fft_hr = result['fft_hr_bpm']
                    
                    # 动态计算当前传感器的采样率
                    if 'timestamp' in all_signals[sensor].columns:
                        current_fs = self.calculate_sampling_rate(all_signals[sensor]['timestamp'].values)
                    else:
                        current_fs = self.default_fs
                    
                    time = np.arange(len(filtered_signal)) / current_fs
                    
                    ax.plot(time[:len(filtered_signal)], filtered_signal, 
                           color=colors[idx % len(colors)], linewidth=1.5, alpha=0.8,
                           label=f'{self.sensor_mapping[sensor]} IR')
                    
                    if len(peaks) > 0:
                        peak_times = peaks / current_fs
                        ax.scatter(peak_times, filtered_signal[peaks], 
                                 color='red', s=40, zorder=5, alpha=0.9)
                        
                        for i, (pt, ps) in enumerate(zip(peak_times, filtered_signal[peaks])):
                            if i % 10 == 0:
                                ax.annotate(f'{i+1}', (pt, ps), xytext=(5, 5), 
                                          textcoords='offset points', fontsize=8)
                    
                    ax.set_title(f'{self.sensor_mapping[sensor]} IR - {quality} - HR: {sensor_results[sensor]["hr_mean"]:.1f} BPM (FFT: {fft_hr:.1f} BPM)', 
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
            
            plot_file = os.path.join(self.current_exp_output_dir, f"ir_peaks_exp_{exp_id}.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"📊 保存IR信号图: {plot_file}")
            
            # 2. PSD可视化（照抄data_processor.py的plot_psd_analysis）
            sensor_dfs = {}
            for sensor in self.sensors:
                if sensor in all_signals and sensor in sensor_results:
                    df = all_signals[sensor][['timestamp', 'ir']].copy()
                    sensor_dfs[sensor] = df
            
            if sensor_dfs:
                n_sensors = len(sensor_dfs)
                channels = ['ir']  # 仅处理IR通道
                fig, axes = plt.subplots(n_sensors, 1, figsize=(15, 4 * n_sensors))
                if n_sensors == 1:
                    axes = [axes]
                
                for i, (sensor, df) in enumerate(sensor_dfs.items()):
                    part = self.sensor_mapping[sensor]
                    ts = df['timestamp'].values
                    tsu = np.unique(ts)
                    ax = axes[i]
                    
                    if len(tsu) < 2:
                        ax.text(0.5, 0.5, '时间戳不足',
                                ha='center', va='center')
                        ax.set_title(f"{part}-ir")
                        continue
                    
                    dt = np.median(np.diff(tsu))
                    fs = 1.0 / dt
                    
                    col_idx = 1  # ir通道
                    if df.shape[1] <= col_idx:
                        ax.text(0.5, 0.5, 'No data',
                                ha='center', va='center', transform=ax.transAxes)
                        ax.set_title(f"{part}-ir")
                        continue
                    
                    y = df.iloc[:, col_idx].values
                    try:
                        p, q = welch(y, fs, nfft=int(1e5/fs), nperseg=min(len(y)-1, 256))
                        bpm = p * 60
                        mask = (bpm >= 30) & (bpm <= 180)
                        
                        ax.plot(bpm[mask], q[mask], linewidth=1.5, color='C0')
                        ax.set_title(f"{part}-ir")
                        ax.grid(True, alpha=0.3)
                        
                        if np.any(mask) and len(q[mask]) > 0:
                            peak_idx = np.argmax(q[mask])
                            peak_bpm = bpm[mask][peak_idx]
                            ax.axvline(peak_bpm, color='red', linestyle='--', alpha=0.5)
                            ax.text(0.98, 0.95, f'{peak_bpm:.1f} BPM',
                                    transform=ax.transAxes,
                                    ha='right', va='top',
                                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
                    except Exception as e:
                        ax.text(0.5, 0.5, f"PSD 失败\n{str(e)[:30]}",
                                ha='center', va='center', transform=ax.transAxes)
                    
                    ax.set_xlabel("Frequency (BPM)")
                    ax.set_ylabel(f"{part}\nPSD", rotation=0, labelpad=30)
                
                plt.suptitle(f"Power Spectral Density Analysis (Experiment {exp_id} - IR signals)", fontsize=16)
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                
                psd_file = os.path.join(self.current_exp_output_dir, f"psd_exp_{exp_id}.png")
                plt.savefig(psd_file, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"📊 保存PSD图: {psd_file}")
            
        except Exception as e:
            print(f"❌ 可视化创建失败: {e}")
    
    def run_analysis(self, experiment_list=None):
        """运行IR通道PTT峰值检测分析（批量处理）"""
        if experiment_list is None:
            experiment_list = [f.split('_')[0] for f in os.listdir(self.data_path) 
                             if f.endswith('_hub_sensor2_aligned.csv')]
            experiment_list = sorted(list(set(experiment_list)))
        
        print(f"\n🔬 开始IR通道PTT峰值检测分析（批量处理）")
        print(f"📋 实验列表: {experiment_list}")
        print(f"🎯 检测策略:")
        print(f"   - 专注IR通道（信号质量最佳）")
        print(f"   - 稳健峰值检测 + IBI质量控制")
        print(f"   - 傅里叶心率分析验证（照抄data_processor.py）")
        print(f"   - 心率范围: {self.min_hr}-{self.max_hr} BPM")
        print(f"   - 滤波范围: {self.filter_lowcut}-{self.filter_highcut} Hz")
        print(f"   - 输出5个标准CSV文件 + PSD图，按expX子文件夹存储")
        
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
        print(f"📁 结果保存在: {self.output_dir}/exp_X")
        print(f"\n📊 输出文件说明:")
        print(f"   1. sensor_summary_exp_X.csv - 传感器质量汇总（含傅里叶心率）")
        print(f"   2. all_peaks_exp_X.csv - 所有峰值详细信息")
        print(f"   3. all_ibi_exp_X.csv - 所有IBI详细信息")
        print(f"   4. ptt_matrix_exp_X.csv - PTT矩阵汇总")
        print(f"   5. ptt_timeseries_exp_X.csv - PTT时间序列（用于建模）")
        print(f"   6. psd_exp_X.png - 各传感器IR通道PSD图（与data_processor.py一致）")
        print(f"\n🎯 下一步: 使用ptt_timeseries_exp_X.csv进行血压建模，检查fft_hr_bpm验证心率一致性")
        
        return all_results

def main():
    """主函数"""
    print("🩺 IR通道专门的PTT峰值检测器（批量处理版，含傅里叶分析）")
    print("=" * 60)
    print("📖 优化特性:")
    print("   • 专注IR通道峰值检测")
    print("   • 稳健的IBI计算和质量控制")
    print("   • 傅里叶心率分析验证（照抄data_processor.py）")
    print("   • 智能心跳匹配")
    print("   • 标准化CSV输出便于建模")
    print("   • 批量处理所有实验，存储到expX子文件夹")
    print("=" * 60)
    
    detector = IRBasedPTTPeakDetector()
    results = detector.run_analysis()
    
    print("\n🎯 分析完成，建议下一步:")
    print("1. 检查每个exp_X/sensor_summary_exp_X.csv了解传感器质量和傅里叶心率")
    print("2. 使用exp_X/ptt_timeseries_exp_X.csv进行血压建模")
    print("3. 验证PTT与血压的相关性 (a*PTT + b)")
    print("4. 检查exp_X/psd_exp_X.png确认傅里叶分析结果")

if __name__ == "__main__":
    main()