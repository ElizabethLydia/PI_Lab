#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🩺 Advanced PTT Peak Detection with Professional IBI Analysis
高级PTT峰值检测器 - 专业IBI分析版本

核心功能：
✅ 专业IBI库函数处理 (neurokit2)
✅ 按sensor分别存储峰值详细信息
✅ 全英文绘图显示  
✅ 生理心跳配对算法
✅ 配对sensor间PTT计算
✅ 完整的生理信号质量评估

基于师兄建议：
- 专注IR通道（信号质量最佳）
- 使用库函数处理IBI验证
- 心跳级别的sensor间配对
- 4传感器 → 6个PTT组合用于建模
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, filtfilt, butter
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# 尝试导入专业生理信号处理库
try:
    import neurokit2 as nk
    HAS_NEUROKIT = True
    print("📦 Successfully imported NeuroKit2 for professional IBI analysis")
except ImportError:
    HAS_NEUROKIT = False
    print("⚠️  NeuroKit2 not available, using basic IBI analysis")

# 设置英文字体
plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

class AdvancedPTTPeakDetector:
    """高级PTT峰值检测器 - 专业IBI分析版本"""
    
    def __init__(self, data_path="/root/PI_Lab/output/csv_output"):
        self.data_path = data_path
        self.output_dir = "/root/PI_Lab/ptt_output"
        self.sensors = ['sensor2', 'sensor3', 'sensor4', 'sensor5']  # 4个传感器
        self.sensor_mapping = {
            'sensor2': 'Nose', 'sensor3': 'Finger', 
            'sensor4': 'Wrist', 'sensor5': 'Ear'
        }
        
        # 生理参数
        self.fs = 100  # 采样率100Hz
        self.min_hr = 50   # 最小心率BPM
        self.max_hr = 150  # 最大心率BPM (放宽到150)
        
        # 滤波参数：专门针对PPG心率频段
        self.filter_lowcut = 0.5   
        self.filter_highcut = 4.0  # 扩展到4Hz
        self.filter_order = 4
        
        # 心跳配对参数
        self.heartbeat_window = 0.3  # 心跳配对窗口（秒）
        self.min_sensors_per_beat = 2  # 每个心跳最少需要的sensor数量
        
        os.makedirs(self.output_dir, exist_ok=True)
        
    def advanced_ppg_filter(self, signal, fs=100):
        """高级PPG滤波 - 多级滤波"""
        try:
            # 1. 带通滤波 (0.5-4Hz)
            nyquist = fs / 2
            low = self.filter_lowcut / nyquist
            high = self.filter_highcut / nyquist
            
            low = max(low, 0.01)
            high = min(high, 0.98)
            
            b, a = butter(self.filter_order, [low, high], btype='band')
            filtered = filtfilt(b, a, signal)
            
            # # 2. 移动平均去噪 (可选)
            # window_size = int(0.05 * fs)  # 50ms窗口
            # if window_size > 1:
            #     filtered = np.convolve(filtered, np.ones(window_size)/window_size, mode='same')
            
            return filtered
        except Exception as e:
            print(f"⚠️  Advanced filtering failed: {e}")
            return signal
    
    def professional_ibi_analysis(self, signal, peaks, fs=100):
        """专业IBI分析 - 使用NeuroKit2或自定义高级算法"""
        try:
            if HAS_NEUROKIT and len(peaks) > 3:
                # 使用NeuroKit2进行专业分析
                # 创建时间向量
                time_vector = np.arange(len(signal)) / fs
                
                # 创建R峰时间
                rpeaks_time = peaks / fs
                
                # 计算IBI和HRV
                ibi_ms = np.diff(rpeaks_time) * 1000  # 转换为毫秒
                
                if len(ibi_ms) > 1:
                    # 使用NeuroKit2计算HRV指标
                    hrv_dict = nk.hrv_time(ibi_ms, sampling_rate=1000, show=False)
                    
                    # 提取关键指标
                    mean_ibi = np.mean(ibi_ms)
                    rmssd = hrv_dict['HRV_RMSSD'].iloc[0] if 'HRV_RMSSD' in hrv_dict.columns else np.std(ibi_ms)
                    pnn50 = hrv_dict['HRV_pNN50'].iloc[0] if 'HRV_pNN50' in hrv_dict.columns else 0
                    
                    # 质量评估
                    cv = np.std(ibi_ms) / mean_ibi  # 变异系数
                    quality_score = self._calculate_signal_quality(ibi_ms, cv, rmssd)
                    
                    return {
                        'method': 'NeuroKit2',
                        'ibi_ms': ibi_ms,
                        'mean_ibi_ms': mean_ibi,
                        'heart_rate_bpm': 60000 / mean_ibi,
                        'rmssd': rmssd,
                        'pnn50': pnn50,
                        'cv': cv,
                        'quality_score': quality_score,
                        'is_valid': quality_score > 0.7,
                        'n_beats': len(ibi_ms)
                    }
                    
            # 回退到基础分析
            return self._basic_ibi_analysis(peaks, fs)
            
        except Exception as e:
            print(f"⚠️  Professional IBI analysis failed: {e}")
            return self._basic_ibi_analysis(peaks, fs)
    
    def _basic_ibi_analysis(self, peaks, fs=100):
        """基础IBI分析方法"""
        if len(peaks) < 2:
            return {
                'method': 'Basic',
                'ibi_ms': np.array([]),
                'mean_ibi_ms': 0,
                'heart_rate_bpm': 0,
                'rmssd': 0,
                'pnn50': 0,
                'cv': 0,
                'quality_score': 0,
                'is_valid': False,
                'n_beats': 0
            }
        
        # 计算IBI
        intervals = np.diff(peaks) / fs  # 转换为秒
        ibi_ms = intervals * 1000  # 转换为毫秒
        
        # 过滤生理范围内的IBI (300ms - 1200ms, 即50-200 BPM)
        valid_mask = (ibi_ms >= 300) & (ibi_ms <= 1200)
        valid_ibi = ibi_ms[valid_mask]
        
        if len(valid_ibi) == 0:
            return {
                'method': 'Basic',
                'ibi_ms': ibi_ms,
                'mean_ibi_ms': 0,
                'heart_rate_bpm': 0,
                'rmssd': 0,
                'pnn50': 0,
                'cv': 0,
                'quality_score': 0,
                'is_valid': False,
                'n_beats': len(ibi_ms)
            }
        
        # 计算指标
        mean_ibi = np.mean(valid_ibi)
        heart_rate = 60000 / mean_ibi
        rmssd = np.sqrt(np.mean(np.diff(valid_ibi)**2))
        pnn50 = np.sum(np.abs(np.diff(valid_ibi)) > 50) / len(valid_ibi) * 100
        cv = np.std(valid_ibi) / mean_ibi
        
        # 质量评估
        quality_score = self._calculate_signal_quality(valid_ibi, cv, rmssd)
        
        return {
            'method': 'Basic',
            'ibi_ms': valid_ibi,
            'mean_ibi_ms': mean_ibi,
            'heart_rate_bpm': heart_rate,
            'rmssd': rmssd,
            'pnn50': pnn50,
            'cv': cv,
            'quality_score': quality_score,
            'is_valid': quality_score > 0.6,  # 基础方法略低阈值
            'n_beats': len(valid_ibi)
        }
    
    def _calculate_signal_quality(self, ibi_ms, cv, rmssd):
        """计算信号质量评分 (0-1)"""
        try:
            # 1. 变异系数评分 (CV < 0.3 为好)
            cv_score = max(0, 1 - cv / 0.3)
            
            # 2. IBI数量评分 (> 30个IBI为好)
            count_score = min(1, len(ibi_ms) / 30)
            
            # 3. RMSSD评分 (10-100ms为正常范围)
            rmssd_score = 1.0 if 10 <= rmssd <= 100 else 0.5
            
            # 4. 生理范围评分
            mean_ibi = np.mean(ibi_ms)
            physio_score = 1.0 if 400 <= mean_ibi <= 1000 else 0.7
            
            # 综合评分
            quality_score = (cv_score * 0.3 + count_score * 0.3 + 
                           rmssd_score * 0.2 + physio_score * 0.2)
            
            return np.clip(quality_score, 0, 1)
            
        except:
            return 0.5  # 默认中等质量
    
    def advanced_peak_detection(self, signal, fs=100):
        """高级峰值检测 - 结合传统方法和NeuroKit2"""
        try:
            if HAS_NEUROKIT:
                # 使用NeuroKit2进行峰值检测
                try:
                    # NeuroKit2的PPG峰值检测
                    _, info = nk.ppg_peaks(signal, sampling_rate=fs, method="elgendi")
                    peaks = info["PPG_Peaks"]
                    
                    if len(peaks) > 0:
                        return peaks
                except:
                    pass  # 如果NeuroKit失败，回退到传统方法
            
            # 传统峰值检测方法
            return self._traditional_peak_detection(signal, fs)
            
        except Exception as e:
            print(f"⚠️  Peak detection failed: {e}")
            return np.array([])
    
    def _traditional_peak_detection(self, signal, fs=100):
        """传统峰值检测方法"""
        try:
            # 自适应阈值
            signal_std = np.std(signal)
            signal_mean = np.mean(signal)
            
            # 动态参数
            min_distance = int(60 / self.max_hr * fs)  # 最小间隔
            height_threshold = signal_mean + 0.3 * signal_std
            prominence_threshold = 0.1 * signal_std
            
            # 峰值检测
            peaks, _ = find_peaks(
                signal,
                height=height_threshold,
                distance=min_distance,
                prominence=prominence_threshold
            )
            
            return peaks
            
        except Exception as e:
            print(f"⚠️  Traditional peak detection failed: {e}")
            return np.array([])
    
    def detect_sensor_peaks(self, sensor, exp_id):
        """检测单个传感器的峰值"""
        try:
            # 读取数据
            file_path = os.path.join(self.data_path, f"{exp_id}_hub_{sensor}_aligned.csv")
            if not os.path.exists(file_path):
                return None
            
            df = pd.read_csv(file_path)
            if len(df.columns) < 3:  # 需要timestamp + 3个通道
                return None
            
            # 获取IR通道 (第2列：red=1, ir=2, green=3)
            ir_signal = df.iloc[:, 2].values
            timestamp = df.iloc[:, 0].values
            
            # 滤波
            filtered_signal = self.advanced_ppg_filter(ir_signal, self.fs)
            
            # 峰值检测
            peaks = self.advanced_peak_detection(filtered_signal, self.fs)
            
            if len(peaks) == 0:
                return None
            
            # IBI分析
            ibi_result = self.professional_ibi_analysis(filtered_signal, peaks, self.fs)
            
            # 时间戳
            peak_times = timestamp[peaks] if len(timestamp) > max(peaks) else peaks / self.fs
            
            return {
                'sensor': sensor,
                'sensor_name': self.sensor_mapping[sensor],
                'peaks': peaks,
                'peak_times': peak_times,
                'filtered_signal': filtered_signal,
                'original_signal': ir_signal,
                'timestamp': timestamp,
                'ibi_result': ibi_result,
                'sampling_rate': self.fs
            }
            
        except Exception as e:
            print(f"❌ Sensor {sensor} detection failed: {e}")
            return None
    
    def match_heartbeats_across_sensors(self, sensor_results):
        """高级心跳匹配算法 - 基于时间窗口和生理约束"""
        try:
            # 1. 找到有效的传感器
            valid_sensors = []
            for sensor, result in sensor_results.items():
                if result and result['ibi_result']['is_valid']:
                    valid_sensors.append(sensor)
            
            if len(valid_sensors) < 2:
                print("⚠️  Not enough valid sensors for heartbeat matching")
                return {}
            
            print(f"📍 Valid sensors for matching: {[self.sensor_mapping[s] for s in valid_sensors]}")
            
            # 2. 选择参考传感器 (优先级: finger > nose > wrist > ear)
            sensor_priority = ['sensor3', 'sensor2', 'sensor4', 'sensor5']
            reference_sensor = None
            for sensor in sensor_priority:
                if sensor in valid_sensors:
                    reference_sensor = sensor
                    break
            
            if not reference_sensor:
                reference_sensor = valid_sensors[0]
            
            print(f"📖 Reference sensor: {self.sensor_mapping[reference_sensor]}")
            
            # 3. 获取参考峰值时间
            ref_peak_times = sensor_results[reference_sensor]['peak_times']
            
            # 4. 为每个心跳创建匹配窗口
            matched_heartbeats = []
            
            for i, ref_time in enumerate(ref_peak_times):
                heartbeat = {
                    'heartbeat_id': i + 1,
                    'reference_sensor': reference_sensor,
                    'reference_time': ref_time,
                    'sensor_peaks': {reference_sensor: ref_time},
                    'sensor_peak_indices': {reference_sensor: sensor_results[reference_sensor]['peaks'][i]}
                }
                
                # 5. 为其他传感器找到匹配的峰值
                for sensor in valid_sensors:
                    if sensor == reference_sensor:
                        continue
                    
                    sensor_peak_times = sensor_results[sensor]['peak_times']
                    sensor_peak_indices = sensor_results[sensor]['peaks']
                    
                    # 在时间窗口内寻找最近的峰值
                    time_diffs = np.abs(sensor_peak_times - ref_time)
                    valid_mask = time_diffs <= self.heartbeat_window
                    
                    if np.any(valid_mask):
                        # 选择最近的峰值
                        closest_idx = np.argmin(time_diffs[valid_mask])
                        actual_idx = np.where(valid_mask)[0][closest_idx]
                        
                        heartbeat['sensor_peaks'][sensor] = sensor_peak_times[actual_idx]
                        heartbeat['sensor_peak_indices'][sensor] = sensor_peak_indices[actual_idx]
                
                # 6. 只保留有足够传感器的心跳
                if len(heartbeat['sensor_peaks']) >= self.min_sensors_per_beat:
                    matched_heartbeats.append(heartbeat)
            
            print(f"📊 Matched heartbeats: {len(matched_heartbeats)}/{len(ref_peak_times)}")
            
            return {
                'matched_heartbeats': matched_heartbeats,
                'valid_sensors': valid_sensors,
                'reference_sensor': reference_sensor,
                'total_heartbeats': len(ref_peak_times),
                'matched_count': len(matched_heartbeats)
            }
            
        except Exception as e:
            print(f"❌ Heartbeat matching failed: {e}")
            return {}
    
    def calculate_ptt_matrix(self, matching_result):
        """计算PTT矩阵 - 所有传感器组合"""
        try:
            matched_heartbeats = matching_result['matched_heartbeats']
            valid_sensors = matching_result['valid_sensors']
            
            if len(matched_heartbeats) == 0 or len(valid_sensors) < 2:
                return {}
            
            # 计算所有传感器对的PTT
            ptt_data = []
            ptt_timeseries = []
            
            for i, sensor1 in enumerate(valid_sensors):
                for j, sensor2 in enumerate(valid_sensors):
                    if i >= j:  # 避免重复计算
                        continue
                    
                    # 提取两个传感器的时间序列
                    times1 = []
                    times2 = []
                    heartbeat_ids = []
                    
                    for hb in matched_heartbeats:
                        if sensor1 in hb['sensor_peaks'] and sensor2 in hb['sensor_peaks']:
                            times1.append(hb['sensor_peaks'][sensor1])
                            times2.append(hb['sensor_peaks'][sensor2])
                            heartbeat_ids.append(hb['heartbeat_id'])
                    
                    if len(times1) > 0:
                        times1 = np.array(times1)
                        times2 = np.array(times2)
                        
                        # 计算PTT (传播方向：sensor1 → sensor2)
                        ptt_values = (times2 - times1) * 1000  # 转换为毫秒
                        
                        # 过滤异常值 (PTT应该在-500ms到+500ms之间)
                        valid_mask = (ptt_values >= -500) & (ptt_values <= 500)
                        valid_ptt = ptt_values[valid_mask]
                        valid_hb_ids = np.array(heartbeat_ids)[valid_mask]
                        
                        if len(valid_ptt) > 0:
                            # 统计信息
                            ptt_summary = {
                                'sensor_pair': f"{sensor1}-{sensor2}",
                                'sensor_names': f"{self.sensor_mapping[sensor1]}-{self.sensor_mapping[sensor2]}",
                                'from_sensor': sensor1,
                                'to_sensor': sensor2,
                                'from_name': self.sensor_mapping[sensor1],
                                'to_name': self.sensor_mapping[sensor2],
                                'n_heartbeats': len(valid_ptt),
                                'mean_ptt_ms': np.mean(valid_ptt),
                                'std_ptt_ms': np.std(valid_ptt),
                                'median_ptt_ms': np.median(valid_ptt),
                                'min_ptt_ms': np.min(valid_ptt),
                                'max_ptt_ms': np.max(valid_ptt),
                                'correlation': pearsonr(times1[valid_mask], times2[valid_mask])[0]
                            }
                            ptt_data.append(ptt_summary)
                            
                            # 时间序列数据
                            for k, (ptt_val, hb_id) in enumerate(zip(valid_ptt, valid_hb_ids)):
                                ptt_timeseries.append({
                                    'heartbeat_id': hb_id,
                                    'sensor_pair': f"{sensor1}-{sensor2}",
                                    'from_sensor': sensor1,
                                    'to_sensor': sensor2,
                                    'ptt_ms': ptt_val,
                                    'time1': times1[valid_mask][k],
                                    'time2': times2[valid_mask][k]
                                })
            
            return {
                'ptt_summary': ptt_data,
                'ptt_timeseries': ptt_timeseries,
                'n_sensor_pairs': len(ptt_data),
                'total_ptt_measurements': len(ptt_timeseries)
            }
            
        except Exception as e:
            print(f"❌ PTT calculation failed: {e}")
            return {}
    
    def save_detailed_results(self, exp_id, sensor_results, matching_result, ptt_result):
        """保存详细结果 - 按sensor分别存储"""
        try:
            # 1. 保存每个sensor的详细峰值信息
            for sensor, result in sensor_results.items():
                if not result:
                    continue
                
                # 准备峰值详细数据
                peaks_detail = []
                for i, (peak_idx, peak_time) in enumerate(zip(result['peaks'], result['peak_times'])):
                    ibi_ms = (result['peak_times'][i] - result['peak_times'][i-1]) * 1000 if i > 0 else np.nan
                    
                    peaks_detail.append({
                        'peak_number': i + 1,
                        'peak_index': peak_idx,
                        'peak_time': peak_time,
                        'ibi_ms': ibi_ms,
                        'is_valid_ibi': 300 <= ibi_ms <= 1200 if not np.isnan(ibi_ms) else False
                    })
                
                # 保存sensor详细文件
                sensor_detail_df = pd.DataFrame(peaks_detail)
                sensor_file = os.path.join(self.output_dir, f"sensor_{sensor}_peaks_exp_{exp_id}.csv")
                sensor_detail_df.to_csv(sensor_file, index=False)
                print(f"💾 Saved {sensor} details: {sensor_file}")
            
            # 2. 保存传感器质量汇总
            sensor_summary = []
            for sensor, result in sensor_results.items():
                if result:
                    ibi = result['ibi_result']
                    sensor_summary.append({
                        'sensor': sensor,
                        'sensor_name': result['sensor_name'],
                        'peak_count': len(result['peaks']),
                        'is_valid': ibi['is_valid'],
                        'quality_score': ibi['quality_score'],
                        'heart_rate_bpm': ibi['heart_rate_bpm'],
                        'mean_ibi_ms': ibi['mean_ibi_ms'],
                        'rmssd': ibi['rmssd'],
                        'cv': ibi['cv'],
                        'analysis_method': ibi['method']
                    })
            
            if sensor_summary:
                summary_df = pd.DataFrame(sensor_summary)
                summary_file = os.path.join(self.output_dir, f"sensor_summary_exp_{exp_id}.csv")
                summary_df.to_csv(summary_file, index=False)
                print(f"💾 Saved sensor summary: {summary_file}")
            
            # 3. 保存心跳匹配结果
            if matching_result and 'matched_heartbeats' in matching_result:
                heartbeat_data = []
                for hb in matching_result['matched_heartbeats']:
                    row = {
                        'heartbeat_id': hb['heartbeat_id'],
                        'reference_sensor': hb['reference_sensor'],
                        'reference_time': hb['reference_time'],
                        'n_sensors': len(hb['sensor_peaks'])
                    }
                    
                    # 添加每个传感器的峰值信息
                    for sensor in self.sensors:
                        row[f'{sensor}_time'] = hb['sensor_peaks'].get(sensor, np.nan)
                        row[f'{sensor}_index'] = hb['sensor_peak_indices'].get(sensor, np.nan)
                    
                    heartbeat_data.append(row)
                
                heartbeat_df = pd.DataFrame(heartbeat_data)
                heartbeat_file = os.path.join(self.output_dir, f"matched_heartbeats_exp_{exp_id}.csv")
                heartbeat_df.to_csv(heartbeat_file, index=False)
                print(f"💾 Saved heartbeat matching: {heartbeat_file}")
            
            # 4. 保存PTT结果
            if ptt_result and 'ptt_summary' in ptt_result:
                # PTT汇总
                ptt_summary_df = pd.DataFrame(ptt_result['ptt_summary'])
                ptt_summary_file = os.path.join(self.output_dir, f"ptt_matrix_exp_{exp_id}.csv")
                ptt_summary_df.to_csv(ptt_summary_file, index=False)
                print(f"💾 Saved PTT matrix: {ptt_summary_file}")
                
                # PTT时间序列
                ptt_timeseries_df = pd.DataFrame(ptt_result['ptt_timeseries'])
                ptt_timeseries_file = os.path.join(self.output_dir, f"ptt_timeseries_exp_{exp_id}.csv")
                ptt_timeseries_df.to_csv(ptt_timeseries_file, index=False)
                print(f"💾 Saved PTT timeseries: {ptt_timeseries_file}")
            
        except Exception as e:
            print(f"❌ Save results failed: {e}")
    
    def create_english_visualizations(self, exp_id, sensor_results, matching_result, ptt_result):
        """创建全英文可视化"""
        try:
            # 1. 传感器峰值检测图
            valid_sensors = [s for s, r in sensor_results.items() if r is not None]
            
            if len(valid_sensors) > 0:
                fig, axes = plt.subplots(len(valid_sensors), 1, figsize=(16, 4*len(valid_sensors)), sharex=True)
                if len(valid_sensors) == 1:
                    axes = [axes]
                
                colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']
                
                for idx, sensor in enumerate(valid_sensors):
                    ax = axes[idx]
                    result = sensor_results[sensor]
                    
                    # 时间轴（前60秒用于显示）
                    time = result['timestamp'] - result['timestamp'][0]
                    display_mask = time <= 60  # 只显示前60秒
                    
                    # 绘制滤波信号
                    ax.plot(time[display_mask], result['filtered_signal'][display_mask], 
                           color=colors[idx % len(colors)], linewidth=1.5, alpha=0.8,
                           label=f'{result["sensor_name"]} IR Channel')
                    
                    # 标记峰值
                    peaks = result['peaks']
                    peak_times = result['peak_times'] - result['timestamp'][0]
                    display_peaks = peak_times <= 60
                    
                    if np.any(display_peaks):
                        displayed_peaks = peaks[display_peaks]
                        displayed_peak_times = peak_times[display_peaks]
                        ax.scatter(displayed_peak_times, result['filtered_signal'][displayed_peaks], 
                                 color='red', s=30, zorder=5, alpha=0.9, marker='o', label='Detected Peaks')
                    
                    # 设置标题和标签
                    ibi = result['ibi_result']
                    status = "✓ Valid" if ibi['is_valid'] else "✗ Invalid"
                    ax.set_title(f'{result["sensor_name"]} IR Channel - {status} - HR: {ibi["heart_rate_bpm"]:.1f} BPM ' +
                                f'(Quality: {ibi["quality_score"]:.2f})', fontsize=12, fontweight='bold')
                    ax.set_ylabel('Filtered Signal', fontsize=10)
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                
                axes[-1].set_xlabel('Time (seconds)', fontsize=12)
                plt.suptitle(f'Experiment {exp_id} - Advanced IR Channel Peak Detection', 
                           fontsize=16, fontweight='bold')
                plt.tight_layout()
                
                plot_file = os.path.join(self.output_dir, f"advanced_peaks_exp_{exp_id}.png")
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"📊 Saved peak detection plot: {plot_file}")
            
            # 2. 心跳匹配和PTT可视化
            if matching_result and ptt_result and 'matched_heartbeats' in matching_result:
                self._create_ptt_visualization(exp_id, matching_result, ptt_result)
            
        except Exception as e:
            print(f"❌ Visualization failed: {e}")
    
    def _create_ptt_visualization(self, exp_id, matching_result, ptt_result):
        """创建PTT专门的可视化"""
        try:
            if 'ptt_summary' not in ptt_result or len(ptt_result['ptt_summary']) == 0:
                return
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # 1. PTT矩阵热力图
            ptt_data = ptt_result['ptt_summary']
            sensors = list(set([d['from_sensor'] for d in ptt_data] + [d['to_sensor'] for d in ptt_data]))
            sensor_names = [self.sensor_mapping[s] for s in sensors]
            
            ptt_matrix = np.full((len(sensors), len(sensors)), np.nan)
            
            for data in ptt_data:
                i = sensors.index(data['from_sensor'])
                j = sensors.index(data['to_sensor'])
                ptt_matrix[i, j] = data['mean_ptt_ms']
                ptt_matrix[j, i] = -data['mean_ptt_ms']  # 反向PTT
            
            im = ax1.imshow(ptt_matrix, cmap='RdBu_r', aspect='auto')
            ax1.set_xticks(range(len(sensor_names)))
            ax1.set_yticks(range(len(sensor_names)))
            ax1.set_xticklabels(sensor_names)
            ax1.set_yticklabels(sensor_names)
            ax1.set_title('PTT Matrix (ms)', fontweight='bold')
            
            # 添加数值标注
            for i in range(len(sensors)):
                for j in range(len(sensors)):
                    if not np.isnan(ptt_matrix[i, j]):
                        ax1.text(j, i, f'{ptt_matrix[i, j]:.1f}', ha="center", va="center", 
                               color="white" if abs(ptt_matrix[i, j]) > np.nanmax(np.abs(ptt_matrix))/2 else "black",
                               fontweight='bold')
            
            plt.colorbar(im, ax=ax1, label='PTT (ms)')
            
            # 2. PTT分布柱状图
            sensor_pairs = [d['sensor_names'] for d in ptt_data]
            mean_ptts = [d['mean_ptt_ms'] for d in ptt_data]
            std_ptts = [d['std_ptt_ms'] for d in ptt_data]
            
            bars = ax2.bar(range(len(sensor_pairs)), mean_ptts, yerr=std_ptts, 
                          capsize=5, alpha=0.7, color=['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6', '#F1C40F'][:len(sensor_pairs)])
            ax2.set_xticks(range(len(sensor_pairs)))
            ax2.set_xticklabels(sensor_pairs, rotation=45)
            ax2.set_ylabel('PTT (ms)')
            ax2.set_title('PTT Distribution Across Sensor Pairs', fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            # 添加数值标签
            for bar, mean_ptt in zip(bars, mean_ptts):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + (height*0.1 if height > 0 else height*0.1),
                        f'{mean_ptt:.1f}', ha='center', va='bottom' if height > 0 else 'top', fontweight='bold')
            
            # 3. 心跳匹配质量
            matched_hb = matching_result['matched_heartbeats']
            sensor_counts = [len(hb['sensor_peaks']) for hb in matched_hb]
            
            count_hist, bins = np.histogram(sensor_counts, bins=range(2, max(sensor_counts)+2))
            ax3.bar(bins[:-1], count_hist, alpha=0.7, color='#3498DB')
            ax3.set_xlabel('Number of Sensors per Heartbeat')
            ax3.set_ylabel('Number of Heartbeats')
            ax3.set_title('Heartbeat Matching Quality', fontweight='bold')
            ax3.grid(True, alpha=0.3)
            
            # 4. PTT时间序列示例
            if 'ptt_timeseries' in ptt_result and len(ptt_result['ptt_timeseries']) > 0:
                timeseries = pd.DataFrame(ptt_result['ptt_timeseries'])
                
                # 选择数据最多的sensor pair
                pair_counts = timeseries['sensor_pair'].value_counts()
                if len(pair_counts) > 0:
                    best_pair = pair_counts.index[0]
                    pair_data = timeseries[timeseries['sensor_pair'] == best_pair].head(50)  # 只显示前50个
                    
                    ax4.plot(pair_data['heartbeat_id'], pair_data['ptt_ms'], 'o-', alpha=0.7, linewidth=2, markersize=4)
                    ax4.set_xlabel('Heartbeat ID')
                    ax4.set_ylabel('PTT (ms)')
                    ax4.set_title(f'PTT Time Series: {best_pair}', fontweight='bold')
                    ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            ptt_plot_file = os.path.join(self.output_dir, f"ptt_analysis_exp_{exp_id}.png")
            plt.savefig(ptt_plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"📊 Saved PTT analysis plot: {ptt_plot_file}")
            
        except Exception as e:
            print(f"❌ PTT visualization failed: {e}")
    
    def process_experiment(self, exp_id):
        """处理单个实验 - 完整流程"""
        print(f"\n🔍 Processing Experiment {exp_id} - Advanced Analysis")
        
        # 1. 检测每个传感器的峰值
        sensor_results = {}
        for sensor in self.sensors:
            print(f"   📡 Processing {self.sensor_mapping[sensor]} ({sensor})...")
            result = self.detect_sensor_peaks(sensor, exp_id)
            if result:
                ibi = result['ibi_result']
                status = "✓" if ibi['is_valid'] else "✗"
                print(f"      {status} {ibi['n_beats']} beats, HR={ibi['heart_rate_bpm']:.1f} BPM, " +
                      f"Quality={ibi['quality_score']:.2f} ({ibi['method']})")
            else:
                print(f"      ✗ Detection failed")
            sensor_results[sensor] = result
        
        # 2. 心跳匹配
        print(f"\n   🔗 Matching heartbeats across sensors...")
        matching_result = self.match_heartbeats_across_sensors(sensor_results)
        
        # 3. 计算PTT
        print(f"\n   ⏱️  Calculating PTT matrix...")
        ptt_result = self.calculate_ptt_matrix(matching_result)
        
        if ptt_result and 'ptt_summary' in ptt_result:
            print(f"      📊 Generated {ptt_result['n_sensor_pairs']} PTT pairs, " +
                  f"{ptt_result['total_ptt_measurements']} measurements")
        
        # 4. 保存结果
        print(f"\n   💾 Saving detailed results...")
        self.save_detailed_results(exp_id, sensor_results, matching_result, ptt_result)
        
        # 5. 生成可视化
        print(f"\n   📊 Creating visualizations...")
        self.create_english_visualizations(exp_id, sensor_results, matching_result, ptt_result)
        
        return {
            'sensor_results': sensor_results,
            'matching_result': matching_result,
            'ptt_result': ptt_result
        }
    
    def run_analysis(self, experiment_list=None):
        """运行完整的高级PTT分析"""
        if experiment_list is None:
            # 自动检测可用实验
            experiment_list = []
            for file in os.listdir(self.data_path):
                if file.endswith('_hub_sensor2_aligned.csv'):
                    exp_id = file.split('_')[0]
                    experiment_list.append(exp_id)
            experiment_list = sorted(list(set(experiment_list)))
        
        print(f"\n🔬 Advanced PTT Analysis with Professional IBI Processing")
        print(f"📋 Experiments: {experiment_list}")
        print(f"🎯 Features:")
        print(f"   • Professional IBI analysis with {'NeuroKit2' if HAS_NEUROKIT else 'basic methods'}")
        print(f"   • Individual sensor peak storage")
        print(f"   • Advanced heartbeat matching")
        print(f"   • Complete PTT matrix calculation")
        print(f"   • English-only visualizations")
        
        all_results = {}
        
        for exp_id in tqdm(experiment_list, desc="Processing experiments"):
            try:
                results = self.process_experiment(exp_id)
                all_results[exp_id] = results
            except Exception as e:
                print(f"❌ Experiment {exp_id} failed: {e}")
                continue
        
        print(f"\n✅ Advanced PTT analysis complete!")
        print(f"📁 Results saved in: {self.output_dir}")
        print(f"🎯 Ready for PTT-based blood pressure modeling!")
        
        return all_results

def main():
    """主函数"""
    print("🩺 Advanced PTT Peak Detection System")
    print("=" * 60)
    print("🚀 Features:")
    print("   • Professional IBI analysis with NeuroKit2")
    print("   • Individual sensor peak storage")
    print("   • Advanced heartbeat matching algorithm")
    print("   • Complete PTT matrix calculation")
    print("   • English-only visualizations")
    print("   • Ready for blood pressure modeling")
    print("=" * 60)
    
    # 创建检测器
    detector = AdvancedPTTPeakDetector()
    
    # 运行分析（测试实验1）
    results = detector.run_analysis(['1'])
    
    print("\n🎯 Next steps:")
    print("1. Review individual sensor peak files")
    print("2. Check heartbeat matching quality")
    print("3. Analyze PTT matrix for sensor relationships")
    print("4. Use PTT timeseries for blood pressure modeling")

if __name__ == "__main__":
    main() 