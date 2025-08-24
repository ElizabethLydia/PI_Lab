#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化版PI-Lab数据预处理脚本 - 使用PhysioNet2025_Calibrated文件夹中的校准血压数据
智能处理：Biopac降采样 + HUB插值去重 + 插值对齐 + 双格式保存 + 校准血压数据
"""

import os
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
import warnings
import time
import multiprocessing
import warnings
from multiprocessing import Manager
warnings.filterwarnings('ignore')

def extract_sbp_dbp_from_waveform(bp_waveform, window_size=2000, min_peak_distance=500):
    """
    从ABP波形中提取SBP和DBP
    使用优化的峰值检测方法，专门针对动脉血压波形
    """
    if len(bp_waveform) < window_size:
        # 数据太少，使用更小的窗口
        window_size = min(1000, len(bp_waveform) // 2)
        if window_size < 100:
            print(f"        警告: 数据点太少 ({len(bp_waveform)})，无法提取峰值")
            return bp_waveform, bp_waveform
    
    print(f"        使用窗口大小 {window_size} 从ABP波形中提取SBP和DBP...")
    
    # 初始化SBP和DBP数组
    sbp_values = []
    dbp_values = []
    timestamps = []
    
    # 使用较小的步长来获得更精确的峰值
    step_size = max(100, window_size // 8)  # 12.5%重叠，提高精度
    
    # 滑动窗口处理
    for i in range(0, len(bp_waveform) - window_size + 1, step_size):
        window_data = bp_waveform.iloc[i:i+window_size]
        
        if len(window_data) < 200:  # 窗口太小，跳过
            continue
            
        # 在窗口内寻找峰值（SBP）和谷值（DBP）
        window_values = window_data['bp_value'].values
        
        try:
            # 使用更精确的峰值检测
            # 找到窗口内的局部最大值和最小值
            from scipy.signal import find_peaks
            
            # 寻找峰值（SBP）- 寻找局部最大值
            peaks, _ = find_peaks(window_values, height=None, distance=min_peak_distance//2)
            if len(peaks) > 0:
                # 选择最高的峰值
                peak_heights = window_values[peaks]
                max_peak_idx = peaks[np.argmax(peak_heights)]
                max_bp = window_values[max_peak_idx]
            else:
                # 如果没有找到峰值，使用窗口内的最大值
                max_bp = np.max(window_values)
            
            # 寻找谷值（DBP）- 寻找局部最小值
            valleys, _ = find_peaks(-window_values, height=None, distance=min_peak_distance//2)
            if len(valleys) > 0:
                # 选择最低的谷值
                valley_heights = window_values[valleys]
                min_valley_idx = valleys[np.argmin(valley_heights)]
                min_bp = window_values[min_valley_idx]
            else:
                # 如果没有找到谷值，使用窗口内的最小值
                min_bp = np.min(window_values)
            
            # 计算窗口中心时间戳
            center_idx = i + window_size // 2
            if center_idx < len(bp_waveform):
                center_timestamp = bp_waveform.iloc[center_idx]['timestamp']
                
                sbp_values.append(max_bp)
                dbp_values.append(min_bp)
                timestamps.append(center_timestamp)
                
        except Exception as e:
            print(f"        窗口 {i} 处理错误: {e}")
            continue
    
    # 创建SBP和DBP的DataFrame
    if sbp_values:
        sbp_df = pd.DataFrame({
            'timestamp': timestamps,
            'sbp': sbp_values
        })
        dbp_df = pd.DataFrame({
            'timestamp': timestamps,
            'dbp': dbp_values
        })
        
        # 验证SBP和DBP的合理性
        sbp_mean = np.mean(sbp_values)
        dbp_mean = np.mean(dbp_values)
        if sbp_mean <= dbp_mean:
            print(f"        警告: SBP ({sbp_mean:.1f}) <= DBP ({dbp_mean:.1f})，可能检测错误")
        
        print(f"        检测完成: SBP {len(sbp_df):,} 点 (均值: {sbp_mean:.1f}), DBP {len(dbp_df):,} 点 (均值: {dbp_mean:.1f})")
        return sbp_df, dbp_df
    else:
        # 如果没有检测到峰值，返回原始数据
        print(f"        警告: 未检测到峰值，返回原始数据")
        return bp_waveform, bp_waveform

def load_calibrated_bp_data(subject_id, experiment_numbers=None):
    """
    加载PhysioNet2025_Calibrated文件夹中的校准血压数据
    """
    calibrated_root = '/root/shared/PhysioNet2025_Calibrated/'
    subject_path = os.path.join(calibrated_root, subject_id)
    
    if not os.path.exists(subject_path):
        print(f"  警告: PhysioNet2025_Calibrated文件夹中不存在subject {subject_id}")
        return {}
    
    # 如果没有指定实验编号，获取所有可用的实验
    if experiment_numbers is None:
        available_experiments = [d for d in os.listdir(subject_path) 
                               if os.path.isdir(os.path.join(subject_path, d)) and d.isdigit()]
        experiment_numbers = [int(exp) for exp in available_experiments]
        experiment_numbers.sort()
        print(f"    发现可用实验: {experiment_numbers}")
    
    bp_data = {}
    
    for exp_num in experiment_numbers:
        exp_dir = os.path.join(subject_path, str(exp_num))
        bp_file_path = os.path.join(exp_dir, 'Biopac', 'bp.csv')
        
        if os.path.exists(bp_file_path):
            try:
                # 读取校准后的血压数据
                data = pd.read_csv(bp_file_path)
                print(f"    加载实验 {exp_num} 校准血压数据: {len(data):,} 行")
                
                # 检查数据列结构 - 新数据没有列名，需要添加
                if len(data.columns) == 2:
                    # 数据有两列但没有列名，添加列名
                    data.columns = ['timestamp', 'bp_value']
                    print(f"      添加列名: timestamp, bp_value")
                
                if 'timestamp' in data.columns and 'bp_value' in data.columns:
                    # 数据现在是正确的格式，但需要从ABP波形中提取SBP和DBP
                    print(f"      从ABP波形中提取SBP和DBP...")
                    sbp_df, dbp_df = extract_sbp_dbp_from_waveform(data)
                    
                    bp_data[str(exp_num)] = {
                        'waveform': data,
                        'sbp': sbp_df,
                        'dbp': dbp_df
                    }
                    print(f"      提取完成: SBP {len(sbp_df):,} 点, DBP {len(dbp_df):,} 点")
                else:
                    # 尝试提取SBP和DBP
                    print(f"      提取SBP和DBP...")
                    sbp_df, dbp_df = extract_sbp_dbp_from_waveform(data)
                    
                    # 存储原始波形、SBP和DBP数据
                    bp_data[str(exp_num)] = {
                        'waveform': data,
                        'sbp': sbp_df,
                        'dbp': dbp_df
                    }
                    
                    print(f"      提取完成: SBP {len(sbp_df):,} 点, DBP {len(dbp_df):,} 点")
                
            except Exception as e:
                print(f"    错误: 读取实验 {exp_num} 校准血压数据失败: {e}")
                continue
        else:
            print(f"    警告: 实验 {exp_num} 校准血压文件不存在: {bp_file_path}")
    
    return bp_data

def interpolate_duplicate_timestamps(df, time_col='timestamp'):
    """
    插值处理重复时间戳 - 用于HUB数据保持精度
    """
    if df.empty or time_col not in df.columns:
        return df
    
    df = df.copy()
    unique_times = df[time_col].unique()
    
    # 检查是否有重复
    if len(unique_times) == len(df):
        return df
    
    print(f"      插值处理重复时间戳...")
    
    new_timestamps = []
    for t in unique_times:
        indices = df[df[time_col] == t].index
        n_points = len(indices)

        if n_points > 1:
            current_idx = np.where(unique_times == t)[0][0]
            if current_idx == len(unique_times) - 1:
                delta = 0.001  # 最后一个时间点使用默认小间隔
            else:
                next_t = unique_times[current_idx + 1]
                delta = (next_t - t) / n_points

            for i, idx in enumerate(indices):
                new_timestamps.append((t + i * delta, idx))
        else:
            new_timestamps.append((t, indices[0]))

    # 按原始顺序重新排列时间戳
    new_timestamps.sort(key=lambda x: x[1])
    df[time_col] = [t for t, _ in new_timestamps]
    
    duplicates_fixed = len(df) - len(unique_times)
    if duplicates_fixed > 0:
        print(f"      修复了 {duplicates_fixed} 个重复时间戳")
    
    return df

def quick_downsample_biopac(df, time_col='timestamp', target_freq=100):
    """对Biopac高频数据降采样"""
    if df.empty or time_col not in df.columns:
        return df
    
    original_len = len(df)
    
    if len(df) > 1:
        time_range = df[time_col].max() - df[time_col].min()
        current_freq = len(df) / time_range if time_range > 0 else 1
        
        print(f"      估算频率: {current_freq:.1f}Hz -> 目标: {target_freq}Hz")
        
        if current_freq > target_freq * 1.5:  # 只有明显高频才降采样
            step = max(1, int(current_freq / target_freq))
            result = df.iloc[::step].copy()
            print(f"      降采样: {original_len:,} -> {len(result):,} 行 (步长: {step})")
            return result
        else:
            print(f"      频率适中，跳过降采样")
    
    return df

def interpolate_with_reftime(time, data, reftime):
    """
    使用插值对齐到参考时间戳
    """
    if len(time) < 2 or len(data) < 2:
        return pd.DataFrame(columns=data.columns if hasattr(data, 'columns') else ['value'])
    
    # 确保数据类型正确
    time = np.asarray(time, dtype=float)
    reftime = np.asarray(reftime, dtype=float)
    
    # 限制插值范围到数据实际范围内
    min_time, max_time = time.min(), time.max()
    valid_reftime_mask = (reftime >= min_time) & (reftime <= max_time)
    valid_reftime = reftime[valid_reftime_mask]
    
    if len(valid_reftime) == 0:
        return pd.DataFrame(columns=data.columns if hasattr(data, 'columns') else ['value'])
    
    try:
        # 使用线性插值
        interp_func = interp1d(time, data, axis=0, kind='linear', bounds_error=False, fill_value=np.nan)
        interpolated_data = interp_func(valid_reftime)
        
        # 创建完整结果DataFrame
        if interpolated_data.ndim == 1:
            interpolated_data = interpolated_data.reshape(-1, 1)
        
        # 创建与参考时间长度相同的结果
        full_result = np.full((len(reftime), interpolated_data.shape[1]), np.nan)
        full_result[valid_reftime_mask] = interpolated_data
        
        result_df = pd.DataFrame(full_result, columns=data.columns if hasattr(data, 'columns') else [f'col_{i}' for i in range(full_result.shape[1])])
        result_df['timestamp'] = reftime
        
        return result_df
    except Exception as e:
        print(f"        插值错误: {e}")
        return pd.DataFrame(columns=data.columns if hasattr(data, 'columns') else ['value'])

def process_biopac_file(file_path, target_freq=100):
    """处理Biopac文件 - 先降采样再插值处理重复"""
    try:
        file_name = os.path.basename(file_path)
        try:
            data = pd.read_csv(file_path)
        except UnicodeDecodeError:
            data = pd.read_csv(file_path, encoding='latin1')
        original_rows = len(data)
        
        if data.empty:
            return pd.DataFrame()
        
        print(f"    Biopac {file_name} ({original_rows:,} 行)")
        
        # 1. 先降采样到目标频率（不去重）
        data = quick_downsample_biopac(data, target_freq=target_freq)
        
        # 2. 再用插值处理重复时间戳
        data = interpolate_duplicate_timestamps(data)
        
        # 3. 排序
        if data is not None and not data.empty and 'timestamp' in data.columns:
            data = data.sort_values('timestamp').reset_index(drop=True)
        
        final_rows = len(data) if data is not None else 0
        compression_ratio = original_rows / final_rows if final_rows > 0 else 1
        print(f"      最终: {final_rows:,} 行 (压缩比: {compression_ratio:.1f}:1)")
        
        return data if data is not None else pd.DataFrame()
        
    except Exception as e:
        print(f"      错误: {e}")
        return pd.DataFrame()

def process_hub_file(file_path):
    """处理HUB文件 - 插值策略"""
    try:
        file_name = os.path.basename(file_path)
        try:
            data = pd.read_csv(file_path)
        except UnicodeDecodeError:
            data = pd.read_csv(file_path, encoding='latin1')
        original_rows = len(data)
        
        if data.empty:
            return pd.DataFrame()
        
        print(f"    HUB {file_name} ({original_rows:,} 行)")
        
        # 1. 插值处理重复时间戳（保持HUB数据精度）
        processed_data = interpolate_duplicate_timestamps(data)
        
        # 2. 排序
        if processed_data is not None and not processed_data.empty and 'timestamp' in processed_data.columns:
            processed_data = processed_data.sort_values('timestamp').reset_index(drop=True)
        
        final_rows = len(processed_data) if processed_data is not None else 0
        print(f"      最终: {final_rows:,} 行")
        
        return processed_data if processed_data is not None else pd.DataFrame()
        
    except Exception as e:
        print(f"      错误: {e}")
        return pd.DataFrame()

def load_experiment_smart(experiment_path, calibrated_bp_data, target_freq=100):
    """智能加载单个实验，包含校准血压数据和原始biopac数据"""
    experiment_name = os.path.basename(experiment_path)
    print(f"\n{'='*50}")
    print(f"处理实验 {experiment_name}")
    print(f"{'='*50}")
    
    result = {'biopac': {}, 'hub': {}}
    
    # 处理校准血压数据
    if experiment_name in calibrated_bp_data:
        bp_data_dict = calibrated_bp_data[experiment_name]
        print(f"\n血压数据 (PhysioNet2025_Calibrated):")
        print(f"      原始波形: {len(bp_data_dict['waveform']):,} 行")
        print(f"      SBP: {len(bp_data_dict['sbp']):,} 点")
        print(f"      DBP: {len(bp_data_dict['dbp']):,} 点")
        
        # 将血压数据存储到biopac部分
        result['biopac']['waveform'] = bp_data_dict['waveform']
        result['biopac']['sbp'] = bp_data_dict['sbp']
        result['biopac']['dbp'] = bp_data_dict['dbp']
    
    # 处理Biopac数据 (高频，先降采样再插值处理重复)
    biopac_path = os.path.join(experiment_path, 'Biopac')
    if os.path.isdir(biopac_path):
        biopac_files = [f for f in os.listdir(biopac_path) if f.endswith('.csv')]
        print(f"\nBiopac文件 ({len(biopac_files)} 个) - 降采样+插值策略:")
        
        for file in biopac_files:
            file_path = os.path.join(biopac_path, file)
            data = process_biopac_file(file_path, target_freq)
            if data is not None and not data.empty:
                key = file.split('-')[0] if '-' in file else file.replace('.csv', '')
                result['biopac'][key] = data
    
    # 处理HUB数据 (低频，用插值)
    hub_path = os.path.join(experiment_path, 'HUB')
    if os.path.isdir(hub_path):
        hub_files = [f for f in os.listdir(hub_path) if f.endswith('.csv')]
        print(f"\nHUB文件 ({len(hub_files)} 个) - 使用插值策略:")
        
        for file in hub_files:
            file_path = os.path.join(hub_path, file)
            data = process_hub_file(file_path)
            if not data.empty:
                key = file.replace('.csv', '')
                result['hub'][key] = data
    
    # 统计结果
    biopac_count = len(result['biopac'])
    hub_count = len(result['hub'])
    print(f"\n实验 {experiment_name} 完成: {biopac_count} 个Biopac文件, {hub_count} 个HUB文件")
    
    return result

def align_data_with_interpolation(data_dict, calibrated_bp_data, output_dir, csv_dir, subject):
    """使用插值进行精确数据对齐，包含校准血压数据"""
    aligned_data = {}
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*50}")
    print("插值对齐阶段 (包含moved血压数据)")
    print(f"{'='*50}")
    
    for experiment_name, experiment_data in data_dict.items():
        try:
            print(f"\n对齐实验 {experiment_name}...")
            
            # 查找参考时间序列（优先使用sensor2）
            ref_data = None
            ref_name = ""
            
            try:
                hub_sensor2 = experiment_data['hub'].get('sensor2', pd.DataFrame())
                if not hub_sensor2.empty and 'timestamp' in hub_sensor2.columns:
                    ref_data = hub_sensor2
                    ref_name = "sensor2"
            except KeyError:
                print(f"  警告: sensor2 缺少 timestamp 列")
            
            if ref_data is None:
                min_len = float('inf')
                for data_type in ['hub', 'biopac']:
                    for key, data in experiment_data[data_type].items():
                        if isinstance(data, pd.DataFrame) and not data.empty and 'timestamp' in data.columns:
                            if len(data) < min_len:
                                min_len = len(data)
                                ref_data = data
                                ref_name = f"{data_type}_{key}"
            
            if ref_data is None or ref_data.empty or 'timestamp' not in ref_data.columns:
                print(f"  警告: experiment {experiment_name} 无有效参考数据或缺少 timestamp 列，跳过")
                continue
            
            try:
                ref_timestamps = ref_data['timestamp'].values
            except KeyError:
                print(f"  错误: 参考数据缺少 timestamp 列，跳过 experiment {experiment_name}")
                continue
        
            print(f"  使用 {ref_name} 作为参考 ({len(ref_data):,} 行)")
            aligned_experiment = {'biopac': {}, 'hub': {}}
            
            # 插值对齐所有数据
            for data_type in ['biopac', 'hub']:
                for key, data in experiment_data[data_type].items():
                    if isinstance(data, pd.DataFrame) and not data.empty and 'timestamp' in data.columns:
                        print(f"    对齐 {data_type}_{key}...")
                        
                        # 提取需要插值的列（除了timestamp）
                        data_columns = [col for col in data.columns if col != 'timestamp']
                        if data_columns:
                            # 使用插值对齐
                            interpolated_data = interpolate_with_reftime(
                                data['timestamp'].values,
                                data[data_columns].values,
                                ref_timestamps
                            )
                            
                            if not interpolated_data.empty:
                                # 重新设置列名
                                interpolated_data.columns = data_columns + ['timestamp']
                                aligned_experiment[data_type][key] = interpolated_data
                                print(f"      对齐完成: {len(data):,} -> {len(interpolated_data):,} 行")
                            else:
                                print(f"      对齐失败，跳过")
            
            aligned_data[experiment_name] = aligned_experiment
            
            # 保存pkl格式
            pkl_path = os.path.join(output_dir, f'experiment_{experiment_name}_aligned.pkl')
            with open(pkl_path, 'wb') as f:
                pickle.dump({experiment_name: aligned_experiment}, f)
            print(f"  保存PKL: {pkl_path}")
            
            # 保存npy格式
            npy_path = os.path.join(output_dir, f'experiment_{experiment_name}_aligned.npy')
            npy_data = {
                experiment_name: {
                    data_type: {
                        key: df.to_dict() if isinstance(df, pd.DataFrame) else df
                        for key, df in type_data.items()
                    }
                    for data_type, type_data in aligned_experiment.items()
                }
            }
            np.save(npy_path, npy_data, allow_pickle=True)
            file_size = os.path.getsize(npy_path) / (1024 * 1024)  # MB
            print(f"  保存NPY (单文件): {npy_path}, 大小: {file_size:.2f} MB")
        except Exception as e:
            print(f'对齐 experiment {experiment_name} 失败: {e}')
            continue
    
    # CSV 生成循环 - 为每个 experiment 生成
    for exp_name, exp_data in aligned_data.items():
        # 整合Biopac数据为单文件CSV
        biopac_data = exp_data['biopac']
        if biopac_data:
            # 安全地获取参考时间戳
            ref_timestamps = None
            hub_sensor2 = exp_data['hub'].get('sensor2', pd.DataFrame())
            
            # 检查hub_sensor2是否有效
            if not hub_sensor2.empty and 'timestamp' in hub_sensor2.columns:
                ref_timestamps = hub_sensor2['timestamp'].values
                print(f"      使用HUB sensor2作为参考时间戳: {len(ref_timestamps):,} 点")
            
            # 如果没有有效的hub_sensor2，尝试使用biopac数据
            if ref_timestamps is None or len(ref_timestamps) == 0:
                for key, df in biopac_data.items():
                    if isinstance(df, pd.DataFrame) and not df.empty and 'timestamp' in df.columns:
                        ref_timestamps = df['timestamp'].values
                        print(f"      使用Biopac {key}作为参考时间戳: {len(ref_timestamps):,} 点")
                        break
            
            # 如果仍然没有有效的时间戳，跳过这个实验
            if ref_timestamps is None or len(ref_timestamps) == 0:
                print(f"      警告: 无法找到有效的时间戳参考，跳过实验 {exp_name}")
                continue
            
            merged_biopac = pd.DataFrame({'timestamp': ref_timestamps})
            
            # 处理moved血压数据
            if 'sbp' in biopac_data and 'dbp' in biopac_data:
                # 对齐SBP和DBP数据到参考时间戳
                sbp_data = biopac_data['sbp']
                dbp_data = biopac_data['dbp']
                
                # 使用插值对齐SBP和DBP
                if not sbp_data.empty and not dbp_data.empty:
                    aligned_sbp = interpolate_with_reftime(
                        sbp_data['timestamp'].values,
                        sbp_data[['sbp']].values,
                        ref_timestamps
                    )
                    aligned_dbp = interpolate_with_reftime(
                        dbp_data['timestamp'].values,
                        dbp_data[['dbp']].values,
                        ref_timestamps
                    )
                    
                    if not aligned_sbp.empty and not aligned_dbp.empty:
                        merged_biopac['moved_sbp'] = aligned_sbp.iloc[:, 0]  # 第一列是sbp值
                        merged_biopac['moved_dbp'] = aligned_dbp.iloc[:, 0]  # 第一列是dbp值
                        print(f"      对齐血压数据: SBP {len(aligned_sbp):,} 点, DBP {len(aligned_dbp):,} 点")
            
            # 处理其他biopac数据（原始biopac数据）
            for key, df in biopac_data.items():
                if key not in ['sbp', 'dbp', 'waveform'] and isinstance(df, pd.DataFrame) and not df.empty:
                    # 对于原始biopac数据，需要先对齐到参考时间戳
                    if 'timestamp' in df.columns:
                        # 提取需要插值的列（除了timestamp）
                        data_columns = [col for col in df.columns if col != 'timestamp']
                        if data_columns:
                            # 使用插值对齐原始biopac数据
                            aligned_biopac = interpolate_with_reftime(
                                df['timestamp'].values,
                                df[data_columns].values,
                                ref_timestamps
                            )
                            
                            if not aligned_biopac.empty:
                                # 将对齐后的数据添加到合并的DataFrame中
                                for i, col in enumerate(data_columns):
                                    merged_biopac[f'biopac_{key}_{col}'] = aligned_biopac.iloc[:, i]
                                print(f"      对齐原始biopac数据 {key}: {len(df):,} -> {len(aligned_biopac):,} 行")
                    else:
                        # 如果没有timestamp列，直接合并
                        merged_biopac = merged_biopac.merge(df, on='timestamp', how='left')
            
            # 填充缺失值
            merged_biopac = merged_biopac.fillna(method='ffill').fillna(method='bfill')
            biopac_csv_path = os.path.join(csv_dir, f'{subject}_{exp_name}_calibrated_bp_aligned.csv')
            merged_biopac.to_csv(biopac_csv_path, index=False)
            print(f'  保存整合校准血压CSV: {biopac_csv_path}')
            
            # 保存单独的校准血压数据CSV，方便人工检查
            if 'sbp' in biopac_data and 'dbp' in biopac_data:
                bp_calibrated_df = pd.DataFrame({
                    'timestamp': ref_timestamps,
                    'calibrated_sbp': merged_biopac['moved_sbp'],
                    'calibrated_dbp': merged_biopac['moved_dbp']
                })
                bp_calibrated_csv_path = os.path.join(csv_dir, f'{subject}_{exp_name}_biopac_calibrated_aligned.csv')
                bp_calibrated_df.to_csv(bp_calibrated_csv_path, index=False)
                print(f'  保存单独校准血压CSV: {bp_calibrated_csv_path}')
        
        # 保存HUB数据为独立CSV文件
        for key, df in exp_data['hub'].items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                columns = ['timestamp'] + [col for col in df.columns if col != 'timestamp']
                df_reordered = df[columns]
                hub_csv_path = os.path.join(csv_dir, f'{subject}_{exp_name}_hub_{key}_aligned.csv')
                df_reordered.to_csv(hub_csv_path, index=False)
                print(f'  保存HUB CSV: {hub_csv_path}')
    
    return aligned_data

def process_subject(date_folder, subject):
    dataset_root = '/root/shared/PhysioNet2025/'  # 添加常量定义
    MAX_EXPERIMENTS = None  # 添加常量定义
    TARGET_FREQ = 100  # 添加常量定义
    date_path = os.path.join(dataset_root, date_folder)
    subject_path = os.path.join(date_path, subject)
    output_dir = os.path.join('/root/autodl-tmp/', subject, 'output_calibrated')
    csv_dir = os.path.join('/root/autodl-tmp/', subject, 'csv_output_calibrated')
    
    # 检查是否已经处理过
    if os.path.exists(output_dir) and os.path.exists(csv_dir):
        # 检查是否有完整的输出文件
        pkl_files = [f for f in os.listdir(output_dir) if f.endswith('.pkl')]
        csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
        
        if pkl_files and csv_files:
            print(f"⏭️  {subject}: 已处理完成，跳过 (PKL: {len(pkl_files)}个, CSV: {len(csv_files)}个)")
            return "SKIPPED_ALREADY_PROCESSED"
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    
    print(f"\n处理subject: {subject} 在 {date_folder} (使用校准血压数据)")
    
    # 加载校准血压数据
    print(f"加载PhysioNet2025_Calibrated文件夹中的校准血压数据...")
    calibrated_bp_data = load_calibrated_bp_data(subject)
    
    if not calibrated_bp_data:
        print(f"  警告: subject {subject} 在PhysioNet2025_Calibrated文件夹中没有找到校准血压数据")
        return None
    
    # 获取所有可用的实验
    all_folders = os.listdir(subject_path)
    experiment_folders = [f for f in all_folders if f.isdigit() and os.path.isdir(os.path.join(subject_path, f))]
    experiment_folders.sort(key=lambda x: int(x))
    
    # 只处理有校准数据的实验
    available_calibrated_experiments = list(calibrated_bp_data.keys())
    experiment_folders = [f for f in experiment_folders if f in available_calibrated_experiments]
    
    if MAX_EXPERIMENTS:
        experiment_folders = experiment_folders[:MAX_EXPERIMENTS]
    
    print(f"发现可用实验: {experiment_folders}")
    print(f"有校准数据的实验: {available_calibrated_experiments}")
    
    all_data = {}
    load_start = time.time()
    for experiment in experiment_folders:
        experiment_path = os.path.join(subject_path, experiment)
        experiment_data = load_experiment_smart(experiment_path, calibrated_bp_data, TARGET_FREQ)
        all_data[experiment] = experiment_data
    load_time = time.time() - load_start
    
    align_start = time.time()
    aligned_data = align_data_with_interpolation(all_data, calibrated_bp_data, output_dir, csv_dir, subject)
    align_time = time.time() - align_start
    
    # 统计
    total_time = time.time() - load_start  # 使用 load_start 作为起点
    print(f"\n{'='*60}")
    print("📊 处理完成统计")
    print(f"{'='*60}")
    print(f"成功处理实验: {len(aligned_data)}")
    print(f"数据加载耗时: {load_time:.1f} 秒")
    print(f"数据对齐耗时: {align_time:.1f} 秒")
    print(f"总处理耗时: {total_time:.1f} 秒")
    if len(experiment_folders) > 0:
        print(f"平均每个实验: {total_time/len(experiment_folders):.1f} 秒")
    
    for exp_name, exp_data in aligned_data.items():
        total_biopac = len(exp_data['biopac'])
        total_hub = len(exp_data['hub'])
        print(f"实验 {exp_name}: {total_biopac} 个Biopac + {total_hub} 个HUB文件")
    
    print(f"\n✅ 处理完成！")
    print(f"PKL格式: {output_dir}/experiment_*_aligned.pkl")
    print(f"NPY格式: {output_dir}/experiment_*_aligned.npy")
    print(f"CSV格式: {csv_dir}/*_calibrated_bp_aligned.csv")
    return "SUCCESS"  # 成功返回状态

def main():
    """主函数"""
    start_time = time.time()
    
    dataset_root = '/root/shared/PhysioNet2025/'
    
    # 设置参数
    TARGET_FREQ = 100  # 目标频率
    MAX_EXPERIMENTS = None  # None表示处理所有实验
    
    # 获取所有日期文件夹
    date_folders = [f for f in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, f)) and f.startswith('20')]
    date_folders.sort()
    
    # 收集所有需要处理的subjects
    all_subjects_to_process = []
    for date_folder in date_folders:
        date_path = os.path.join(dataset_root, date_folder)
        all_subject_folders = [f for f in os.listdir(date_path) if os.path.isdir(os.path.join(date_path, f)) and f.startswith('00')]
        all_subject_folders.sort()
        
        for subject in all_subject_folders:
            # 检查PhysioNet2025_Calibrated文件夹中是否存在该subject的数据
            calibrated_subject_path = os.path.join('/root/shared/PhysioNet2025_Calibrated/', subject)
            if os.path.exists(calibrated_subject_path):
                # 检查是否有校准血压数据
                available_experiments = [d for d in os.listdir(calibrated_subject_path) 
                                       if os.path.isdir(os.path.join(calibrated_subject_path, d)) and d.isdigit()]
                if available_experiments:
                    all_subjects_to_process.append((date_folder, subject))
    
    print(f"找到 {len(all_subjects_to_process)} 个subjects需要处理")
    print("Subjects:", [f"{date}/{subject}" for date, subject in all_subjects_to_process])
    
    # 使用8核并行处理
    with multiprocessing.Pool(processes=8) as pool:
        results = pool.starmap(process_subject, all_subjects_to_process)
    
    # 统计成功、失败和跳过的数量
    successful = [r for r in results if r == "SUCCESS"]
    skipped = [r for r in results if r == "SKIPPED_ALREADY_PROCESSED"]
    failed = [r for r in results if r not in ["SUCCESS", "SKIPPED_ALREADY_PROCESSED"]]
    
    print(f"\n{'='*60}")
    print("📊 并行处理完成统计")
    print(f"{'='*60}")
    print(f"总subjects数量: {len(all_subjects_to_process)}")
    print(f"成功处理: {len(successful)}")
    print(f"跳过已处理: {len(skipped)}")
    print(f"处理失败: {len(failed)}")
    
    if failed:
        print(f"\n失败的subjects:")
        for error in failed:
            print(f"  - {error}")
    
    if skipped:
        print(f"\n跳过的subjects (已处理完成):")
        print(f"  共跳过 {len(skipped)} 个subjects")
    
    # 最终统计
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print("📊 总处理完成统计")
    print(f"{'='*60}")
    print(f"总处理耗时: {total_time:.1f} 秒")
    print(f"使用8核并行处理，处理所有有校准数据的实验")
    print(f"使用PhysioNet2025_Calibrated文件夹中的校准血压数据")

if __name__ == "__main__":
    main()
