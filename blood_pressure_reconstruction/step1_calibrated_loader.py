#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于step1_preprocess.py的校准数据加载器
完全照搬step1逻辑，只是BP数据源换成校准后的数据集
实现分层存储：csv、pkl、npy、processing_logs、summary_reports
支持8核并行处理
"""

import os
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from scipy.interpolate import interp1d
import warnings
import time
import multiprocessing
import warnings
from multiprocessing import Manager, Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading
warnings.filterwarnings('ignore')

# 全局配置
N_CORES = 2  # 并行核心数
MAX_WORKERS = min(N_CORES, cpu_count())  # 实际使用的核心数

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

def load_calibrated_bp_data(subject_id, experiment_number):
    """加载校准后的血压数据"""
    calibrated_dir = "/root/shared/PhysioNet2025_Calibrated"
    bp_file = os.path.join(calibrated_dir, subject_id, str(experiment_number), "bp.csv")
    
    if not os.path.exists(bp_file):
        print(f"  校准血压文件不存在: {bp_file}")
        return None, "calibrated_not_found"
    
    try:
        # 读取血压数据（第一列时间戳，第二列血压值）
        bp_data = pd.read_csv(bp_file, header=None, names=['abp', 'timestamp'])
        # 重新排列列顺序：时间戳在前，血压值在后
        bp_data = bp_data[['timestamp', 'abp']]
        print(f"  加载校准血压数据: {len(bp_data)} 行")
        return bp_data, "calibrated"
    except Exception as e:
        print(f"  加载校准血压数据失败: {e}")
        return None, "calibrated_error"

def load_original_bp_data(experiment_path, subject_id, experiment_number):
    """加载原始数据集中的BP数据"""
    biopac_path = os.path.join(experiment_path, 'Biopac')
    if not os.path.exists(biopac_path):
        print(f"  原始Biopac目录不存在: {biopac_path}")
        return None, "original_not_found"
    
    # 查找bp相关的文件
    bp_files = []
    for file in os.listdir(biopac_path):
        if file.endswith('.csv') and ('bp' in file.lower() or 'blood' in file.lower()):
            bp_files.append(file)
    
    if not bp_files:
        print(f"  原始Biopac目录中未找到BP相关文件")
        return None, "original_no_bp_files"
    
    # 优先选择bp.csv，然后是其他包含bp的文件
    bp_file = None
    for file in bp_files:
        if file.lower() == 'bp.csv':
            bp_file = file
            break
    if bp_file is None:
        bp_file = bp_files[0]  # 选择第一个找到的BP文件
    
    try:
        file_path = os.path.join(biopac_path, bp_file)
        data = pd.read_csv(file_path)
        
        # 检查列名，找到血压列
        bp_column = None
        for col in data.columns:
            if 'bp' in col.lower() or 'blood' in col.lower():
                bp_column = col
                break
        
        if bp_column is None:
            print(f"  在{file_path}中未找到血压列")
            return None, "original_no_bp_column"
        
        # 创建标准格式的血压数据
        bp_data = pd.DataFrame({
            'timestamp': data['timestamp'],
            'abp': data[bp_column]
        })
        
        print(f"  加载原始血压数据: {file_path} ({len(bp_data)} 行, 列: {bp_column})")
        return bp_data, "original"
        
    except Exception as e:
        print(f"  加载原始血压数据失败: {e}")
        return None, "original_error"

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

def load_experiment_smart(experiment_path, target_freq=100):
    """智能加载单个实验"""
    experiment_name = os.path.basename(experiment_path)
    print(f"\n{'='*50}")
    print(f"处理实验 {experiment_name}")
    print(f"{'='*50}")
    
    result = {'biopac': {}, 'hub': {}}
    
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

def load_experiment_with_calibrated_bp(experiment_path, subject_id, experiment_number, target_freq=100):
    """智能加载单个实验，优先使用校准后的血压数据，回退到原始数据"""
    experiment_name = os.path.basename(experiment_path)
    print(f"\n{'='*50}")
    print(f"处理实验 {experiment_name} (优先校准血压数据)")
    print(f"{'='*50}")
    
    result = {'biopac': {}, 'hub': {}}
    bp_source = "none"
    bp_status = "failed"
    
    # 1. 优先尝试加载校准后的血压数据
    bp_data, calibrated_status = load_calibrated_bp_data(subject_id, experiment_number)
    if bp_data is not None:
        # 处理校准血压数据（降采样和插值）
        bp_processed = process_biopac_file_dataframe(bp_data, target_freq)
        if bp_processed is not None and not bp_processed.empty:
            result['biopac']['bp'] = bp_processed
            bp_source = "calibrated"
            bp_status = "success"
            print(f"  ✅ 使用校准血压数据: {len(bp_processed)} 行")
        else:
            print(f"  ❌ 校准血压数据处理失败")
            bp_status = "processing_failed"
    else:
        print(f"  ⚠️  校准血压数据不可用: {calibrated_status}")
        
        # 2. 回退到原始数据集中的血压数据
        print(f"  🔄 尝试加载原始血压数据...")
        bp_data, original_status = load_original_bp_data(experiment_path, subject_id, experiment_number)
        if bp_data is not None:
            # 处理原始血压数据（降采样和插值）
            bp_processed = process_biopac_file_dataframe(bp_data, target_freq)
            if bp_processed is not None and not bp_processed.empty:
                result['biopac']['bp'] = bp_processed
                bp_source = "original"
                bp_status = "success"
                print(f"  ✅ 使用原始血压数据: {len(bp_processed)} 行")
            else:
                print(f"  ❌ 原始血压数据处理失败")
                bp_status = "processing_failed"
        else:
            print(f"  ❌ 原始血压数据也不可用: {original_status}")
            bp_status = "all_failed"
    
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
    
    # 返回结果和BP数据源信息
    return result, bp_source, bp_status

def process_biopac_file_dataframe(df, target_freq=100):
    """处理DataFrame格式的Biopac数据"""
    if df.empty:
        return df
    
    original_rows = len(df)
    
    # 1. 先降采样到目标频率
    df = quick_downsample_biopac(df, target_freq=target_freq)
    
    # 2. 再用插值处理重复时间戳
    df = interpolate_duplicate_timestamps(df)
    
    # 3. 排序
    if df is not None and not df.empty and 'timestamp' in df.columns:
        df = df.sort_values('timestamp').reset_index(drop=True)
    
    final_rows = len(df) if df is not None else 0
    compression_ratio = original_rows / final_rows if final_rows > 0 else 1
    print(f"      校准血压数据处理: {original_rows:,} -> {final_rows:,} 行 (压缩比: {compression_ratio:.1f}:1)")
    
    return df if df is not None else pd.DataFrame()

def align_data_with_interpolation(data_dict, output_base_dir, subject):
    """使用插值进行精确数据对齐，并分层存储"""
    aligned_data = {}
    
    # 创建分层存储目录结构
    csv_dir = os.path.join(output_base_dir, 'csv')
    pkl_dir = os.path.join(output_base_dir, 'pkl')
    npy_dir = os.path.join(output_base_dir, 'npy')
    
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(pkl_dir, exist_ok=True)
    os.makedirs(npy_dir, exist_ok=True)
    
    print(f"\n{'='*50}")
    print("插值对齐阶段 - 分层存储")
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
            
            # 分层存储数据
            save_layered_data(experiment_name, aligned_experiment, csv_dir, pkl_dir, npy_dir, subject)
            
        except Exception as e:
            print(f'对齐 experiment {experiment_name} 失败: {e}')
            continue
    
    return aligned_data

def save_layered_data(experiment_name, aligned_experiment, csv_dir, pkl_dir, npy_dir, subject):
    """分层保存数据到不同格式"""
    print(f"  分层存储实验 {experiment_name}...")
    
    # 1. 保存CSV格式（主要使用）
    save_csv_data(experiment_name, aligned_experiment, csv_dir, subject)
    
    # 2. 保存PKL格式（快速加载）
    save_pkl_data(experiment_name, aligned_experiment, pkl_dir, subject)
    
    # 3. 保存NPY格式（数值计算）
    save_npy_data(experiment_name, aligned_experiment, npy_dir, subject)

def save_csv_data(experiment_name, aligned_experiment, csv_dir, subject):
    """保存CSV格式数据"""
    print(f"    保存CSV格式...")
    
    # 保存HUB数据为独立CSV文件
    for key, df in aligned_experiment['hub'].items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            columns = ['timestamp'] + [col for col in df.columns if col != 'timestamp']
            df_reordered = df[columns]
            hub_csv_path = os.path.join(csv_dir, f'{subject}_{experiment_name}_{key}.csv')
            df_reordered.to_csv(hub_csv_path, index=False)
            print(f"      保存HUB CSV: {hub_csv_path}")
    
    # 保存血压数据CSV
    if 'bp' in aligned_experiment['biopac']:
        bp_df = aligned_experiment['biopac']['bp']
        if isinstance(bp_df, pd.DataFrame) and not bp_df.empty:
            # 确保时间戳在第一列
            columns = ['timestamp'] + [col for col in bp_df.columns if col != 'timestamp']
            bp_df_reordered = bp_df[columns]
            bp_csv_path = os.path.join(csv_dir, f'{subject}_{experiment_name}_abp.csv')
            bp_df_reordered.to_csv(bp_csv_path, index=False)
            print(f"      保存血压CSV: {bp_csv_path}")

def save_pkl_data(experiment_name, aligned_experiment, pkl_dir, subject):
    """保存PKL格式数据"""
    print(f"    保存PKL格式...")
    
    # 保存每个传感器/数据类型的独立PKL文件
    for data_type in ['hub', 'biopac']:
        for key, df in aligned_experiment[data_type].items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                # 特殊处理血压数据，改名为abp
                if key == 'bp':
                    pkl_path = os.path.join(pkl_dir, f'{subject}_{experiment_name}_abp.pkl')
                else:
                    pkl_path = os.path.join(pkl_dir, f'{subject}_{experiment_name}_{key}.pkl')
                with open(pkl_path, 'wb') as f:
                    pickle.dump(df, f)
                print(f"      保存PKL: {pkl_path}")

def save_npy_data(experiment_name, aligned_experiment, npy_dir, subject):
    """保存NPY格式数据"""
    print(f"    保存NPY格式...")
    
    # 保存每个传感器/数据类型的独立NPY文件
    for data_type in ['hub', 'biopac']:
        for key, df in aligned_experiment[data_type].items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                # 只保存数值列，不保存timestamp
                numeric_columns = [col for col in df.columns if col != 'timestamp']
                if numeric_columns:
                    numeric_data = df[numeric_columns].values
                    # 特殊处理血压数据，改名为abp
                    if key == 'bp':
                        npy_path = os.path.join(npy_dir, f'{subject}_{experiment_name}_abp.npy')
                    else:
                        npy_path = os.path.join(npy_dir, f'{subject}_{experiment_name}_{key}.npy')
                    np.save(npy_path, numeric_data)
                    print(f"      保存NPY: {npy_path}")

def process_experiment_parallel(args):
    """并行处理单个实验的函数"""
    experiment_path, subject_id, experiment_number, target_freq = args
    
    try:
        # 加载实验数据
        experiment_data, bp_source, bp_status = load_experiment_with_calibrated_bp(
            experiment_path, subject_id, experiment_number, target_freq
        )
        
        return {
            'experiment': experiment_number,
            'data': experiment_data,
            'bp_source': bp_source,
            'bp_status': bp_status,
            'success': True
        }
    except Exception as e:
        return {
            'experiment': experiment_number,
            'data': None,
            'bp_source': 'none',
            'bp_status': 'error',
            'success': False,
            'error': str(e)
        }

def process_subject_parallel(date_folder, subject):
    """并行处理单个受试者的所有实验"""
    dataset_root = '/root/shared/PhysioNet2025/'
    MAX_EXPERIMENTS = None
    TARGET_FREQ = 100
    
    date_path = os.path.join(dataset_root, date_folder)
    subject_path = os.path.join(date_path, subject)
    
    # 新的输出目录结构
    output_base_dir = os.path.join('/root/autodl-tmp/blood_pressure_reconstruction', subject)
    
    # 检查是否已经处理过（跳过已处理的受试者）
    status_file = os.path.join(output_base_dir, 'processing_logs', 'step1_calibrated_succ.txt')
    bp_report_file = os.path.join(output_base_dir, 'processing_logs', f'{subject}_bp_source_report.txt')
    csv_dir = os.path.join(output_base_dir, 'csv')
    
    # 检查多个指标来判断是否已处理
    already_processed = False
    skip_reason = ""
    
    # 1. 检查成功状态文件
    if os.path.exists(status_file):
        try:
            with open(status_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 检查状态是否为SUCCESS
            if 'Status: SUCCESS' in content:
                already_processed = True
                skip_reason = "step1_calibrated_succ.txt exists with SUCCESS status"
        except Exception as e:
            print(f"⚠️  读取状态文件失败，继续检查其他指标: {e}")
    
    # 2. 检查BP源报告文件
    if not already_processed and os.path.exists(bp_report_file):
        try:
            with open(bp_report_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 如果BP源报告存在，说明已经处理过
            already_processed = True
            skip_reason = "bp_source_report.txt exists"
        except Exception as e:
            print(f"⚠️  读取BP源报告失败，继续检查其他指标: {e}")
    
    # 3. 检查CSV目录和文件
    if not already_processed and os.path.exists(csv_dir):
        csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
        if len(csv_files) > 0:
            # 如果有CSV文件，说明已经处理过
            already_processed = True
            skip_reason = f"CSV directory exists with {len(csv_files)} files"
    
    # 如果已经处理过，则跳过
    if already_processed:
        print(f"\n{'='*60}")
        print(f"⏭️  跳过已处理的受试者: {subject}")
        print(f"📁 输出目录: {output_base_dir}")
        print(f"🔍 跳过原因: {skip_reason}")
        print(f"✅ 状态: 已完成")
        print(f"{'='*60}")
        
        # 返回跳过状态
        return {
            'subject': subject,
            'status': 'skipped',
            'total_experiments': 0,
            'successful': 0,
            'failed': 0,
            'parallel_time': 0,
            'align_time': 0,
            'total_time': 0,
            'success_rate': 100.0,
            'skip_reason': skip_reason
        }
    
    # 创建主目录和子目录
    os.makedirs(output_base_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"🚀 并行处理受试者: {subject} 在 {date_folder}")
    print(f"📁 输出目录: {output_base_dir}")
    print(f"⚡ 并行核心数: {MAX_WORKERS}")
    print(f"{'='*60}")
    
    all_folders = os.listdir(subject_path)
    experiment_folders = [f for f in all_folders if f.isdigit() and os.path.isdir(os.path.join(subject_path, f))]
    experiment_folders.sort(key=lambda x: int(x))
    
    if MAX_EXPERIMENTS:
        experiment_folders = experiment_folders[:MAX_EXPERIMENTS]
    
    print(f"📋 发现实验: {experiment_folders}")
    print(f"🔢 总实验数: {len(experiment_folders)}")
    
    # 准备并行处理参数
    parallel_args = []
    for experiment in experiment_folders:
        experiment_path = os.path.join(subject_path, experiment)
        parallel_args.append((experiment_path, subject, experiment, TARGET_FREQ))
    
    # 并行处理所有实验
    print(f"\n⚡ 开始并行处理 {len(experiment_folders)} 个实验...")
    parallel_start = time.time()
    
    all_data = {}
    bp_source_info = {}
    successful_experiments = 0
    failed_experiments = 0
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务
        future_to_experiment = {executor.submit(process_experiment_parallel, args): args for args in parallel_args}
        
        # 使用tqdm显示进度
        with tqdm(total=len(parallel_args), desc="并行处理实验", unit="实验") as pbar:
            for future in as_completed(future_to_experiment):
                result = future.result()
                experiment_num = result['experiment']
                
                if result['success']:
                    all_data[experiment_num] = result['data']
                    bp_source_info[experiment_num] = {
                        'source': result['bp_source'],
                        'status': result['bp_status']
                    }
                    successful_experiments += 1
                    pbar.set_postfix({
                        '成功': successful_experiments,
                        '失败': failed_experiments,
                        '当前': f"实验{experiment_num}"
                    })
                else:
                    failed_experiments += 1
                    print(f"❌ 实验 {experiment_num} 处理失败: {result.get('error', '未知错误')}")
                    pbar.set_postfix({
                        '成功': successful_experiments,
                        '失败': failed_experiments,
                        '当前': f"实验{experiment_num}"
                    })
                
                pbar.update(1)
    
    parallel_time = time.time() - parallel_start
    
    # 数据对齐阶段
    print(f"\n🔄 开始数据对齐阶段...")
    align_start = time.time()
    aligned_data = align_data_with_interpolation(all_data, output_base_dir, subject)
    align_time = time.time() - align_start
    
    # 统计
    total_time = time.time() - parallel_start
    print(f"\n{'='*60}")
    print("📊 并行处理完成统计")
    print(f"{'='*60}")
    print(f"✅ 成功处理实验: {successful_experiments}")
    print(f"❌ 失败实验: {failed_experiments}")
    print(f"📈 成功率: {successful_experiments/(successful_experiments+failed_experiments)*100:.1f}%")
    print(f"⚡ 并行处理耗时: {parallel_time:.1f} 秒")
    print(f"🔄 数据对齐耗时: {align_time:.1f} 秒")
    print(f"⏱️  总处理耗时: {total_time:.1f} 秒")
    if successful_experiments > 0:
        print(f"📊 平均每个实验: {total_time/successful_experiments:.1f} 秒")
        print(f"🚀 并行加速比: {len(experiment_folders)*total_time/successful_experiments/total_time:.1f}x")
    
    for exp_name, exp_data in aligned_data.items():
        total_biopac = len(exp_data['biopac'])
        total_hub = len(exp_data['hub'])
        print(f"实验 {exp_name}: {total_biopac} 个Biopac + {total_hub} 个HUB文件")
    
    print(f"\n✅ 并行处理完成！")
    print(f"📁 输出目录结构:")
    print(f"   CSV格式: {output_base_dir}/csv/")
    print(f"   PKL格式: {output_base_dir}/pkl/")
    print(f"   NPY格式: {output_base_dir}/npy/")
    
    # 生成BP数据源记录
    generate_bp_source_report(subject, bp_source_info, output_base_dir)
    
    # 生成成功状态文件
    create_success_status_file(subject, output_base_dir, len(experiment_folders), successful_experiments, failed_experiments, total_time)
    
    return {
        'subject': subject,
        'status': 'success',
        'total_experiments': len(experiment_folders),
        'successful': successful_experiments,
        'failed': failed_experiments,
        'parallel_time': parallel_time,
        'align_time': align_time,
        'total_time': total_time,
        'success_rate': successful_experiments/(successful_experiments+failed_experiments)*100
    }

def process_subject(date_folder, subject):
    """兼容性函数，调用并行版本"""
    return process_subject_parallel(date_folder, subject)

def generate_bp_source_report(subject_id, bp_source_info, output_base_dir):
    """生成BP数据源使用记录"""
    # 创建processing_logs目录
    logs_dir = os.path.join(output_base_dir, 'processing_logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    report_file = os.path.join(logs_dir, f'{subject_id}_bp_source_report.txt')
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"受试者 {subject_id} BP数据源使用记录\n")
        f.write("="*50 + "\n\n")
        f.write("处理策略: 优先使用校准血压数据，回退到原始数据\n")
        f.write(f"并行核心数: {MAX_WORKERS}\n\n")
        
        # 统计信息
        total_experiments = len(bp_source_info)
        calibrated_count = sum(1 for info in bp_source_info.values() if info['source'] == 'calibrated')
        original_count = sum(1 for info in bp_source_info.values() if info['source'] == 'original')
        failed_count = sum(1 for info in bp_source_info.values() if info['status'] == 'all_failed')
        
        f.write(f"总实验数: {total_experiments}\n")
        f.write(f"使用校准数据: {calibrated_count}\n")
        f.write(f"使用原始数据: {original_count}\n")
        f.write(f"完全失败: {failed_count}\n\n")
        
        # 详细记录
        f.write("详细记录:\n")
        f.write("-"*30 + "\n")
        for exp_num, info in sorted(bp_source_info.items()):
            status_emoji = {
                'success': '✅',
                'processing_failed': '❌',
                'all_failed': '💥'
            }.get(info['status'], '❓')
            
            source_desc = {
                'calibrated': '校准血压数据',
                'original': '原始血压数据',
                'none': '无数据'
            }.get(info['source'], '未知')
            
            f.write(f"实验 {exp_num}: {status_emoji} {source_desc} ({info['status']})\n")
        
        f.write(f"\n报告生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"BP数据源记录已保存到: {report_file}")

def create_success_status_file(subject_id, output_base_dir, total_experiments, successful_experiments, failed_experiments, total_time):
    """生成成功处理的状态文件"""
    # 创建processing_logs目录
    logs_dir = os.path.join(output_base_dir, 'processing_logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    status_file = os.path.join(logs_dir, 'step1_calibrated_succ.txt')
    
    with open(status_file, 'w', encoding='utf-8') as f:
        f.write(f"Step1 Calibrated 处理成功状态\n")
        f.write("="*50 + "\n\n")
        f.write(f"Subject ID: {subject_id}\n")
        f.write(f"Status: SUCCESS\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Output Directory: {output_base_dir}\n\n")
        
        f.write(f"Processing Summary:\n")
        f.write(f"- Total Experiments: {total_experiments}\n")
        f.write(f"- Successful: {successful_experiments}\n")
        f.write(f"- Failed: {failed_experiments}\n")
        f.write(f"- Success Rate: {successful_experiments/(successful_experiments+failed_experiments)*100:.1f}%\n")
        f.write(f"- Total Processing Time: {total_time:.1f} seconds\n\n")
        
        f.write(f"Output Structure:\n")
        f.write(f"- CSV files: {output_base_dir}/csv/\n")
        f.write(f"- PKL files: {output_base_dir}/pkl/\n")
        f.write(f"- NPY files: {output_base_dir}/npy/\n")
        f.write(f"- Processing logs: {output_base_dir}/processing_logs/\n\n")
        
        f.write(f"Note: This subject has been successfully processed.\n")
        f.write(f"Future runs will skip this subject to avoid reprocessing.\n")
    
    print(f"✅ 成功状态文件已生成: {status_file}")

def main():
    """主函数 - 并行处理版本"""
    start_time = time.time()
    
    dataset_root = '/root/shared/PhysioNet2025/'
    
    # 设置参数
    TARGET_FREQ = 100  # 目标频率
    MAX_EXPERIMENTS = None  # None表示处理所有实验
    
    # 处理所有受试者
    print(f"\n{'='*80}")
    print("🚀 PPG-ABP重构数据并行处理系统")
    print(f"{'='*80}")
    print(f"⚡ 并行核心数: {MAX_WORKERS}")
    print(f"🎯 目标: 处理所有受试者")
    print(f"📊 目标频率: {TARGET_FREQ}Hz")
    print(f"📁 输出目录: /root/autodl-tmp/blood_pressure_reconstruction/")
    print(f"{'='*80}")
    
    # 获取所有日期文件夹
    date_folders = [f for f in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, f)) and f.startswith('20')]
    date_folders.sort()
    
    results = []  # 用于收集结果
    total_subjects = 0
    total_experiments = 0
    total_successful = 0
    total_failed = 0
    all_subjects = []  # 记录所有找到的受试者
    
    # 首先收集所有可用的受试者
    print(f"\n🔍 扫描所有可用的受试者...")
    for date_folder in date_folders:
        date_path = os.path.join(dataset_root, date_folder)
        all_subject_folders = [f for f in os.listdir(date_path) if os.path.isdir(os.path.join(date_path, f)) and f.startswith('00')]
        all_subjects.extend(all_subject_folders)
    
    # 去重并排序
    all_subjects = sorted(list(set(all_subjects)))
    print(f"📋 发现受试者: {all_subjects}")
    print(f"🔢 总受试者数: {len(all_subjects)}")
    
    # 准备并行处理参数
    parallel_args = []
    for subject in all_subjects:
        # 找到包含该受试者的日期文件夹
        subject_date_folder = None
        for date_folder in date_folders:
            subject_path = os.path.join(dataset_root, date_folder, subject)
            if os.path.exists(subject_path):
                subject_date_folder = date_folder
                break
        
        if subject_date_folder is None:
            print(f"❌ 受试者 {subject} 在所有日期文件夹中都未找到，跳过")
            continue
            
        print(f"📅 在日期文件夹 {subject_date_folder} 中找到受试者 {subject}")
        parallel_args.append((subject_date_folder, subject))
    
    print(f"\n⚡ 开始真正的跨受试者并行处理...")
    print(f"🎯 将并行处理 {len(parallel_args)} 个受试者，每个受试者内部也并行处理实验")
    
    # 真正的跨受试者并行处理
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有受试者任务
        future_to_subject = {executor.submit(process_subject_parallel, date_folder, subject): (date_folder, subject) 
                           for date_folder, subject in parallel_args}
        
        # 使用tqdm显示进度
        with tqdm(total=len(parallel_args), desc="并行处理受试者", unit="受试者") as pbar:
            for future in as_completed(future_to_subject):
                date_folder, subject = future_to_subject[future]
                try:
                    res = future.result()
                    results.append(res)
                    
                    if res and res.get('status') == 'skipped':
                        print(f"⏭️  受试者 {subject} 已跳过（之前已处理完成）")
                    elif res and res.get('status') == 'success':
                        total_subjects += 1
                        total_experiments += res['total_experiments']
                        total_successful += res['successful']
                        total_failed += res['failed']
                    
                    pbar.set_postfix({
                        '成功': total_subjects,
                        '跳过': len([r for r in results if r and r.get('status') == 'skipped']),
                        '实验': total_experiments,
                        '当前': f"{subject}"
                    })
                except Exception as e:
                    print(f"❌ 受试者 {subject} 处理失败: {e}")
                
                pbar.update(1)
    
    # 最终统计
    total_time = time.time() - start_time
    skipped_subjects = len([r for r in results if r and r.get('status') == 'skipped'])
    
    print(f"\n{'='*80}")
    print("📊 总并行处理完成统计")
    print(f"{'='*80}")
    print(f"⏱️  总处理耗时: {total_time:.1f} 秒")
    print(f"👥 处理受试者: {total_subjects}")
    print(f"⏭️  跳过受试者: {skipped_subjects}")
    print(f"🔬 总实验数: {total_experiments}")
    print(f"✅ 成功实验: {total_successful}")
    print(f"❌ 失败实验: {total_failed}")
    if total_experiments > 0:
        print(f"📈 总体成功率: {total_successful/total_experiments*100:.1f}%")
        print(f"🚀 平均每个实验: {total_time/total_experiments:.1f} 秒")
        print(f"🚀 平均每个受试者: {total_time/total_subjects:.1f} 秒")
    
    # 创建summary_reports目录和总报告
    create_summary_report(all_subjects, total_time, results, skipped_subjects)

def create_summary_report(subjects, total_time, results, skipped_subjects):
    """创建总处理报告 - 包含并行处理统计"""
    summary_dir = '/root/autodl-tmp/blood_pressure_reconstruction/summary_reports'
    os.makedirs(summary_dir, exist_ok=True)
    
    report_file = os.path.join(summary_dir, f'parallel_processing_summary_{time.strftime("%Y%m%d_%H%M%S")}.txt')
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("PPG-ABP重构数据并行处理总报告\n")
        f.write("="*60 + "\n\n")
        f.write(f"处理时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总耗时: {total_time:.1f} 秒\n")
        f.write(f"并行核心数: {MAX_WORKERS}\n")
        f.write(f"处理受试者: {', '.join(subjects)}\n\n")
        
        # 总体统计
        total_experiments = sum(r['total_experiments'] for r in results if r and r.get('status') == 'success')
        total_successful = sum(r['successful'] for r in results if r and r.get('status') == 'success')
        total_failed = sum(r['failed'] for r in results if r and r.get('status') == 'success')
        
        f.write("总体统计:\n")
        f.write("-"*30 + "\n")
        f.write(f"总受试者数: {len(subjects)}\n")
        f.write(f"新处理受试者: {len([r for r in results if r and r.get('status') == 'success'])}\n")
        f.write(f"跳过受试者: {skipped_subjects}\n")
        f.write(f"总实验数: {total_experiments}\n")
        f.write(f"成功实验: {total_successful}\n")
        f.write(f"失败实验: {total_failed}\n")
        if total_experiments > 0:
            f.write(f"成功率: {total_successful/total_experiments*100:.1f}%\n")
        
        f.write("\n受试者详细统计:\n")
        f.write("-"*40 + "\n")
        for result in results:
            if result:
                if result.get('status') == 'skipped':
                    f.write(f"受试者 {result['subject']}: ⏭️ 已跳过（之前已处理完成）\n")
                    f.write(f"  跳过原因: {result.get('skip_reason', 'unknown')}\n\n")
                else:
                    f.write(f"受试者 {result['subject']}:\n")
                    f.write(f"  总实验数: {result['total_experiments']}\n")
                    f.write(f"  成功: {result['successful']}\n")
                    f.write(f"  失败: {result['failed']}\n")
                    f.write(f"  成功率: {result['success_rate']:.1f}%\n")
                    f.write(f"  并行处理时间: {result['parallel_time']:.1f}秒\n")
                    f.write(f"  对齐时间: {result['align_time']:.1f}秒\n")
                    f.write(f"  总时间: {result['total_time']:.1f}秒\n\n")
        
        f.write("输出目录结构:\n")
        f.write("-"*30 + "\n")
        for subject in subjects:
            f.write(f"/root/autodl-tmp/blood_pressure_reconstruction/{subject}/\n")
            f.write(f"  ├── csv/          # CSV格式（主要使用）\n")
            f.write(f"  ├── pkl/          # Pickle格式（快速加载）\n")
            f.write(f"  ├── npy/          # Numpy格式（数值计算）\n")
            f.write(f"  └── processing_logs/  # 处理日志\n")
        
        f.write(f"\n跳过机制说明:\n")
        f.write("-"*30 + "\n")
        f.write(f"系统会自动检查每个受试者的状态文件:\n")
        f.write(f"  /root/autodl-tmp/blood_pressure_reconstruction/{{subject}}/processing_logs/step1_calibrated_succ.txt\n")
        f.write(f"如果状态为 'Status: SUCCESS'，则跳过该受试者以避免重复处理。\n")
        f.write(f"本批次跳过了 {skipped_subjects} 个已处理的受试者。\n\n")
        
        f.write(f"总报告位置: {summary_dir}\n")
    
    print(f"并行处理总报告已保存到: {report_file}")
    
    # 创建类似step3的详细统计报告
    create_detailed_status_report(subjects, results, skipped_subjects)

def create_detailed_status_report(subjects, results, skipped_subjects):
    """创建详细的处理状态报告（类似step3）"""
    print("📊 创建step1_calibrated处理状态汇总报告")
    print("="*50)
    
    # 创建专门的检查文件夹
    check_dir = "/root/PI_Lab/step1_calibrated_check_results"
    os.makedirs(check_dir, exist_ok=True)
    print(f"📁 创建检查文件夹: {check_dir}")
    
    # 收集所有状态信息
    all_status = []
    
    for subject in subjects:
        status_file = os.path.join('/root/autodl-tmp/blood_pressure_reconstruction', subject, 'processing_logs', 'step1_calibrated_succ.txt')
        
        if os.path.exists(status_file):
            try:
                with open(status_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 解析状态文件内容
                status_info = {
                    'subject': subject,
                    'status_file_exists': True,
                    'status': 'SUCCESS',
                    'timestamp': 'UNKNOWN',
                    'output_directory': 'UNKNOWN',
                    'total_experiments': 'UNKNOWN',
                    'successful_experiments': 'UNKNOWN',
                    'failed_experiments': 'UNKNOWN',
                    'success_rate': 'UNKNOWN',
                    'processing_time': 'UNKNOWN',
                    'raw_content': content
                }
                
                # 提取状态信息
                lines = content.strip().split('\n')
                for line in lines:
                    if line.startswith('Timestamp:'):
                        status_info['timestamp'] = line.replace('Timestamp:', '').strip()
                    elif line.startswith('Output Directory:'):
                        status_info['output_directory'] = line.replace('Output Directory:', '').strip()
                    elif line.startswith('Total Experiments:'):
                        status_info['total_experiments'] = line.replace('Total Experiments:', '').strip()
                    elif line.startswith('Successful:'):
                        status_info['successful_experiments'] = line.replace('Successful:', '').strip()
                    elif line.startswith('Failed:'):
                        status_info['failed_experiments'] = line.replace('Failed:', '').strip()
                    elif line.startswith('Success Rate:'):
                        status_info['success_rate'] = line.replace('Success Rate:', '').strip()
                    elif line.startswith('Total Processing Time:'):
                        status_info['processing_time'] = line.replace('Total Processing Time:', '').strip()
                
                all_status.append(status_info)
                print(f"✅ {subject}: 状态文件存在")
                
            except Exception as e:
                print(f"❌ {subject}: 读取状态文件失败 - {e}")
                all_status.append({
                    'subject': subject,
                    'status_file_exists': True,
                    'status': 'READ_ERROR',
                    'timestamp': 'UNKNOWN',
                    'output_directory': 'UNKNOWN',
                    'total_experiments': 'UNKNOWN',
                    'successful_experiments': 'UNKNOWN',
                    'failed_experiments': 'UNKNOWN',
                    'success_rate': 'UNKNOWN',
                    'processing_time': 'UNKNOWN',
                    'error_message': f'读取失败: {str(e)}',
                    'raw_content': ''
                })
        else:
            print(f"⚠️  {subject}: 状态文件不存在")
            all_status.append({
                'subject': subject,
                'status_file_exists': False,
                'status': 'NO_FILE',
                'timestamp': 'N/A',
                'output_directory': 'N/A',
                'total_experiments': 'N/A',
                'successful_experiments': 'N/A',
                'failed_experiments': 'N/A',
                'success_rate': 'N/A',
                'processing_time': 'N/A',
                'error_message': '状态文件不存在',
                'raw_content': ''
            })
    
    # 创建汇总DataFrame
    summary_df = pd.DataFrame(all_status)
    
    # 统计结果
    total_subjects = len(summary_df)
    success_count = len(summary_df[summary_df['status'] == 'SUCCESS'])
    no_file_count = len(summary_df[summary_df['status'] == 'NO_FILE'])
    read_error_count = len(summary_df[summary_df['status'] == 'READ_ERROR'])
    
    print(f"\n📊 处理状态统计:")
    print(f"   📋 总受试者数: {total_subjects}")
    print(f"   ✅ 成功: {success_count}")
    print(f"   📁 无状态文件: {no_file_count}")
    print(f"   🔍 读取错误: {read_error_count}")
    print(f"   ⏭️  本批次跳过: {skipped_subjects}")
    
    # 保存详细汇总报告到检查文件夹
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    summary_file = os.path.join(check_dir, f"step1_calibrated_summary_report_{timestamp}.csv")
    summary_df.to_csv(summary_file, index=False, encoding='utf-8')
    print(f"\n💾 详细汇总报告已保存: {summary_file}")
    
    # 创建简化状态报告
    simple_status = []
    for _, row in summary_df.iterrows():
        simple_status.append({
            'subject': row['subject'],
            'status': row['status'],
            'timestamp': row['timestamp'] if row['timestamp'] != 'UNKNOWN' else 'N/A',
            'total_experiments': row['total_experiments'],
            'successful_experiments': row['successful_experiments'],
            'failed_experiments': row['failed_experiments'],
            'success_rate': row['success_rate'],
            'processing_time': row['processing_time'],
            'has_error': 'error_message' in row and row['error_message'] != '',
            'error_summary': row.get('error_message', '')[:100] + '...' if 'error_message' in row and len(row.get('error_message', '')) > 100 else row.get('error_message', '')
        })
    
    simple_df = pd.DataFrame(simple_status)
    simple_file = os.path.join(check_dir, f"step1_calibrated_simple_status_{timestamp}.csv")
    simple_df.to_csv(simple_file, index=False, encoding='utf-8')
    print(f"💾 简化状态报告已保存: {simple_file}")
    
    # 创建成功/失败列表
    success_subjects = summary_df[summary_df['status'] == 'SUCCESS']['subject'].tolist()
    no_file_subjects = summary_df[summary_df['status'] == 'NO_FILE']['subject'].tolist()
    read_error_subjects = summary_df[summary_df['status'] == 'READ_ERROR']['subject'].tolist()
    
    # 保存分类列表到检查文件夹
    with open(os.path.join(check_dir, f"step1_calibrated_success_subjects_{timestamp}.txt"), 'w', encoding='utf-8') as f:
        f.write(f"Step1 Calibrated 成功处理的受试者列表\n")
        f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"成功数量: {len(success_subjects)}\n")
        f.write(f"{'='*50}\n")
        for subject in success_subjects:
            f.write(f"{subject}\n")
    
    if no_file_subjects:
        with open(os.path.join(check_dir, f"step1_calibrated_no_file_subjects_{timestamp}.txt"), 'w', encoding='utf-8') as f:
            f.write(f"Step1 Calibrated 无状态文件的受试者列表\n")
            f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"无文件数量: {len(no_file_subjects)}\n")
            f.write(f"{'='*50}\n")
            for subject in no_file_subjects:
                f.write(f"{subject}\n")
    
    if read_error_subjects:
        with open(os.path.join(check_dir, f"step1_calibrated_read_error_subjects_{timestamp}.txt"), 'w', encoding='utf-8') as f:
            f.write(f"Step1 Calibrated 读取错误的受试者列表\n")
            f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"错误数量: {len(read_error_subjects)}\n")
            f.write(f"{'='*50}\n")
            for subject in read_error_subjects:
                f.write(f"{subject}\n")
    
    # 创建最终汇总报告
    final_summary_file = os.path.join(check_dir, f"step1_calibrated_final_summary_{timestamp}.txt")
    with open(final_summary_file, 'w', encoding='utf-8') as f:
        f.write(f"Step1 Calibrated 处理结果最终汇总\n")
        f.write(f"{'='*50}\n")
        f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总受试者数: {total_subjects}\n")
        f.write(f"本批次跳过: {skipped_subjects}\n\n")
        
        f.write(f"✅ 成功处理: {success_count} 个受试者 ({success_count/total_subjects*100:.1f}%)\n")
        f.write(f"📁 无状态文件: {no_file_count} 个受试者 ({no_file_count/total_subjects*100:.1f}%)\n")
        f.write(f"🔍 读取错误: {read_error_count} 个受试者 ({read_error_count/total_subjects*100:.1f}%)\n\n")
        
        f.write(f"🎉 成功处理的受试者列表 ({success_count}个):\n")
        for i, subject in enumerate(success_subjects, 1):
            f.write(f"{subject}")
            if i % 10 == 0:
                f.write("\n")
            elif i < len(success_subjects):
                f.write(", ")
        f.write("\n\n")
        
        if no_file_subjects:
            f.write(f"📁 无状态文件的受试者列表 ({no_file_count}个):\n")
            for i, subject in enumerate(no_file_subjects, 1):
                f.write(f"{subject}")
                if i % 10 == 0:
                    f.write("\n")
                elif i < len(no_file_subjects):
                    f.write(", ")
            f.write("\n\n")
        
        f.write(f"📊 处理状态说明:\n")
        f.write(f"- SUCCESS: 分析完成，生成了所有相关文件和图表\n")
        f.write(f"- NO_FILE: 状态文件不存在，可能是新受试者\n")
        f.write(f"- READ_ERROR: 状态文件读取失败\n\n")
        
        f.write(f"💡 建议:\n")
        f.write(f"1. 成功处理的{success_count}个受试者可以直接使用结果\n")
        if no_file_count > 0:
            f.write(f"2. 无状态文件的{no_file_count}个受试者需要重新处理\n")
        if read_error_count > 0:
            f.write(f"3. 读取错误的{read_error_count}个受试者需要检查状态文件\n")
        f.write(f"4. 总体成功率{success_count/total_subjects*100:.1f}%，处理效果良好\n")
        f.write(f"5. 本批次跳过了{skipped_subjects}个已处理的受试者，节省了处理时间\n\n")
        
        f.write(f"📁 本批次所有报告文件:\n")
        f.write(f"- step1_calibrated_summary_report_{timestamp}.csv (完整报告)\n")
        f.write(f"- step1_calibrated_simple_status_{timestamp}.csv (简化状态)\n")
        f.write(f"- step1_calibrated_success_subjects_{timestamp}.txt (成功列表)\n")
        if no_file_count > 0:
            f.write(f"- step1_calibrated_no_file_subjects_{timestamp}.txt (无文件列表)\n")
        if read_error_count > 0:
            f.write(f"- step1_calibrated_read_error_subjects_{timestamp}.txt (读取错误列表)\n")
    
    print(f"\n📝 分类列表已保存到 {check_dir} 文件夹:")
    print(f"   ✅ 成功: step1_calibrated_success_subjects_{timestamp}.txt")
    if no_file_count > 0:
        print(f"   📁 无文件: step1_calibrated_no_file_subjects_{timestamp}.txt")
    if read_error_count > 0:
        print(f"   🔍 读取错误: step1_calibrated_read_error_subjects_{timestamp}.txt")
    print(f"   📋 最终汇总: step1_calibrated_final_summary_{timestamp}.txt")
    
    # 显示成功和失败的受试者
    if success_subjects:
        print(f"\n🎉 成功处理的受试者 ({len(success_subjects)}):")
        for i, subject in enumerate(success_subjects, 1):
            print(f"   {i:2d}. {subject}")
    
    if no_file_subjects:
        print(f"\n📁 无状态文件的受试者 ({len(no_file_subjects)}):")
        for i, subject in enumerate(no_file_subjects, 1):
            print(f"   {i:2d}. {subject}")
    
    if read_error_subjects:
        print(f"\n🔍 读取错误的受试者 ({len(read_error_subjects)}):")
        for i, subject in enumerate(read_error_subjects, 1):
            print(f"   {i:2d}. {subject}")
    
    print(f"\n📁 所有报告文件已保存到: {check_dir}/")
    print(f"🎯 检查完成！")
    
    return summary_df

if __name__ == "__main__":
    main()
