#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化版PI-Lab数据预处理脚本
智能处理：Biopac降采样 + HUB插值去重 + 插值对齐 + 双格式保存
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
from multiprocessing import Manager
warnings.filterwarnings('ignore')

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

def align_data_with_interpolation(data_dict, output_dir, csv_dir, subject):
    """使用插值进行精确数据对齐"""
    aligned_data = {}
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*50}")
    print("插值对齐阶段")
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
            ref_timestamps = exp_data['hub'].get('sensor2', pd.DataFrame())['timestamp'].values
            if len(ref_timestamps) == 0:
                ref_timestamps = biopac_data[next(iter(biopac_data))]['timestamp'].values
            
            merged_biopac = pd.DataFrame({'timestamp': ref_timestamps})
            
            for key, df in biopac_data.items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    merged_biopac = merged_biopac.merge(df[['timestamp', key]], on='timestamp', how='left')
            
            merged_biopac = merged_biopac.fillna(method='ffill').fillna(method='bfill')
            biopac_csv_path = os.path.join(csv_dir, f'{subject}_{exp_name}_biopac_aligned.csv')
            merged_biopac.to_csv(biopac_csv_path, index=False)
            print(f'  保存整合Biopac CSV: {biopac_csv_path}')
        
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
    output_dir = os.path.join('/root/autodl-tmp/', subject, 'output')
    csv_dir = os.path.join('/root/autodl-tmp/', subject, 'csv_output')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    
    if os.listdir(output_dir):
        print(f"\nsubject {subject} 已处理（output目录非空），跳过")
        return None  # 成功返回 None
    
    print(f"\n处理subject: {subject} 在 {date_folder}")
    
    all_folders = os.listdir(subject_path)
    experiment_folders = [f for f in all_folders if f.isdigit() and os.path.isdir(os.path.join(subject_path, f))]
    experiment_folders.sort(key=lambda x: int(x))
    
    if MAX_EXPERIMENTS:
        experiment_folders = experiment_folders[:MAX_EXPERIMENTS]
    
    print(f"发现实验: {experiment_folders}")
    
    all_data = {}
    load_start = time.time()
    for experiment in experiment_folders:
        experiment_path = os.path.join(subject_path, experiment)
        experiment_data = load_experiment_smart(experiment_path, TARGET_FREQ)
        all_data[experiment] = experiment_data
    load_time = time.time() - load_start
    
    align_start = time.time()
    aligned_data = align_data_with_interpolation(all_data, output_dir, csv_dir, subject)
    align_time = time.time() - align_start
    
    # 统计
    total_time = time.time() - load_start  # 使用 load_start 作为起点
    print(f"\n{'='*60}")
    print("�� 处理完成统计")
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
    return None  # 成功返回 None

def main():
    """主函数"""
    start_time = time.time()
    manager = Manager()
    failed_subjects = manager.list()
    
    dataset_root = '/root/shared/PhysioNet2025/'
    
    # 设置参数
    TARGET_FREQ = 100  # 目标频率
    MAX_EXPERIMENTS = None  # None表示处理所有实验
    
    # 获取所有日期文件夹
    date_folders = [f for f in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, f)) and f.startswith('20')]
    date_folders.sort()
    
    for date_folder in date_folders:
        date_path = os.path.join(dataset_root, date_folder)
        subject_folders = [f for f in os.listdir(date_path) if os.path.isdir(os.path.join(date_path, f)) and f.startswith('00')]
        subject_folders.sort()
        
        with multiprocessing.Pool() as p:
            results = p.starmap(process_subject, [(date_folder, subject) for subject in subject_folders])
    
    failed_subjects = [res for res in results if res is not None]
    
    if failed_subjects:
        with open('/root/autodl-tmp/failed_subjects.txt', 'w') as f:
            f.write("失败的 subject:\n")
            for fs in failed_subjects:
                f.write(f"{fs}\n")
        print(f"失败 subject 数量: {len(failed_subjects)}, 详情见 /root/autodl-tmp/failed_subjects.txt")
    else:
        print("所有 subject 处理成功！")
    
    # 最终统计
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print("📊 总处理完成统计")
    print(f"{'='*60}")
    print(f"总处理耗时: {total_time:.1f} 秒")

    # with open('failed_subjects.txt', 'w') as f: # This line is removed as per the new_code, as failed_subjects is now a Manager list
    #     for date, subj in failed_subjects:
    #         f.write(f'{date}/{subj}\n')
    # if failed_subjects: # This line is removed as per the new_code, as failed_subjects is now a Manager list
    #     print(f"失败的 subjects 已写入 failed_subjects.txt") # This line is removed as per the new_code, as failed_subjects is now a Manager list
    # else: # This line is removed as per the new_code, as failed_subjects is now a Manager list
    #     print("所有 subjects 处理成功") # This line is removed as per the new_code, as failed_subjects is now a Manager list

if __name__ == "__main__":
    main()