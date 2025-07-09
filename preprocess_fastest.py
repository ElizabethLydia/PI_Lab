#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最终版PI-Lab数据预处理脚本
经过测试验证的高效处理方案
直接去重，直接降采样，直接对齐
"""

import os
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import warnings
import time
warnings.filterwarnings('ignore')

def quick_downsample(df, time_col='timestamp', target_freq=100):
    """极简降采样：每N行取一行"""
    if df.empty or time_col not in df.columns:
        return df
    
    original_len = len(df)
    
    if len(df) > 1:
        time_range = df[time_col].max() - df[time_col].min()
        current_freq = len(df) / time_range if time_range > 0 else 1
        
        if current_freq > target_freq:
            step = max(1, int(current_freq / target_freq))
            result = df.iloc[::step].copy()
            print(f"    降采样: {original_len:,} -> {len(result):,} 行 (步长: {step})")
            return result
    
    return df

def remove_duplicates_simple(df, time_col='timestamp'):
    """简单去重：保留第一个出现的时间戳"""
    if df.empty or time_col not in df.columns:
        return df
    
    original_len = len(df)
    result = df.drop_duplicates(subset=[time_col], keep='first')
    removed = original_len - len(result)
    
    if removed > 0:
        print(f"    去重: 移除 {removed:,} 个重复 ({removed/original_len*100:.1f}%)")
    
    return result

def process_file_fast(file_path, target_freq=100):
    """快速处理单个文件"""
    try:
        file_name = os.path.basename(file_path)
        
        # 读取数据
        data = pd.read_csv(file_path)
        original_rows = len(data)
        
        if data.empty:
            return pd.DataFrame()
        
        print(f"  处理: {file_name} ({original_rows:,} 行)")
        
        # 1. 快速去重
        data = remove_duplicates_simple(data)
        
        # 2. 快速降采样（只对高频数据降采样）
        if len(data) > target_freq * 100:  # 如果超过10秒的数据量才降采样
            data = quick_downsample(data, target_freq=target_freq)
        
        # 3. 按时间戳排序
        if 'timestamp' in data.columns:
            data = data.sort_values('timestamp').reset_index(drop=True)
        
        final_rows = len(data)
        compression_ratio = original_rows / final_rows if final_rows > 0 else 1
        print(f"    最终: {final_rows:,} 行 (压缩比: {compression_ratio:.1f}:1)")
        
        return data
        
    except Exception as e:
        print(f"    错误: {e}")
        return pd.DataFrame()

def load_experiment_fast(experiment_path, target_freq=100):
    """快速加载单个实验"""
    experiment_name = os.path.basename(experiment_path)
    print(f"\n{'='*50}")
    print(f"处理实验 {experiment_name}")
    print(f"{'='*50}")
    
    result = {'biopac': {}, 'hub': {}}
    
    # 处理Biopac数据
    biopac_path = os.path.join(experiment_path, 'Biopac')
    if os.path.isdir(biopac_path):
        biopac_files = [f for f in os.listdir(biopac_path) if f.endswith('.csv')]
        print(f"\nBiopac文件 ({len(biopac_files)} 个):")
        
        for file in biopac_files:
            file_path = os.path.join(biopac_path, file)
            data = process_file_fast(file_path, target_freq)
            if not data.empty:
                key = file.split('-')[0] if '-' in file else file.replace('.csv', '')
                result['biopac'][key] = data
    
    # 处理HUB数据
    hub_path = os.path.join(experiment_path, 'HUB')
    if os.path.isdir(hub_path):
        hub_files = [f for f in os.listdir(hub_path) if f.endswith('.csv')]
        print(f"\nHUB文件 ({len(hub_files)} 个):")
        
        for file in hub_files:
            file_path = os.path.join(hub_path, file)
            data = process_file_fast(file_path, target_freq)
            if not data.empty:
                key = file.replace('.csv', '')
                result['hub'][key] = data
    
    # 统计结果
    biopac_count = len(result['biopac'])
    hub_count = len(result['hub'])
    print(f"\n实验 {experiment_name} 完成: {biopac_count} 个Biopac文件, {hub_count} 个HUB文件")
    
    return result

def align_data_simple(data_dict):
    """简化的数据对齐"""
    aligned_data = {}
    output_dir = '/root/PI_Lab/output'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*50}")
    print("数据对齐阶段")
    print(f"{'='*50}")
    
    for experiment_name, experiment_data in data_dict.items():
        print(f"\n对齐实验 {experiment_name}...")
        
        # 查找参考时间序列（优先使用sensor2）
        ref_data = None
        ref_name = ""
        
        if 'sensor2' in experiment_data['hub'] and not experiment_data['hub']['sensor2'].empty:
            ref_data = experiment_data['hub']['sensor2']
            ref_name = "sensor2"
        else:
            # 找到行数最少的数据作为参考
            min_len = float('inf')
            for data_type in ['hub', 'biopac']:
                for key, data in experiment_data[data_type].items():
                    if isinstance(data, pd.DataFrame) and not data.empty and 'timestamp' in data.columns:
                        if len(data) < min_len:
                            min_len = len(data)
                            ref_data = data
                            ref_name = f"{data_type}_{key}"
        
        if ref_data is None or ref_data.empty:
            print(f"  警告: 无有效参考数据，跳过对齐")
            continue
        
        print(f"  使用 {ref_name} 作为参考 ({len(ref_data):,} 行)")
        ref_timestamps = ref_data['timestamp'].values
        min_time, max_time = ref_timestamps.min(), ref_timestamps.max()
        
        aligned_experiment = {'biopac': {}, 'hub': {}}
        
        # 对齐所有数据
        for data_type in ['biopac', 'hub']:
            for key, data in experiment_data[data_type].items():
                if isinstance(data, pd.DataFrame) and not data.empty and 'timestamp' in data.columns:
                    # 过滤到参考时间范围内
                    mask = (data['timestamp'] >= min_time) & (data['timestamp'] <= max_time)
                    filtered_data = data[mask].copy()
                    
                    if not filtered_data.empty:
                        aligned_experiment[data_type][key] = filtered_data
                        print(f"    {data_type}_{key}: {len(data):,} -> {len(filtered_data):,} 行")
        
        aligned_data[experiment_name] = aligned_experiment
        
        # 保存结果
        output_path = os.path.join(output_dir, f'experiment_{experiment_name}_aligned.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump({experiment_name: aligned_experiment}, f)
        print(f"  保存到: {output_path}")
    
    return aligned_data

def main():
    """主函数"""
    start_time = time.time()
    
    pi_lab_folder = '/root/PI_Lab/00017'
    
    # 设置参数
    TARGET_FREQ = 100  # 目标频率
    MAX_EXPERIMENTS = None  # None表示处理所有实验，或设置数字限制
    
    print("🚀 PI-Lab数据高效预处理")
    print("="*60)
    print(f"数据路径: {pi_lab_folder}")
    print(f"目标频率: {TARGET_FREQ}Hz")
    print(f"策略: 快速去重 + 直接降采样 + 简单对齐")
    print("="*60)
    
    # 获取实验文件夹
    all_folders = os.listdir(pi_lab_folder)
    experiment_folders = [f for f in all_folders if f.isdigit() and os.path.isdir(os.path.join(pi_lab_folder, f))]
    experiment_folders.sort(key=lambda x: int(x))
    
    if MAX_EXPERIMENTS:
        experiment_folders = experiment_folders[:MAX_EXPERIMENTS]
        print(f"限制处理实验数量: {MAX_EXPERIMENTS}")
    
    print(f"发现实验: {experiment_folders}")
    print(f"总共处理: {len(experiment_folders)} 个实验")
    
    # 加载数据
    all_data = {}
    load_start = time.time()
    
    for i, experiment in enumerate(experiment_folders, 1):
        print(f"\n[{i}/{len(experiment_folders)}] 开始处理实验 {experiment}")
        experiment_path = os.path.join(pi_lab_folder, experiment)
        experiment_data = load_experiment_fast(experiment_path, TARGET_FREQ)
        all_data[experiment] = experiment_data
    
    load_time = time.time() - load_start
    
    # 对齐数据
    align_start = time.time()
    aligned_data = align_data_simple(all_data)
    align_time = time.time() - align_start
    
    # 最终统计
    total_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print("📊 处理完成统计")
    print(f"{'='*60}")
    print(f"成功处理实验: {len(aligned_data)}")
    print(f"数据加载耗时: {load_time:.1f} 秒")
    print(f"数据对齐耗时: {align_time:.1f} 秒")
    print(f"总处理耗时: {total_time:.1f} 秒")
    print(f"平均每个实验: {total_time/len(experiment_folders):.1f} 秒")
    
    for exp_name, exp_data in aligned_data.items():
        total_biopac = len(exp_data['biopac'])
        total_hub = len(exp_data['hub'])
        print(f"实验 {exp_name}: {total_biopac} 个Biopac + {total_hub} 个HUB文件")
    
    print(f"\n✅ 所有处理完成！数据保存在: /root/PI_Lab/output/")

if __name__ == "__main__":
    main() 