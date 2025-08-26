#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
整合脚本：将以前step1的biopac文件与现在的abp.csv整合
生成完整的biopac文件，bp列替换为abp列
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
from scipy.interpolate import interp1d

def interpolate_with_reftime(time, data, reftime):
    """
    使用插值对齐到参考时间戳
    """
    if len(time) < 2 or len(data) < 2:
        return np.full(len(reftime), np.nan)
    
    # 确保数据类型正确
    time = np.asarray(time, dtype=float)
    reftime = np.asarray(reftime, dtype=float)
    
    # 限制插值范围到数据实际范围内
    min_time, max_time = time.min(), time.max()
    valid_reftime_mask = (reftime >= min_time) & (reftime <= max_time)
    valid_reftime = reftime[valid_reftime_mask]
    
    if len(valid_reftime) == 0:
        return np.full(len(reftime), np.nan)
    
    try:
        # 使用线性插值
        interp_func = interp1d(time, data, kind='linear', bounds_error=False, fill_value=np.nan)
        interpolated_data = interp_func(valid_reftime)
        
        # 创建完整结果
        full_result = np.full(len(reftime), np.nan)
        full_result[valid_reftime_mask] = interpolated_data
        
        return full_result
    except Exception as e:
        print(f"        插值错误: {e}")
        return np.full(len(reftime), np.nan)

def integrate_biopac_with_abp(subject_id):
    """整合单个受试者的biopac和abp数据"""
    print(f"\n{'='*60}")
    print(f"🔗 整合受试者 {subject_id} 的biopac和abp数据")
    print(f"{'='*60}")
    
    # 路径定义
    new_output_dir = f'/root/autodl-tmp/blood_pressure_reconstruction/{subject_id}'
    old_csv_dir = f'/root/autodl-tmp/{subject_id}/csv_output'
    new_csv_dir = os.path.join(new_output_dir, 'csv')
    
    # 检查目录是否存在
    if not os.path.exists(new_output_dir):
        print(f"❌ 新的输出目录不存在: {new_output_dir}")
        return False
    
    if not os.path.exists(old_csv_dir):
        print(f"❌ 旧的CSV目录不存在: {old_csv_dir}")
        return False
    
    if not os.path.exists(new_csv_dir):
        print(f"❌ 新的CSV目录不存在: {new_csv_dir}")
        return False
    
    # 查找实验编号
    experiments = set()
    
    # 从新目录查找实验编号
    new_files = [f for f in os.listdir(new_csv_dir) if f.endswith('_abp.csv')]
    for file in new_files:
        # 文件名格式：00003_1_abp.csv
        parts = file.split('_')
        if len(parts) >= 2:
            experiments.add(parts[1])
    
    # 从旧目录查找实验编号
    old_files = [f for f in os.listdir(old_csv_dir) if f.endswith('_biopac_aligned.csv')]
    for file in old_files:
        # 文件名格式：00003_1_biopac_aligned.csv
        parts = file.split('_')
        if len(parts) >= 2:
            experiments.add(parts[1])
    
    experiments = sorted(list(experiments))
    print(f"📋 发现实验: {experiments}")
    
    if not experiments:
        print(f"⚠️  没有找到任何实验数据")
        return False
    
    success_count = 0
    total_count = len(experiments)
    
    for exp_num in experiments:
        print(f"\n🔬 处理实验 {exp_num}...")
        
        # 文件路径
        old_biopac_path = os.path.join(old_csv_dir, f'{subject_id}_{exp_num}_biopac_aligned.csv')
        new_abp_path = os.path.join(new_csv_dir, f'{subject_id}_{exp_num}_abp.csv')
        integrated_output_path = os.path.join(new_csv_dir, f'{subject_id}_{exp_num}_biopac_integrated.csv')
        
        # 检查文件是否存在
        if not os.path.exists(old_biopac_path):
            print(f"  ⚠️  旧biopac文件不存在: {old_biopac_path}")
            continue
        
        if not os.path.exists(new_abp_path):
            print(f"  ⚠️  新abp文件不存在: {new_abp_path}")
            continue
        
        # 检查是否已经存在整合文件
        if os.path.exists(integrated_output_path):
            print(f"  ⚠️  整合文件已存在，跳过: {integrated_output_path}")
            continue
        
        try:
            # 读取旧biopac文件
            print(f"  📖 读取旧biopac文件: {old_biopac_path}")
            old_biopac = pd.read_csv(old_biopac_path)
            print(f"    旧文件列: {list(old_biopac.columns)}")
            print(f"    旧文件行数: {len(old_biopac)}")
            
            # 读取新abp文件
            print(f"  📖 读取新abp文件: {new_abp_path}")
            new_abp = pd.read_csv(new_abp_path)
            print(f"    新文件列: {list(new_abp.columns)}")
            print(f"    新文件行数: {len(new_abp)}")
            
            # 创建整合后的DataFrame
            integrated_df = old_biopac.copy()
            
            # 检查是否有bp列需要替换
            if 'bp' in integrated_df.columns:
                print(f"  🔄 替换bp列为abp列...")
                
                # 使用插值对齐abp数据到旧的时间戳
                abp_interpolated = interpolate_with_reftime(
                    new_abp['timestamp'].values,
                    new_abp['abp'].values,
                    integrated_df['timestamp'].values
                )
                
                # 替换bp列为abp列
                integrated_df['abp'] = abp_interpolated
                
                # 删除原来的bp列
                integrated_df = integrated_df.drop('bp', axis=1)
                
                print(f"    ✅ bp列已替换为abp列")
            else:
                print(f"  ⚠️  旧文件中没有bp列，直接添加abp列...")
                
                # 如果没有bp列，直接添加abp列
                abp_interpolated = interpolate_with_reftime(
                    new_abp['timestamp'].values,
                    new_abp['abp'].values,
                    integrated_df['timestamp'].values
                )
                
                integrated_df['abp'] = abp_interpolated
            
            # 重新排列列顺序，确保timestamp在第一列，abp在第二列
            other_columns = [col for col in integrated_df.columns if col not in ['timestamp', 'abp']]
            columns = ['timestamp', 'abp'] + other_columns
            integrated_df = integrated_df[columns]
            
            # 保存整合后的文件（新文件，不覆盖任何原有文件）
            integrated_df.to_csv(integrated_output_path, index=False)
            
            print(f"  💾 保存整合文件: {integrated_output_path}")
            print(f"    最终列: {list(integrated_df.columns)}")
            print(f"    最终行数: {len(integrated_df)}")
            
            # 检查数据质量
            abp_nan_count = integrated_df['abp'].isna().sum()
            abp_total_count = len(integrated_df)
            abp_quality = (abp_total_count - abp_nan_count) / abp_total_count * 100
            
            print(f"    📊 abp数据质量: {abp_quality:.1f}% ({abp_total_count - abp_nan_count}/{abp_total_count} 有效值)")
            
            success_count += 1
            
        except Exception as e:
            print(f"  ❌ 处理实验 {exp_num} 失败: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"📊 整合完成统计")
    print(f"{'='*60}")
    print(f"✅ 成功整合: {success_count}/{total_count}")
    print(f"📈 成功率: {success_count/total_count*100:.1f}%")
    
    if success_count > 0:
        print(f"\n🎯 整合文件已保存到: {new_csv_dir}/")
        print(f"📁 文件命名格式: {subject_id}_<实验编号>_biopac_integrated.csv")
        print(f"🔍 包含列: timestamp, abp, 以及其他原始biopac列")
        print(f"💡 注意: 这些是新生成的文件，不会覆盖任何原有文件")
    
    return success_count > 0

def main():
    """主函数"""
    print("🚀 开始整合biopac和abp数据")
    print("="*80)
    
    # 获取所有已处理的受试者
    base_dir = '/root/autodl-tmp/blood_pressure_reconstruction'
    if not os.path.exists(base_dir):
        print(f"❌ 基础目录不存在: {base_dir}")
        return
    
    subjects = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('00')]
    subjects.sort()
    
    print(f"📋 发现受试者: {len(subjects)} 个")
    print(f"🔢 受试者列表: {subjects}")
    
    if not subjects:
        print("⚠️  没有找到任何受试者目录")
        return
    
    # 询问用户是否继续
    print(f"\n⚠️  即将处理 {len(subjects)} 个受试者")
    print("💡 这将为每个受试者生成整合后的biopac文件")
    print("📁 输出位置: /root/autodl-tmp/blood_pressure_reconstruction/<subject>/csv/")
    print("🔒 安全保证: 不会覆盖任何原有文件，只生成新的整合文件")
    print("📋 新文件命名: <subject>_<实验编号>_biopac_integrated.csv")
    
    # 开始处理
    start_time = time.time()
    success_count = 0
    
    for subject in tqdm(subjects, desc="整合进度", unit="受试者"):
        try:
            if integrate_biopac_with_abp(subject):
                success_count += 1
        except Exception as e:
            print(f"❌ 处理受试者 {subject} 时发生错误: {e}")
            continue
    
    # 最终统计
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print("🎉 整合任务完成！")
    print(f"{'='*80}")
    print(f"⏱️  总耗时: {total_time:.1f} 秒")
    print(f"👥 总受试者数: {len(subjects)}")
    print(f"✅ 成功整合: {success_count}")
    print(f"❌ 失败: {len(subjects) - success_count}")
    print(f"📈 成功率: {success_count/len(subjects)*100:.1f}%")
    
    if success_count > 0:
        print(f"\n📁 整合文件已保存到各受试者的csv目录中")
        print(f"🔍 文件命名: <subject>_<实验编号>_biopac_integrated.csv")
        print(f"💡 这些文件包含了完整的biopac数据，bp列已替换为abp列")
        print(f"🔒 安全提醒: 所有原有文件都保持不变，包括abp.csv和旧的biopac文件")
    
    print(f"\n🎯 任务完成！")

if __name__ == "__main__":
    main()
