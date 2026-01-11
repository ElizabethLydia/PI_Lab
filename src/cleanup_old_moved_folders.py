#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清理旧的moved相关文件夹
"""

import os
import shutil
from pathlib import Path

def cleanup_old_moved_folders():
    """清理旧的moved相关文件夹"""
    print("🧹 开始清理旧的moved相关文件夹...")
    
    # 要清理的文件夹路径
    folders_to_clean = [
        '/root/autodl-tmp/moved',
        '/root/autodl-tmp/integrated_analysis_moved',
        '/root/autodl-tmp/integrated_analysis_moved_filtered'
    ]
    
    # 要清理的文件夹模式
    patterns_to_clean = [
        'output_moved',
        'csv_output_moved',
        'ptt_bp_analysis_moved',
        'static_experiments_analysis_moved'
    ]
    
    total_cleaned = 0
    
    # 清理根目录下的文件夹
    for folder_path in folders_to_clean:
        if os.path.exists(folder_path):
            try:
                shutil.rmtree(folder_path)
                print(f"✅ 已删除: {folder_path}")
                total_cleaned += 1
            except Exception as e:
                print(f"❌ 删除失败 {folder_path}: {e}")
    
    # 清理每个subject下的moved相关文件夹
    autodl_tmp = '/root/autodl-tmp'
    if os.path.exists(autodl_tmp):
        for item in os.listdir(autodl_tmp):
            item_path = os.path.join(autodl_tmp, item)
            if os.path.isdir(item_path) and item.startswith('00'):
                # 这是一个subject文件夹
                for pattern in patterns_to_clean:
                    pattern_path = os.path.join(item_path, pattern)
                    if os.path.exists(pattern_path):
                        try:
                            shutil.rmtree(pattern_path)
                            print(f"✅ 已删除: {pattern_path}")
                            total_cleaned += 1
                        except Exception as e:
                            print(f"❌ 删除失败 {pattern_path}: {e}")
    
    print(f"\n🧹 清理完成！")
    print(f"总共删除了 {total_cleaned} 个文件夹")
    
    # 显示新的校准数据文件夹信息
    calibrated_root = '/root/shared/PhysioNet2025_Calibrated'
    if os.path.exists(calibrated_root):
        subjects = [d for d in os.listdir(calibrated_root) 
                   if os.path.isdir(os.path.join(calibrated_root, d)) and d.startswith('00')]
        print(f"\n📁 新的校准数据文件夹: {calibrated_root}")
        print(f"包含 {len(subjects)} 个subjects")
        print(f"前10个subjects: {subjects[:10]}")
        
        # 检查一个subject的实验数量
        if subjects:
            sample_subject = subjects[0]
            sample_path = os.path.join(calibrated_root, sample_subject)
            experiments = [d for d in os.listdir(sample_path) 
                         if os.path.isdir(os.path.join(sample_path, d)) and d.isdigit()]
            print(f"示例subject {sample_subject} 包含实验: {sorted(experiments)}")
    
    print(f"\n💡 现在可以运行修改后的step1_moved.py来处理校准数据了！")

if __name__ == "__main__":
    cleanup_old_moved_folders()

