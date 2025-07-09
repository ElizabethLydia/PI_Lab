#!/usr/bin/env python3
# 测试不同对齐策略的效果

from preprocessing_simple import SimplePreprocessor
import numpy as np
import matplotlib.pyplot as plt

def test_alignment_strategies():
    """测试不同的对齐策略"""
    print("🧪 测试不同的时间戳对齐策略")
    print("=" * 50)
    
    preprocessor = SimplePreprocessor()
    
    # 测试策略1: HUB sensor2作为参考，100Hz
    print("\n策略1: HUB sensor2参考，100Hz")
    try:
        result1 = preprocessor.process_single_condition(
            '1', './test_alignment_hub100', 
            target_sampling_rate=100, 
            reference_device='hub_sensor2'
        )
        print("✅ 策略1完成")
    except Exception as e:
        print(f"❌ 策略1失败: {e}")
    
    # 测试策略2: BIOPAC HR作为参考，200Hz  
    print("\n策略2: BIOPAC HR参考，200Hz")
    try:
        result2 = preprocessor.process_single_condition(
            '1', './test_alignment_biopac200',
            target_sampling_rate=200,
            reference_device='biopac_hr'
        )
        print("✅ 策略2完成")
    except Exception as e:
        print(f"❌ 策略2失败: {e}")
    
    # 测试策略3: 全局时间范围，50Hz
    print("\n策略3: 全局时间范围，50Hz")
    try:
        result3 = preprocessor.process_single_condition(
            '1', './test_alignment_global50',
            target_sampling_rate=50,
            reference_device='nonexistent'  # 强制使用全局范围
        )
        print("✅ 策略3完成")
    except Exception as e:
        print(f"❌ 策略3失败: {e}")
    
    print("\n🎯 对齐策略建议:")
    print("策略1 (HUB参考100Hz): 推荐用于PPG分析和PTT计算")
    print("策略2 (BIOPAC参考200Hz): 推荐用于高精度心率分析") 
    print("策略3 (全局50Hz): 推荐用于快速原型和资源受限环境")

if __name__ == "__main__":
    test_alignment_strategies() 