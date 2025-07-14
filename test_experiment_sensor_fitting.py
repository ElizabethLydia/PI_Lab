#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试实验×传感器对单独拟合分析
每个实验的每个传感器对对每个生理指标建立独立模型
"""

from step4_ptt_bp_analysis import PTTBloodPressureAnalyzer

def main():
    """运行实验×传感器对单独拟合分析"""
    print("🎯 实验×传感器对单独拟合分析测试")
    print("="*60)
    print("📋 拟合策略说明:")
    print("   每个实验的每个传感器对对每个生理指标建立独立的单变量回归模型")
    print("   例如: exp_1的sensor2-sensor3对systolic_bp建一个模型")
    print("        exp_1的sensor2-sensor4对systolic_bp建一个模型")
    print("        exp_2的sensor2-sensor3对systolic_bp建一个模型")
    print("        ... 以此类推")
    
    # 创建分析器
    analyzer = PTTBloodPressureAnalyzer()
    
    # 运行实验×传感器对单独拟合分析
    exp_sensor_models = analyzer.run_individual_experiment_sensor_regression_analysis()
    
    if exp_sensor_models:
        print(f"\n✅ 实验×传感器对单独拟合完成!")
        print(f"📁 结果保存在: {analyzer.output_dir}")
        print(f"📊 成功分析的实验数量: {len(exp_sensor_models)}")
        
        print(f"\n📋 生成的文件:")
        print(f"   • experiment_sensor_models.csv - 每个模型的详细性能")
        print(f"   • experiment_sensor_performance_comparison.png - 多维度性能对比图")
        print(f"   • best_sensors_across_experiments.csv - 跨实验最佳传感器对排名")
        
        # 显示详细的实验×传感器对概况
        print(f"\n📡 各实验的传感器对模型统计:")
        total_models = 0
        for exp_key, exp_data in exp_sensor_models.items():
            exp_total = 0
            print(f"   🔬 {exp_key}:")
            for sensor_pair, sensor_models in exp_data.items():
                sensor_label = analyzer._format_sensor_pair_label_en(sensor_pair)
                model_count = len(sensor_models)
                exp_total += model_count
                print(f"      • {sensor_pair} ({sensor_label}): {model_count}个模型")
            print(f"      小计: {exp_total}个模型")
            total_models += exp_total
        
        print(f"\n📊 总计: {total_models}个独立的单变量回归模型")
        print(f"\n🔍 模型详情说明:")
        print(f"   • 每个模型都是: 单个实验的单个传感器对PTT → 单个生理指标")
        print(f"   • 例如: exp_1的Nose→Finger的PTT值 → 收缩压")
        print(f"   • 这样可以分析每个传感器对在不同实验中的表现差异")
    else:
        print("❌ 实验×传感器对分析失败")

if __name__ == "__main__":
    main() 