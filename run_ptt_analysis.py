#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行PTT与血压相关性分析的简化脚本
"""

from ptt_bp_analysis import PTTBloodPressureAnalyzer

def run_analysis():
    """运行PTT-血压相关性分析"""
    print("🔬 启动PTT与血压相关性分析")
    print("="*50)
    
    # 创建分析器
    analyzer = PTTBloodPressureAnalyzer()
    
    # 运行分析
    results = analyzer.run_comprehensive_analysis()
    
    print("\n✅ 分析完成！")
    print("\n📊 生成的分析结果：")
    print("1. ptt_bp_correlation_heatmap_整体分析.png - 相关性热图")
    print("2. ptt_bp_regression_analysis.png - 回归分析图") 
    print("3. synchronized_ptt_bp_data.csv - 同步的PTT-血压数据")
    print("4. ptt_bp_correlations.csv - 详细相关性统计")
    print("5. ptt_bp_model_evaluation.csv - 模型评估结果")
    
    print("\n📋 分析要点：")
    print("• 使用了时频域心率误差≤5BPM的有效窗口")
    print("• 分析了6个PTT组合与4种血压指标的关系")
    print("• 建立了多元回归模型预测血压")
    print("• 评估了统计显著性和临床准确性")

if __name__ == "__main__":
    run_analysis() 