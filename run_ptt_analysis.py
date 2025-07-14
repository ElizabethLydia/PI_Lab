#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced PTT-Cardiovascular Parameters Correlation Analysis
Multi-level Analysis: Overall + Individual Experiments + Comparison
"""

from ptt_bp_analysis import PTTBloodPressureAnalyzer

def run_enhanced_analysis():
    """运行增强版PTT-心血管参数相关性分析"""
    print("🔬 Enhanced PTT-Cardiovascular Parameters Analysis")
    print("="*60)
    
    # 创建分析器
    analyzer = PTTBloodPressureAnalyzer()
    
    # 运行多层次分析
    results = analyzer.run_comprehensive_analysis()
    
    print("\n✅ Multi-level analysis completed!")
    print("\n📊 Generated Analysis Results:")
    
    print("\n🎯 OVERALL ANALYSIS:")
    print("1. ptt_cardiovascular_correlation_heatmap_(整体分析).png - Full correlation matrix")
    print("2. ptt_cardiovascular_correlation_focused_整体分析_聚焦.png - Key parameters heatmap")
    print("3. ptt_cardiovascular_regression_analysis.png - Prediction models")
    print("4. synchronized_ptt_cardiovascular_data.csv - Combined dataset")
    print("5. ptt_cardiovascular_correlations.csv - Detailed correlations")
    
    print("\n🔍 INDIVIDUAL EXPERIMENTS:")
    print("6. ptt_cardiovascular_correlation_focused_实验[X].png - Per-experiment heatmaps")
    print("7. ptt_cardiovascular_correlations_exp_[X].csv - Per-experiment correlations")
    
    print("\n📈 COMPARISON ANALYSIS:")
    print("8. experiment_comparison.csv - Cross-experiment comparison")
    
    print("\n📋 Analysis Features:")
    print("• 🎯 Multi-level approach: Overall + Individual + Comparison")
    print("• 📊 9 cardiovascular parameters analyzed comprehensively")
    print("• 🔍 Focused heatmaps for better visualization clarity")
    print("• 🧪 Individual experiment analysis to detect variations")
    print("• 📈 Cross-experiment comparison for consistency validation")
    print("• 🏥 Professional English terminology for journal publication")
    print("• 📐 Quality-controlled windows (HR error ≤5 BPM)")
    print("• 🎨 Publication-ready visualizations")
    
    print("\n🔬 Research Insights:")
    print("• Pooled analysis provides overall population trends")
    print("• Individual analysis reveals experiment-specific patterns")
    print("• Comparison analysis validates result consistency")
    print("• Focused heatmaps highlight key physiological relationships")
    print("• Multi-dimensional approach strengthens scientific rigor")
    
    print("\n💡 Clinical Applications:")
    print("• Overall results: General population monitoring guidelines")
    print("• Individual results: Subject-specific monitoring optimization")
    print("• Comparison results: Monitoring system reliability assessment")
    print("• Focused visualization: Clinical decision support systems")

if __name__ == "__main__":
    run_enhanced_analysis() 