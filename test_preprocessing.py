# 测试预处理单个条件
from preprocessing_simple import SimplePreprocessor

def test_single_condition():
    """测试处理单个条件（条件1 - 静止状态）"""
    print("🧪 测试预处理条件1...")
    
    # 初始化预处理器
    preprocessor = SimplePreprocessor()
    
    # 处理条件1
    result = preprocessor.process_single_condition('1', './preprocessed_data')
    
    if result:
        print(f"\n🎉 预处理成功!")
        print(f"📁 输出路径: {result['output_path']}")
        
        # 显示质量报告摘要
        quality_report = result['quality_report']
        print(f"\n📊 数据质量摘要:")
        print(f"  BIOPAC信号数: {len(quality_report['biopac_quality'])}")
        print(f"  HUB信号数: {len(quality_report['hub_quality'])}")
        
        return result
    else:
        print("❌ 预处理失败")
        return None

if __name__ == "__main__":
    test_single_condition() 