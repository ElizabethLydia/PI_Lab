# PI_Lab 简单预处理代码
# 专门用于数据加载、时间戳对齐和保存为npy文件

import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d

class SimplePreprocessor:
    """简单的预处理器，专注于基础数据处理"""
    
    def __init__(self, data_root="/root/PI_Lab/00017"):
        self.data_root = data_root
        self.conditions = [str(i) for i in range(1, 12)]  # 1-11个条件
        
    def interpolate_duplicate_timestamps(self, df, time_col='timestamp'):
        """
        处理重复时间戳问题，基于清华项目的方法
        
        参数:
            df: 原始 DataFrame
            time_col: 时间戳列名
            
        返回:
            处理后的 DataFrame，时间戳不再重复
        """
        print(f"    🔧 处理重复时间戳...")
        df = df.copy()
        original_len = len(df)
        duplicate_count = original_len - len(df[time_col].unique())
        
        if duplicate_count == 0:
            print(f"      ✅ 无重复时间戳")
            return df
            
        print(f"      ⚠️ 发现 {duplicate_count} 个重复时间戳 ({duplicate_count/original_len*100:.1f}%)")
        
        unique_times = df[time_col].unique()
        new_timestamps = []
        
        for t in unique_times:
            # 当前时间戳对应的行索引
            indices = df[df[time_col] == t].index
            n_points = len(indices)

            if n_points > 1:
                # 多个点共享同一时间戳，需要插值
                current_idx = np.where(unique_times == t)[0][0]
                if current_idx == len(unique_times) - 1:
                    delta = 0.0005  # 最后一个时间点使用默认小间隔（0.5ms）
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
        
        print(f"      ✅ 重复时间戳处理完成")
        return df
    
    def interpolate_to_reference(self, timestamps, values, ref_timestamps):
        """
        将数据插值到参考时间戳，基于清华项目的方法
        
        参数:
            timestamps: 原始时间戳
            values: 原始数值
            ref_timestamps: 参考时间戳
            
        返回:
            插值后的数值
        """
        # 使用线性插值，超出范围的用边界值填充
        interp_func = interp1d(timestamps, values, 
                              kind='linear', 
                              bounds_error=False,
                              fill_value='extrapolate')
        return interp_func(ref_timestamps)
        
    def load_condition_data(self, condition):
        """加载单个条件的所有数据"""
        print(f"\n📁 加载条件 {condition} 的数据...")
        
        condition_path = os.path.join(self.data_root, condition)
        if not os.path.exists(condition_path):
            print(f"❌ 条件 {condition} 路径不存在")
            return None
        
        data = {
            'condition': condition,
            'biopac': {},
            'hub': {},
            'timestamps': {}
        }
        
        # 1. 加载BIOPAC数据
        biopac_files = {
            'hr': f'hr-{condition}.csv',
            'systolic_bp': f'systolic_bp-{condition}.csv',
            'diastolic_bp': f'diastolic_bp-{condition}.csv',
            'mean_bp': f'mean_bp-{condition}.csv',
            'bp': f'bp-{condition}.csv',
            'cardiac_output': f'cardiac_output-{condition}.csv',
            'cardiac_index': f'cardiac_index-{condition}.csv',
            'systemic_vascular_resistance': f'systemic_vascular_resistance-{condition}.csv',
            'rsp': f'rsp-{condition}.csv'
        }
        
        biopac_path = os.path.join(condition_path, 'Biopac')
        for signal_name, filename in biopac_files.items():
            filepath = os.path.join(biopac_path, filename)
            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath)
                    # 处理重复时间戳
                    df_clean = self.interpolate_duplicate_timestamps(df, 'timestamp')
                    data['biopac'][signal_name] = {
                        'values': df_clean.iloc[:, 1].values,  # 第二列是数值
                        'timestamps': df_clean['timestamp'].values
                    }
                    print(f"  ✓ BIOPAC {signal_name}: {len(df)} -> {len(df_clean)} 个数据点")
                except Exception as e:
                    print(f"  ❌ 加载 {signal_name} 失败: {e}")
        
        # 2. 加载HUB数据
        hub_files = ['sensor2.csv', 'sensor3.csv', 'sensor4.csv', 'sensor5.csv']
        hub_path = os.path.join(condition_path, 'HUB')
        
        for sensor_file in hub_files:
            filepath = os.path.join(hub_path, sensor_file)
            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath)
                    # 处理重复时间戳
                    df_clean = self.interpolate_duplicate_timestamps(df, 'timestamp')
                    sensor_name = sensor_file.replace('.csv', '')
                    data['hub'][sensor_name] = {
                        'timestamp': df_clean['timestamp'].values,
                        'red': df_clean['red'].values,
                        'ir': df_clean['ir'].values,
                        'green': df_clean['green'].values,
                        'ax': df_clean['ax'].values,
                        'ay': df_clean['ay'].values,
                        'az': df_clean['az'].values,
                        'rx': df_clean['rx'].values,
                        'ry': df_clean['ry'].values,
                        'rz': df_clean['rz'].values,
                        'mx': df_clean['mx'].values,
                        'my': df_clean['my'].values,
                        'mz': df_clean['mz'].values,
                        'temp': df_clean['temp'].values
                    }
                    print(f"  ✓ HUB {sensor_name}: {len(df)} -> {len(df_clean)} 个数据点")
                except Exception as e:
                    print(f"  ❌ 加载 {sensor_file} 失败: {e}")
        
        # 3. 分析时间戳范围
        self._analyze_timestamps(data)
        
        return data
    
    def _analyze_timestamps(self, data):
        """分析时间戳范围和对齐情况"""
        print(f"\n🕐 分析时间戳...")
        
        all_timestamps = []
        device_ranges = {}
        
        # BIOPAC时间戳
        for signal_name, signal_data in data['biopac'].items():
            if 'timestamps' in signal_data:
                timestamps = signal_data['timestamps']
                device_ranges[f'biopac_{signal_name}'] = {
                    'start': timestamps[0],
                    'end': timestamps[-1],
                    'count': len(timestamps),
                    'duration': timestamps[-1] - timestamps[0],
                    'sampling_rate': len(timestamps) / (timestamps[-1] - timestamps[0])
                }
                all_timestamps.extend(timestamps)
        
        # HUB时间戳
        for sensor_name, sensor_data in data['hub'].items():
            if 'timestamp' in sensor_data:
                timestamps = sensor_data['timestamp']
                device_ranges[f'hub_{sensor_name}'] = {
                    'start': timestamps[0],
                    'end': timestamps[-1],
                    'count': len(timestamps),
                    'duration': timestamps[-1] - timestamps[0],
                    'sampling_rate': len(timestamps) / (timestamps[-1] - timestamps[0])
                }
                all_timestamps.extend(timestamps)
        
        # 总体时间范围
        if all_timestamps:
            global_start = min(all_timestamps)
            global_end = max(all_timestamps)
            data['timestamps'] = {
                'global_start': global_start,
                'global_end': global_end,
                'global_duration': global_end - global_start,
                'device_ranges': device_ranges
            }
            
            print(f"  📊 全局时间范围: {global_end - global_start:.2f} 秒")
            print(f"  📊 开始时间: {global_start}")
            print(f"  📊 结束时间: {global_end}")
            
            # 显示每个设备的采样率
            for device_name, info in device_ranges.items():
                print(f"  📊 {device_name}: {info['sampling_rate']:.1f} Hz")
            
            # 检查设备对齐情况
            self._check_device_alignment(device_ranges)
    
    def _check_device_alignment(self, device_ranges):
        """检查设备时间对齐情况"""
        print(f"\n🔄 检查设备对齐情况...")
        
        starts = [info['start'] for info in device_ranges.values()]
        ends = [info['end'] for info in device_ranges.values()]
        
        start_diff = max(starts) - min(starts)
        end_diff = max(ends) - min(ends)
        
        print(f"  📏 开始时间差: {start_diff:.3f} 秒")
        print(f"  📏 结束时间差: {end_diff:.3f} 秒")
        
        if start_diff < 1.0 and end_diff < 1.0:
            print(f"  ✅ 设备时间对齐良好")
        else:
            print(f"  ⚠️ 设备时间存在偏差，需要对齐处理")
        
        return start_diff, end_diff
    
    def align_timestamps_advanced(self, data, target_sampling_rate=100, reference_device='hub_sensor2'):
        """
        高级时间戳对齐，基于清华项目的策略
        
        参数:
            data: 原始数据
            target_sampling_rate: 目标采样率
            reference_device: 参考设备（默认使用HUB sensor2作为参考）
        """
        print(f"\n⏰ 高级时间戳对齐 (参考设备: {reference_device}, 目标采样率: {target_sampling_rate} Hz)...")
        
        # 1. 确定参考时间轴
        ref_timestamps = None
        if reference_device.startswith('hub_'):
            sensor_name = reference_device.replace('hub_', '')
            if sensor_name in data['hub'] and 'timestamp' in data['hub'][sensor_name]:
                ref_timestamps = data['hub'][sensor_name]['timestamp']
        elif reference_device.startswith('biopac_'):
            signal_name = reference_device.replace('biopac_', '')
            if signal_name in data['biopac'] and 'timestamps' in data['biopac'][signal_name]:
                ref_timestamps = data['biopac'][signal_name]['timestamps']
        
        if ref_timestamps is None:
            print(f"  ❌ 找不到参考设备 {reference_device}，使用全局时间范围")
            # 使用全局时间范围创建统一时间轴
            global_start = data['timestamps']['global_start']
            global_end = data['timestamps']['global_end']
            duration = global_end - global_start
            unified_time = np.linspace(global_start, global_end, 
                                      int(duration * target_sampling_rate))
        else:
            print(f"  ✅ 使用 {reference_device} 作为时间参考")
            # 重采样参考时间戳到目标采样率
            ref_start = ref_timestamps[0]
            ref_end = ref_timestamps[-1]
            duration = ref_end - ref_start
            unified_time = np.linspace(ref_start, ref_end, 
                                      int(duration * target_sampling_rate))
        
        aligned_data = {
            'condition': data['condition'],
            'unified_time': unified_time,
            'sampling_rate': target_sampling_rate,
            'duration': duration,
            'reference_device': reference_device,
            'biopac_aligned': {},
            'hub_aligned': {}
        }
        
        # 2. 对齐BIOPAC数据
        print(f"  🔧 对齐BIOPAC数据...")
        for signal_name, signal_data in data['biopac'].items():
            if 'timestamps' in signal_data and 'values' in signal_data:
                timestamps = signal_data['timestamps']
                values = signal_data['values']
                
                # 插值到统一时间轴
                aligned_values = self.interpolate_to_reference(timestamps, values, unified_time)
                aligned_data['biopac_aligned'][signal_name] = aligned_values
                print(f"    ✓ {signal_name}: {len(values)} -> {len(aligned_values)} 点")
                
        print(f"  ✅ BIOPAC数据对齐完成")
        
        # 3. 对齐HUB数据
        print(f"  🔧 对齐HUB数据...")
        for sensor_name, sensor_data in data['hub'].items():
            if 'timestamp' in sensor_data:
                timestamps = sensor_data['timestamp']
                
                # 对所有信号进行插值
                for signal_name in ['red', 'ir', 'green', 'ax', 'ay', 'az', 
                                   'rx', 'ry', 'rz', 'mx', 'my', 'mz', 'temp']:
                    if signal_name in sensor_data:
                        values = sensor_data[signal_name]
                        aligned_values = self.interpolate_to_reference(timestamps, values, unified_time)
                        
                        # 使用sensor_signal格式保存，便于区分不同传感器
                        key = f'{sensor_name}_{signal_name}'
                        aligned_data['hub_aligned'][key] = aligned_values
                
                print(f"    ✓ {sensor_name}: 13个信号 -> {len(unified_time)} 点")
        
        print(f"  ✅ HUB数据对齐完成")
        print(f"  📊 统一数据长度: {len(unified_time)} 个采样点")
        
        return aligned_data
    
    def save_preprocessed_data(self, aligned_data, output_dir):
        """保存预处理后的数据为npy文件"""
        condition = aligned_data['condition']
        print(f"\n💾 保存条件 {condition} 的预处理数据...")
        
        # 创建输出目录
        output_path = Path(output_dir) / f"condition_{condition}"
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 保存基本信息
        metadata = {
            'condition': condition,
            'sampling_rate': aligned_data['sampling_rate'],
            'duration': aligned_data['duration'],
            'num_samples': len(aligned_data['unified_time']),
            'reference_device': aligned_data.get('reference_device', 'unknown')
        }
        np.save(output_path / 'metadata.npy', metadata, allow_pickle=True)
        
        # 保存统一时间轴
        np.save(output_path / 'timestamps.npy', aligned_data['unified_time'])
        
        # 保存BIOPAC数据
        biopac_path = output_path / 'biopac'
        biopac_path.mkdir(exist_ok=True)
        
        for signal_name, values in aligned_data['biopac_aligned'].items():
            np.save(biopac_path / f'{signal_name}.npy', values)
            
        print(f"  ✓ BIOPAC数据保存到: {biopac_path}")
        
        # 保存HUB数据
        hub_path = output_path / 'hub'
        hub_path.mkdir(exist_ok=True)
        
        for signal_name, values in aligned_data['hub_aligned'].items():
            np.save(hub_path / f'{signal_name}.npy', values)
            
        print(f"  ✓ HUB数据保存到: {hub_path}")
        
        # 创建汇总文件
        summary = {
            'biopac_signals': list(aligned_data['biopac_aligned'].keys()),
            'hub_signals': list(aligned_data['hub_aligned'].keys()),
            'total_duration': aligned_data['duration'],
            'sampling_rate': aligned_data['sampling_rate'],
            'reference_device': aligned_data.get('reference_device', 'unknown')
        }
        np.save(output_path / 'summary.npy', summary, allow_pickle=True)
        
        print(f"  ✅ 条件 {condition} 预处理完成，保存到: {output_path}")
        
        return output_path
    
    def check_data_quality(self, aligned_data):
        """检查数据质量"""
        print(f"\n🔍 检查数据质量...")
        
        quality_report = {
            'condition': aligned_data['condition'],
            'biopac_quality': {},
            'hub_quality': {}
        }
        
        # 检查BIOPAC数据质量
        for signal_name, values in aligned_data['biopac_aligned'].items():
            nan_count = np.isnan(values).sum()
            inf_count = np.isinf(values).sum()
            
            quality_report['biopac_quality'][signal_name] = {
                'length': len(values),
                'nan_count': nan_count,
                'inf_count': inf_count,
                'mean': np.nanmean(values),
                'std': np.nanstd(values),
                'quality_score': 1.0 - (nan_count + inf_count) / len(values)
            }
            
            print(f"  BIOPAC {signal_name}: 质量分数 {quality_report['biopac_quality'][signal_name]['quality_score']:.3f}")
        
        # 检查HUB数据质量
        for signal_name, values in aligned_data['hub_aligned'].items():
            nan_count = np.isnan(values).sum()
            inf_count = np.isinf(values).sum()
            
            quality_report['hub_quality'][signal_name] = {
                'length': len(values),
                'nan_count': nan_count,
                'inf_count': inf_count,
                'mean': np.nanmean(values),
                'std': np.nanstd(values),
                'quality_score': 1.0 - (nan_count + inf_count) / len(values)
            }
            
            print(f"  HUB {signal_name}: 质量分数 {quality_report['hub_quality'][signal_name]['quality_score']:.3f}")
        
        return quality_report
    
    def process_single_condition(self, condition, output_dir="./preprocessed_data", 
                                target_sampling_rate=100, reference_device='hub_sensor2'):
        """处理单个条件的完整流程"""
        print(f"🎯 开始处理条件 {condition}")
        print("=" * 50)
        
        # 1. 加载数据
        raw_data = self.load_condition_data(condition)
        if raw_data is None:
            return None
        
        # 2. 高级时间戳对齐
        aligned_data = self.align_timestamps_advanced(raw_data, target_sampling_rate, reference_device)
        
        # 3. 检查数据质量
        quality_report = self.check_data_quality(aligned_data)
        
        # 4. 保存数据
        output_path = self.save_preprocessed_data(aligned_data, output_dir)
        
        # 5. 保存质量报告
        np.save(output_path / 'quality_report.npy', quality_report, allow_pickle=True)
        
        print(f"✅ 条件 {condition} 处理完成!")
        
        return {
            'output_path': output_path,
            'aligned_data': aligned_data,
            'quality_report': quality_report
        }
    
    def process_all_conditions(self, output_dir="./preprocessed_data", 
                              target_sampling_rate=100, reference_device='hub_sensor2'):
        """处理所有条件"""
        print("🚀 开始处理所有条件...")
        print("=" * 60)
        
        results = {}
        
        for condition in self.conditions:
            try:
                result = self.process_single_condition(condition, output_dir, 
                                                     target_sampling_rate, reference_device)
                if result:
                    results[condition] = result
                print()  # 空行分隔
            except Exception as e:
                print(f"❌ 处理条件 {condition} 时出错: {e}")
                continue
        
        # 生成总体报告
        self._generate_overall_report(results, output_dir)
        
        print(f"🎉 所有条件处理完成! 结果保存在: {output_dir}")
        
        return results
    
    def _generate_overall_report(self, results, output_dir):
        """生成总体报告"""
        print("📋 生成总体报告...")
        
        overall_report = {
            'processed_conditions': list(results.keys()),
            'total_conditions': len(results),
            'condition_summaries': {}
        }
        
        for condition, result in results.items():
            quality_report = result['quality_report']
            overall_report['condition_summaries'][condition] = {
                'duration': result['aligned_data']['duration'],
                'sampling_rate': result['aligned_data']['sampling_rate'],
                'reference_device': result['aligned_data'].get('reference_device', 'unknown'),
                'biopac_signals': len(quality_report['biopac_quality']),
                'hub_signals': len(quality_report['hub_quality']),
                'avg_quality_score': np.mean([
                    info['quality_score'] 
                    for info in list(quality_report['biopac_quality'].values()) + 
                               list(quality_report['hub_quality'].values())
                ])
            }
        
        # 保存总体报告
        output_path = Path(output_dir)
        np.save(output_path / 'overall_report.npy', overall_report, allow_pickle=True)
        
        print(f"  ✓ 总体报告保存到: {output_path / 'overall_report.npy'}")


def main():
    """主函数 - 改进的演示"""
    print("🔧 PI_Lab 高级预处理工具 (基于清华项目经验)")
    print("=" * 60)
    
    # 初始化预处理器
    preprocessor = SimplePreprocessor()
    
    # 选择要处理的条件
    print("选择处理方式:")
    print("1. 处理单个条件（条件1 - 静止状态）- 高级对齐")
    print("2. 处理所有条件 - 高级对齐")
    print("3. 自定义参数处理")
    
    choice = input("请输入选择 (1, 2 或 3): ").strip()
    
    if choice == "1":
        # 处理单个条件
        result = preprocessor.process_single_condition('1', './preprocessed_data_v2')
        if result:
            print(f"\n✅ 高级预处理完成!")
            print(f"📁 输出路径: {result['output_path']}")
    
    elif choice == "2":
        # 处理所有条件
        results = preprocessor.process_all_conditions('./preprocessed_data_v2')
        print(f"\n✅ 全部高级预处理完成!")
        print(f"📁 成功处理 {len(results)} 个条件")
    
    elif choice == "3":
        # 自定义参数
        sampling_rate = int(input("目标采样率 (Hz, 默认100): ") or "100")
        ref_device = input("参考设备 (默认hub_sensor2): ") or "hub_sensor2"
        
        result = preprocessor.process_single_condition('1', './preprocessed_data_custom',
                                                     sampling_rate, ref_device)
        if result:
            print(f"\n✅ 自定义预处理完成!")
            print(f"📁 输出路径: {result['output_path']}")
    
    else:
        print("❌ 无效选择")


if __name__ == "__main__":
    main() 