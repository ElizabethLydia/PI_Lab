# PTT计算核心演示代码
# 展示如何从PI_Lab数据计算脉搏传输时间

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

def load_single_condition(condition='1', data_root="/root/PI_Lab/00017"):
    """加载单个实验条件的数据"""
    print(f"📂 加载条件{condition}的数据...")
    
    condition_path = os.path.join(data_root, condition)
    
    # 加载Biopac心率数据
    hr_path = os.path.join(condition_path, 'Biopac', f'hr-{condition}.csv')
    hr_data = pd.read_csv(hr_path)
    print(f"✓ Biopac心率数据: {len(hr_data)}个数据点")
    
    # 加载Biopac血压数据
    bp_path = os.path.join(condition_path, 'Biopac', f'systolic_bp-{condition}.csv')
    bp_data = pd.read_csv(bp_path)
    print(f"✓ Biopac血压数据: {len(bp_data)}个数据点")
    
    # 加载HUB PPG数据
    hub_path = os.path.join(condition_path, 'HUB', 'sensor2.csv')
    hub_data = pd.read_csv(hub_path)
    print(f"✓ HUB PPG数据: {len(hub_data)}个数据点")
    
    return hr_data, bp_data, hub_data

def detect_r_waves_from_hr(hr_data, max_points=1000):
    """从心率数据推算R波时间"""
    print("💓 从心率数据推算R波时间...")
    
    # 取前1000个点进行演示
    hr_subset = hr_data.head(max_points)
    hr_values = hr_subset['hr'].values
    timestamps = hr_subset['timestamp'].values
    
    # 计算RR间期
    rr_intervals = 60.0 / hr_values  # 秒
    
    # 推算R波时间戳
    r_timestamps = []
    for i in range(len(rr_intervals)-1):
        time_diff = timestamps[i+1] - timestamps[i]
        num_beats = max(1, int(time_diff / rr_intervals[i]))
        
        for j in range(num_beats):
            r_time = timestamps[i] + j * rr_intervals[i]
            if r_time <= timestamps[i+1]:
                r_timestamps.append(r_time)
    
    print(f"✓ 推算出{len(r_timestamps)}个R波时间点")
    return np.array(r_timestamps)

def detect_ppg_peaks(hub_data, max_points=1000):
    """从HUB PPG信号检测脉搏峰值"""
    print("🔍 检测PPG脉搏峰值...")
    
    # 取前1000个点进行演示
    hub_subset = hub_data.head(max_points)
    
    # 使用绿光PPG信号
    ppg_signal = hub_subset['green'].values
    timestamps = hub_subset['timestamp'].values
    
    # 简单的峰值检测
    # 1. 滤波
    try:
        sampling_rate = 1.0 / np.mean(np.diff(timestamps))
        nyquist = sampling_rate / 2
        low_cutoff = 0.5 / nyquist
        high_cutoff = 5 / nyquist
        b, a = signal.butter(4, [low_cutoff, high_cutoff], btype='band')
        filtered_ppg = signal.filtfilt(b, a, ppg_signal)
    except:
        filtered_ppg = ppg_signal
    
    # 2. 寻找峰值
    height_threshold = np.mean(filtered_ppg) + 0.3 * np.std(filtered_ppg)
    distance = int(0.4 * len(filtered_ppg) / (timestamps[-1] - timestamps[0]))  # 最小心跳间隔
    
    peaks, _ = signal.find_peaks(filtered_ppg, 
                                height=height_threshold,
                                distance=max(distance, 5))
    
    peak_timestamps = timestamps[peaks]
    print(f"✓ 检测到{len(peak_timestamps)}个PPG峰值")
    
    return peaks, peak_timestamps, filtered_ppg

def calculate_ptt(r_timestamps, ppg_timestamps, max_ptt=1.0):
    """计算脉搏传输时间"""
    print("⏱️ 计算PTT...")
    
    ptt_values = []
    matched_pairs = []
    
    for r_time in r_timestamps:
        # 找到R波后的第一个PPG峰值
        future_ppg = ppg_timestamps[ppg_timestamps > r_time]
        
        if len(future_ppg) > 0:
            ppg_time = future_ppg[0]
            ptt = ppg_time - r_time
            
            # 过滤合理的PTT值(50ms - 1000ms)
            if 0.05 <= ptt <= max_ptt:
                ptt_values.append(ptt)
                matched_pairs.append((r_time, ppg_time))
    
    ptt_values = np.array(ptt_values)
    print(f"✓ 计算出{len(ptt_values)}个有效PTT值")
    print(f"✓ PTT范围: {np.min(ptt_values)*1000:.1f} - {np.max(ptt_values)*1000:.1f} ms")
    print(f"✓ PTT均值: {np.mean(ptt_values)*1000:.1f} ± {np.std(ptt_values)*1000:.1f} ms")
    
    return ptt_values, matched_pairs

def visualize_ptt_analysis(hr_data, hub_data, r_timestamps, ppg_peaks, ppg_timestamps, 
                          filtered_ppg, ptt_values, condition='1'):
    """可视化PTT分析结果"""
    print("📈 生成可视化图表...")
    
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    
    # 1. 原始心率数据
    hr_subset = hr_data.head(1000)
    axes[0].plot(hr_subset['timestamp'], hr_subset['hr'], 'r-', alpha=0.7)
    axes[0].set_title(f'条件{condition} - Biopac心率监测')
    axes[0].set_ylabel('心率 (BPM)')
    axes[0].grid(True)
    
    # 2. PPG信号和峰值检测
    hub_subset = hub_data.head(1000)
    axes[1].plot(hub_subset['timestamp'], hub_subset['green'], 'g-', alpha=0.5, label='原始PPG')
    axes[1].plot(hub_subset['timestamp'], filtered_ppg, 'b-', alpha=0.8, label='滤波后PPG')
    axes[1].scatter(ppg_timestamps, filtered_ppg[ppg_peaks], color='red', s=50, 
                   zorder=5, label=f'检测峰值({len(ppg_timestamps)}个)')
    axes[1].set_title('HUB PPG信号和峰值检测')
    axes[1].set_ylabel('PPG幅度')
    axes[1].legend()
    axes[1].grid(True)
    
    # 3. PTT值分布
    axes[2].hist(ptt_values * 1000, bins=20, alpha=0.7, color='purple', edgecolor='black')
    axes[2].axvline(np.mean(ptt_values) * 1000, color='red', linestyle='--', 
                   label=f'均值: {np.mean(ptt_values)*1000:.1f}ms')
    axes[2].set_title('PTT值分布')
    axes[2].set_xlabel('PTT (ms)')
    axes[2].set_ylabel('频次')
    axes[2].legend()
    axes[2].grid(True)
    
    # 4. PTT时间序列
    if len(ptt_values) > 0:
        axes[3].plot(ptt_values * 1000, 'bo-', alpha=0.7, markersize=3)
        axes[3].set_title('PTT时间序列')
        axes[3].set_xlabel('心跳序号')
        axes[3].set_ylabel('PTT (ms)')
        axes[3].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'ptt_analysis_condition_{condition}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 图表已保存为: ptt_analysis_condition_{condition}.png")

def analyze_bp_ptt_relationship(bp_data, ptt_values, condition='1'):
    """分析血压与PTT的关系"""
    print("🩺 分析血压-PTT关系...")
    
    # 取同时间段的血压数据
    bp_subset = bp_data.head(len(ptt_values))
    bp_values = bp_subset['systolic_bp'].values[:len(ptt_values)]
    
    if len(bp_values) > 0 and len(ptt_values) > 0:
        min_len = min(len(bp_values), len(ptt_values))
        bp_subset = bp_values[:min_len]
        ptt_subset = ptt_values[:min_len] * 1000  # 转换为ms
        
        # 计算相关性
        correlation = np.corrcoef(bp_subset, ptt_subset)[0, 1]
        
        print(f"✓ 血压范围: {np.min(bp_subset):.1f} - {np.max(bp_subset):.1f} mmHg")
        print(f"✓ PTT-血压相关性: {correlation:.3f}")
        
        # 可视化关系
        plt.figure(figsize=(10, 6))
        plt.scatter(ptt_subset, bp_subset, alpha=0.6)
        plt.xlabel('PTT (ms)')
        plt.ylabel('收缩压 (mmHg)')
        plt.title(f'条件{condition} - PTT与血压关系 (相关性: {correlation:.3f})')
        
        # 添加趋势线
        z = np.polyfit(ptt_subset, bp_subset, 1)
        p = np.poly1d(z)
        plt.plot(ptt_subset, p(ptt_subset), 'r--', alpha=0.8)
        
        plt.grid(True)
        plt.savefig(f'ptt_bp_relationship_condition_{condition}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 关系图已保存为: ptt_bp_relationship_condition_{condition}.png")
        
        return correlation
    
    return None

def run_ptt_demo(condition='1'):
    """运行PTT计算演示"""
    print("🚀 开始PTT计算演示...")
    print("=" * 60)
    
    # 1. 加载数据
    hr_data, bp_data, hub_data = load_single_condition(condition)
    
    # 2. 检测R波
    r_timestamps = detect_r_waves_from_hr(hr_data)
    
    # 3. 检测PPG峰值
    ppg_peaks, ppg_timestamps, filtered_ppg = detect_ppg_peaks(hub_data)
    
    # 4. 计算PTT
    ptt_values, matched_pairs = calculate_ptt(r_timestamps, ppg_timestamps)
    
    # 5. 可视化分析
    if len(ptt_values) > 0:
        visualize_ptt_analysis(hr_data, hub_data, r_timestamps, ppg_peaks, 
                             ppg_timestamps, filtered_ppg, ptt_values, condition)
        
        # 6. 分析血压关系
        correlation = analyze_bp_ptt_relationship(bp_data, ptt_values, condition)
    
    print("=" * 60)
    print("✅ PTT计算演示完成!")
    
    # 返回结果摘要
    return {
        'condition': condition,
        'num_r_waves': len(r_timestamps),
        'num_ppg_peaks': len(ppg_timestamps),
        'num_ptt_values': len(ptt_values),
        'ptt_mean_ms': np.mean(ptt_values) * 1000 if len(ptt_values) > 0 else None,
        'ptt_std_ms': np.std(ptt_values) * 1000 if len(ptt_values) > 0 else None,
        'bp_ptt_correlation': correlation if 'correlation' in locals() else None
    }

if __name__ == "__main__":
    # 演示静止状态(条件1)的PTT计算
    print("🎯 演示条件1(静止状态)的PTT计算")
    result_1 = run_ptt_demo('1')
    
    print("\n🎯 演示条件7(静止状态)的PTT计算")
    result_7 = run_ptt_demo('7')
    
    # 对比两个静止状态
    print("\n📊 静止状态对比:")
    print(f"条件1 PTT: {result_1['ptt_mean_ms']:.1f}±{result_1['ptt_std_ms']:.1f} ms")
    print(f"条件7 PTT: {result_7['ptt_mean_ms']:.1f}±{result_7['ptt_std_ms']:.1f} ms")
    
    if result_1['bp_ptt_correlation'] and result_7['bp_ptt_correlation']:
        print(f"条件1 血压相关性: {result_1['bp_ptt_correlation']:.3f}")
        print(f"条件7 血压相关性: {result_7['bp_ptt_correlation']:.3f}")
    
    print("\n🎉 多设备PTT血压预测演示完成!") 