#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IR通道PPG信号与ABP信号相似性分析
基于论文方法论的实现
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.stats import pearsonr, spearmanr
from scipy.signal import butter, filtfilt, find_peaks
from scipy.fft import fft, fftfreq
from scipy.interpolate import interp1d
import pywt
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class PPGABPSimilarityAnalyzer:
    """PPG与ABP信号相似性分析器"""
    
    def __init__(self, data_dir="/root/PI_Lab"):
        self.data_dir = data_dir
        self.results_dir = "/root/PI_Lab/blood_pressure_reconstruction"
        os.makedirs(self.results_dir, exist_ok=True)
        
    def load_calibrated_data(self, subject_id, experiment_number):
        """加载校准后的数据"""
        # 导入数据加载器
        from data_loader import PILabDataLoader
        
        loader = PILabDataLoader(self.data_dir)
        data = loader.load_experiment_data(subject_id, experiment_number)
        
        if data is not None:
            # 确保数据格式正确
            if 'ir_signal' in data and 'abp_signal' in data and 'timestamp' in data:
                return data
            else:
                print("数据格式不正确，缺少必要的信号列")
                return None
        
        return None
    
    def preprocess_ir_signal(self, ir_signal, fs=100, method='butterworth'):
        """IR信号预处理"""
        if method == 'butterworth':
            # Butterworth滤波
            nyquist = fs / 2
            low_freq = 0.5  # 0.5 Hz
            high_freq = 8.0  # 8 Hz
            
            # 低通滤波
            b, a = butter(4, high_freq / nyquist, btype='low')
            filtered = filtfilt(b, a, ir_signal)
            
            # 高通滤波
            b, a = butter(4, low_freq / nyquist, btype='high')
            filtered = filtfilt(b, a, filtered)
            
        elif method == 'wavelet':
            # 小波去噪
            coeffs = pywt.wavedec(ir_signal, 'db4', level=4)
            threshold = np.std(coeffs[-1]) * np.sqrt(2 * np.log(len(ir_signal)))
            coeffs[1:] = [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
            filtered = pywt.waverec(coeffs, 'db4')
            
        elif method == 'morphological':
            # 形态学滤波
            from scipy.ndimage import grey_opening, grey_closing
            # 开运算去除小峰值
            opened = grey_opening(ir_signal, size=5)
            # 闭运算去除小谷值
            filtered = grey_closing(opened, size=5)
            
        else:
            filtered = ir_signal
            
        return filtered
    
    def remove_motion_artifacts(self, ir_signal, acc_data, fs=100):
        """基于加速度数据去除运动伪影"""
        # 计算加速度幅度
        acc_magnitude = np.sqrt(acc_data['ax']**2 + acc_data['ay']**2 + acc_data['az']**2)
        
        # 检测运动阈值
        motion_threshold = np.percentile(acc_magnitude, 75)
        motion_mask = acc_magnitude > motion_threshold
        
        # 对运动段进行插值
        if np.any(motion_mask):
            # 找到运动段的边界
            motion_starts = np.where(np.diff(motion_mask.astype(int)) == 1)[0]
            motion_ends = np.where(np.diff(motion_mask.astype(int)) == -1)[0]
            
            # 处理边界情况
            if len(motion_starts) > 0 and len(motion_ends) > 0:
                if motion_starts[0] > motion_ends[0]:
                    motion_starts = np.concatenate([[0], motion_starts])
                if motion_ends[-1] < motion_starts[-1]:
                    motion_ends = np.concatenate([motion_ends, [len(ir_signal)-1]])
                
                # 对每个运动段进行插值
                for start, end in zip(motion_starts, motion_ends):
                    if start < end and start > 0 and end < len(ir_signal)-1:
                        # 使用前后非运动段的值进行线性插值
                        ir_signal[start:end+1] = np.linspace(
                            ir_signal[start-1], ir_signal[end+1], end-start+1
                        )
        
        return ir_signal
    
    def calculate_similarity_metrics(self, ppg_signal, abp_signal):
        """计算相似性指标（基于文献方法）"""
        # 确保信号长度一致
        min_len = min(len(ppg_signal), len(abp_signal))
        ppg = ppg_signal[:min_len]
        abp = abp_signal[:min_len]
        
        # 标准化信号
        ppg_norm = (ppg - np.mean(ppg)) / np.std(ppg)
        abp_norm = (abp - np.mean(abp)) / np.std(abp)
        
        # 1. 时域相关性 (Correlation r) - 文献核心指标
        pearson_corr, pearson_p = pearsonr(ppg_norm, abp_norm)
        spearman_corr, spearman_p = spearmanr(ppg_norm, abp_norm)
        
        # 2. 频域相关性 (Frequency Domain Correlation)
        ppg_fft = np.abs(fft(ppg_norm))
        abp_fft = np.abs(fft(abp_norm))
        freq_corr, _ = pearsonr(ppg_fft, abp_fft)
        
        # 3. 相干性 (Coherence) - 文献关键指标
        coherence = self.calculate_coherence(ppg_norm, abp_norm)
        
        # 4. 部分相干性 (Partial Coherence) - 文献关键指标
        partial_coherence = self.calculate_partial_coherence(ppg_norm, abp_norm)
        
        # 5. 互信息 (Mutual Information) - 非线性依赖分析
        ppg_binned = pd.cut(ppg_norm, bins=20, labels=False)
        abp_binned = pd.cut(abp_norm, bins=20, labels=False)
        mutual_info = mutual_info_score(ppg_binned, abp_binned)
        
        # 6. DTW距离 (形态学相似性)
        dtw_distance = self.calculate_dtw_distance(ppg_norm, abp_norm)
        
        # 7. 相位相关性 (Phase Correlation)
        phase_corr = self.calculate_phase_correlation(ppg_norm, abp_norm)
        
        return {
            'pearson_correlation': pearson_corr,      # 文献核心指标r
            'spearman_correlation': spearman_corr,    # 单调相关性
            'frequency_correlation': freq_corr,       # 频域相关性
            'coherence': coherence,                   # 相干性COH
            'partial_coherence': partial_coherence,   # 部分相干性pCOH
            'mutual_information': mutual_info,        # 互信息
            'phase_correlation': phase_corr,          # 相位相关性
            'dtw_distance': dtw_distance,             # DTW距离
            'pearson_p_value': pearson_p,
            'spearman_p_value': spearman_p
        }
    
    def calculate_coherence(self, signal1, signal2):
        """计算相干性 (Coherence) - 文献关键指标"""
        try:
            # 使用scipy.signal.coherence计算
            from scipy.signal import coherence
            f, coh = coherence(signal1, signal2, fs=100, nperseg=min(256, len(signal1)//4))
            
            # 返回平均相干性（在0.5-8Hz频带内）
            freq_mask = (f >= 0.5) & (f <= 8.0)
            if np.any(freq_mask):
                mean_coherence = np.mean(coh[freq_mask])
            else:
                mean_coherence = np.mean(coh)
            
            return mean_coherence
        except:
            # 如果scipy.signal.coherence不可用，使用简化计算
            return self.calculate_simple_coherence(signal1, signal2)
    
    def calculate_simple_coherence(self, signal1, signal2):
        """简化相干性计算"""
        # 计算互功率谱
        ppg_fft = fft(signal1)
        abp_fft = fft(signal2)
        
        # 互功率谱
        cross_power = np.abs(ppg_fft * np.conj(abp_fft))
        
        # 自功率谱
        ppg_power = np.abs(ppg_fft)**2
        abp_power = np.abs(abp_fft)**2
        
        # 相干性
        coherence = cross_power / np.sqrt(ppg_power * abp_power)
        
        # 避免除零
        coherence = np.nan_to_num(coherence, nan=0.0, posinf=1.0, neginf=0.0)
        
        # 返回平均相干性
        return np.mean(coherence)
    
    def calculate_partial_coherence(self, signal1, signal2):
        """计算部分相干性 (Partial Coherence) - 文献关键指标"""
        try:
            # 简化版本：使用去趋势后的信号计算相干性
            from scipy.signal import detrend
            ppg_detrended = detrend(signal1)
            abp_detrended = detrend(signal2)
            
            return self.calculate_coherence(ppg_detrended, abp_detrended)
        except:
            # 如果detrend不可用，返回普通相干性
            return self.calculate_coherence(signal1, signal2)
    
    def calculate_phase_correlation(self, signal1, signal2):
        """计算相位相关性"""
        try:
            # 计算相位
            ppg_phase = np.angle(fft(signal1))
            abp_phase = np.angle(fft(signal2))
            
            # 相位相关性
            phase_corr = np.corrcoef(ppg_phase, abp_phase)[0, 1]
            
            return phase_corr if not np.isnan(phase_corr) else 0.0
        except:
            return 0.0
    
    def calculate_dtw_distance(self, signal1, signal2, max_window=100):
        """计算DTW距离（简化版本）"""
        n, m = len(signal1), len(signal2)
        
        # 限制搜索窗口以提高效率
        window = min(max_window, abs(n - m))
        
        # 初始化DTW矩阵
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0
        
        # 填充DTW矩阵
        for i in range(1, n + 1):
            start = max(1, i - window)
            end = min(m + 1, i + window + 1)
            for j in range(start, end):
                cost = abs(signal1[i-1] - signal2[j-1])
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i-1, j],    # 插入
                    dtw_matrix[i, j-1],    # 删除
                    dtw_matrix[i-1, j-1]   # 匹配
                )
        
        return dtw_matrix[n, m]
    
    def calculate_snr(self, signal, fs=100):
        """计算信噪比"""
        # 使用FFT分析
        fft_vals = np.abs(fft(signal))
        freqs = fftfreq(len(signal), 1/fs)
        
        # 定义信号频带（0.5-8 Hz）
        signal_mask = (freqs >= 0.5) & (freqs <= 8.0)
        noise_mask = (freqs < 0.5) | (freqs > 8.0)
        
        signal_power = np.sum(fft_vals[signal_mask]**2)
        noise_power = np.sum(fft_vals[noise_mask]**2)
        
        if noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
        else:
            snr = np.inf
            
        return snr
    
    def visualize_signals(self, ppg_signals, abp_signal, timestamps, subject_id, exp_num):
        """可视化信号对比"""
        fig, axes = plt.subplots(len(ppg_signals) + 1, 1, figsize=(15, 4*(len(ppg_signals) + 1)))
        
        # 绘制ABP信号
        axes[0].plot(timestamps, abp_signal, 'r-', linewidth=1, label='ABP信号')
        axes[0].set_title(f'Subject {subject_id} - Experiment {exp_num} - ABP信号')
        axes[0].set_ylabel('血压值 (mmHg)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 绘制不同预处理的PPG信号
        for i, (method, ppg_signal) in enumerate(ppg_signals.items()):
            axes[i+1].plot(timestamps, ppg_signal, 'b-', linewidth=1, label=f'{method}预处理')
            axes[i+1].set_title(f'{method}预处理后的IR信号')
            axes[i+1].set_ylabel('信号强度')
            axes[i+1].legend()
            axes[i+1].grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('时间 (s)')
        plt.tight_layout()
        
        # 保存图片
        save_path = os.path.join(self.results_dir, f'subject_{subject_id}_exp_{exp_num}_signals.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return save_path
    
    def create_correlation_heatmap(self, similarity_results, subject_id, exp_num):
        """创建相关性热图（包含文献关键指标）"""
        # 提取相关性指标（按重要性排序）
        methods = list(similarity_results.keys())
        metrics = [
            'pearson_correlation',      # 文献核心指标r
            'coherence',                # 相干性COH
            'partial_coherence',        # 部分相干性pCOH
            'spearman_correlation',     # 单调相关性
            'frequency_correlation',    # 频域相关性
            'mutual_information',       # 互信息
            'phase_correlation',        # 相位相关性
            'dtw_distance'              # DTW距离
        ]
        
        # 创建相关性矩阵
        corr_matrix = np.zeros((len(methods), len(metrics)))
        for i, method in enumerate(methods):
            for j, metric in enumerate(metrics):
                if metric in similarity_results[method]:
                    value = similarity_results[method][metric]
                    # 对于DTW距离，转换为相似性分数（距离越小，相似性越高）
                    if metric == 'dtw_distance':
                        # 归一化DTW距离到0-1范围，然后转换为相似性
                        max_dtw = max([r.get('dtw_distance', 0) for r in similarity_results.values()])
                        if max_dtw > 0:
                            value = 1 - (value / max_dtw)
                        else:
                            value = 1.0
                    corr_matrix[i, j] = value
                else:
                    corr_matrix[i, j] = np.nan
        
        # 创建热图
        plt.figure(figsize=(14, 10))
        
        # 自定义颜色映射
        cmap = sns.diverging_palette(10, 220, sep=80, n=7)
        
        sns.heatmap(corr_matrix, 
                   xticklabels=[self.get_metric_label(m) for m in metrics], 
                   yticklabels=methods,
                   annot=True, 
                   fmt='.3f', 
                   cmap=cmap,
                   center=0.5,
                   vmin=0, vmax=1,
                   cbar_kws={'label': '相似性分数'})
        
        plt.title(f'Subject {subject_id} - Experiment {exp_num} - 相似性指标热图\n(基于文献方法论的PPG-ABP相似性分析)', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('相似性指标', fontsize=12)
        plt.ylabel('预处理方法', fontsize=12)
        plt.tight_layout()
        
        # 保存图片
        save_path = os.path.join(self.results_dir, f'subject_{subject_id}_exp_{exp_num}_correlation_heatmap.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return save_path
    
    def get_metric_label(self, metric):
        """获取指标的显示标签"""
        labels = {
            'pearson_correlation': '相关系数r',
            'coherence': '相干性COH',
            'partial_coherence': '部分相干性pCOH',
            'spearman_correlation': 'Spearman相关',
            'frequency_correlation': '频域相关',
            'mutual_information': '互信息',
            'phase_correlation': '相位相关',
            'dtw_distance': 'DTW相似性'
        }
        return labels.get(metric, metric)
    
    def analyze_experiment(self, subject_id, experiment_number):
        """分析单个实验"""
        print(f"分析 Subject {subject_id} - Experiment {experiment_number}")
        
        # 加载数据
        data = self.load_calibrated_data(subject_id, experiment_number)
        if data is None:
            print(f"无法加载数据: Subject {subject_id} - Experiment {experiment_number}")
            return None
        
        # 提取信号
        if 'ir_signal' in data and 'abp_signal' in data:
            ir_signal = data['ir_signal']
            abp_signal = data['abp_signal']
            timestamps = data['timestamp']  # 修复键名
            acc_data = data.get('acc_data', None)
        else:
            print("数据格式不正确，需要包含ir_signal, abp_signal, timestamp")
            return None
        
        # 不同预处理方法
        preprocessing_methods = {
            '原始信号': ir_signal,
            'Butterworth滤波': self.preprocess_ir_signal(ir_signal, method='butterworth'),
            '小波去噪': self.preprocess_ir_signal(ir_signal, method='wavelet'),
            '形态学滤波': self.preprocess_ir_signal(ir_signal, method='morphological')
        }
        
        # 如果有加速度数据，添加运动伪影去除
        if acc_data and len(acc_data) > 0:
            # 检查是否有加速度数据
            has_acc = False
            for key in acc_data.keys():
                if any(axis in key for axis in ['ax', 'ay', 'az']):
                    has_acc = True
                    break
            
            if has_acc:
                motion_removed = self.remove_motion_artifacts(
                    preprocessing_methods['Butterworth滤波'].copy(), 
                    acc_data
                )
                preprocessing_methods['运动伪影去除'] = motion_removed
        
        # 计算相似性指标
        similarity_results = {}
        for method_name, ppg_signal in preprocessing_methods.items():
            print(f"  计算 {method_name} 的相似性指标...")
            similarity_results[method_name] = self.calculate_similarity_metrics(ppg_signal, abp_signal)
            
            # 计算SNR
            snr = self.calculate_snr(ppg_signal)
            similarity_results[method_name]['snr'] = snr
            print(f"    SNR: {snr:.2f} dB")
        
        # 可视化
        print("  生成可视化图表...")
        self.visualize_signals(preprocessing_methods, abp_signal, timestamps, subject_id, experiment_number)
        self.create_correlation_heatmap(similarity_results, subject_id, experiment_number)
        
        # 保存结果
        results_summary = {
            'subject_id': subject_id,
            'experiment_number': experiment_number,
            'similarity_metrics': similarity_results,
            'signal_length': len(ir_signal),
            'sampling_frequency': 100  # 假设100Hz
        }
        
        save_path = os.path.join(self.results_dir, f'subject_{subject_id}_exp_{experiment_number}_results.pkl')
        with open(save_path, 'wb') as f:
            pd.to_pickle(results_summary, f)
        
        print(f"  结果已保存到: {save_path}")
        return results_summary
    
    def run_analysis(self, subject_ids=None, experiment_numbers=None):
        """运行完整分析"""
        if subject_ids is None:
            # 默认分析所有可用受试者
            subject_ids = ['00017', '00018', '00019']  # 根据你的数据调整
        
        if experiment_numbers is None:
            experiment_numbers = [1, 2, 3]  # 默认分析前3个实验
        
        all_results = []
        
        for subject_id in subject_ids:
            for exp_num in experiment_numbers:
                try:
                    result = self.analyze_experiment(subject_id, exp_num)
                    if result:
                        all_results.append(result)
                except Exception as e:
                    print(f"分析失败: Subject {subject_id} - Experiment {exp_num}: {e}")
                    continue
        
        # 生成汇总报告
        if all_results:
            self.generate_summary_report(all_results)
        
        return all_results
    
    def generate_summary_report(self, results):
        """生成汇总报告（包含文献关键指标）"""
        print("\n" + "="*50)
        print("PPG-ABP相似性分析汇总报告")
        print("基于文献方法论的实现")
        print("="*50)
        
        # 计算平均相关性（按文献重要性排序）
        methods = list(results[0]['similarity_metrics'].keys())
        metrics = [
            'pearson_correlation',      # 文献核心指标r
            'coherence',                # 相干性COH
            'partial_coherence',        # 部分相干性pCOH
            'spearman_correlation',     # 单调相关性
            'frequency_correlation',    # 频域相关性
            'mutual_information',       # 互信息
            'phase_correlation',        # 相位相关性
            'dtw_distance',             # DTW距离
            'snr'                       # 信噪比
        ]
        
        summary_data = []
        for method in methods:
            method_results = []
            for result in results:
                if method in result['similarity_metrics']:
                    method_results.append(result['similarity_metrics'][method])
            
            if method_results:
                avg_metrics = {}
                for metric in metrics:
                    if metric in method_results[0]:
                        values = [r[metric] for r in method_results if not np.isnan(r[metric])]
                        if values:
                            avg_metrics[metric] = np.mean(values)
                        else:
                            avg_metrics[metric] = np.nan
                
                summary_data.append({
                    '预处理方法': method,
                    **avg_metrics
                })
        
        # 创建汇总表格
        summary_df = pd.DataFrame(summary_data)
        
        # 重新排列列顺序
        column_order = ['预处理方法'] + metrics
        summary_df = summary_df[column_order]
        
        print("\n平均相似性指标:")
        print(summary_df.round(4))
        
        # 文献关键指标分析
        print("\n" + "="*50)
        print("文献关键指标分析")
        print("="*50)
        
        # 检查相关系数r是否>0.9（文献标准）
        if 'pearson_correlation' in summary_df.columns:
            best_method = summary_df.loc[summary_df['pearson_correlation'].idxmax(), '预处理方法']
            best_corr = summary_df['pearson_correlation'].max()
            print(f"最佳相关系数r: {best_corr:.4f} ({best_method})")
            
            if best_corr > 0.9:
                print("✓ 达到文献标准: r > 0.9 (PPG和ABP形态高度相似)")
            elif best_corr > 0.8:
                print("⚠ 接近文献标准: r > 0.8 (PPG和ABP形态相似)")
            else:
                print("✗ 未达到文献标准: r < 0.8 (需要改进预处理方法)")
        
        # 检查相干性
        if 'coherence' in summary_df.columns:
            best_coherence = summary_df['coherence'].max()
            best_method_coherence = summary_df.loc[summary_df['coherence'].idxmax(), '预处理方法']
            print(f"最佳相干性COH: {best_coherence:.4f} ({best_method_coherence})")
        
        # 保存汇总报告
        summary_path = os.path.join(self.results_dir, 'similarity_analysis_summary.csv')
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        print(f"\n汇总报告已保存到: {summary_path}")
        
        # 生成文献对比报告
        self.generate_literature_comparison(summary_df)
        
        return summary_df
    
    def generate_literature_comparison(self, summary_df):
        """生成与文献的对比报告"""
        print("\n" + "="*50)
        print("与文献结果对比")
        print("="*50)
        
        # 文献中的关键结果
        literature_results = {
            '相关系数r': '> 0.9 (所有群体)',
            '相干性COH': '正常组 > 0.8, 高血压组 < 0.6',
            '部分相干性pCOH': '能显著区分正常和高血压群体 (p < 0.01)',
            'ABP→PPG因果性': '强 (> 0.7)',
            'PPG→ABP因果性': '弱 (< 0.3)',
            '分类准确率': '87.5% (神经网络分类器)'
        }
        
        print("文献报道的关键结果:")
        for metric, value in literature_results.items():
            print(f"  {metric}: {value}")
        
        # 你的结果与文献对比
        if 'pearson_correlation' in summary_df.columns:
            your_best_corr = summary_df['pearson_correlation'].max()
            print(f"\n你的最佳结果 vs 文献标准:")
            print(f"  相关系数r: {your_best_corr:.4f} vs > 0.9")
            
            if your_best_corr > 0.9:
                print("  ✓ 达到文献标准，PPG可以替代ABP")
            elif your_best_corr > 0.8:
                print("  ⚠ 接近文献标准，需要轻微改进")
            else:
                print("  ✗ 未达到文献标准，需要显著改进预处理方法")
        
        # 保存对比报告
        comparison_path = os.path.join(self.results_dir, 'literature_comparison_report.txt')
        with open(comparison_path, 'w', encoding='utf-8') as f:
            f.write("PPG-ABP相似性分析文献对比报告\n")
            f.write("="*50 + "\n\n")
            f.write("文献报道的关键结果:\n")
            for metric, value in literature_results.items():
                f.write(f"  {metric}: {value}\n")
            
            if 'pearson_correlation' in summary_df.columns:
                your_best_corr = summary_df['pearson_correlation'].max()
                f.write(f"\n你的最佳结果 vs 文献标准:\n")
                f.write(f"  相关系数r: {your_best_corr:.4f} vs > 0.9\n")
        
        print(f"\n对比报告已保存到: {comparison_path}")

def main():
    """主函数"""
    analyzer = PPGABPSimilarityAnalyzer()
    
    # 运行分析
    print("开始PPG-ABP信号相似性分析...")
    results = analyzer.run_analysis()
    
    print(f"\n分析完成！共分析了 {len(results)} 个实验。")
    print(f"结果保存在: {analyzer.results_dir}")

if __name__ == "__main__":
    main()
