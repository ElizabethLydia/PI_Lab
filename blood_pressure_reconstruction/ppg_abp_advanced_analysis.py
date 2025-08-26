#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级PPG-ABP信号分析脚本
可以分析不同实验、不同信号段，并且可以调整参数
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.stats import pearsonr, spearmanr
from scipy.interpolate import interp1d
from sklearn.metrics import mutual_info_score
import pywt
from scipy.ndimage import grey_opening, grey_closing
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和图表样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

class AdvancedPPGABPAnalyzer:
    def __init__(self, subject_id="00017"):
        """
        初始化高级分析器
        
        Args:
            subject_id: 受试者ID
        """
        self.subject_id = subject_id
        self.base_dir = f'/root/autodl-tmp/blood_pressure_reconstruction/{subject_id}/csv'
        
        # 获取可用的实验列表
        self.available_experiments = self.get_available_experiments()
        
        print(f"📋 受试者 {subject_id} 可用实验: {self.available_experiments}")
    
    def get_available_experiments(self):
        """获取可用的实验列表"""
        if not os.path.exists(self.base_dir):
            return []
        
        experiments = set()
        files = os.listdir(self.base_dir)
        
        for file in files:
            if file.endswith('_abp.csv'):
                parts = file.split('_')
                if len(parts) >= 2:
                    experiments.add(parts[1])
        
        return sorted(list(experiments))
    
    def analyze_experiment(self, experiment, segment_length=2000, start_idx=None, 
                          butterworth_params=(0.5, 8.0, 100, 4),
                          wavelet_params=('db4', 4),
                          morphological_params=(5,),
                          show_plots=True):
        """
        分析单个实验
        
        Args:
            experiment: 实验编号
            segment_length: 绘图段长度
            start_idx: 开始索引（None表示自动选择）
            butterworth_params: (lowcut, highcut, fs, order)
            wavelet_params: (wavelet, level)
            morphological_params: (size,)
            show_plots: 是否显示图表
        """
        print(f"\n{'='*80}")
        print(f"🔬 分析实验 {experiment}")
        print(f"{'='*80}")
        
        # 创建分析器实例
        analyzer = PPGABPAnalyzer(self.subject_id, experiment)
        
        # 自定义预处理参数
        analyzer.butterworth_params = butterworth_params
        analyzer.wavelet_params = wavelet_params
        analyzer.morphological_params = morphological_params
        
        # 运行分析
        try:
            analyzer.run_complete_analysis(
                segment_length=segment_length,
                start_idx=start_idx
            )
            
            # 返回分析结果
            return {
                'experiment': experiment,
                'analyzer': analyzer,
                'correlation_metrics': analyzer.correlation_metrics,
                'processed_signals': analyzer.processed_signals
            }
            
        except Exception as e:
            print(f"❌ 实验 {experiment} 分析失败: {e}")
            return None
    
    def analyze_multiple_experiments(self, experiments=None, segment_length=2000, 
                                   start_idx=None, show_plots=True):
        """
        分析多个实验
        
        Args:
            experiments: 实验列表（None表示分析所有实验）
            segment_length: 绘图段长度
            start_idx: 开始索引
            show_plots: 是否显示图表
        """
        if experiments is None:
            experiments = self.available_experiments
        
        print(f"\n🚀 开始分析多个实验...")
        print(f"📋 目标实验: {experiments}")
        print(f"📏 绘图段长度: {segment_length}")
        
        results = {}
        
        for exp in experiments:
            if exp in self.available_experiments:
                result = self.analyze_experiment(
                    experiment=exp,
                    segment_length=segment_length,
                    start_idx=start_idx,
                    show_plots=show_plots
                )
                if result:
                    results[exp] = result
            else:
                print(f"⚠️  实验 {exp} 不可用，跳过")
        
        return results
    
    def compare_experiments(self, experiments, segment_length=2000, start_idx=None):
        """
        比较多个实验的相关性指标
        """
        print(f"\n📊 比较多个实验的相关性指标...")
        
        # 分析所有实验
        results = self.analyze_multiple_experiments(
            experiments=experiments,
            segment_length=segment_length,
            start_idx=start_idx,
            show_plots=False  # 不显示图表，只计算指标
        )
        
        if not results:
            print("❌ 没有可用的分析结果")
            return None
        
        # 创建比较表格
        comparison_data = []
        
        for exp, result in results.items():
            metrics = result['correlation_metrics']
            
            for method_name, method_metrics in metrics.items():
                comparison_data.append({
                    'experiment': exp,
                    'method': method_name,
                    'pearson_r': method_metrics.get('pearson_r', np.nan),
                    'spearman_r': method_metrics.get('spearman_r', np.nan),
                    'mutual_info': method_metrics.get('mutual_info', np.nan),
                    'freq_correlation': method_metrics.get('freq_correlation', np.nan),
                    'coherence_mean': method_metrics.get('coherence_mean', np.nan),
                    'ppg_snr': method_metrics.get('ppg_snr', np.nan),
                    'abp_snr': method_metrics.get('abp_snr', np.nan)
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # 保存比较结果
        output_dir = f'/root/autodl-tmp/blood_pressure_reconstruction/{self.subject_id}/analysis_results'
        os.makedirs(output_dir, exist_ok=True)
        
        comparison_file = os.path.join(output_dir, f'{self.subject_id}_experiments_comparison.csv')
        comparison_df.to_csv(comparison_file, index=False, encoding='utf-8')
        print(f"💾 实验比较结果已保存: {comparison_file}")
        
        # 显示最佳结果
        print(f"\n🏆 最佳相关性结果:")
        best_pearson = comparison_df.loc[comparison_df['pearson_r'].idxmax()]
        print(f"  Pearson相关系数最高: 实验{best_pearson['experiment']} - {best_pearson['method']} (r={best_pearson['pearson_r']:.4f})")
        
        best_spearman = comparison_df.loc[comparison_df['spearman_r'].idxmax()]
        print(f"  Spearman相关系数最高: 实验{best_spearman['experiment']} - {best_spearman['method']} (r={best_spearman['spearman_r']:.4f})")
        
        best_mutual_info = comparison_df.loc[comparison_df['mutual_info'].idxmax()]
        print(f"  互信息最高: 实验{best_mutual_info['experiment']} - {best_mutual_info['method']} (MI={best_mutual_info['mutual_info']:.4f})")
        
        return comparison_df
    
    def plot_experiment_comparison(self, comparison_df):
        """绘制实验比较图"""
        if comparison_df is None or comparison_df.empty:
            print("❌ 没有比较数据可绘制")
            return
        
        print(f"\n📊 绘制实验比较图...")
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Pearson相关系数比较
        ax1 = axes[0, 0]
        sns.boxplot(data=comparison_df, x='method', y='pearson_r', ax=ax1)
        ax1.set_title('Pearson相关系数比较', fontweight='bold')
        ax1.set_xlabel('预处理方法')
        ax1.set_ylabel('Pearson相关系数')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Spearman相关系数比较
        ax2 = axes[0, 1]
        sns.boxplot(data=comparison_df, x='method', y='spearman_r', ax=ax2)
        ax2.set_title('Spearman相关系数比较', fontweight='bold')
        ax2.set_xlabel('预处理方法')
        ax2.set_ylabel('Spearman相关系数')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. 互信息比较
        ax3 = axes[1, 0]
        sns.boxplot(data=comparison_df, x='method', y='mutual_info', ax=ax3)
        ax3.set_title('互信息比较', fontweight='bold')
        ax3.set_xlabel('预处理方法')
        ax3.set_ylabel('互信息')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. 信噪比比较
        ax4 = axes[1, 1]
        sns.boxplot(data=comparison_df, x='method', y='ppg_snr', ax=ax4)
        ax4.set_title('PPG信噪比比较', fontweight='bold')
        ax4.set_xlabel('预处理方法')
        ax4.set_ylabel('PPG信噪比 (dB)')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # 保存图片
        output_dir = f'/root/autodl-tmp/blood_pressure_reconstruction/{self.subject_id}/analysis_results'
        os.makedirs(output_dir, exist_ok=True)
        
        plot_file = os.path.join(output_dir, f'{self.subject_id}_experiments_comparison.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"  💾 实验比较图已保存: {plot_file}")
        
        plt.show()
        
        return fig

class PPGABPAnalyzer:
    def __init__(self, subject_id="00017", experiment="1"):
        """
        初始化分析器
        
        Args:
            subject_id: 受试者ID
            experiment: 实验编号
        """
        self.subject_id = subject_id
        self.experiment = experiment
        self.base_dir = f'/root/autodl-tmp/blood_pressure_reconstruction/{subject_id}/csv'
        
        # 数据文件路径
        self.ppg_file = f'{subject_id}_{experiment}_sensor2.csv'
        self.abp_file = f'{subject_id}_{experiment}_abp.csv'
        
        # 默认预处理参数
        self.butterworth_params = (0.5, 8.0, 100, 4)
        self.wavelet_params = ('db4', 4)
        self.morphological_params = (5,)
        
        # 加载数据
        self.ppg_data = None
        self.abp_data = None
        self.load_data()
        
        # 预处理后的数据
        self.processed_signals = {}
        
        # 相关性指标
        self.correlation_metrics = {}
        
    def load_data(self):
        """加载PPG和ABP数据"""
        print(f"📖 加载数据...")
        print(f"  PPG文件: {self.ppg_file}")
        print(f"  ABP文件: {self.abp_file}")
        
        try:
            # 加载PPG数据
            ppg_path = os.path.join(self.base_dir, self.ppg_file)
            self.ppg_data = pd.read_csv(ppg_path)
            print(f"  ✅ PPG数据加载成功: {len(self.ppg_data)} 行")
            
            # 加载ABP数据
            abp_path = os.path.join(self.base_dir, self.abp_file)
            self.abp_data = pd.read_csv(abp_path)
            print(f"  ✅ ABP数据加载成功: {len(self.abp_data)} 行")
            
            # 检查数据长度
            print(f"  📊 数据长度对比:")
            print(f"    PPG: {len(self.ppg_data)} 行")
            print(f"    ABP: {len(self.abp_data)} 行")
            
            # 时间范围
            ppg_time_range = self.ppg_data['timestamp'].max() - self.ppg_data['timestamp'].min()
            abp_time_range = self.abp_data['timestamp'].max() - self.abp_data['timestamp'].min()
            print(f"  ⏱️  时间范围:")
            print(f"    PPG: {ppg_time_range:.2f} 秒")
            print(f"    ABP: {abp_time_range:.2f} 秒")
            
        except Exception as e:
            print(f"  ❌ 数据加载失败: {e}")
            raise
    
    def align_signals(self):
        """对齐PPG和ABP信号到相同的时间戳"""
        print(f"\n🔄 对齐信号...")
        
        # 使用PPG时间戳作为参考
        ref_timestamps = self.ppg_data['timestamp'].values
        ppg_ir = self.ppg_data['ir'].values
        ppg_red = self.ppg_data['red'].values
        ppg_green = self.ppg_data['green'].values
        
        # 插值ABP数据到PPG时间戳
        abp_interpolated = np.interp(
            ref_timestamps, 
            self.abp_data['timestamp'].values, 
            self.abp_data['abp'].values
        )
        
        # 创建对齐后的数据
        self.aligned_data = pd.DataFrame({
            'timestamp': ref_timestamps,
            'ppg_ir': ppg_ir,
            'ppg_red': ppg_red,
            'ppg_green': ppg_green,
            'abp': abp_interpolated
        })
        
        # 添加加速度数据
        if 'ax' in self.ppg_data.columns:
            self.aligned_data['ax'] = np.interp(
                ref_timestamps,
                self.ppg_data['timestamp'].values,
                self.ppg_data['ax'].values
            )
            self.aligned_data['ay'] = np.interp(
                ref_timestamps,
                self.ppg_data['timestamp'].values,
                self.ppg_data['ay'].values
            )
            self.aligned_data['az'] = np.interp(
                ref_timestamps,
                self.ppg_data['timestamp'].values,
                self.ppg_data['az'].values
            )
        
        print(f"  ✅ 信号对齐完成: {len(self.aligned_data)} 行")
        
        # 检查数据质量
        abp_nan_count = self.aligned_data['abp'].isna().sum()
        if abp_nan_count > 0:
            print(f"  ⚠️  ABP数据中有 {abp_nan_count} 个NaN值")
        
        return self.aligned_data
    
    def butterworth_filter(self, signal_data):
        """Butterworth带通滤波器"""
        lowcut, highcut, fs, order = self.butterworth_params
        nyquist = fs / 2
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = signal.butter(order, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, signal_data)
        return filtered
    
    def wavelet_denoising(self, signal_data):
        """小波去噪"""
        try:
            wavelet, level = self.wavelet_params
            # 小波分解
            coeffs = pywt.wavedec(signal_data, wavelet, level=level)
            
            # 阈值处理
            threshold = np.std(coeffs[-1]) * np.sqrt(2 * np.log(len(signal_data)))
            coeffs[1:] = [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
            
            # 小波重构
            denoised = pywt.waverec(coeffs, wavelet)
            
            # 确保长度一致
            if len(denoised) > len(signal_data):
                denoised = denoised[:len(signal_data)]
            elif len(denoised) < len(signal_data):
                denoised = np.pad(denoised, (0, len(signal_data) - len(denoised)), 'edge')
            
            return denoised
        except Exception as e:
            print(f"    小波去噪失败: {e}")
            return signal_data
    
    def morphological_filter(self, signal_data):
        """形态学滤波"""
        try:
            size = self.morphological_params[0]
            # 开运算（先腐蚀后膨胀）
            opened = grey_opening(signal_data, size=size)
            # 闭运算（先膨胀后腐蚀）
            closed = grey_closing(opened, size=size)
            return closed
        except Exception as e:
            print(f"    形态学滤波失败: {e}")
            return signal_data
    
    def remove_motion_artifacts(self, ppg_signal, acc_data):
        """使用加速度数据去除运动伪影"""
        try:
            if acc_data is None or len(acc_data) == 0:
                return ppg_signal
            
            # 计算加速度幅度
            acc_magnitude = np.sqrt(acc_data['ax']**2 + acc_data['ay']**2 + acc_data['az']**2)
            
            # 计算加速度阈值（动态阈值）
            acc_threshold = np.mean(acc_magnitude) + 2 * np.std(acc_magnitude)
            
            # 创建运动掩码
            motion_mask = acc_magnitude > acc_threshold
            
            # 对运动段进行插值
            ppg_cleaned = ppg_signal.copy()
            if np.any(motion_mask):
                # 找到非运动段的索引
                non_motion_indices = np.where(~motion_mask)[0]
                motion_indices = np.where(motion_mask)[0]
                
                if len(non_motion_indices) > 0:
                    # 使用非运动段的值插值运动段
                    ppg_cleaned[motion_indices] = np.interp(
                        motion_indices, 
                        non_motion_indices, 
                        ppg_signal[non_motion_indices]
                    )
            
            return ppg_cleaned
        except Exception as e:
            print(f"    运动伪影去除失败: {e}")
            return ppg_signal
    
    def apply_preprocessing_methods(self):
        """应用多种预处理方法"""
        print(f"\n🔧 应用预处理方法...")
        
        # 获取对齐后的数据
        if not hasattr(self, 'aligned_data'):
            self.align_signals()
        
        # 提取信号
        ppg_ir = self.aligned_data['ppg_ir'].values
        abp = self.aligned_data['abp'].values
        
        # 1. 原始信号
        self.processed_signals['原始信号'] = {
            'ppg': ppg_ir,
            'abp': abp
        }
        
        # 2. Butterworth滤波
        print(f"  🔧 Butterworth滤波...")
        ppg_butter = self.butterworth_filter(ppg_ir)
        self.processed_signals['Butterworth滤波'] = {
            'ppg': ppg_butter,
            'abp': abp
        }
        
        # 3. 小波去噪
        print(f"  🔧 小波去噪...")
        ppg_wavelet = self.wavelet_denoising(ppg_ir)
        self.processed_signals['小波去噪'] = {
            'ppg': ppg_wavelet,
            'abp': abp
        }
        
        # 4. 形态学滤波
        print(f"  🔧 形态学滤波...")
        ppg_morph = self.morphological_filter(ppg_ir)
        self.processed_signals['形态学滤波'] = {
            'ppg': ppg_morph,
            'abp': abp
        }
        
        # 5. 组合滤波（Butterworth + 小波）
        print(f"  🔧 组合滤波...")
        ppg_combined = self.butterworth_filter(ppg_wavelet)
        self.processed_signals['组合滤波'] = {
            'ppg': ppg_combined,
            'abp': abp
        }
        
        # 6. 运动伪影去除
        if 'ax' in self.aligned_data.columns:
            print(f"  🔧 运动伪影去除...")
            acc_data = self.aligned_data[['ax', 'ay', 'az']]
            ppg_motion_removed = self.remove_motion_artifacts(ppg_ir, acc_data)
            self.processed_signals['运动伪影去除'] = {
                'ppg': ppg_motion_removed,
                'abp': abp
            }
        
        print(f"  ✅ 预处理完成，共 {len(self.processed_signals)} 种方法")
        
        return self.processed_signals
    
    def calculate_correlation_metrics(self):
        """计算各种相关性指标"""
        print(f"\n📊 计算相关性指标...")
        
        for method_name, signals in self.processed_signals.items():
            ppg = signals['ppg']
            abp = signals['abp']
            
            # 去除NaN值
            valid_mask = ~(np.isnan(ppg) | np.isnan(abp))
            if np.sum(valid_mask) < 10:  # 至少需要10个有效点
                continue
            
            ppg_valid = ppg[valid_mask]
            abp_valid = abp[valid_mask]
            
            metrics = {}
            
            try:
                # 1. Pearson相关系数
                pearson_r, pearson_p = pearsonr(ppg_valid, abp_valid)
                metrics['pearson_r'] = pearson_r
                metrics['pearson_p'] = pearson_p
                
                # 2. Spearman相关系数
                spearman_r, spearman_p = spearmanr(ppg_valid, abp_valid)
                metrics['spearman_r'] = spearman_r
                metrics['spearman_p'] = spearman_p
                
                # 3. 互信息
                # 将连续值分箱以计算互信息
                ppg_binned = pd.cut(ppg_valid, bins=20, labels=False)
                abp_binned = pd.cut(abp_valid, bins=20, labels=False)
                mutual_info = mutual_info_score(ppg_binned, abp_binned)
                metrics['mutual_info'] = mutual_info
                
                # 4. 频率域相关性
                ppg_fft = np.abs(np.fft.fft(ppg_valid))
                abp_fft = np.abs(np.fft.fft(abp_valid))
                freq_corr, _ = pearsonr(ppg_fft, abp_fft)
                metrics['freq_correlation'] = freq_corr
                
                # 5. 相干性
                if len(ppg_valid) > 100:
                    f, coh = signal.coherence(ppg_valid, abp_valid, fs=100)
                    metrics['coherence_mean'] = np.mean(coh)
                    metrics['coherence_max'] = np.max(coh)
                else:
                    metrics['coherence_mean'] = np.nan
                    metrics['coherence_max'] = np.nan
                
                # 6. 信号质量指标
                ppg_snr = self.calculate_snr(ppg_valid)
                abp_snr = self.calculate_snr(abp_valid)
                metrics['ppg_snr'] = ppg_snr
                metrics['abp_snr'] = abp_snr
                
                self.correlation_metrics[method_name] = metrics
                
            except Exception as e:
                print(f"    ❌ {method_name} 相关性计算失败: {e}")
                continue
        
        print(f"  ✅ 相关性计算完成，共 {len(self.correlation_metrics)} 种方法")
        return self.correlation_metrics
    
    def calculate_snr(self, signal_data):
        """计算信噪比"""
        try:
            # 使用FFT计算信噪比
            fft = np.fft.fft(signal_data)
            power = np.abs(fft)**2
            
            # 假设低频部分是信号，高频部分是噪声
            mid_freq = len(power) // 2
            signal_power = np.mean(power[:mid_freq])
            noise_power = np.mean(power[mid_freq:])
            
            if noise_power > 0:
                snr = 10 * np.log10(signal_power / noise_power)
                return snr
            else:
                return np.nan
        except:
            return np.nan
    
    def plot_signals_comparison(self, segment_length=1000, start_idx=None):
        """绘制信号对比图"""
        print(f"\n📊 绘制信号对比图...")
        
        if not self.processed_signals:
            self.apply_preprocessing_methods()
        
        if not self.correlation_metrics:
            self.calculate_correlation_metrics()
        
        # 确定绘图段
        total_length = len(self.aligned_data)
        if start_idx is None:
            start_idx = total_length // 4  # 从1/4处开始
        
        end_idx = min(start_idx + segment_length, total_length)
        
        print(f"  📏 绘图段: {start_idx} - {end_idx} (共 {end_idx - start_idx} 个点)")
        
        # 创建子图
        n_methods = len(self.processed_signals)
        fig, axes = plt.subplots(n_methods, 2, figsize=(20, 4*n_methods))
        
        if n_methods == 1:
            axes = axes.reshape(1, -1)
        
        # 时间轴
        time_segment = self.aligned_data['timestamp'].iloc[start_idx:end_idx]
        time_relative = time_segment - time_segment.iloc[0]
        
        # 绘制每种预处理方法的结果
        for i, (method_name, signals) in enumerate(self.processed_signals.items()):
            ppg = signals['ppg'][start_idx:end_idx]
            abp = signals['abp'][start_idx:end_idx]
            
            # 获取相关性指标
            metrics = self.correlation_metrics.get(method_name, {})
            
            # PPG信号
            ax1 = axes[i, 0]
            ax1.plot(time_relative, ppg, 'b-', linewidth=1, alpha=0.8, label='PPG (IR)')
            ax1.set_title(f'{method_name} - PPG信号', fontsize=12, fontweight='bold')
            ax1.set_xlabel('时间 (秒)')
            ax1.set_ylabel('PPG值')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # 添加相关性信息
            if metrics:
                corr_text = f"Pearson r: {metrics.get('pearson_r', 'N/A'):.3f}\n"
                corr_text += f"Spearman r: {metrics.get('spearman_r', 'N/A'):.3f}\n"
                corr_text += f"互信息: {metrics.get('mutual_info', 'N/A'):.3f}"
                ax1.text(0.02, 0.98, corr_text, transform=ax1.transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # ABP信号
            ax2 = axes[i, 1]
            ax2.plot(time_relative, abp, 'r-', linewidth=1, alpha=0.8, label='ABP')
            ax2.set_title(f'{method_name} - ABP信号', fontsize=12, fontweight='bold')
            ax2.set_xlabel('时间 (秒)')
            ax2.set_ylabel('血压 (mmHg)')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # 添加信号质量信息
            if metrics:
                quality_text = f"PPG SNR: {metrics.get('ppg_snr', 'N/A'):.1f} dB\n"
                quality_text += f"ABP SNR: {metrics.get('abp_snr', 'N/A'):.1f} dB\n"
                quality_text += f"频率相关性: {metrics.get('freq_correlation', 'N/A'):.3f}"
                ax2.text(0.02, 0.98, quality_text, transform=ax2.transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        
        plt.tight_layout()
        
        # 保存图片
        output_dir = f'/root/autodl-tmp/blood_pressure_reconstruction/{self.subject_id}/analysis_results'
        os.makedirs(output_dir, exist_ok=True)
        
        plot_file = os.path.join(output_dir, f'{self.subject_id}_{self.experiment}_signals_comparison.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"  💾 信号对比图已保存: {plot_file}")
        
        plt.show()
        
        return fig
    
    def plot_correlation_heatmap(self):
        """绘制相关性热力图"""
        print(f"\n📊 绘制相关性热力图...")
        
        if not self.correlation_metrics:
            self.calculate_correlation_metrics()
        
        # 准备数据
        methods = list(self.correlation_metrics.keys())
        metrics_names = ['pearson_r', 'spearman_r', 'mutual_info', 'freq_correlation', 'coherence_mean']
        
        # 创建相关性矩阵
        corr_matrix = np.zeros((len(methods), len(metrics_names)))
        
        for i, method in enumerate(methods):
            for j, metric in enumerate(metrics_names):
                corr_matrix[i, j] = self.correlation_metrics[method].get(metric, np.nan)
        
        # 创建热力图
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 使用seaborn绘制热力图
        sns.heatmap(corr_matrix, 
                   xticklabels=metrics_names, 
                   yticklabels=methods,
                   annot=True, 
                   fmt='.3f', 
                   cmap='RdYlBu_r',
                   center=0,
                   ax=ax)
        
        ax.set_title(f'受试者{self.subject_id} 实验{self.experiment} - 相关性指标热力图', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('相关性指标', fontsize=12)
        ax.set_ylabel('预处理方法', fontsize=12)
        
        # 旋转x轴标签
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        # 保存图片
        output_dir = f'/root/autodl-tmp/blood_pressure_reconstruction/{self.subject_id}/analysis_results'
        os.makedirs(output_dir, exist_ok=True)
        
        heatmap_file = os.path.join(output_dir, f'{self.subject_id}_{self.experiment}_correlation_heatmap.png')
        plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
        print(f"  💾 相关性热力图已保存: {heatmap_file}")
        
        plt.show()
        
        return fig
    
    def generate_summary_report(self):
        """生成分析总结报告"""
        print(f"\n📝 生成分析总结报告...")
        
        if not self.correlation_metrics:
            self.calculate_correlation_metrics()
        
        # 创建输出目录
        output_dir = f'/root/autodl-tmp/blood_pressure_reconstruction/{self.subject_id}/analysis_results'
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成报告
        report_file = os.path.join(output_dir, f'{self.subject_id}_{self.experiment}_analysis_report.txt')
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"PPG-ABP信号分析报告\n")
            f.write("="*50 + "\n\n")
            f.write(f"受试者ID: {self.subject_id}\n")
            f.write(f"实验编号: {self.experiment}\n")
            f.write(f"分析时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("预处理参数:\n")
            f.write("-"*20 + "\n")
            f.write(f"Butterworth滤波: 低截止={self.butterworth_params[0]}Hz, 高截止={self.butterworth_params[1]}Hz, 采样率={self.butterworth_params[2]}Hz, 阶数={self.butterworth_params[3]}\n")
            f.write(f"小波去噪: 小波={self.wavelet_params[0]}, 分解层数={self.wavelet_params[1]}\n")
            f.write(f"形态学滤波: 结构元素大小={self.morphological_params[0]}\n\n")
            
            f.write("数据概览:\n")
            f.write("-"*20 + "\n")
            f.write(f"PPG数据长度: {len(self.aligned_data)} 行\n")
            f.write(f"时间范围: {self.aligned_data['timestamp'].max() - self.aligned_data['timestamp'].min():.2f} 秒\n")
            f.write(f"采样频率: 约 {len(self.aligned_data) / (self.aligned_data['timestamp'].max() - self.aligned_data['timestamp'].min()):.1f} Hz\n\n")
            
            f.write("相关性分析结果:\n")
            f.write("-"*30 + "\n")
            
            # 按Pearson相关系数排序
            sorted_methods = sorted(
                self.correlation_metrics.items(),
                key=lambda x: x[1].get('pearson_r', -1),
                reverse=True
            )
            
            for method_name, metrics in sorted_methods:
                f.write(f"\n{method_name}:\n")
                f.write(f"  Pearson相关系数: {metrics.get('pearson_r', 'N/A'):.4f}\n")
                f.write(f"  Spearman相关系数: {metrics.get('spearman_r', 'N/A'):.4f}\n")
                f.write(f"  互信息: {metrics.get('mutual_info', 'N/A'):.4f}\n")
                f.write(f"  频率相关性: {metrics.get('freq_correlation', 'N/A'):.4f}\n")
                f.write(f"  平均相干性: {metrics.get('coherence_mean', 'N/A'):.4f}\n")
                f.write(f"  PPG信噪比: {metrics.get('ppg_snr', 'N/A'):.1f} dB\n")
                f.write(f"  ABP信噪比: {metrics.get('abp_snr', 'N/A'):.4f} dB\n")
            
            f.write(f"\n最佳预处理方法:\n")
            f.write("-"*25 + "\n")
            best_method = sorted_methods[0][0]
            best_pearson = sorted_methods[0][1].get('pearson_r', 0)
            f.write(f"方法: {best_method}\n")
            f.write(f"Pearson相关系数: {best_pearson:.4f}\n")
            
            if best_pearson > 0.9:
                f.write("评价: 相关性极强 (r > 0.9)\n")
            elif best_pearson > 0.7:
                f.write("评价: 相关性强 (r > 0.7)\n")
            elif best_pearson > 0.5:
                f.write("评价: 相关性中等 (r > 0.5)\n")
            else:
                f.write("评价: 相关性较弱 (r < 0.5)\n")
            
            f.write(f"\n输出文件:\n")
            f.write("-"*15 + "\n")
            f.write(f"信号对比图: {self.subject_id}_{self.experiment}_signals_comparison.png\n")
            f.write(f"相关性热力图: {self.subject_id}_{self.experiment}_correlation_heatmap.png\n")
            f.write(f"分析报告: {self.subject_id}_{self.experiment}_analysis_report.txt\n")
        
        print(f"  💾 分析报告已保存: {report_file}")
        
        return report_file
    
    def run_complete_analysis(self, segment_length=1000, start_idx=None):
        """运行完整分析流程"""
        print(f"\n🚀 开始完整分析流程...")
        print(f"{'='*60}")
        
        try:
            # 1. 应用预处理方法
            self.apply_preprocessing_methods()
            
            # 2. 计算相关性指标
            self.calculate_correlation_metrics()
            
            # 3. 绘制信号对比图
            self.plot_signals_comparison(segment_length, start_idx)
            
            # 4. 绘制相关性热力图
            self.plot_correlation_heatmap()
            
            # 5. 生成分析报告
            self.generate_summary_report()
            
            print(f"\n🎉 完整分析流程完成！")
            print(f"📁 结果保存在: /root/autodl-tmp/blood_pressure_reconstruction/{self.subject_id}/analysis_results/")
            
        except Exception as e:
            print(f"❌ 分析过程中发生错误: {e}")
            raise

def main():
    """主函数"""
    print("🚀 高级PPG-ABP信号分析系统")
    print("="*60)
    
    # 创建高级分析器
    analyzer = AdvancedPPGABPAnalyzer(subject_id="00017")
    
    # 示例1: 分析单个实验，使用自定义参数
    print(f"\n🔬 示例1: 分析实验1，使用自定义参数")
    custom_analyzer = PPGABPAnalyzer(subject_id="00017", experiment="1")
    
    # 自定义预处理参数
    custom_analyzer.butterworth_params = (0.3, 10.0, 100, 6)  # 更宽的频带，更高阶数
    custom_analyzer.wavelet_params = ('db6', 6)  # 不同的小波和分解层数
    custom_analyzer.morphological_params = (7,)  # 更大的结构元素
    
    # 运行分析
    custom_analyzer.run_complete_analysis(segment_length=1500, start_idx=20000)
    
    # 示例2: 比较多个实验
    print(f"\n🔬 示例2: 比较多个实验的相关性")
    comparison_df = analyzer.compare_experiments(
        experiments=['1', '2', '3'],  # 比较前3个实验
        segment_length=1000,
        start_idx=None
    )
    
    # 绘制比较图
    if comparison_df is not None:
        analyzer.plot_experiment_comparison(comparison_df)

if __name__ == "__main__":
    main()
