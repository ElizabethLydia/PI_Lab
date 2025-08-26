#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PPG-ABP信号分析和可视化脚本
针对17号受试者，实现多种预处理方法并计算相关性指标
交互式可视化，图表标签使用英文便于汇报
支持批量处理所有实验和所有传感器
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

# 设置matplotlib后端，确保图表正确显示（服务器环境兼容）
import matplotlib
try:
    # 尝试使用TkAgg后端（如果有GUI环境）
    matplotlib.use('TkAgg')
    print("✅ 使用TkAgg后端，支持弹窗显示")
except ImportError:
    try:
        # 回退到Agg后端（服务器环境）
        matplotlib.use('Agg')
        print("⚠️  使用Agg后端，图表将保存到文件")
    except:
        # 最后回退到默认后端
        print("⚠️  使用默认matplotlib后端")
        pass

# 设置英文字体和图表样式，便于汇报使用
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# 设置seaborn样式，美化图表
sns.set_palette("husl")
sns.set_context("notebook", font_scale=1.2)

class PPGABPAnalyzer:
    def __init__(self, subject_id="00017", experiment="1", sensor_name="sensor2"):
        """
        初始化分析器
        
        Args:
            subject_id: 受试者ID
            experiment: 实验编号
            sensor_name: 传感器名称
        """
        self.subject_id = subject_id
        self.experiment = experiment
        self.sensor_name = sensor_name
        self.base_dir = f'/root/autodl-tmp/blood_pressure_reconstruction/{subject_id}/csv'
        
        # 数据文件路径
        self.ppg_file = f'{subject_id}_{experiment}_{sensor_name}.csv'
        self.abp_file = f'{subject_id}_{experiment}_abp.csv'
        
        # 加载数据
        self.ppg_data = None
        self.abp_data = None
        self.load_data()
        
        # 预处理后的数据
        self.processed_signals = {}
        
        # 相关性指标
        self.correlation_metrics = {}
        
        # 运动检测缓存
        self.acc_available = False
        self.acc_magnitude = None
        self.motion_mask = None
        
    def load_data(self):
        """加载PPG和ABP数据"""
        print(f"📖 加载数据...")
        print(f"  PPG文件: {self.ppg_file}  (传感器: {self.sensor_name})")
        print(f"  ABP文件: {self.abp_file}")
        
        try:
            # 加载PPG数据
            ppg_path = os.path.join(self.base_dir, self.ppg_file)
            self.ppg_data = pd.read_csv(ppg_path)
            print(f"  ✅ PPG数据加载成功: {len(self.ppg_data)} 行")
            print(f"    列: {list(self.ppg_data.columns)}")
            
            # 加载ABP数据
            abp_path = os.path.join(self.base_dir, self.abp_file)
            self.abp_data = pd.read_csv(abp_path)
            print(f"  ✅ ABP数据加载成功: {len(self.abp_data)} 行")
            print(f"    列: {list(self.abp_data.columns)}")
            
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
    
    def _compute_motion_mask(self):
        """基于加速度幅值计算运动掩码并缓存。"""
        if not self.acc_available:
            self.motion_mask = None
            return
        try:
            acc_mag = self.acc_magnitude
            threshold = np.mean(acc_mag) + 2.0 * np.std(acc_mag)
            self.motion_mask = acc_mag > threshold
            print(f"  🏃 运动检测: 使用传感器{self.sensor_name}的加速度(ax, ay, az)，阈值={threshold:.3f}，运动占比={(np.mean(self.motion_mask)*100):.1f}%")
        except Exception as e:
            print(f"  ⚠️  运动掩码计算失败: {e}")
            self.motion_mask = None
    
    def _spec(self, x, fs=None):
        """完全按照MATLAB spec函数实现的频谱分析
        Input:  x   - 1xN pulse signal
                fs  - 1x1 camera frame rate
        Output: spc - 1xM pulse spectrogram
                rate - 1xM heart rate signal
        """
        if fs is None:
            fs = 100  # 默认采样率
        
        # 确保x是1D数组
        x = np.asarray(x).flatten()
        
        # 定义参数 (MATLAB: L = fps * 10)
        L = int(fs * 10)
        
        # 初始化S矩阵 (MATLAB: S = zeros(size(signal,2)-L+1,L))
        S = np.zeros((len(x) - L + 1, L))
        
        # 滑动窗口处理 (MATLAB: for idx = 1:size(signal,2)-L+1)
        for idx in range(len(x) - L + 1):
            p = x[idx:idx + L]
            # 标准化 (MATLAB: p = (p-mean(p))/(eps+std(p)))
            S[idx, :] = (p - np.mean(p)) / (np.std(p) + np.finfo(float).eps)
        
        # 去除均值 (MATLAB: S = S-repmat(mean(S,2),[1,L]))
        S = S - np.mean(S, axis=1, keepdims=True)
        
        # 应用Hann窗 (MATLAB: S = S .* repmat(hann(L)',[size(S,1),L]))
        hann_window = signal.windows.hann(L, sym=False)
        S = S * hann_window
        
        # FFT (MATLAB: S = abs(fft(S,fps*60,2)))
        fft_size = fs * 60
        S_fft = np.abs(np.fft.fft(S, fft_size, axis=1))
        
        # 取前半部分并转置 (MATLAB: spc = S(:,1:fps*60/2)')
        spc = S_fft[:, :fft_size//2].T
        
        # 计算心率 (MATLAB: [~, rate] = max(spc,[],1); rate = rate - 1)
        rate = np.argmax(spc, axis=0)
        rate = rate - 1
        
        return spc, rate
    
    def pca_extraction(self, rgb_data):
        """PCA方法提取PPG信号"""
        try:
            from sklearn.decomposition import PCA
            
            # 处理NaN值
            rgb_clean = rgb_data.copy()
            if np.any(np.isnan(rgb_clean)):
                # 用前一个有效值填充NaN
                for i in range(rgb_clean.shape[1]):
                    rgb_clean[:, i] = pd.Series(rgb_clean[:, i]).fillna(method='ffill').fillna(method='bfill').values
                if np.any(np.isnan(rgb_clean)):
                    return None, None
            
            # 预处理每个通道
            processed_data = np.zeros_like(rgb_clean)
            for i in range(3):
                processed_data[:, i] = self.preprocess_signal(rgb_clean[:, i])
            
            # 应用PCA
            pca = PCA(n_components=3)
            pca_result = pca.fit_transform(processed_data)
            
            # 第一主成分通常包含PPG信号
            ppg_signal = pca_result[:, 0]
            explained_variance = pca.explained_variance_ratio_
            
            return ppg_signal, explained_variance
            
        except Exception as e:
            print(f"    PCA提取失败: {e}")
            return None, None
    
    def svd_extraction(self, rgb_data):
        """SVD方法提取PPG信号"""
        try:
            from scipy.linalg import svd
            
            # 处理NaN值
            rgb_clean = rgb_data.copy()
            if np.any(np.isnan(rgb_clean)):
                # 用前一个有效值填充NaN
                for i in range(rgb_clean.shape[1]):
                    rgb_clean[:, i] = pd.Series(rgb_clean[:, i]).fillna(method='ffill').fillna(method='bfill').values
                if np.any(np.isnan(rgb_clean)):
                    return None, None
            
            # 预处理每个通道
            processed_data = np.zeros_like(rgb_clean)
            for i in range(3):
                processed_data[:, i] = self.preprocess_signal(rgb_clean[:, i])
            
            # 应用SVD
            U, s, Vt = svd(processed_data, full_matrices=False)
            
            # 第一奇异值对应的左奇异向量通常包含PPG信号
            ppg_signal = U[:, 0] * s[0]
            singular_values = s
            
            return ppg_signal, singular_values
            
        except Exception as e:
            print(f"    SVD提取失败: {e}")
            return None, None
    
    def pos_extraction(self, rgb_data):
        """POS (Plane Orthogonal to Skin) 方法提取PPG信号"""
        try:
            # 处理NaN值
            rgb_clean = rgb_data.copy()
            if np.any(np.isnan(rgb_clean)):
                # 用前一个有效值填充NaN
                for i in range(rgb_clean.shape[1]):
                    rgb_clean[:, i] = pd.Series(rgb_clean[:, i]).fillna(method='ffill').fillna(method='bfill').values
                if np.any(np.isnan(rgb_clean)):
                    return None
            
            # 预处理每个通道
            processed_data = np.zeros_like(rgb_clean)
            for i in range(3):
                processed_data[:, i] = self.preprocess_signal(rgb_clean[:, i])
            
            # POS算法实现
            # 1. 计算RGB通道的归一化值
            r_norm = processed_data[:, 0] / np.mean(processed_data[:, 0])
            g_norm = processed_data[:, 1] / np.mean(processed_data[:, 1])
            b_norm = processed_data[:, 2] / np.mean(processed_data[:, 2])
            
            # 2. 计算POS信号
            # POS = (r_norm - g_norm) + (r_norm - b_norm)
            pos_signal = (r_norm - g_norm) + (r_norm - b_norm)
            
            # 3. 再次应用带通滤波
            ppg_signal = self.butterworth_filter(pos_signal, lowcut=0.5, highcut=4.0, order=4)
            
            return ppg_signal
            
        except Exception as e:
            print(f"    POS提取失败: {e}")
            return None
    
    def chrom_extraction(self, rgb_data):
        """CHROM (Chrominance-based) 方法提取PPG信号"""
        try:
            # 处理NaN值
            rgb_clean = rgb_data.copy()
            if np.any(np.isnan(rgb_clean)):
                # 用前一个有效值填充NaN
                for i in range(rgb_clean.shape[1]):
                    rgb_clean[:, i] = pd.Series(rgb_clean[:, i]).fillna(method='ffill').fillna(method='bfill').values
                if np.any(np.isnan(rgb_clean)):
                    return None
            
            # 预处理每个通道
            processed_data = np.zeros_like(rgb_clean)
            for i in range(3):
                processed_data[:, i] = self.preprocess_signal(rgb_clean[:, i])
            
            # CHROM算法实现
            # 1. 计算归一化RGB值
            r_norm = processed_data[:, 0] / np.mean(processed_data[:, 0])
            g_norm = processed_data[:, 1] / np.mean(processed_data[:, 1])
            b_norm = processed_data[:, 2] / np.mean(processed_data[:, 2])
            
            # 2. 计算色度信号
            # CHROM = (r_norm - g_norm) / (r_norm + g_norm - 2*b_norm)
            denominator = r_norm + g_norm - 2 * b_norm
            
            # 避免除零
            epsilon = 1e-10
            denominator = np.where(np.abs(denominator) < epsilon, epsilon, denominator)
            
            chrom_signal = (r_norm - g_norm) / denominator
            
            # 3. 再次应用带通滤波
            ppg_signal = self.butterworth_filter(chrom_signal, lowcut=0.5, highcut=4.0, order=4)
            
            return ppg_signal
            
        except Exception as e:
            print(f"    CHROM提取失败: {e}")
            return None
    
    def ica_extraction(self, rgb_data):
        """ICA (独立分量分析) 方法提取PPG信号"""
        try:
            from sklearn.decomposition import FastICA
            
            # 处理NaN值
            rgb_clean = rgb_data.copy()
            if np.any(np.isnan(rgb_clean)):
                # 用前一个有效值填充NaN
                for i in range(rgb_clean.shape[1]):
                    rgb_clean[:, i] = pd.Series(rgb_clean[:, i]).fillna(method='ffill').fillna(method='bfill').values
                if np.any(np.isnan(rgb_clean)):
                    return None
            
            # 预处理每个通道
            processed_data = np.zeros_like(rgb_clean)
            for i in range(3):
                processed_data[:, i] = self.preprocess_signal(rgb_clean[:, i])
            
            # 应用ICA
            ica = FastICA(n_components=3, random_state=42)
            ica_result = ica.fit_transform(processed_data)
            
            # 选择方差最大的分量作为PPG信号
            variances = np.var(ica_result, axis=0)
            best_component = np.argmax(variances)
            ppg_signal = ica_result[:, best_component]
            
            return ppg_signal
            
        except Exception as e:
            print(f"    ICA提取失败: {e}")
            return None
    
    def nmf_extraction(self, rgb_data):
        """NMF (非负矩阵分解) 方法提取PPG信号"""
        try:
            from sklearn.decomposition import NMF
            
            # 处理NaN值
            rgb_clean = rgb_data.copy()
            if np.any(np.isnan(rgb_clean)):
                # 用前一个有效值填充NaN
                for i in range(rgb_clean.shape[1]):
                    rgb_clean[:, i] = pd.Series(rgb_clean[:, i]).fillna(method='ffill').fillna(method='bfill').values
                if np.any(np.isnan(rgb_clean)):
                    return None
            
            # 预处理：确保数据非负
            processed_data = np.abs(rgb_clean)
            
            # 应用NMF
            nmf = NMF(n_components=3, random_state=42, max_iter=200)
            nmf_result = nmf.fit_transform(processed_data)
            
            # 选择重构误差最小的分量作为PPG信号
            reconstruction_errors = []
            for i in range(3):
                # 重构单个分量 - 修复矩阵维度问题
                component_i = nmf.components_[i:i+1]  # (1, 3)
                scores_i = nmf_result[:, i:i+1]      # (n_samples, 1)
                reconstructed = scores_i @ component_i  # (n_samples, 3)
                error = np.mean((processed_data - reconstructed) ** 2)
                reconstruction_errors.append(error)
            
            best_component = np.argmin(reconstruction_errors)
            ppg_signal = nmf_result[:, best_component]
            
            # 应用带通滤波
            ppg_signal = self.butterworth_filter(ppg_signal, lowcut=0.5, highcut=4.0, order=4)
            
            return ppg_signal
            
        except Exception as e:
            print(f"    NMF提取失败: {e}")
            return None
    
    def preprocess_signal(self, data):
        """预处理信号：归一化 + 带通滤波"""
        # 1. 归一化
        normalized = (data - np.mean(data)) / (np.std(data) + 1e-10)
        
        # 2. 带通滤波
        filtered = self.butterworth_filter(normalized, lowcut=0.5, highcut=4.0, order=4)
        
        return filtered
    
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
        
        # 添加加速度数据并计算运动掩码
        if all(col in self.ppg_data.columns for col in ['ax', 'ay', 'az']):
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
            # 计算加速度幅度
            self.acc_magnitude = np.sqrt(self.aligned_data['ax']**2 + self.aligned_data['ay']**2 + self.aligned_data['az']**2).values
            self.acc_available = True
            print(f"  ✅ 加速度列存在(ax, ay, az)，将用于运动伪影检测和去除")
            self._compute_motion_mask()
        else:
            self.acc_available = False
            self.acc_magnitude = None
            self.motion_mask = None
            print(f"  ⚠️  未发现加速度列(ax, ay, az)，跳过运动伪影去除")
        
        print(f"  ✅ 信号对齐完成: {len(self.aligned_data)} 行")
        
        # 检查数据质量
        abp_nan_count = self.aligned_data['abp'].isna().sum()
        if abp_nan_count > 0:
            print(f"  ⚠️  ABP数据中有 {abp_nan_count} 个NaN值")
        
        return self.aligned_data
    
    def _auto_detect_fs(self):
        """自动检测采样率"""
        try:
            if hasattr(self, 'aligned_data') and len(self.aligned_data) > 1:
                timestamps = self.aligned_data['timestamp'].values
                time_diff = np.diff(timestamps)
                # 计算平均时间间隔
                mean_interval = np.mean(time_diff)
                # 采样率 = 1 / 时间间隔
                fs = 1.0 / mean_interval
                print(f"  📊 自动检测采样率: {fs:.2f} Hz")
                return fs
            else:
                print(f"  ⚠️  无法检测采样率，使用默认值: 100 Hz")
                return 100.0
        except Exception as e:
            print(f"  ⚠️  采样率检测失败: {e}，使用默认值: 100 Hz")
            return 100.0
    
    def butterworth_filter(self, signal_data, lowcut=0.5, highcut=4.0, fs=None, order=4):
        """Butterworth带通滤波器 - 去除低频漂移和高频噪声"""
        if fs is None:
            fs = self._auto_detect_fs()
        
        nyquist = fs / 2
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = signal.butter(order, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, signal_data)
        return filtered
    
    def pca_denoising(self, signal_data, n_components=3):
        """PCA去噪 - 先带通滤波，再进行PCA降噪"""
        try:
            from sklearn.decomposition import PCA
            
            # 步骤1: 先进行带通滤波
            fs = self._auto_detect_fs()
            filtered_signal = self.butterworth_filter(signal_data, lowcut=0.5, highcut=4.0, fs=fs, order=4)
            
            # 步骤2: 将信号分段，每段100个点
            segment_length = 100
            n_segments = len(filtered_signal) // segment_length
            
            if n_segments < 2:
                return filtered_signal
            
            # 创建信号矩阵
            signal_matrix = []
            for i in range(n_segments):
                segment = filtered_signal[i*segment_length:(i+1)*segment_length]
                signal_matrix.append(segment)
            
            # 如果最后一段不完整，用前一段填充
            if len(filtered_signal) % segment_length != 0:
                last_segment = filtered_signal[n_segments*segment_length:]
                if len(last_segment) > 0:
                    # 用前一段的对应部分填充
                    padding = signal_matrix[-1][:segment_length-len(last_segment)]
                    last_segment = np.concatenate([last_segment, padding])
                    signal_matrix.append(last_segment)
            
            signal_matrix = np.array(signal_matrix)
            
            # 步骤3: 应用PCA
            pca = PCA(n_components=min(n_components, signal_matrix.shape[1]))
            signal_reconstructed = pca.fit_transform(signal_matrix)
            signal_denoised = pca.inverse_transform(signal_reconstructed)
            
            # 重构完整信号
            denoised_signal = []
            for i in range(n_segments):
                denoised_signal.extend(signal_denoised[i][:segment_length])
            
            # 处理剩余部分
            if len(filtered_signal) % segment_length != 0:
                remaining = filtered_signal[n_segments*segment_length:]
                denoised_signal.extend(remaining)
            
            return np.array(denoised_signal[:len(filtered_signal)])
            
        except Exception as e:
            print(f"    PCA去噪失败: {e}")
            return signal_data
    

    
    def emd_denoising(self, signal_data, max_imfs=5):
        """经验模态分解(EMD)去噪"""
        try:
            from PyEMD import EMD
            
            # 创建EMD对象
            emd = EMD()
            emd.emd(signal_data, max_imf=max_imfs)
            
            # 获取IMF分量
            imfs = emd.imfs
            
            # 计算每个IMF的能量
            energies = []
            for imf in imfs:
                energy = np.sum(imf**2)
                energies.append(energy)
            
            # 选择能量较大的IMF作为信号，较小的作为噪声
            threshold = np.mean(energies) * 0.1
            signal_imfs = []
            
            for i, energy in enumerate(energies):
                if energy > threshold:
                    signal_imfs.append(imfs[i])
            
            # 重构信号
            if signal_imfs:
                denoised = np.sum(signal_imfs, axis=0)
            else:
                denoised = signal_data
            
            return denoised
            
        except Exception as e:
            print(f"    EMD去噪失败: {e}")
            return signal_data
    
    def kalman_filter(self, signal_data, Q=0.1, R=1.0):
        """卡尔曼滤波 - 状态估计去噪"""
        try:
            n = len(signal_data)
            
            # 初始化
            x_hat = np.zeros(n)  # 状态估计
            P = np.zeros(n)      # 误差协方差
            
            # 初始值
            x_hat[0] = signal_data[0]
            P[0] = 1.0
            
            # 卡尔曼滤波迭代
            for k in range(1, n):
                # 预测步骤
                x_hat_minus = x_hat[k-1]
                P_minus = P[k-1] + Q
                
                # 更新步骤
                K = P_minus / (P_minus + R)  # 卡尔曼增益
                x_hat[k] = x_hat_minus + K * (signal_data[k] - x_hat_minus)
                P[k] = (1 - K) * P_minus
            
            return x_hat
            
        except Exception as e:
            print(f"    卡尔曼滤波失败: {e}")
            return signal_data
    
    def wavelet_denoising(self, signal_data, wavelet='db4', level=4):
        """小波去噪 - 去除随机噪声，保持信号特征"""
        try:
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
    
    def morphological_filter(self, signal_data, size=5):
        """形态学滤波 - 去除基线漂移，平滑信号"""
        try:
            # 开运算（先腐蚀后膨胀）
            opened = grey_opening(signal_data, size=size)
            # 闭运算（先膨胀后腐蚀）
            closed = grey_closing(opened, size=size)
            return closed
        except Exception as e:
            print(f"    形态学滤波失败: {e}")
            return signal_data
    
    def improved_motion_removal(self, ppg_signal, acc_data, window_size=100):
        """改进的运动伪影去除 - 先带通滤波，再进行运动检测和去除"""
        try:
            if acc_data is None or len(acc_data) == 0:
                return ppg_signal
            
            # 步骤1: 先进行带通滤波
            fs = self._auto_detect_fs()
            filtered_ppg = self.butterworth_filter(ppg_signal, lowcut=0.5, highcut=4.0, fs=fs, order=4)
            
            # 步骤2: 计算加速度幅度
            acc_magnitude = np.sqrt(acc_data['ax']**2 + acc_data['ay']**2 + acc_data['az']**2)
            
            # 步骤3: 使用滑动窗口计算动态阈值
            acc_threshold = np.zeros_like(acc_magnitude)
            for i in range(len(acc_magnitude)):
                start_idx = max(0, i - window_size // 2)
                end_idx = min(len(acc_magnitude), i + window_size // 2)
                local_mean = np.mean(acc_magnitude[start_idx:end_idx])
                local_std = np.std(acc_magnitude[start_idx:end_idx])
                acc_threshold[i] = local_mean + 1.5 * local_std  # 降低阈值
            
            # 步骤4: 创建运动掩码
            motion_mask = acc_magnitude > acc_threshold
            
            # 步骤5: 形态学操作：去除孤立的运动点
            from scipy.ndimage import binary_opening, binary_closing
            motion_mask = binary_opening(motion_mask, structure=np.ones(5))
            motion_mask = binary_closing(motion_mask, structure=np.ones(10))
            
            # 步骤6: 对运动段进行更智能的插值
            ppg_cleaned = filtered_ppg.copy()
            if np.any(motion_mask):
                # 找到连续的运动段
                motion_changes = np.diff(np.concatenate([[False], motion_mask, [False]]))
                motion_starts = np.where(motion_changes)[0][::2]
                motion_ends = np.where(motion_changes)[0][1::2]
                
                for start, end in zip(motion_starts, motion_ends):
                    if start < len(filtered_ppg) and end <= len(filtered_ppg):
                        # 获取运动段前后的有效数据
                        pre_motion = filtered_ppg[max(0, start-50):start]
                        post_motion = filtered_ppg[end:min(len(filtered_ppg), end+50)]
                        
                        if len(pre_motion) > 0 and len(post_motion) > 0:
                            # 使用三次样条插值
                            from scipy.interpolate import CubicSpline
                            try:
                                pre_indices = np.arange(max(0, start-50), start)
                                post_indices = np.arange(end, min(len(filtered_ppg), end+50))
                                
                                # 创建插值函数
                                all_indices = np.concatenate([pre_indices, post_indices])
                                all_values = np.concatenate([pre_motion, post_motion])
                                
                                if len(all_indices) > 3:
                                    cs = CubicSpline(all_indices, all_values)
                                    motion_indices = np.arange(start, end)
                                    ppg_cleaned[motion_indices] = cs(motion_indices)
                            except:
                                # 如果样条插值失败，使用线性插值
                                motion_indices = np.arange(start, end)
                                ppg_cleaned[motion_indices] = np.interp(
                                    motion_indices, 
                                    np.concatenate([pre_indices, post_indices]), 
                                    np.concatenate([pre_motion, post_motion])
                                )
            
            return ppg_cleaned
            
        except Exception as e:
            print(f"    改进运动伪影去除失败: {e}")
            return ppg_signal
    
    def apply_preprocessing_methods(self):
        """应用多种预处理方法"""
        print(f"\n🔧 应用预处理方法...")
        
        # 获取对齐后的数据
        if not hasattr(self, 'aligned_data'):
            self.align_signals()
        
        # 提取信号
        ppg_ir = self.aligned_data['ppg_ir'].values
        ppg_red = self.aligned_data['ppg_red'].values
        ppg_green = self.aligned_data['ppg_green'].values
        abp = self.aligned_data['abp'].values
        
        # 创建RGB数据矩阵
        rgb_data = np.column_stack([ppg_ir, ppg_red, ppg_green])
        
        # 1. 原始信号
        self.processed_signals['Original Signal'] = {
            'ppg': ppg_ir,
            'abp': abp
        }
        
        # 2. Butterworth滤波 (0.5-4Hz，适合PPG信号)
        print(f"  🔧 Butterworth滤波 (0.5-4Hz)...")
        ppg_butter = self.butterworth_filter(ppg_ir, lowcut=0.5, highcut=4.0, order=4)
        self.processed_signals['Butterworth Filter'] = {
            'ppg': ppg_butter,
            'abp': abp
        }
        
        # 3. 小波去噪
        print(f"  🔧 小波去噪...")
        ppg_wavelet = self.wavelet_denoising(ppg_ir)
        self.processed_signals['Wavelet Denoising'] = {
            'ppg': ppg_wavelet,
            'abp': abp
        }
        
        # 4. 形态学滤波
        print(f"  🔧 形态学滤波...")
        ppg_morph = self.morphological_filter(ppg_ir)
        self.processed_signals['Morphological Filter'] = {
            'ppg': ppg_morph,
            'abp': abp
        }
        
        # 5. PCA去噪（带通滤波后）
        print(f"  🔧 PCA去噪（带通滤波后）...")
        ppg_pca = self.pca_denoising(ppg_ir, n_components=3)
        self.processed_signals['PCA Denoising (Bandpass)'] = {
            'ppg': ppg_pca,
            'abp': abp
        }
        
        # 6. 卡尔曼滤波
        print(f"  🔧 卡尔曼滤波...")
        ppg_kalman = self.kalman_filter(ppg_ir, Q=0.1, R=1.0)
        self.processed_signals['Kalman Filter'] = {
            'ppg': ppg_kalman,
            'abp': abp
        }
        
        # 7. 组合滤波（Butterworth + 小波）
        print(f"  🔧 组合滤波 (Butterworth + 小波)...")
        ppg_combined = self.butterworth_filter(ppg_wavelet, lowcut=0.5, highcut=4.0, order=4)
        self.processed_signals['Combined Filter'] = {
            'ppg': ppg_combined,
            'abp': abp
        }
        
        # 8. 改进的运动伪影去除（带通滤波后）
        if self.acc_available:
            print(f"  🔧 改进运动伪影去除（带通滤波后，使用{self.sensor_name}的ax, ay, az）...")
            acc_data = self.aligned_data[['ax', 'ay', 'az']]
            ppg_motion_removed = self.improved_motion_removal(ppg_ir, acc_data)
            self.processed_signals['Motion Removal (Bandpass)'] = {
                'ppg': ppg_motion_removed,
                'abp': abp
            }
        else:
            print(f"  ⚠️  无加速度数据，跳过运动伪影去除")
        
        # 9. 多通道融合（IR + Red + Green）
        print(f"  🔧 多通道融合 (IR + Red + Green)...")
        # 简单加权平均
        ppg_fusion = 0.6 * ppg_ir + 0.3 * ppg_red + 0.1 * ppg_green
        self.processed_signals['Multi-Channel Fusion'] = {
            'ppg': ppg_fusion,
            'abp': abp
        }
        
        # 10. PCA提取方法
        print(f"  🔧 PCA提取方法...")
        ppg_pca_extracted, pca_variance = self.pca_extraction(rgb_data)
        if ppg_pca_extracted is not None:
            self.processed_signals['PCA Extraction'] = {
                'ppg': ppg_pca_extracted,
                'abp': abp
            }
        
        # 11. SVD提取方法
        print(f"  🔧 SVD提取方法...")
        ppg_svd_extracted, svd_values = self.svd_extraction(rgb_data)
        if ppg_svd_extracted is not None:
            self.processed_signals['SVD Extraction'] = {
                'ppg': ppg_svd_extracted,
                'abp': abp
            }
        
        # 12. POS提取方法
        print(f"  🔧 POS提取方法...")
        ppg_pos_extracted = self.pos_extraction(rgb_data)
        if ppg_pos_extracted is not None:
            self.processed_signals['POS Extraction'] = {
                'ppg': ppg_pos_extracted,
                'abp': abp
            }
        
        # 13. CHROM提取方法
        print(f"  🔧 CHROM提取方法...")
        ppg_chrom_extracted = self.chrom_extraction(rgb_data)
        if ppg_chrom_extracted is not None:
            self.processed_signals['CHROM Extraction'] = {
                'ppg': ppg_chrom_extracted,
                'abp': abp
            }
        
        # 14. ICA提取方法
        print(f"  🔧 ICA提取方法...")
        ppg_ica_extracted = self.ica_extraction(rgb_data)
        if ppg_ica_extracted is not None:
            self.processed_signals['ICA Extraction'] = {
                'ppg': ppg_ica_extracted,
                'abp': abp
            }
        
        # 15. NMF提取方法
        print(f"  🔧 NMF提取方法...")
        ppg_nmf_extracted = self.nmf_extraction(rgb_data)
        if ppg_nmf_extracted is not None:
            self.processed_signals['NMF Extraction'] = {
                'ppg': ppg_nmf_extracted,
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
        """计算信噪比 - 使用BVPSNR方法 (G. de Haan, TBME, 2013)"""
        try:
            from scipy.signal import periodogram
            from scipy.signal.windows import hamming
            
            # 估计心率（使用FFT找到最大功率频率）
            fft = np.fft.fft(signal_data)
            freqs = np.fft.fftfreq(len(signal_data), 1/100)  # 假设100Hz采样率
            
            # 只考虑正频率和0.5-4Hz范围
            pos_mask = (freqs > 0) & (freqs >= 0.5) & (freqs <= 4)
            pos_freqs = freqs[pos_mask]
            pos_power = np.abs(fft[pos_mask])**2
            
            if len(pos_power) == 0:
                return np.nan
            
            # 找到最大功率对应的频率
            max_power_idx = np.argmax(pos_power)
            hr_freq = pos_freqs[max_power_idx]  # Hz
            
            # 转换为BPM
            hr_bpm = hr_freq * 60
            
            # 使用BVPSNR方法计算SNR
            return self._calculate_bvpsnr(signal_data, 100, hr_bpm)
            
        except Exception as e:
            print(f"    SNR计算失败: {e}")
            return np.nan
    
    def _calculate_bvpsnr(self, bvp, fs, hr_bpm, plot_tf=False):
        """BVPSNR方法计算信噪比 (G. de Haan, TBME, 2013)"""
        try:
            from scipy.signal import periodogram
            from scipy.signal.windows import hamming
            
            # 处理NaN值
            bvp_clean = bvp.copy()
            if np.any(np.isnan(bvp_clean)):
                # 用前一个有效值填充NaN
                bvp_clean = pd.Series(bvp_clean).fillna(method='ffill').fillna(method='bfill').values
                if np.any(np.isnan(bvp_clean)):
                    return np.nan
            
            # 转换心率为Hz
            hr_f = hr_bpm / 60
            
            # 计算功率谱密度
            nyquist_f = fs / 2
            f_res_bpm = 0.5  # 分辨率 (bpm)
            n = round((60 * 2 * nyquist_f) / f_res_bpm)  # 功率谱中的bin数量
            
            # 构造周期图
            f, pxx = periodogram(bvp_clean, fs=fs)
            
            # 创建掩码
            # 1. 心率峰值区域 (±0.1 Hz)
            gt_mask1 = (f >= hr_f - 0.1) & (f <= hr_f + 0.1)
            
            # 2. 一次谐波区域 (±0.2 Hz)
            gt_mask2 = (f >= hr_f * 2 - 0.2) & (f <= hr_f * 2 + 0.2)
            
            # 3. 信号功率 (心率峰值 + 一次谐波)
            s_power = np.sum(pxx[gt_mask1 | gt_mask2])
            
            # 4. 总功率 (0.5-4 Hz)
            f_mask2 = (f >= 0.5) & (f <= 4)
            all_power = np.sum(pxx[f_mask2])
            
            # 5. 计算SNR
            if (all_power - s_power) > 0:
                snr = 10 * np.log10(s_power / (all_power - s_power))
            else:
                snr = np.nan
            
            # 可选：绘制功率谱和SNR区域
            if plot_tf:
                self._plot_snr_regions(f, pxx, hr_f)
            
            return snr
            
        except Exception as e:
            print(f"    BVPSNR计算失败: {e}")
            return np.nan
    
    def _plot_snr_regions(self, f, pxx, hr_f):
        """绘制功率谱和SNR区域"""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # 绘制功率谱
            ax.plot(f, 10 * np.log10(pxx + 1e-12))
            ax.set_title('Power Spectrum and SNR Regions')
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Power (dB)')
            ax.set_xlim([0.5, 4])
            
            ylim_reg = ax.get_ylim()
            ax.hold = True
            
            # 心率峰值区域
            ax.axvline(x=hr_f-0.1, color='red', linestyle='--', label='HR-0.1Hz')
            ax.axvline(x=hr_f+0.1, color='red', linestyle='--', label='HR+0.1Hz')
            
            # 一次谐波区域
            ax.axvline(x=hr_f*2-0.2, color='red', linestyle='--', label='2HR-0.2Hz')
            ax.axvline(x=hr_f*2+0.2, color='red', linestyle='--', label='2HR+0.2Hz')
            
            # 总功率区域
            ax.axvline(x=0.5, color='black', linestyle='-', label='0.5Hz')
            ax.axvline(x=4.0, color='black', linestyle='-', label='4Hz')
            
            ax.set_xlim([0, 4.5])
            ax.set_ylim(ylim_reg)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"    SNR区域绘图失败: {e}")
    
    def _shade_motion_regions(self, ax, time_relative, start_idx, end_idx):
        """在图上高亮标注运动段。"""
        if self.motion_mask is None:
            return
        mask_segment = self.motion_mask[start_idx:end_idx]
        if not np.any(mask_segment):
            return
        # 找连续区域
        idx = np.where(mask_segment)[0]
        if len(idx) == 0:
            return
        # 分段
        splits = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
        for seg in splits:
            t0 = time_relative.iloc[seg[0]]
            t1 = time_relative.iloc[seg[-1]]
            ax.axvspan(t0, t1, color='red', alpha=0.08, linewidth=0)
    
    def plot_full_length_signals(self):
        """绘制全长度信号对比图"""
        print(f"\n📈 绘制全长度信号对比图...")
        
        if not self.processed_signals:
            self.apply_preprocessing_methods()
        
        if not self.correlation_metrics:
            self.calculate_correlation_metrics()
        
        # 创建图形
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        
        # 时间轴（相对时间）
        time_relative = self.aligned_data['timestamp'] - self.aligned_data['timestamp'].iloc[0]
        
        # 获取原始信号
        ppg_original = self.processed_signals['Original Signal']['ppg']
        abp_original = self.processed_signals['Original Signal']['abp']
        
        # 获取相关性指标用于标题
        metrics = self.correlation_metrics.get('Original Signal', {})
        title_suffix = ""
        if metrics:
            title_suffix = f" | Pearson r={metrics.get('pearson_r', 'N/A'):.3f}, Spearman r={metrics.get('spearman_r', 'N/A'):.3f}, MI={metrics.get('mutual_info', 'N/A'):.3f}"
        
        # PPG信号
        ax1 = axes[0]
        ax1.plot(time_relative, ppg_original, 'b-', linewidth=0.8, alpha=0.8, label=f'PPG (IR, {self.sensor_name})')
        ax1.set_title(f'Subject {self.subject_id} Experiment {self.experiment} - PPG Signal (Full Length){title_suffix}', 
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time (seconds)', fontsize=12)
        ax1.set_ylabel('PPG Value', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=11)
        
        # ABP信号
        ax2 = axes[1]
        ax2.plot(time_relative, abp_original, 'r-', linewidth=0.8, alpha=0.8, label='ABP (Arterial Blood Pressure)')
        ax2.set_title(f'Subject {self.subject_id} Experiment {self.experiment} - ABP Signal (Full Length){title_suffix}', 
                     fontsize=14, fontweight='bold')
        ax2.set_xlabel('Time (seconds)', fontsize=12)
        ax2.set_ylabel('Blood Pressure (mmHg)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=11)
        
        plt.tight_layout()
        
        # 保存并显示
        output_dir = f'/root/autodl-tmp/blood_pressure_reconstruction/{self.subject_id}/analysis_results'
        os.makedirs(output_dir, exist_ok=True)
        
        plot_file = os.path.join(output_dir, f'{self.subject_id}_exp{self.experiment}_{self.sensor_name}_full_length_signals.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"  💾 全长度信号图已保存: {plot_file}")
        
        # 尝试显示图表（如果环境支持）
        try:
            plt.show()
        except:
            print("  ⚠️  图表已保存，但无法在终端中显示")
        
        return fig
    
    def plot_segment_signals(self, segment_length=4000, start_idx=None):
        """绘制分段信号对比图 - 包含信噪比信息，并高亮运动段"""
        print(f"\n📊 绘制分段信号对比图...")
        
        if not self.processed_signals:
            self.apply_preprocessing_methods()
        
        if not self.correlation_metrics:
            self.calculate_correlation_metrics()
        
        # 确定绘图段
        total_length = len(self.aligned_data)
        if start_idx is None:
            start_idx = total_length // 4  # 从1/4处开始
        
        end_idx = min(start_idx + segment_length, total_length)
        
        print(f"  🧮 绘图段: {start_idx} - {end_idx} (共 {end_idx - start_idx} 个点)")
        
        # 创建子图 - 每个方法3个子图：PPG处理结果、原始ABP、PPG和ABP对比
        n_methods = len(self.processed_signals)
        fig, axes = plt.subplots(n_methods, 3, figsize=(24, 4*n_methods))
        
        if n_methods == 1:
            axes = axes.reshape(1, -1)
        
        # 时间轴
        time_segment = self.aligned_data['timestamp'].iloc[start_idx:end_idx]
        time_relative = time_segment - time_segment.iloc[0]
        
        # 获取原始ABP信号（用于对比）
        abp_original = self.aligned_data['abp'].values[start_idx:end_idx]
        
        # 绘制每种预处理方法的结果
        for i, (method_name, signals) in enumerate(self.processed_signals.items()):
            ppg = signals['ppg'][start_idx:end_idx]
            abp = signals['abp'][start_idx:end_idx]
            
            # 获取相关性指标
            metrics = self.correlation_metrics.get(method_name, {})
            
            # 构建包含相关性信息和信噪比的标题
            title_ppg = f'{method_name} - PPG Signal ({self.sensor_name})\n'
            title_abp = f'Original ABP Signal\n'
            title_compare = f'{method_name} - PPG vs ABP Comparison\n'
            
            if metrics:
                # PPG标题：相关性指标
                title_ppg += f'Pearson r={metrics.get("pearson_r", "N/A"):.3f}, '
                title_ppg += f'Spearman r={metrics.get("spearman_r", "N/A"):.3f}, '
                title_ppg += f'MI={metrics.get("mutual_info", "N/A"):.3f}'
                
                # ABP标题：信噪比
                title_abp += f'ABP SNR={metrics.get("abp_snr", "N/A"):.1f}dB'
                
                # 对比标题：相关性指标
                title_compare += f'Correlation: r={metrics.get("pearson_r", "N/A"):.3f}, SNR={metrics.get("ppg_snr", "N/A"):.1f}dB'
            
            # 第一列：PPG处理结果
            ax1 = axes[i, 0]
            ax1.plot(time_relative, ppg, 'b-', linewidth=1, alpha=0.85, label=f'PPG (IR, {self.sensor_name})')
            ax1.set_title(title_ppg, fontsize=11, fontweight='bold')
            ax1.set_xlabel('Time (seconds)', fontsize=10)
            ax1.set_ylabel('PPG Value', fontsize=10)
            ax1.grid(True, alpha=0.3)
            ax1.legend(fontsize=9)
            # 高亮运动段
            self._shade_motion_regions(ax1, time_relative, start_idx, end_idx)
            
            # 在PPG图上添加信噪比信息
            if metrics and not np.isnan(metrics.get('ppg_snr', np.nan)):
                snr_text = f"PPG SNR: {metrics.get('ppg_snr', 'N/A'):.1f} dB"
                ax1.text(0.02, 0.98, snr_text, transform=ax1.transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            # 第二列：原始ABP信号
            ax2 = axes[i, 1]
            ax2.plot(time_relative, abp_original, 'r-', linewidth=1, alpha=0.85, label='ABP (Arterial Blood Pressure)')
            ax2.set_title(title_abp, fontsize=11, fontweight='bold')
            ax2.set_xlabel('Time (seconds)', fontsize=10)
            ax2.set_ylabel('Blood Pressure (mmHg)', fontsize=10)
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=9)
            # 高亮运动段（同一时间区域）
            self._shade_motion_regions(ax2, time_relative, start_idx, end_idx)
            
            # 在ABP图上添加信噪比信息
            if metrics and not np.isnan(metrics.get('abp_snr', np.nan)):
                snr_text = f"ABP SNR: {metrics.get('abp_snr', 'N/A'):.1f} dB"
                ax2.text(0.02, 0.98, snr_text, transform=ax2.transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
            
            # 第三列：PPG和ABP对比图
            ax3 = axes[i, 2]
            # 双Y轴显示
            ax3_twin = ax3.twinx()
            
            # 绘制PPG信号（左Y轴）
            line1 = ax3.plot(time_relative, ppg, 'b-', linewidth=1.5, alpha=0.8, label=f'PPG ({self.sensor_name})')
            ax3.set_xlabel('Time (seconds)', fontsize=10)
            ax3.set_ylabel('PPG Value', fontsize=10, color='blue')
            ax3.tick_params(axis='y', labelcolor='blue')
            
            # 绘制ABP信号（右Y轴）
            line2 = ax3_twin.plot(time_relative, abp_original, 'r-', linewidth=1.5, alpha=0.8, label='ABP')
            ax3_twin.set_ylabel('Blood Pressure (mmHg)', fontsize=10, color='red')
            ax3_twin.tick_params(axis='y', labelcolor='red')
            
            # 设置标题
            ax3.set_title(title_compare, fontsize=11, fontweight='bold')
            
            # 合并图例
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax3.legend(lines, labels, loc='upper right', fontsize=9)
            
            # 高亮运动段
            self._shade_motion_regions(ax3, time_relative, start_idx, end_idx)
            
            # 添加相关性信息
            if metrics and not np.isnan(metrics.get('pearson_r', np.nan)):
                corr_text = f"r={metrics.get('pearson_r', 'N/A'):.3f}"
                ax3.text(0.02, 0.98, corr_text, transform=ax3.transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout()
        
        # 保存并显示
        output_dir = f'/root/autodl-tmp/blood_pressure_reconstruction/{self.subject_id}/analysis_results'
        os.makedirs(output_dir, exist_ok=True)
        
        plot_file = os.path.join(output_dir, f'{self.subject_id}_exp{self.experiment}_{self.sensor_name}_segment_signals.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"  💾 分段信号图已保存: {plot_file}")
        
        # 尝试显示图表（如果环境支持）
        try:
            plt.show()
        except:
            print("  ⚠️  图表已保存，但无法在终端中显示")
        
        return fig
    
    def plot_correlation_summary(self):
        """绘制相关性总结图 - 使用seaborn美化"""
        print(f"\n📊 绘制相关性总结图...")
        
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
        
        # 创建DataFrame用于seaborn
        df_corr = pd.DataFrame(corr_matrix, 
                              index=methods, 
                              columns=['Pearson r', 'Spearman r', 'Mutual Info', 'Freq Corr', 'Coherence'])
        
        # 创建美化后的热力图 - 增大图表尺寸
        fig, ax = plt.subplots(figsize=(18, 14))
        
        # 使用seaborn绘制美化热力图
        sns.heatmap(df_corr, 
                   annot=True, 
                   fmt='.3f', 
                   cmap='RdYlBu_r',  # 红蓝配色，更美观
                   center=0,
                   square=True,      # 正方形单元格
                   linewidths=0.8,   # 增加网格线宽度
                   cbar_kws={'shrink': 0.8},  # 颜色条设置
                   ax=ax,
                   annot_kws={'size': 10})  # 增大注释字体
        
        # 设置标题和标签 - 增大字体
        ax.set_title(f'Subject {self.subject_id} Experiment {self.experiment} - Correlation Metrics Summary ({self.sensor_name})', 
                    fontsize=18, fontweight='bold', pad=25)
        ax.set_xlabel('Correlation Metrics', fontsize=16, fontweight='bold')
        ax.set_ylabel('Preprocessing Methods', fontsize=16, fontweight='bold')
        
        # 旋转标签并增大字体
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(rotation=0, fontsize=11)
        
        # 调整布局，增加边距
        plt.tight_layout(pad=2.0)
        
        # 保存并显示
        output_dir = f'/root/autodl-tmp/blood_pressure_reconstruction/{self.subject_id}/analysis_results'
        os.makedirs(output_dir, exist_ok=True)
        
        heatmap_file = os.path.join(output_dir, f'{self.subject_id}_exp{self.experiment}_{self.sensor_name}_correlation_heatmap.png')
        plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
        print(f"  💾 相关性热力图已保存: {heatmap_file}")
        
        # 尝试显示图表（如果环境支持）
        try:
            plt.show()
        except:
            print("  ⚠️  图表已保存，但无法在终端中显示")
        
        return fig
    
    def run_visualization_analysis(self, segment_length=2000, start_idx=None):
        """运行可视化分析"""
        print(f"\n🚀 开始可视化分析...")
        print(f"{'='*60}")
        print(f"  受试者: {self.subject_id}")
        print(f"  实验编号: {self.experiment}")
        print(f"  使用传感器: {self.sensor_name}")
        
        try:
            # 1. 应用预处理方法
            self.apply_preprocessing_methods()
            
            # 2. 计算相关性指标
            self.calculate_correlation_metrics()
            
            # 3. 绘制全长度信号对比图
            self.plot_full_length_signals()
            
            # 4. 绘制分段信号对比图
            self.plot_segment_signals(segment_length, start_idx)
            
            # 5. 绘制相关性总结图
            self.plot_correlation_summary()
            
            print(f"\n🎉 可视化分析完成！")
            print(f"📁 结果保存在: /root/autodl-tmp/blood_pressure_reconstruction/{self.subject_id}/analysis_results/")
            
        except Exception as e:
            print(f"❌ 分析过程中发生错误: {e}")
            raise

class BatchAnalyzer:
    """批量分析器 - 处理所有实验和所有传感器"""
    
    def __init__(self, subject_id="00017"):
        self.subject_id = subject_id
        self.base_dir = f'/root/autodl-tmp/blood_pressure_reconstruction/{subject_id}/csv'
        
    def get_available_experiments_and_sensors(self):
        """获取可用的实验编号和传感器列表"""
        if not os.path.exists(self.base_dir):
            print(f"❌ 目录不存在: {self.base_dir}")
            return [], []
        
        # 获取所有文件
        files = os.listdir(self.base_dir)
        
        # 提取实验编号
        experiments = set()
        sensors = set()
        
        for file in files:
            if file.endswith('.csv'):
                parts = file.split('_')
                if len(parts) >= 3:
                    # 格式: 00017_1_sensor2.csv 或 00017_1_abp.csv
                    exp_num = parts[1]
                    if exp_num.isdigit():
                        experiments.add(exp_num)
                    
                    if len(parts) >= 3:
                        sensor_part = parts[2].replace('.csv', '')
                        if sensor_part.startswith('sensor'):
                            sensors.add(sensor_part)
        
        experiments = sorted(list(experiments), key=int)
        sensors = sorted(list(sensors))
        
        return experiments, sensors
    
    def run_batch_analysis(self, experiments=None, sensors=None, segment_length=2000):
        """运行批量分析"""
        print(f"\n🚀 开始批量分析 - 受试者 {self.subject_id}")
        print(f"{'='*80}")
        
        # 获取可用的实验和传感器
        available_experiments, available_sensors = self.get_available_experiments_and_sensors()
        
        if not available_experiments:
            print("❌ 未找到任何实验数据")
            return
        
        if not available_sensors:
            print("❌ 未找到任何传感器数据")
            return
        
        # 使用指定的实验和传感器，如果没有指定则使用所有可用的
        if experiments is None:
            experiments = available_experiments
        if sensors is None:
            sensors = available_sensors
        
        print(f"📋 可用实验: {available_experiments}")
        print(f"📋 可用传感器: {available_sensors}")
        print(f"🎯 将分析实验: {experiments}")
        print(f"🎯 将分析传感器: {sensors}")
        
        total_combinations = len(experiments) * len(sensors)
        current = 0
        
        results = []
        
        for exp in experiments:
            for sensor in sensors:
                current += 1
                print(f"\n{'='*60}")
                print(f"🔬 进度: {current}/{total_combinations}")
                print(f"📊 分析: 实验{exp} - {sensor}")
                print(f"{'='*60}")
                
                try:
                    # 检查文件是否存在
                    ppg_file = f'{self.subject_id}_{exp}_{sensor}.csv'
                    abp_file = f'{self.subject_id}_{exp}_abp.csv'
                    
                    ppg_path = os.path.join(self.base_dir, ppg_file)
                    abp_path = os.path.join(self.base_dir, abp_file)
                    
                    if not os.path.exists(ppg_path):
                        print(f"  ⚠️  PPG文件不存在: {ppg_file}")
                        continue
                    
                    if not os.path.exists(abp_path):
                        print(f"  ⚠️  ABP文件不存在: {abp_file}")
                        continue
                    
                    # 创建分析器并运行分析
                    analyzer = PPGABPAnalyzer(self.subject_id, exp, sensor)
                    analyzer.run_visualization_analysis(segment_length=segment_length)
                    
                    results.append({
                        'experiment': exp,
                        'sensor': sensor,
                        'status': 'success'
                    })
                    
                    print(f"  ✅ 实验{exp} - {sensor} 分析完成")
                    
                except Exception as e:
                    print(f"  ❌ 实验{exp} - {sensor} 分析失败: {e}")
                    results.append({
                        'experiment': exp,
                        'sensor': sensor,
                        'status': 'failed',
                        'error': str(e)
                    })
        
        # 生成批量分析报告
        self._generate_batch_report(results)
        
        print(f"\n🎉 批量分析完成！")
        print(f"📊 成功: {len([r for r in results if r['status'] == 'success'])}")
        print(f"❌ 失败: {len([r for r in results if r['status'] == 'failed'])}")
    
    def _generate_batch_report(self, results):
        """生成批量分析报告"""
        output_dir = f'/root/autodl-tmp/blood_pressure_reconstruction/{self.subject_id}/analysis_results'
        os.makedirs(output_dir, exist_ok=True)
        
        report_file = os.path.join(output_dir, f'{self.subject_id}_batch_analysis_report.txt')
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"受试者 {self.subject_id} 批量分析报告\n")
            f.write("="*60 + "\n\n")
            f.write(f"分析时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总组合数: {len(results)}\n\n")
            
            # 成功和失败的统计
            successful = [r for r in results if r['status'] == 'success']
            failed = [r for r in results if r['status'] == 'failed']
            
            f.write(f"成功分析: {len(successful)} 个组合\n")
            f.write(f"失败分析: {len(failed)} 个组合\n")
            f.write(f"成功率: {len(successful)/len(results)*100:.1f}%\n\n")
            
            # 成功的组合
            if successful:
                f.write("成功分析的组合:\n")
                f.write("-"*30 + "\n")
                for result in successful:
                    f.write(f"  实验{result['experiment']} - {result['sensor']}\n")
                f.write("\n")
            
            # 失败的组合
            if failed:
                f.write("失败分析的组合:\n")
                f.write("-"*30 + "\n")
                for result in failed:
                    f.write(f"  实验{result['experiment']} - {result['sensor']}: {result.get('error', '未知错误')}\n")
                f.write("\n")
            
            f.write("输出文件命名规则:\n")
            f.write("-"*25 + "\n")
            f.write(f"  全长度信号图: {self.subject_id}_exp<实验编号>_<传感器>_full_length_signals.png\n")
            f.write(f"  分段信号图: {self.subject_id}_exp<实验编号>_<传感器>_segment_signals.png\n")
            f.write(f"  相关性热力图: {self.subject_id}_exp<实验编号>_<传感器>_correlation_heatmap.png\n")
        
        print(f"📝 批量分析报告已保存: {report_file}")

def main():
    """主函数"""
    print("🚀 PPG-ABP信号可视化分析系统")
    print("="*60)
    
    # 选择分析模式
    print("请选择分析模式:")
    print("1. 单个分析 (指定实验和传感器)")
    print("2. 批量分析 (所有实验和传感器)")
    
    choice = input("请输入选择 (1 或 2): ").strip()
    
    if choice == "2":
        # 批量分析
        batch_analyzer = BatchAnalyzer(subject_id="00017")
        batch_analyzer.run_batch_analysis()
    else:
        # 单个分析
        print("\n单个分析模式:")
        experiment = input("请输入实验编号 (默认1): ").strip() or "1"
        sensor = input("请输入传感器名称 (默认sensor2): ").strip() or "sensor2"
        
        print(f"\n开始分析: 实验{experiment} - {sensor}")
        
        # 创建分析器
        analyzer = PPGABPAnalyzer(subject_id="00017", experiment=experiment, sensor_name=sensor)
        
        # 运行可视化分析
        analyzer.run_visualization_analysis(
            segment_length=2000,  # 分段长度
            start_idx=None        # 自动选择起始位置
        )

if __name__ == "__main__":
    main()
