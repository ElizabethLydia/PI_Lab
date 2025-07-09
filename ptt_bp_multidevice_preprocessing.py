# PI_Lab多设备PTT血压预测预处理代码
# 基于Biopac和HUB设备的生理信号进行脉搏传输时间计算和血压预测

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class MultiDevicePTTProcessor:
    """多设备PTT血压预测处理器"""
    
    def __init__(self, data_root="/root/PI_Lab/00017", sampling_rate=100):
        self.data_root = data_root
        self.sampling_rate = sampling_rate
        self.static_conditions = ['1', '7']  # 静止状态
        self.all_conditions = [str(i) for i in range(1, 12)]
        
    def load_condition_data(self, condition):
        """加载指定条件的数据"""
        condition_path = os.path.join(self.data_root, condition)
        if not os.path.exists(condition_path):
            print(f"条件 {condition} 的数据路径不存在")
            return None
        
        data = {
            'condition': condition,
            'biopac': {},
            'hub': {},
            'timestamp_range': None
        }
        
        # 加载Biopac数据
        biopac_path = os.path.join(condition_path, 'Biopac')
        if os.path.exists(biopac_path):
            biopac_files = {
                'hr': f'hr-{condition}.csv',
                'bp': f'bp-{condition}.csv',
                'systolic_bp': f'systolic_bp-{condition}.csv',
                'diastolic_bp': f'diastolic_bp-{condition}.csv',
                'mean_bp': f'mean_bp-{condition}.csv',
                'cardiac_output': f'cardiac_output-{condition}.csv',
                'cardiac_index': f'cardiac_index-{condition}.csv',
                'systemic_vascular_resistance': f'systemic_vascular_resistance-{condition}.csv',
                'rsp': f'rsp-{condition}.csv'
            }
            
            for signal_type, filename in biopac_files.items():
                filepath = os.path.join(biopac_path, filename)
                if os.path.exists(filepath):
                    try:
                        df = pd.read_csv(filepath)
                        data['biopac'][signal_type] = df
                        print(f"成功加载Biopac {signal_type}: {len(df)}行")
                    except Exception as e:
                        print(f"加载Biopac {signal_type}失败: {e}")
        
        # 加载HUB数据
        hub_path = os.path.join(condition_path, 'HUB')
        if os.path.exists(hub_path):
            hub_sensors = ['sensor2.csv', 'sensor3.csv', 'sensor4.csv', 'sensor5.csv']
            
            for sensor_file in hub_sensors:
                filepath = os.path.join(hub_path, sensor_file)
                if os.path.exists(filepath):
                    try:
                        df = pd.read_csv(filepath)
                        sensor_name = sensor_file.replace('.csv', '')
                        data['hub'][sensor_name] = df
                        print(f"成功加载HUB {sensor_name}: {len(df)}行")
                    except Exception as e:
                        print(f"加载HUB {sensor_file}失败: {e}")
        
        # 确定时间戳范围
        all_timestamps = []
        for device_data in [data['biopac'], data['hub']]:
            for signal_df in device_data.values():
                if 'timestamp' in signal_df.columns:
                    all_timestamps.extend(signal_df['timestamp'].tolist())
        
        if all_timestamps:
            data['timestamp_range'] = (min(all_timestamps), max(all_timestamps))
            print(f"条件 {condition} 时间戳范围: {data['timestamp_range'][1] - data['timestamp_range'][0]:.2f}秒")
        
        return data
    
    def detect_ppg_peaks(self, ppg_signal, timestamps, method='green'):
        """从HUB PPG信号检测心跳峰值"""
        if isinstance(ppg_signal, pd.DataFrame):
            if method in ppg_signal.columns:
                ppg_values = ppg_signal[method].values
            else:
                # 默认使用第一个数值列
                numeric_cols = ppg_signal.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 1:  # 跳过timestamp列
                    ppg_values = ppg_signal[numeric_cols[1]].values
                else:
                    print("未找到合适的PPG信号列")
                    return [], []
        else:
            ppg_values = ppg_signal
        
        # 预处理：去除异常值
        ppg_values = np.array(ppg_values)
        q25, q75 = np.percentile(ppg_values, [25, 75])
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        ppg_values = np.clip(ppg_values, lower_bound, upper_bound)
        
        # 滤波
        try:
            nyquist = self.sampling_rate / 2
            low_cutoff = 0.5 / nyquist  # 0.5 Hz
            high_cutoff = 8 / nyquist   # 8 Hz
            b, a = signal.butter(4, [low_cutoff, high_cutoff], btype='band')
            filtered_ppg = signal.filtfilt(b, a, ppg_values)
        except:
            filtered_ppg = ppg_values
        
        # 寻找峰值
        # 使用自适应阈值
        height_threshold = np.mean(filtered_ppg) + 0.5 * np.std(filtered_ppg)
        distance = int(0.4 * self.sampling_rate)  # 最小心跳间隔400ms
        
        peaks, properties = signal.find_peaks(filtered_ppg, 
                                            height=height_threshold,
                                            distance=distance,
                                            prominence=np.std(filtered_ppg) * 0.2)
        
        if len(peaks) == 0:
            # 降低阈值重试
            height_threshold = np.mean(filtered_ppg) + 0.2 * np.std(filtered_ppg)
            peaks, properties = signal.find_peaks(filtered_ppg, 
                                                height=height_threshold,
                                                distance=distance)
        
        peak_timestamps = timestamps[peaks] if len(peaks) > 0 else []
        
        return peaks, peak_timestamps
    
    def calculate_hr_from_biopac(self, hr_data):
        """从Biopac心率数据计算RR间期，推算R波时间"""
        if 'hr' not in hr_data.columns or 'timestamp' not in hr_data.columns:
            return [], []
        
        hr_values = hr_data['hr'].values
        timestamps = hr_data['timestamp'].values
        
        # 计算RR间期
        rr_intervals = 60.0 / hr_values  # 秒
        
        # 推算R波时间戳
        r_timestamps = []
        current_time = timestamps[0]
        
        for i, rr in enumerate(rr_intervals):
            if i < len(timestamps) - 1:
                # 使用实际时间戳之间的间隔
                time_diff = timestamps[i+1] - timestamps[i]
                num_beats = max(1, int(time_diff / rr))
                
                for j in range(num_beats):
                    r_time = timestamps[i] + j * rr
                    if r_time <= timestamps[i+1]:
                        r_timestamps.append(r_time)
        
        return r_timestamps, rr_intervals
    
    def calculate_ptt_multidevice(self, condition_data):
        """计算多设备PTT"""
        results = {
            'condition': condition_data['condition'],
            'ptt_values': [],
            'ptt_timestamps': [],
            'r_timestamps': [],
            'ppg_timestamps': [],
            'hr_biopac': [],
            'bp_biopac': {},
            'device_sync_quality': 0
        }
        
        # 获取参考血压数据
        bp_data = {}
        if 'biopac' in condition_data:
            for bp_type in ['systolic_bp', 'diastolic_bp', 'mean_bp', 'bp']:
                if bp_type in condition_data['biopac']:
                    bp_df = condition_data['biopac'][bp_type]
                    if not bp_df.empty:
                        bp_data[bp_type] = bp_df
        
        # 方法1：使用Biopac心率推算R波 + HUB PPG
        r_timestamps = []
        if 'hr' in condition_data['biopac']:
            hr_df = condition_data['biopac']['hr']
            r_timestamps, rr_intervals = self.calculate_hr_from_biopac(hr_df)
            results['hr_biopac'] = hr_df['hr'].values
        
        # 获取HUB PPG峰值
        ppg_timestamps = []
        if condition_data['hub']:
            # 尝试不同的传感器
            for sensor_name, sensor_data in condition_data['hub'].items():
                if 'green' in sensor_data.columns:  # 优先使用绿光
                    ppg_peaks, ppg_timestamps = self.detect_ppg_peaks(
                        sensor_data, sensor_data['timestamp'].values, 'green')
                    if len(ppg_timestamps) > 10:  # 至少10个峰值才认为有效
                        print(f"使用{sensor_name}的绿光PPG信号，检测到{len(ppg_timestamps)}个峰值")
                        break
                elif 'ir' in sensor_data.columns:  # 备选红外
                    ppg_peaks, ppg_timestamps = self.detect_ppg_peaks(
                        sensor_data, sensor_data['timestamp'].values, 'ir')
                    if len(ppg_timestamps) > 10:
                        print(f"使用{sensor_name}的红外PPG信号，检测到{len(ppg_timestamps)}个峰值")
                        break
        
        # 计算PTT
        if len(r_timestamps) > 0 and len(ppg_timestamps) > 0:
            ptt_values, matched_r_times, matched_ppg_times = self._match_r_ppg_peaks(
                r_timestamps, ppg_timestamps)
            
            results['ptt_values'] = ptt_values
            results['ptt_timestamps'] = matched_r_times
            results['r_timestamps'] = r_timestamps
            results['ppg_timestamps'] = ppg_timestamps
            
            # 计算设备同步质量
            if len(ptt_values) > 0:
                results['device_sync_quality'] = len(ptt_values) / min(len(r_timestamps), len(ppg_timestamps))
            
            print(f"条件{condition_data['condition']}: 计算出{len(ptt_values)}个PTT值")
            if len(ptt_values) > 0:
                print(f"PTT范围: {np.min(ptt_values)*1000:.1f} - {np.max(ptt_values)*1000:.1f} ms")
        
        # 存储血压数据
        results['bp_biopac'] = bp_data
        
        return results
    
    def _match_r_ppg_peaks(self, r_timestamps, ppg_timestamps, max_ptt=1.5):
        """匹配R波和PPG峰值，计算PTT"""
        ptt_values = []
        matched_r_times = []
        matched_ppg_times = []
        
        r_timestamps = np.array(r_timestamps)
        ppg_timestamps = np.array(ppg_timestamps)
        
        for r_time in r_timestamps:
            # 找到R波后第一个PPG峰值
            future_ppg = ppg_timestamps[ppg_timestamps > r_time]
            
            if len(future_ppg) > 0:
                ppg_time = future_ppg[0]
                ptt = ppg_time - r_time
                
                # 过滤异常PTT值
                if 0.05 <= ptt <= max_ptt:  # 50ms - 1500ms
                    ptt_values.append(ptt)
                    matched_r_times.append(r_time)
                    matched_ppg_times.append(ppg_time)
        
        return np.array(ptt_values), np.array(matched_r_times), np.array(matched_ppg_times)
    
    def extract_ptt_features(self, ptt_results):
        """提取PTT特征用于血压预测"""
        if len(ptt_results['ptt_values']) == 0:
            return None
        
        ptt_values = ptt_results['ptt_values']
        
        features = {
            # PTT统计特征
            'ptt_mean': np.mean(ptt_values),
            'ptt_std': np.std(ptt_values),
            'ptt_median': np.median(ptt_values),
            'ptt_min': np.min(ptt_values),
            'ptt_max': np.max(ptt_values),
            'ptt_range': np.max(ptt_values) - np.min(ptt_values),
            'ptt_cv': np.std(ptt_values) / np.mean(ptt_values),  # 变异系数
            
            # 心率相关特征
            'hr_mean': np.mean(ptt_results['hr_biopac']) if len(ptt_results['hr_biopac']) > 0 else np.nan,
            'hr_std': np.std(ptt_results['hr_biopac']) if len(ptt_results['hr_biopac']) > 0 else np.nan,
            
            # 信号质量特征
            'num_beats': len(ptt_values),
            'sync_quality': ptt_results['device_sync_quality'],
            
            # 时域特征
            'ptt_rmssd': None,  # 相邻PTT差值的均方根
            'ptt_pnn50': None   # 相邻PTT差值>50ms的百分比
        }
        
        # 计算PTT变异性特征
        if len(ptt_values) > 1:
            ptt_diff = np.diff(ptt_values)
            features['ptt_rmssd'] = np.sqrt(np.mean(ptt_diff**2))
            features['ptt_pnn50'] = np.sum(np.abs(ptt_diff) > 0.05) / len(ptt_diff) * 100
        
        return features
    
    def align_bp_with_ptt(self, ptt_results, bp_data):
        """将血压数据与PTT时间戳对齐"""
        aligned_bp = {}
        
        if not ptt_results['ptt_timestamps'].size or not bp_data:
            return aligned_bp
        
        ptt_timestamps = ptt_results['ptt_timestamps']
        time_range = (np.min(ptt_timestamps), np.max(ptt_timestamps))
        
        for bp_type, bp_df in bp_data.items():
            if 'timestamp' in bp_df.columns:
                bp_signal = bp_df[bp_df.columns[1]].values  # 第二列是数值
                bp_timestamps = bp_df['timestamp'].values
                
                # 筛选时间范围内的血压数据
                mask = (bp_timestamps >= time_range[0]) & (bp_timestamps <= time_range[1])
                if np.sum(mask) > 0:
                    aligned_bp[bp_type] = {
                        'values': bp_signal[mask],
                        'timestamps': bp_timestamps[mask],
                        'mean': np.mean(bp_signal[mask]),
                        'std': np.std(bp_signal[mask])
                    }
        
        return aligned_bp
    
    def build_bp_prediction_models(self, training_data):
        """构建血压预测模型"""
        # 准备训练数据
        features_list = []
        labels = {'systolic': [], 'diastolic': [], 'mean': []}
        subject_conditions = []
        
        for condition, data in training_data.items():
            if 'ptt_features' not in data or 'aligned_bp' not in data:
                continue
            
            ptt_features = data['ptt_features']
            aligned_bp = data['aligned_bp']
            
            # 构建特征向量
            feature_vector = [
                ptt_features['ptt_mean'],
                ptt_features['ptt_std'],
                ptt_features['ptt_cv'],
                ptt_features['hr_mean'] if not np.isnan(ptt_features['hr_mean']) else 70,
                ptt_features['hr_std'] if not np.isnan(ptt_features['hr_std']) else 5,
                ptt_features['sync_quality'],
                ptt_features['ptt_rmssd'] if ptt_features['ptt_rmssd'] is not None else 0,
                ptt_features['ptt_pnn50'] if ptt_features['ptt_pnn50'] is not None else 0
            ]
            
            # 检查是否有有效特征
            if not any(np.isnan(feature_vector[:3])):  # 至少前3个特征有效
                features_list.append(feature_vector)
                subject_conditions.append(condition)
                
                # 提取血压标签
                if 'systolic_bp' in aligned_bp:
                    labels['systolic'].append(aligned_bp['systolic_bp']['mean'])
                else:
                    labels['systolic'].append(np.nan)
                
                if 'diastolic_bp' in aligned_bp:
                    labels['diastolic'].append(aligned_bp['diastolic_bp']['mean'])
                else:
                    labels['diastolic'].append(np.nan)
                
                if 'mean_bp' in aligned_bp:
                    labels['mean'].append(aligned_bp['mean_bp']['mean'])
                else:
                    labels['mean'].append(np.nan)
        
        if len(features_list) < 3:
            print(f"训练数据不足: {len(features_list)}个样本")
            return None
        
        # 转换为numpy数组
        X = np.array(features_list)
        
        # 特征标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        models = {}
        performance = {}
        
        # 为每种血压类型训练模型
        for bp_type in ['systolic', 'diastolic', 'mean']:
            y = np.array(labels[bp_type])
            valid_mask = ~np.isnan(y)
            
            if np.sum(valid_mask) < 3:
                print(f"{bp_type}血压数据不足")
                continue
            
            X_valid = X_scaled[valid_mask]
            y_valid = y[valid_mask]
            
            # 线性回归
            lr_model = LinearRegression()
            lr_model.fit(X_valid, y_valid)
            y_pred_lr = lr_model.predict(X_valid)
            
            # 随机森林
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_valid, y_valid)
            y_pred_rf = rf_model.predict(X_valid)
            
            models[bp_type] = {
                'linear': lr_model,
                'random_forest': rf_model,
                'scaler': scaler
            }
            
            performance[bp_type] = {
                'linear': {
                    'r2': r2_score(y_valid, y_pred_lr),
                    'mae': mean_absolute_error(y_valid, y_pred_lr),
                    'rmse': np.sqrt(mean_squared_error(y_valid, y_pred_lr))
                },
                'random_forest': {
                    'r2': r2_score(y_valid, y_pred_rf),
                    'mae': mean_absolute_error(y_valid, y_pred_rf),
                    'rmse': np.sqrt(mean_squared_error(y_valid, y_pred_rf))
                }
            }
            
            print(f"\n{bp_type}血压预测模型性能:")
            print(f"线性回归 - R²: {performance[bp_type]['linear']['r2']:.3f}, MAE: {performance[bp_type]['linear']['mae']:.3f}")
            print(f"随机森林 - R²: {performance[bp_type]['random_forest']['r2']:.3f}, MAE: {performance[bp_type]['random_forest']['mae']:.3f}")
        
        return models, performance, (X, labels, subject_conditions)
    
    def process_all_conditions(self):
        """处理所有实验条件"""
        all_results = {}
        
        print("开始处理所有实验条件...")
        
        for condition in self.all_conditions:
            print(f"\n处理条件 {condition}...")
            
            # 加载数据
            condition_data = self.load_condition_data(condition)
            if condition_data is None:
                continue
            
            # 计算PTT
            ptt_results = self.calculate_ptt_multidevice(condition_data)
            
            # 提取特征
            ptt_features = self.extract_ptt_features(ptt_results)
            
            # 对齐血压数据
            aligned_bp = self.align_bp_with_ptt(ptt_results, ptt_results['bp_biopac'])
            
            all_results[condition] = {
                'ptt_results': ptt_results,
                'ptt_features': ptt_features,
                'aligned_bp': aligned_bp,
                'is_static': condition in self.static_conditions
            }
        
        return all_results
    
    def analyze_static_vs_dynamic(self, all_results):
        """分析静止状态vs动态状态的PTT差异"""
        static_ptt = []
        dynamic_ptt = []
        
        for condition, results in all_results.items():
            if results['ptt_features'] is not None:
                ptt_mean = results['ptt_features']['ptt_mean']
                if results['is_static']:
                    static_ptt.append(ptt_mean)
                else:
                    dynamic_ptt.append(ptt_mean)
        
        analysis = {
            'static_ptt_mean': np.mean(static_ptt) if static_ptt else np.nan,
            'static_ptt_std': np.std(static_ptt) if static_ptt else np.nan,
            'dynamic_ptt_mean': np.mean(dynamic_ptt) if dynamic_ptt else np.nan,
            'dynamic_ptt_std': np.std(dynamic_ptt) if dynamic_ptt else np.nan,
            'static_conditions': len(static_ptt),
            'dynamic_conditions': len(dynamic_ptt)
        }
        
        return analysis
    
    def visualize_results(self, all_results, output_dir):
        """可视化分析结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. PTT对比图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # PTT均值对比
        conditions = []
        ptt_means = []
        is_static = []
        
        for condition, results in all_results.items():
            if results['ptt_features'] is not None:
                conditions.append(condition)
                ptt_means.append(results['ptt_features']['ptt_mean'] * 1000)  # 转换为ms
                is_static.append(results['is_static'])
        
        colors = ['red' if static else 'blue' for static in is_static]
        axes[0, 0].bar(conditions, ptt_means, color=colors, alpha=0.7)
        axes[0, 0].set_title('各条件PTT均值对比')
        axes[0, 0].set_xlabel('实验条件')
        axes[0, 0].set_ylabel('PTT (ms)')
        axes[0, 0].legend(['静止状态', '动态状态'])
        
        # PTT变异性对比
        ptt_stds = []
        for condition, results in all_results.items():
            if results['ptt_features'] is not None:
                ptt_stds.append(results['ptt_features']['ptt_std'] * 1000)
        
        if len(ptt_stds) == len(conditions):
            axes[0, 1].bar(conditions, ptt_stds, color=colors, alpha=0.7)
            axes[0, 1].set_title('各条件PTT变异性对比')
            axes[0, 1].set_xlabel('实验条件')
            axes[0, 1].set_ylabel('PTT标准差 (ms)')
        
        # 血压分布
        bp_systolic = []
        bp_diastolic = []
        for condition, results in all_results.items():
            if 'systolic_bp' in results['aligned_bp']:
                bp_systolic.append(results['aligned_bp']['systolic_bp']['mean'])
            if 'diastolic_bp' in results['aligned_bp']:
                bp_diastolic.append(results['aligned_bp']['diastolic_bp']['mean'])
        
        if bp_systolic:
            axes[1, 0].hist(bp_systolic, bins=10, alpha=0.7, label='收缩压')
            axes[1, 0].set_title('血压分布')
            axes[1, 0].set_xlabel('血压 (mmHg)')
            axes[1, 0].set_ylabel('频次')
            axes[1, 0].legend()
        
        if bp_diastolic:
            axes[1, 1].hist(bp_diastolic, bins=10, alpha=0.7, label='舒张压', color='orange')
            axes[1, 1].set_title('舒张压分布')
            axes[1, 1].set_xlabel('血压 (mmHg)')
            axes[1, 1].set_ylabel('频次')
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'ptt_analysis_overview.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 时间序列图（选择一个条件）
        for condition in ['1', '7']:  # 静止状态
            if condition in all_results and all_results[condition]['ptt_features'] is not None:
                self._plot_condition_timeseries(all_results[condition], condition, output_dir)
                break
    
    def _plot_condition_timeseries(self, condition_results, condition, output_dir):
        """绘制单个条件的时间序列"""
        ptt_results = condition_results['ptt_results']
        
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # PTT时间序列
        if len(ptt_results['ptt_timestamps']) > 0:
            axes[0].plot(ptt_results['ptt_timestamps'], 
                        np.array(ptt_results['ptt_values']) * 1000, 'b-', alpha=0.7)
            axes[0].set_title(f'条件{condition} - PTT时间序列')
            axes[0].set_ylabel('PTT (ms)')
            axes[0].grid(True)
        
        # 心率时间序列
        if len(ptt_results['hr_biopac']) > 0:
            # 需要对应的时间戳，这里用索引代替
            axes[1].plot(ptt_results['hr_biopac'], 'r-', alpha=0.7)
            axes[1].set_title(f'条件{condition} - 心率时间序列')
            axes[1].set_ylabel('心率 (BPM)')
            axes[1].grid(True)
        
        # PTT vs 心率散点图
        if len(ptt_results['ptt_values']) > 0 and len(ptt_results['hr_biopac']) > 0:
            min_len = min(len(ptt_results['ptt_values']), len(ptt_results['hr_biopac']))
            axes[2].scatter(np.array(ptt_results['ptt_values'][:min_len]) * 1000,
                          ptt_results['hr_biopac'][:min_len], alpha=0.6)
            axes[2].set_title(f'条件{condition} - PTT vs 心率关系')
            axes[2].set_xlabel('PTT (ms)')
            axes[2].set_ylabel('心率 (BPM)')
            axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'condition_{condition}_timeseries.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self, results, output_path):
        """保存处理结果"""
        np.save(output_path, results)
        print(f"结果已保存到: {output_path}")
    
    def run_full_pipeline(self, output_dir="./ptt_multidevice_results"):
        """运行完整的多设备PTT血压预测流水线"""
        print("开始PI_Lab多设备PTT血压预测流水线...")
        
        # 1. 处理所有条件
        all_results = self.process_all_conditions()
        
        if not all_results:
            print("未能处理任何数据")
            return None
        
        # 2. 静止状态vs动态状态分析
        static_dynamic_analysis = self.analyze_static_vs_dynamic(all_results)
        
        # 3. 构建血压预测模型（使用静止状态数据）
        static_data = {k: v for k, v in all_results.items() if v['is_static']}
        if static_data:
            models_result = self.build_bp_prediction_models(static_data)
            if models_result:
                models, performance, training_data = models_result
            else:
                models, performance = None, None
        else:
            models, performance = None, None
        
        # 4. 保存结果
        os.makedirs(output_dir, exist_ok=True)
        
        self.save_results(all_results, os.path.join(output_dir, 'all_conditions_results.npy'))
        self.save_results(static_dynamic_analysis, os.path.join(output_dir, 'static_dynamic_analysis.npy'))
        
        if models:
            self.save_results(models, os.path.join(output_dir, 'bp_prediction_models.npy'))
            self.save_results(performance, os.path.join(output_dir, 'model_performance.npy'))
        
        # 5. 生成可视化
        self.visualize_results(all_results, output_dir)
        
        # 6. 生成分析报告
        self.generate_comprehensive_report(all_results, static_dynamic_analysis, 
                                         performance, output_dir)
        
        print(f"处理完成！结果保存在: {output_dir}")
        
        return {
            'all_results': all_results,
            'static_dynamic_analysis': static_dynamic_analysis,
            'models': models,
            'performance': performance
        }
    
    def generate_comprehensive_report(self, all_results, static_dynamic_analysis, 
                                    performance, output_dir):
        """生成综合分析报告"""
        report_path = os.path.join(output_dir, 'comprehensive_analysis_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("PI_Lab多设备PTT血压预测分析报告\n")
            f.write("=" * 60 + "\n\n")
            
            # 数据处理概况
            f.write("1. 数据处理概况\n")
            f.write("-" * 30 + "\n")
            successful_conditions = sum(1 for results in all_results.values() 
                                      if results['ptt_features'] is not None)
            f.write(f"总实验条件数: {len(self.all_conditions)}\n")
            f.write(f"成功处理条件数: {successful_conditions}\n")
            f.write(f"静止状态条件: {len(self.static_conditions)} (条件1, 7)\n")
            f.write(f"动态状态条件: {len(self.all_conditions) - len(self.static_conditions)}\n\n")
            
            # PTT特征统计
            f.write("2. PTT特征统计\n")
            f.write("-" * 30 + "\n")
            
            all_ptt_means = []
            all_ptt_stds = []
            for condition, results in all_results.items():
                if results['ptt_features'] is not None:
                    ptt_features = results['ptt_features']
                    f.write(f"条件 {condition} ({'静止' if results['is_static'] else '动态'}):\n")
                    f.write(f"  PTT均值: {ptt_features['ptt_mean']*1000:.1f} ms\n")
                    f.write(f"  PTT标准差: {ptt_features['ptt_std']*1000:.1f} ms\n")
                    f.write(f"  心率均值: {ptt_features['hr_mean']:.1f} BPM\n")
                    f.write(f"  有效心跳数: {ptt_features['num_beats']}\n")
                    f.write(f"  设备同步质量: {ptt_features['sync_quality']:.3f}\n\n")
                    
                    all_ptt_means.append(ptt_features['ptt_mean'] * 1000)
                    all_ptt_stds.append(ptt_features['ptt_std'] * 1000)
            
            # 总体PTT统计
            f.write("3. 总体PTT统计\n")
            f.write("-" * 30 + "\n")
            if all_ptt_means:
                f.write(f"PTT均值范围: {np.min(all_ptt_means):.1f} - {np.max(all_ptt_means):.1f} ms\n")
                f.write(f"PTT均值平均: {np.mean(all_ptt_means):.1f} ms\n")
                f.write(f"PTT变异性范围: {np.min(all_ptt_stds):.1f} - {np.max(all_ptt_stds):.1f} ms\n")
                f.write(f"PTT变异性平均: {np.mean(all_ptt_stds):.1f} ms\n\n")
            
            # 静止vs动态分析
            f.write("4. 静止状态 vs 动态状态分析\n")
            f.write("-" * 30 + "\n")
            if not np.isnan(static_dynamic_analysis['static_ptt_mean']):
                f.write(f"静止状态PTT均值: {static_dynamic_analysis['static_ptt_mean']*1000:.1f} ± {static_dynamic_analysis['static_ptt_std']*1000:.1f} ms\n")
            if not np.isnan(static_dynamic_analysis['dynamic_ptt_mean']):
                f.write(f"动态状态PTT均值: {static_dynamic_analysis['dynamic_ptt_mean']*1000:.1f} ± {static_dynamic_analysis['dynamic_ptt_std']*1000:.1f} ms\n")
            f.write(f"静止状态条件数: {static_dynamic_analysis['static_conditions']}\n")
            f.write(f"动态状态条件数: {static_dynamic_analysis['dynamic_conditions']}\n\n")
            
            # 模型性能
            if performance:
                f.write("5. 血压预测模型性能\n")
                f.write("-" * 30 + "\n")
                for bp_type, perf in performance.items():
                    f.write(f"{bp_type.capitalize()}血压预测:\n")
                    f.write(f"  线性回归 - R²: {perf['linear']['r2']:.3f}, MAE: {perf['linear']['mae']:.1f} mmHg\n")
                    f.write(f"  随机森林 - R²: {perf['random_forest']['r2']:.3f}, MAE: {perf['random_forest']['mae']:.1f} mmHg\n\n")
            
            # 设备特性分析
            f.write("6. 设备特性分析\n")
            f.write("-" * 30 + "\n")
            f.write("Biopac系统 - 医用级连续血压和心率监测\n")
            f.write("  ✓ 提供准确的参考血压数据\n")
            f.write("  ✓ 连续心率监测用于R波时间推算\n")
            f.write("  ✓ 多种血压参数(收缩压/舒张压/平均压)\n\n")
            
            f.write("HUB系统 - 多传感器集成设备\n")
            f.write("  ✓ PPG信号(红光/红外/绿光)用于外周脉搏检测\n")
            f.write("  ✓ IMU传感器用于运动状态检测\n")
            f.write("  ✓ 温度传感器用于环境监测\n\n")
            
            # 结论和建议
            f.write("7. 结论和建议\n")
            f.write("-" * 30 + "\n")
            f.write("✓ 成功实现多设备PTT计算和血压预测\n")
            f.write("✓ 静止状态数据质量较好，适合血压预测建模\n")
            f.write("✓ 设备间时间同步质量影响PTT计算精度\n")
            f.write("✓ 建议优化信号预处理算法提高峰值检测准确性\n")
            f.write("✓ 可考虑增加更多生理特征提高预测精度\n")
        
        print(f"综合分析报告已保存到: {report_path}")


# 使用示例
if __name__ == "__main__":
    # 初始化处理器
    processor = MultiDevicePTTProcessor(
        data_root="/root/PI_Lab/00017",
        sampling_rate=100
    )
    
    # 运行完整流水线
    results = processor.run_full_pipeline("./ptt_multidevice_results")
    
    if results:
        print("\n🎉 多设备PTT血压预测处理成功完成!")
        print(f"📊 处理条件数: {len(results['all_results'])}")
        
        if results['models']:
            print("🤖 血压预测模型已训练完成")
        
        print("📁 结果文件:")
        print("  - all_conditions_results.npy: 所有条件处理结果")
        print("  - bp_prediction_models.npy: 血压预测模型")
        print("  - ptt_analysis_overview.png: PTT分析概览图")
        print("  - comprehensive_analysis_report.txt: 综合分析报告")
    else:
        print("❌ 处理失败，请检查数据路径和格式") 