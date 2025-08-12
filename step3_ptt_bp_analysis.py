#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PTT与血压相关性分析
基于师兄建议：使用合理区间的PTT数据分析与血压的相关性
"""

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 设置非交互模式，不显示弹窗
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和图片保存模式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.ioff()  # 关闭交互模式

class PTTBloodPressureAnalyzer:
    """PTT与血压相关性分析器"""
    
    def __init__(self, output_dir="ptt_bp_analysis"):
        self.output_dir = output_dir
        self.ptt_output_dir = "ptt_output"  # 窗口化PTT数据目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 血压相关生理指标映射（英文专业术语）
        self.physiological_indicators = {
            'systolic_bp': 'Systolic BP (mmHg)',
            'diastolic_bp': 'Diastolic BP (mmHg)', 
            'mean_bp': 'Mean Arterial Pressure (mmHg)',
        }
        
        # PTT传感器组合（英文标签）
        self.ptt_combinations_en = {
            'sensor2-sensor3': 'Nose→Finger',
            'sensor2-sensor4': 'Nose→Wrist', 
            'sensor2-sensor5': 'Nose→Ear',
            'sensor3-sensor4': 'Finger→Wrist',
            'sensor3-sensor5': 'Finger→Ear',
            'sensor4-sensor5': 'Wrist→Ear'
        }
        
        print("🔬 Enhanced PTT-Cardiovascular Parameters Correlation Analyzer")
        print(f"📁 Results will be saved to: {self.output_dir}")
        print(f"📊 Analyzing {len(self.physiological_indicators)} physiological indicators")
        print(f"🎯 Using {len(self.ptt_combinations_en)} PTT sensor combinations")
    
    def load_ground_truth_bp(self, exp_id):
        """加载生理指标数据（从CSV文件）"""
        try:
            # 加载CSV文件（修正文件名）
            csv_file = os.path.join(self.csv_output_dir, f'{self.subject}_{exp_id}_biopac_aligned.csv')
            if not os.path.exists(csv_file):
                print(f"❌ 生理数据文件不存在: {csv_file}")
                return None
            
            # 读取完整生理数据
            df = pd.read_csv(csv_file)
            print(f"✅ 加载生理数据: {len(df)}条记录")
            
            # 显示可用的生理指标
            available_indicators = [col for col in df.columns if col in self.physiological_indicators.keys()]
            print(f"📊 可用生理指标: {available_indicators}")
            
            return df
            
        except Exception as e:
            print(f"❌ 加载生理数据失败: {e}")
            return None
    
    def load_ptt_data(self, exp_id):
        """加载有效窗口的PTT数据"""
        try:
            # 加载窗口验证数据
            window_file = f"{self.ptt_output_dir}/exp_{exp_id}/window_validation_exp_{exp_id}.csv"
            ptt_file = f"{self.ptt_output_dir}/exp_{exp_id}/ptt_windowed_exp_{exp_id}.csv"
            
            if not (os.path.exists(window_file) and os.path.exists(ptt_file)):
                print(f"❌ PTT数据文件不存在: exp_{exp_id}")
                return None
            
            # 加载窗口验证信息
            window_df = pd.read_csv(window_file)
            # 筛选有效窗口（时频域心率误差合理）
            valid_windows = window_df[
                (window_df['is_valid'] == True) & 
                (window_df['hr_diff_bpm'].abs() <= 5)  # 心率误差≤5BPM
            ]
            
            # 加载PTT数据
            ptt_df = pd.read_csv(ptt_file)
            
            # 新增：基于IBI筛选 abs(PTT) < 0.5 * reference_mean_ibi_ms
            if 'reference_mean_ibi_ms' in ptt_df.columns:
                mask = np.abs(ptt_df['ptt_ms']) < 0.5 * ptt_df['reference_mean_ibi_ms']
                filtered_ptt = ptt_df[mask | ptt_df['reference_mean_ibi_ms'].isna()]  # 如果IBI NaN则保留
                print(f"🆕 IBI-based筛选: 原始{len(ptt_df)} → 筛选后{len(filtered_ptt)}")
                print(f"筛选合理比例: {len(filtered_ptt)/len(ptt_df)*100:.1f}%")  # 新增：输出筛选比例
            else:
                print("⚠️ 无reference_mean_ibi_ms列，跳过IBI筛选")
                filtered_ptt = ptt_df
            
            # 只保留有效窗口的PTT数据
            valid_ptt = filtered_ptt[filtered_ptt['window_id'].isin(valid_windows['window_id'])]
            
            print(f"📊 实验{exp_id}: 总窗口{len(window_df)}, 有效窗口{len(valid_windows)}, 有效PTT数据{len(valid_ptt)}")
            
            return {
                'window_info': valid_windows,
                'ptt_data': valid_ptt
            }
            
        except Exception as e:
            print(f"❌ 加载PTT数据失败: {e}")
            return None
    
    def remove_outliers_iqr(self, data_series):
        """使用IQR方法去除极值"""
        q1 = data_series.quantile(0.25)
        q3 = data_series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return data_series[(data_series >= lower_bound) & (data_series <= upper_bound)]
    
    def synchronize_data(self, ptt_data, physio_data, exp_id):
        """时间同步PTT和生理数据"""
        synchronized_data = []
        
        for _, ptt_row in ptt_data['ptt_data'].iterrows():
            # PTT数据的时间信息（修正列名）
            start_time = ptt_row['window_start_s']
            end_time = ptt_row['window_end_s']
            window_center = (start_time + end_time) / 2
            
            # 转换为时间戳（假设生理数据的timestamp是绝对时间戳）
            # 需要找到生理数据时间戳的起始点
            physio_start_time = physio_data['timestamp'].iloc[0]
            start_timestamp = physio_start_time + start_time
            end_timestamp = physio_start_time + end_time
            
            # 找到时间窗口内的生理数据
            time_mask = (physio_data['timestamp'] >= start_timestamp) & (physio_data['timestamp'] <= end_timestamp)
            window_physio = physio_data[time_mask]
            
            if len(window_physio) == 0:
                continue  # 跳过没有生理数据的窗口
            
            # 计算窗口内所有生理指标的统计量（只计算mean）
            physio_values = {}
            for indicator in self.physiological_indicators.keys():
                if indicator in physio_data.columns:
                    physio_values[f'{indicator}_mean'] = window_physio[indicator].mean()
                    physio_values[f'{indicator}_count'] = len(window_physio)
            
            # 构建同步数据行
            sync_row = {
                'exp_id': exp_id,
                'window_id': ptt_row['window_id'],
                'start_time': start_time,
                'end_time': end_time,
                'window_center': window_center,
                'sensor_pair': ptt_row['sensor_pair'],
                'ptt_ms': ptt_row['ptt_ms'],
                **physio_values
            }
            
            synchronized_data.append(sync_row)
        
        sync_df = pd.DataFrame(synchronized_data)
        print(f"📊 同步完成: {len(sync_df)}个有效窗口")
        
        # 新增：IQR去除极值（窗口级？但这里是心跳级，需分组）
        # 假设分组计算mean after IQR
        grouped = sync_df.groupby(['window_id', 'sensor_pair'])
        cleaned_data = []
        for name, group in grouped:
            clean_ptt = self.remove_outliers_iqr(group['ptt_ms'])
            if not clean_ptt.empty:
                mean_ptt = clean_ptt.mean()
                row = group.iloc[0].copy()
                row['ptt_ms'] = mean_ptt
                cleaned_data.append(row)
        cleaned_df = pd.DataFrame(cleaned_data)
        
        # 生成箱线图
        self.create_ptt_boxplot(cleaned_df, exp_id)  # 修改：传入exp_id
        
        return cleaned_df
    
    def create_ptt_boxplot(self, df, exp_id=None):
        """生成PTT箱线图"""
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='sensor_pair', y='ptt_ms', data=df)
        title = 'PTT Boxplot per Sensor Pair'
        if exp_id:
            title += f' (Exp {exp_id})'
            # 创建实验特定的文件夹
            exp_output_dir = os.path.join(self.output_dir, f'exp_{exp_id}')
            os.makedirs(exp_output_dir, exist_ok=True)
            filename = f'exp_{exp_id}_ptt_boxplot.png'
            filepath = os.path.join(exp_output_dir, filename)
        else:
            title += ' (Overall)'
            filename = 'overall_ptt_boxplot.png'
            filepath = os.path.join(self.output_dir, filename)
        if hasattr(self, 'subject'):
            title += f' (Subject {self.subject})'
        plt.title(title)
        plt.savefig(filepath)
        plt.close()
    
    def calculate_correlations(self, sync_df):
        """计算PTT与所有生理指标的相关性"""
        correlations = {}
        
        # 生理指标（只处理mean）
        physio_metrics = []
        for indicator in self.physiological_indicators.keys():
            col_name = f'{indicator}_mean'
            if col_name in sync_df.columns:
                physio_metrics.append(col_name)
        
        # 获取所有传感器对
        sensor_pairs = sync_df['sensor_pair'].unique()
        
        print(f"\n📊 计算相关性：{len(sensor_pairs)}个传感器对 × {len(physio_metrics)}个生理指标")
        
        for sensor_pair in sensor_pairs:
            correlations[sensor_pair] = {}
            
            # 提取该传感器对的数据
            pair_data = sync_df[sync_df['sensor_pair'] == sensor_pair]
            
            if len(pair_data) < 10:  # 至少10个数据点
                continue
            
            for physio_col in physio_metrics:
                # 提取有效数据
                mask = ~(pair_data['ptt_ms'].isna() | pair_data[physio_col].isna())
                if mask.sum() < 10:
                    continue
                
                ptt_vals = pair_data.loc[mask, 'ptt_ms']
                physio_vals = pair_data.loc[mask, physio_col]
                
                # 计算皮尔逊相关系数
                try:
                    corr_coef, p_value = stats.pearsonr(ptt_vals, physio_vals)
                    
                    correlations[sensor_pair][physio_col] = {
                        'correlation': corr_coef,
                        'p_value': p_value,
                        'n_samples': len(ptt_vals),
                        'significant': p_value < 0.05
                    }
                except Exception as e:
                    print(f"⚠️ 计算相关性失败 {sensor_pair}-{physio_col}: {e}")
                    continue
        
        return correlations
    
    def create_correlation_heatmap(self, correlations, title_suffix=""):
        """创建相关性热图（英文专业版）"""
        # 准备数据
        sensor_pairs = list(correlations.keys())
        physio_cols = set()
        for pair_data in correlations.values():
            physio_cols.update(pair_data.keys())
        physio_cols = sorted(list(physio_cols))
        
        if len(sensor_pairs) == 0 or len(physio_cols) == 0:
            print("⚠️ 没有足够的相关性数据来创建热图")
            return None
        
        # 创建相关性矩阵
        corr_matrix = np.full((len(sensor_pairs), len(physio_cols)), np.nan)
        p_matrix = np.full((len(sensor_pairs), len(physio_cols)), np.nan)
        
        for i, sensor_pair in enumerate(sensor_pairs):
            for j, physio_col in enumerate(physio_cols):
                if physio_col in correlations[sensor_pair]:
                    corr_matrix[i, j] = correlations[sensor_pair][physio_col]['correlation']
                    p_matrix[i, j] = correlations[sensor_pair][physio_col]['p_value']
        
        # 新增：预格式化annot字符串
        annot_matrix = np.full((len(sensor_pairs), len(physio_cols)), '', dtype=object)
        for i in range(len(sensor_pairs)):
            for j in range(len(physio_cols)):
                if not np.isnan(corr_matrix[i, j]):
                    corr_str = f"{corr_matrix[i, j]:.3f}"
                    if p_matrix[i, j] < 0.05:
                        corr_str += '*'
                    annot_matrix[i, j] = corr_str
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # 绘制热图
        mask = np.isnan(corr_matrix)
        im = sns.heatmap(corr_matrix, 
                        xticklabels=[self._format_physio_label_en(col) for col in physio_cols],
                        yticklabels=[self._format_sensor_pair_label_en(pair) for pair in sensor_pairs],
                        annot=annot_matrix, fmt='', cmap='RdBu_r', center=0,
                        mask=mask, square=False, linewidths=0.5,
                        cbar_kws={'label': 'Correlation Coefficient'},
                        annot_kws={'size': 8})
        
        # 移除旧的ax.text
        
        plt.title(f'PTT-Cardiovascular Parameters Correlation Analysis{title_suffix}', 
                 fontsize=16, fontweight='bold', pad=20)
        if hasattr(self, 'subject'):
            plt.suptitle(f'Subject {self.subject}', y=1.02, fontsize=12)
        plt.xlabel('Physiological Parameters', fontsize=12, fontweight='bold')
        plt.ylabel('PTT Sensor Combinations', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # 保存图像
        if title_suffix and title_suffix.startswith("_exp"):
            # 提取实验ID
            exp_id = title_suffix.replace("_exp", "").replace(" ", "_")
            exp_output_dir = os.path.join(self.output_dir, f'exp_{exp_id}')
            os.makedirs(exp_output_dir, exist_ok=True)
            filename = f"{exp_output_dir}/ptt_cardiovascular_correlation_heatmap_{exp_id}.png"
        else:
            filename = f"{self.output_dir}/ptt_cardiovascular_correlation_heatmap{title_suffix.replace(' ', '_')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"💾 保存相关性热图: {filename}")
        
        return fig
    
    def _format_sensor_pair_label(self, sensor_pair):
        """格式化传感器对标签"""
        # sensor2-sensor3 -> nose→finger
        sensor_map = {'sensor2': 'nose', 'sensor3': 'finger', 'sensor4': 'wrist', 'sensor5': 'ear'}
        if '-' in sensor_pair:
            parts = sensor_pair.split('-')
            if len(parts) == 2:
                return f"{sensor_map.get(parts[0], parts[0])}→{sensor_map.get(parts[1], parts[1])}"
        return sensor_pair
    
    def _format_physio_label_en(self, physio_col):
        """格式化生理指标标签（英文专业版）"""
        # 提取基础指标名称和统计量
        for indicator, label in self.physiological_indicators.items():
            if physio_col.startswith(indicator):
                stat_part = physio_col.replace(indicator, '').replace('_', ' ')
                if stat_part == ' mean':
                    return label
                elif stat_part == ' std':
                    return f"{label} (SD)"
                elif stat_part == ' min':
                    return f"{label} (Min)"
                elif stat_part == ' max':
                    return f"{label} (Max)"
                else:
                    return f"{label}{stat_part}"
        return physio_col
    
    def _format_sensor_pair_label_en(self, sensor_pair):
        """格式化传感器对标签（英文版）"""
        return self.ptt_combinations_en.get(sensor_pair, sensor_pair)
    
    def build_regression_models(self, sync_df, correlations, exp_id=None):
        """构建PTT→生理指标的回归模型，并返回模型和数据"""
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)

        all_corrs = []
        for sensor_pair, physio_data in correlations.items():
            for physio_col, stats_data in physio_data.items():
                if stats_data['significant']:
                    all_corrs.append((abs(stats_data['correlation']), 
                                    self._format_sensor_pair_label_en(sensor_pair),
                                    self._format_physio_label_en(physio_col),
                                    stats_data['correlation'],
                                    stats_data['p_value'],
                                    stats_data['n_samples']))
                
        all_corrs.sort(reverse=True)
        
        # 创建相关性映射，方便后续查找
        corr_map = {}
        for sensor_pair, physio_data in correlations.items():
            for physio_col, stats_data in physio_data.items():
                key = f"{sensor_pair}→{physio_col}"
                corr_map[key] = {
                    'correlation': stats_data['correlation'],
                    'p_value': stats_data['p_value']
                }
        
        # 主要血压指标（均值）
        main_physio_cols = []
        for indicator in ['systolic_bp', 'diastolic_bp', 'mean_bp']:
            col_name = f'{indicator}_mean'
            if col_name in sync_df.columns:
                main_physio_cols.append(col_name)
        
        # 获取所有传感器对
        sensor_pairs = sync_df['sensor_pair'].unique()
        print(f"🔍 发现传感器对: {sensor_pairs}")
        
        # 创建结果数据结构
        all_models = {}
        all_model_data = {}
        metrics_list = []  # 用于存储 CSV 的指标数据
        
        # 为每个传感器对单独处理
        for sensor_pair in sensor_pairs:
            print(f"\n🔧 处理传感器对: {sensor_pair}")
            
            # 过滤当前传感器对的数据
            pair_df = sync_df[sync_df['sensor_pair'] == sensor_pair].copy()
            
            # 创建数据透视表 - 每个窗口一个PTT值
            ptt_pivot = pair_df.pivot_table(
                index=['exp_id', 'window_id'], 
                values='ptt_ms',
                aggfunc='mean'
            ).reset_index().rename(columns={'ptt_ms': f'ptt_{sensor_pair}'})
            
            # 合并生理数据（取平均值）
            physio_agg = pair_df.groupby(['exp_id', 'window_id']).agg({
                col: 'mean' for col in main_physio_cols if col in pair_df.columns
            }).reset_index()
            
            # 合并PTT和生理数据
            model_data = pd.merge(ptt_pivot, physio_agg, on=['exp_id', 'window_id'], how='inner')
            
            # 检查数据量
            if len(model_data) < 10:
                print(f"⚠️ 数据不足: {sensor_pair} 只有{len(model_data)}个样本")
                continue
                
            # 获取PTT特征列
            ptt_col = f'ptt_{sensor_pair}'
            
            # 检查PTT列的NaN比例
            nan_ratio = model_data[ptt_col].isna().mean()
            print(f"📊 PTT列 {ptt_col} NaN比例: {nan_ratio:.2%}")
            
            # 为每个生理指标单独建模
            for physio_col in main_physio_cols:
                if physio_col not in model_data.columns:
                    continue
                    
                # 准备数据 - 移除NaN
                mask = ~model_data[physio_col].isna() & ~model_data[ptt_col].isna()
                
                if mask.sum() < 5:  # 至少5个样本
                    print(f"⚠️ 数据不足: {sensor_pair}→{physio_col} 有效样本={mask.sum()}")
                    continue
                
                X = model_data.loc[mask, ptt_col].values.reshape(-1, 1)
                y = model_data.loc[mask, physio_col].values
                
                # 数据标准化
                scaler_X = StandardScaler()
                scaler_y = StandardScaler()
                X_scaled = scaler_X.fit_transform(X)
                y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
                
                # 训练模型
                model = LinearRegression()
                model.fit(X_scaled, y_scaled)
                
                # 预测
                y_pred_scaled = model.predict(X_scaled)
                y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
                
                # 评估
                r2 = r2_score(y, y_pred)
                mae = mean_absolute_error(y, y_pred)
                
                # 获取相关性数据
                model_key = f"{sensor_pair}→{physio_col}"
                corr_data = corr_map.get(model_key, {'correlation': float('nan'), 'p_value': float('nan')})
                ptt_correlation = corr_data['correlation']
                ptt_p_value = corr_data['p_value']
                
                # 存储模型
                all_models[model_key] = {
                    'model': model,
                    'scaler_X': scaler_X,
                    'scaler_y': scaler_y,
                    'feature_names': [ptt_col],
                    'r2_score': r2,
                    'mae': mae,
                    'n_samples': len(y),
                    'y_true': y,
                    'y_pred': y_pred,
                    'sensor_pair': sensor_pair,
                    'physio_col': physio_col,
                    'ptt_correlation': ptt_correlation,
                    'ptt_p_value': ptt_p_value
                }
                
                print(f"📈 {model_key}模型: R²={r2:.3f}, MAE={mae:.2f}, N={len(y)}")
                print(f"   📊 PTT相关性: r={ptt_correlation:.3f}, p={ptt_p_value:.2e}")
                
                # 创建图表
                plt.figure(figsize=(10, 8))  # 增加图表高度以容纳更多信息
                
                # 1. 绘制原始数据点
                plt.scatter(X, y, alpha=0.6, color='blue', label='Data Points')
                
                # 2. 绘制拟合直线
                x_min, x_max = np.min(X), np.max(X)
                x_range = np.linspace(x_min, x_max, 100).reshape(-1, 1)
                
                x_range_scaled = scaler_X.transform(x_range)
                y_range_scaled = model.predict(x_range_scaled)
                y_range = scaler_y.inverse_transform(y_range_scaled.reshape(-1, 1)).flatten()
                
                plt.plot(x_range, y_range, 'r-', linewidth=2, label='Fitted Line')
                
                # 3. 添加图例和标签
                plt.xlabel(f'PTT ({sensor_pair}) (ms)')
                plt.ylabel(f'{self._format_physio_label_en(physio_col)}')
                
                # 获取方程系数
                coef = model.coef_[0]
                intercept = model.intercept_
                
                # 计算原始数据空间的斜率和截距
                coef_original = coef * (scaler_y.scale_[0] / scaler_X.scale_[0])
                intercept_original = scaler_y.mean_[0] - coef * (scaler_X.mean_[0] * scaler_y.scale_[0] / scaler_X.scale_[0]) + intercept * scaler_y.scale_[0]
                
                # 4. 更新标题，包含相关性信息
                title = f'{self._format_physio_label_en(physio_col)} vs PTT ({sensor_pair}) (Subject {self.subject})\n'
                title += f'Equation: y = {coef_original:.3f}x + {intercept_original:.3f} | Correlation: r={ptt_correlation:.3f}, p={ptt_p_value:.2e}\n'
                title += f'R²={r2:.3f}, MAE={mae:.2f}, n={len(y)}'
                plt.title(title)
                
                plt.legend()
                plt.grid(alpha=0.3)
                
                # 保存图表
                safe_physio = physio_col.replace(' ', '_').replace('/', '_')
                safe_pair = sensor_pair.replace(' ', '_').replace('/', '_')
                if exp_id is not None:
                    # 创建实验特定的文件夹
                    exp_output_dir = os.path.join(self.output_dir, f'exp_{exp_id}')
                    os.makedirs(exp_output_dir, exist_ok=True)
                    plot_path = os.path.join(exp_output_dir, f"exp_{exp_id}_{safe_physio}_vs_{safe_pair}_fit.png")
                else:
                    plot_path = os.path.join(self.output_dir, f"{safe_physio}_vs_{safe_pair}_fit.png")
                plt.savefig(plot_path, bbox_inches='tight', dpi=150)
                plt.close()
                
                print(f"💾 保存特征拟合图: {plot_path}")
                
                # 存储模型数据
                all_model_data[model_key] = model_data.loc[mask, [ptt_col, physio_col]]
                
                # 收集指标数据用于 CSV
                if exp_id is not None:
                    metrics_list.append({
                        'exp_id': exp_id,
                        'sensor_pair': sensor_pair,
                        'sensor_combination': self._format_sensor_pair_label_en(sensor_pair),
                        'physiological_parameter': physio_col,
                        'parameter_label': self._format_physio_label_en(physio_col),
                        'r2_score': r2,
                        'mae': mae,
                        'n_samples': len(y),
                        'slope': coef_original,
                        'intercept': intercept_original,
                        'ptt_correlation': ptt_correlation,
                        'ptt_p_value': ptt_p_value,
                        'correlation_significant': ptt_p_value < 0.05
                    })
                else:
                    metrics_list.append({
                        'sensor_pair': sensor_pair,
                        'sensor_combination': self._format_sensor_pair_label_en(sensor_pair),
                        'physiological_parameter': physio_col,
                        'parameter_label': self._format_physio_label_en(physio_col),
                        'r2_score': r2,
                        'mae': mae,
                        'n_samples': len(y),
                        'slope': coef_original,
                        'intercept': intercept_original,
                        'ptt_correlation': ptt_correlation,
                        'ptt_p_value': ptt_p_value,
                        'correlation_significant': ptt_p_value < 0.05
                    })
        
        # 如果没有 exp_id，保存指标到 CSV
        if exp_id is None:
            print("保存整体模型评估：")
            csv_path = os.path.join(self.output_dir, "overall_regression_metrics.csv")
            metrics_df = pd.DataFrame(metrics_list)
            metrics_df.to_csv(csv_path, index=False)
        else:
            print("保存单个实验模型评估：")
            csv_path = os.path.join(self.output_dir, "all_experiments_regression_metrics.csv")
            metrics_df = pd.DataFrame(metrics_list)
            # 如果文件存在，追加数据；否则创建新文件
            if os.path.exists(csv_path):
                existing_df = pd.read_csv(csv_path)
                combined_df = pd.concat([existing_df, metrics_df], ignore_index=True)
                combined_df.to_csv(csv_path, index=False)
            else:
                metrics_df.to_csv(csv_path, index=False)
        
        return all_models, all_model_data

    def analyze_experiment(self, exp_id):
        """分析单个实验"""
        print(f"\n🔍 分析实验 {exp_id}")
        
        # 1. 加载数据
        physio_data = self.load_ground_truth_bp(exp_id)
        ptt_data = self.load_ptt_data(exp_id)
        
        if physio_data is None or ptt_data is None:
            print(f"❌ 实验 {exp_id} 数据加载失败")
            return None
        
        # 2. 时间同步
        sync_df = self.synchronize_data(ptt_data, physio_data, exp_id)
        print(f"📊 同步数据: {len(sync_df)}个时间窗口")
        
        # 3. 相关性分析
        correlations = self.calculate_correlations(sync_df)
        
        # 4. 回归建模
        models, model_data = self.build_regression_models(sync_df, correlations, exp_id=exp_id)
        
        return {
            'sync_data': sync_df,
            'correlations': correlations,
            'models': models
        }
    
    def analyze_experiment_cross(self, exp_id):
        # 1. 加载数据
        physio_data = self.load_ground_truth_bp(exp_id)
        ptt_data = self.load_ptt_data(exp_id)
        
        if physio_data is None or ptt_data is None:
            print(f"❌ 实验 {exp_id} 数据加载失败")
            return None
        
        # 2. 时间同步
        sync_df = self.synchronize_data(ptt_data, physio_data, exp_id)
        print(f"📊 同步数据: {len(sync_df)}个时间窗口")

        return {
            'sync_data': sync_df,
        }
    
    def run_individual_experiment_analysis(self):
        """运行单个实验的分析"""
        print("🔬 开始单个实验分析...")
        
        individual_results = {}
        all_experiments = []
        
        for exp_id in range(1, 12):
            print(f"\n🔍 单独分析实验 {exp_id}")
            
            # 分析单个实验
            exp_result = self.analyze_experiment(exp_id)
            if exp_result:
                individual_results[exp_id] = exp_result['sync_data']
                
                # 计算相关性
                correlations = self.calculate_correlations(exp_result['sync_data'])
                
                # 创建单个实验的热图
                self.create_focused_correlation_heatmap(correlations, f"_exp{exp_id}")
                
                # 创建单个实验的Bland-Altman图
                self.create_bland_altman_plots(exp_result['sync_data'], exp_id=exp_id)
                
                # 保存单个实验结果
                self.save_individual_experiment_results(exp_result['sync_data'], correlations, exp_id)

                # 用于结果合并
                all_experiments.append(exp_result['sync_data'])
        
        if not all_experiments:
            print("❌ 没有有效的实验数据")
            return None
        
        # 合并所有实验的数据
        combined_df = pd.concat(all_experiments, ignore_index=True)
        print(combined_df.head())
        print(f"\n📊 合并数据: {len(combined_df)}个样本，来自{len(all_experiments)}个实验")
        
        return individual_results, combined_df
    
    def create_focused_correlation_heatmap(self, correlations, title_suffix=""):
        """创建聚焦的相关性热图（只显示重要指标）"""
        # 选择重要的血压指标（减少图像大小）
        important_indicators = [
            'systolic_bp_mean', 'diastolic_bp_mean', 'mean_bp_mean'
        ]
        
        # 准备数据
        sensor_pairs = list(correlations.keys())
        filtered_correlations = {}
        
        for sensor_pair in sensor_pairs:
            filtered_correlations[sensor_pair] = {
                col: correlations[sensor_pair][col] 
                for col in important_indicators 
                if col in correlations[sensor_pair]
            }
        
        # 创建相关性矩阵
        physio_cols = list(set().union(*[pair_data.keys() for pair_data in filtered_correlations.values()]))
        physio_cols = sorted(physio_cols)
        
        if len(sensor_pairs) == 0 or len(physio_cols) == 0:
            print("⚠️ 没有足够的数据创建聚焦热图")
            return None
        
        corr_matrix = np.full((len(sensor_pairs), len(physio_cols)), np.nan)
        p_matrix = np.full((len(sensor_pairs), len(physio_cols)), np.nan)
        
        for i, sensor_pair in enumerate(sensor_pairs):
            for j, physio_col in enumerate(physio_cols):
                if physio_col in filtered_correlations[sensor_pair]:
                    corr_matrix[i, j] = filtered_correlations[sensor_pair][physio_col]['correlation']
                    p_matrix[i, j] = filtered_correlations[sensor_pair][physio_col]['p_value']
        
        # 新增：预格式化annot字符串
        annot_matrix = np.full((len(sensor_pairs), len(physio_cols)), '', dtype=object)
        for i in range(len(sensor_pairs)):
            for j in range(len(physio_cols)):
                if not np.isnan(corr_matrix[i, j]):
                    corr_str = f"{corr_matrix[i, j]:.3f}"
                    if p_matrix[i, j] < 0.05:
                        corr_str += '*'
                    annot_matrix[i, j] = corr_str
        
        # 创建图形（更小更清晰）
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 绘制热图
        mask = np.isnan(corr_matrix)
        im = sns.heatmap(corr_matrix, 
                        xticklabels=[self._format_physio_label_en(col) for col in physio_cols],
                        yticklabels=[self._format_sensor_pair_label_en(pair) for pair in sensor_pairs],
                        annot=annot_matrix, fmt='', cmap='RdBu_r', center=0,
                        mask=mask, square=False, linewidths=0.5,
                        cbar_kws={'label': 'Correlation Coefficient'},
                        annot_kws={'size': 10})
        
        # 移除旧的ax.text
        
        plt.title(f'PTT-Cardiovascular Correlation Analysis (Key Parameters){title_suffix}', 
                 fontsize=14, fontweight='bold', pad=20)
        if hasattr(self, 'subject'):
            plt.suptitle(f'Subject {self.subject}', y=1.02, fontsize=12)
        plt.xlabel('Physiological Parameters', fontsize=12, fontweight='bold')
        plt.ylabel('PTT Sensor Combinations', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # 保存图像
        if title_suffix and title_suffix.startswith("_exp"):
            # 提取实验ID
            exp_id = title_suffix.replace("_exp", "").replace(" ", "_")
            exp_output_dir = os.path.join(self.output_dir, f'exp_{exp_id}')
            os.makedirs(exp_output_dir, exist_ok=True)
            filename = f"{exp_output_dir}/ptt_cardiovascular_correlation_focused_{exp_id}.png"
        else:
            filename = f"{self.output_dir}/ptt_cardiovascular_correlation_focused{title_suffix.replace(' ', '_')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"💾 保存聚焦热图: {filename}")
        
        return fig
    
    def save_individual_experiment_results(self, sync_data, correlations, exp_id):
        """保存单个实验的结果"""
        # 保存相关性结果
        corr_results = []
        for sensor_pair, physio_data in correlations.items():
            for physio_col, stats_data in physio_data.items():
                corr_results.append({
                    'experiment_id': exp_id,
                    'sensor_pair': sensor_pair,
                    'sensor_combination': self._format_sensor_pair_label_en(sensor_pair),
                    'physiological_parameter': physio_col,
                    'parameter_label': self._format_physio_label_en(physio_col),
                    'correlation_coefficient': stats_data['correlation'],
                    'p_value': stats_data['p_value'],
                    'n_samples': stats_data['n_samples'],
                    'statistically_significant': stats_data['significant']
                })
        
        corr_df = pd.DataFrame(corr_results)
        # 创建实验特定的文件夹
        exp_output_dir = os.path.join(self.output_dir, f'exp_{exp_id}')
        os.makedirs(exp_output_dir, exist_ok=True)
        corr_file = f"{exp_output_dir}/ptt_cardiovascular_correlations_exp_{exp_id}.csv"
        corr_df.to_csv(corr_file, index=False)
        print(f"💾 保存实验{exp_id}相关性: {corr_file}")
    
    def run_comprehensive_analysis(self):
        """运行综合分析（单个+跨实验实验）"""
        print("🔬 开始PTT与生理参数综合分析")
        print("📋 分析实验列表: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]")

        # 1. 单个实验分析（新增功能）
        print("\n=== 第一部分：单个实验分析 ===")
        individual_results, combined_df = self.run_individual_experiment_analysis()
        
        # 1. 整体分析（原有功能）
        print("\n=== 第二部分：整体分析 ===")
        overall_results = self.run_overall_analysis(combined_df)
        
        # 3. 创建聚焦热图（优化显示）
        print("\n=== 第三部分：创建聚焦热图（只显示重要指标）===")
        if overall_results:
            self.create_focused_correlation_heatmap(overall_results['correlations'], "_overall_focus")
        
        # 4. 创建整体Bland-Altman图
        print("\n=== 第四部分：创建整体Bland-Altman图 ===")
        if overall_results:
            self.create_bland_altman_plots(overall_results['combined_data'], exp_id=None)
        
        return {
            'overall': overall_results,
            'individual': individual_results
        }
    
    def run_overall_analysis(self, combined_df):
        """运行整体分析（原有功能重命名）"""
        # 计算整体相关性
        print("\n📈 计算整体相关性...")
        correlations = self.calculate_correlations(combined_df)
        
        # 创建相关性热图
        self.create_correlation_heatmap(correlations, "_overall")
        
        # 构建回归模型
        print("\n🎯 构建整体回归模型...")
        models = self.build_regression_models(combined_df, correlations, exp_id=None)
        
        # 保存结果
        self.save_analysis_results(combined_df, correlations, models)
        
        # 新增：整体箱线图
        self.create_ptt_boxplot(combined_df, None)
        
        return {
            'combined_data': combined_df,
            'correlations': correlations,
            'models': models
        }
    
    def run_cross_experiments_analysis(self):
        """跨实验构建回归模型"""
        print("\n🎯 开始跨实验拟合分析...")
        # 为每个实验单独建模
        all_experiments = []
        
        for exp_id in range(1, 12):
            # 分析单个实验
            exp_result = self.analyze_experiment_cross(exp_id)
            if exp_result:
                # 用于结果合并
                all_experiments.append(exp_result['sync_data'])
        
        if not all_experiments:
            print("❌ 没有有效的实验数据")
            return None
        
        # 合并所有实验的数据
        combined_df = pd.concat(all_experiments, ignore_index=True)
        print(combined_df.head())
        print(f"\n📊 合并数据: {len(combined_df)}个样本，来自{len(all_experiments)}个实验")
            
        overall_results = self.run_overall_analysis(combined_df)

        if overall_results:
            self.create_focused_correlation_heatmap(overall_results['correlations'], "_overall_focus")
        
        return {
            'overall': overall_results,
        }
    
    def run_individual_regression_analysis(self):
        """为每个实验单独构建回归模型并绘制拟合直线"""
        print("\n🎯 开始单独实验回归分析...")
        individual_models = {}
        model_summary = []
        
        for exp_id in range(1, 12):  # 实验1-11
            print(f"\n📊 构建实验{exp_id}的回归模型")
            exp_data = self.analyze_experiment(exp_id)
            
            if not exp_data or len(exp_data['sync_data']) < 20:
                print(f"❌ 实验{exp_id}数据不足（<20样本）")
                continue
            
            # 直接从analyze_experiment的结果中获取模型
            if 'models' in exp_data and exp_data['models']:
                exp_models = exp_data['models']
                individual_models[f'exp_{exp_id}'] = exp_models
                
                # 收集模型性能统计
                for model_key, model_info in exp_models.items():
                    # 从model_key中提取生理指标名称
                    # model_key格式: "sensor_pair→physio_col"
                    physio_param = model_key.split('→')[1] if '→' in model_key else model_key
                    
                    model_summary.append({
                        'experiment': exp_id,
                        'physiological_parameter': physio_param,
                        'parameter_label': self._format_physio_label_en(physio_param),
                        'r2_score': model_info['r2_score'],
                        'mae': model_info['mae'],
                        'n_samples': model_info['n_samples'],
                        'sensor_pair': model_info.get('sensor_pair', ''),
                        'sensor_label': self._format_sensor_pair_label_en(model_info.get('sensor_pair', ''))
                    })
        
        # 保存单独实验的模型评估
        if model_summary:
            model_df = pd.DataFrame(model_summary)
            model_file = f"{self.output_dir}/individual_experiment_models.csv"
            model_df.to_csv(model_file, index=False)
            print(f"💾 保存单独实验模型评估: {model_file}")

            # 对每个实验的每个生理参数，选择最佳模型（按R²）
            # 首先按实验和生理参数分组，然后在每个组内取R²最大的行
            best_model_df = model_df.loc[model_df.groupby(['experiment', 'physiological_parameter'])['r2_score'].idxmax()]
            
            # 创建模型性能对比可视化
            self.create_individual_model_comparison(best_model_df)
        
        return individual_models

    def create_individual_model_comparison(self, model_df):
        """创建单独实验模型性能对比图"""
        if model_df.empty:
            return
        
        # 创建MAE和R²对比图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # MAE对比
        pivot_mae = model_df.pivot(index='experiment', columns='parameter_label', values='mae')
        im1 = ax1.imshow(pivot_mae.values.T, cmap='Reds', aspect='auto')
        ax1.set_title('MAE Comparison Across Experiments (Lower is Better)', fontsize=14, fontweight='bold')
        if hasattr(self, 'subject'):
            ax1.set_suptitle(f'Subject {self.subject}', y=1.02, fontsize=12)
        ax1.set_xlabel('Experiment Number')
        ax1.set_ylabel('Physiological Parameter')
        ax1.set_xticks(range(len(pivot_mae.index)))
        ax1.set_xticklabels(pivot_mae.index)
        ax1.set_yticks(range(len(pivot_mae.columns)))
        ax1.set_yticklabels(pivot_mae.columns, fontsize=10)
        
        # 添加MAE数值标注
        for i in range(len(pivot_mae.columns)):
            for j in range(len(pivot_mae.index)):
                if not np.isnan(pivot_mae.iloc[j, i]):
                    ax1.text(j, i, f'{pivot_mae.iloc[j, i]:.1f}', 
                            ha='center', va='center', fontweight='bold')
        
        plt.colorbar(im1, ax=ax1, label='MAE')
        
        # R²对比
        pivot_r2 = model_df.pivot(index='experiment', columns='parameter_label', values='r2_score')
        im2 = ax2.imshow(pivot_r2.values.T, cmap='Blues', aspect='auto')
        ax2.set_title('R² Comparison Across Experiments (Higher is Better)', fontsize=14, fontweight='bold')
        if hasattr(self, 'subject'):
            ax2.set_suptitle(f'Subject {self.subject}', y=1.02, fontsize=12)
        ax2.set_xlabel('Experiment Number')
        ax2.set_ylabel('Physiological Parameter')
        ax2.set_xticks(range(len(pivot_r2.index)))
        ax2.set_xticklabels(pivot_r2.index)
        ax2.set_yticks(range(len(pivot_r2.columns)))
        ax2.set_yticklabels(pivot_r2.columns, fontsize=10)
        
        # 添加R²数值标注
        for i in range(len(pivot_r2.columns)):
            for j in range(len(pivot_r2.index)):
                if not np.isnan(pivot_r2.iloc[j, i]):
                    ax2.text(j, i, f'{pivot_r2.iloc[j, i]:.2f}', 
                            ha='center', va='center', fontweight='bold')
        
        plt.colorbar(im2, ax=ax2, label='R² Score')
        
        plt.tight_layout()
        comparison_file = f"{self.output_dir}/individual_model_performance_comparison.png"
        plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
        plt.close()  # 关闭图形，不显示
        print(f"💾 保存模型性能对比图: {comparison_file}")

    def save_analysis_results(self, combined_df, correlations, models):
        """保存分析结果（英文版）"""
        # 1. 保存同步数据
        sync_file = f"{self.output_dir}/synchronized_ptt_cardiovascular_data.csv"
        combined_df.to_csv(sync_file, index=False)
        print(f"💾 保存同步数据: {sync_file}")
        
        # 2. 保存相关性结果
        corr_results = []
        for sensor_pair, physio_data in correlations.items():
            for physio_col, stats_data in physio_data.items():
                corr_results.append({
                    'sensor_pair': sensor_pair,
                    'sensor_combination': self._format_sensor_pair_label_en(sensor_pair),
                    'physiological_parameter': physio_col,
                    'parameter_label': self._format_physio_label_en(physio_col),
                    'correlation_coefficient': stats_data['correlation'],
                    'p_value': stats_data['p_value'],
                    'n_samples': stats_data['n_samples'],
                    'statistically_significant': stats_data['significant']
                })
        
        corr_df = pd.DataFrame(corr_results)
        corr_file = f"{self.output_dir}/ptt_cardiovascular_correlations.csv"
        corr_df.to_csv(corr_file, index=False)
        print(f"💾 保存相关性数据: {corr_file}")

    def create_bland_altman_plots(self, sync_df, exp_id=None):
        """创建Bland-Altman图 - 六个传感器对的SBP、DBP和Mean BP"""
        try:
            # 血压指标 - 包括Mean BP
            bp_indicators = ['systolic_bp_mean', 'diastolic_bp_mean', 'mean_bp_mean']
            bp_labels = ['Systolic BP', 'Diastolic BP', 'Mean BP']
            
            # 传感器对
            sensor_pairs = list(self.ptt_combinations_en.keys())
            
            # 创建6x6的子图布局 - 6行（传感器对）x 6列（每个生理指标左右两个图）
            fig, axes = plt.subplots(6, 6, figsize=(24, 20))
            fig.suptitle(f'PTT vs Reference BP Analysis{"" if exp_id is None else f" (Exp {exp_id})"}', 
                        fontsize=16, fontweight='bold', y=1)
            
            if hasattr(self, 'subject'):
                fig.suptitle(f'PTT vs Reference BP Analysis - Subject {self.subject}{"" if exp_id is None else f" (Exp {exp_id})"}', 
                            fontsize=16, fontweight='bold', y=0.99)
            
            # 存储误差带统计信息
            error_band_stats = []
            
            for sensor_idx, sensor_pair in enumerate(sensor_pairs):
                for bp_idx, (bp_indicator, bp_label) in enumerate(zip(bp_indicators, bp_labels)):
                    # 计算子图位置 - 6行x6列布局
                    row = sensor_idx  # 0-5 for 6 sensor pairs
                    col_left = bp_idx * 2      # 0, 2, 4 for left plots (regression)
                    col_right = bp_idx * 2 + 1 # 1, 3, 5 for right plots (bland-altman)
                    
                    ax_left = axes[row, col_left]   # 左侧回归图
                    ax_right = axes[row, col_right] # 右侧Bland-Altman图
                    
                    # 获取该传感器对的数据
                    pair_data = sync_df[sync_df['sensor_pair'] == sensor_pair].copy()
                    
                    if len(pair_data) < 10:
                        ax_left.text(0.5, 0.5, 'Insufficient Data', ha='center', va='center', 
                                   transform=ax_left.transAxes, fontsize=10)
                        ax_right.text(0.5, 0.5, 'Insufficient Data', ha='center', va='center', 
                                    transform=ax_right.transAxes, fontsize=10)
                        ax_left.set_title(f'{self.ptt_combinations_en[sensor_pair]}\n{bp_label}')
                        continue
                    
                    # 准备数据
                    mask = ~(pair_data['ptt_ms'].isna() | pair_data[bp_indicator].isna())
                    if mask.sum() < 10:
                        ax_left.text(0.5, 0.5, 'Insufficient Data', ha='center', va='center', 
                                   transform=ax_left.transAxes, fontsize=10)
                        ax_right.text(0.5, 0.5, 'Insufficient Data', ha='center', va='center', 
                                    transform=ax_right.transAxes, fontsize=10)
                        ax_left.set_title(f'{self.ptt_combinations_en[sensor_pair]}\n{bp_label}')
                        continue
                    
                    ptt_vals = pair_data.loc[mask, 'ptt_ms'].values
                    bp_vals = pair_data.loc[mask, bp_indicator].values
                    
                    # 构建简单的线性回归模型进行预测
                    try:
                        # 标准化数据
                        scaler_ptt = StandardScaler()
                        scaler_bp = StandardScaler()
                        ptt_scaled = scaler_ptt.fit_transform(ptt_vals.reshape(-1, 1))
                        bp_scaled = scaler_bp.fit_transform(bp_vals.reshape(-1, 1))
                        
                        # 训练模型
                        model = LinearRegression()
                        model.fit(ptt_scaled, bp_scaled.flatten())
                        
                        # 预测
                        bp_pred_scaled = model.predict(ptt_scaled)
                        bp_pred = scaler_bp.inverse_transform(bp_pred_scaled.reshape(-1, 1)).flatten()
                        
                    except Exception as e:
                        print(f"⚠️ 模型训练失败 {sensor_pair}-{bp_indicator}: {e}")
                        ax_left.text(0.5, 0.5, 'Model Error', ha='center', va='center', 
                                   transform=ax_left.transAxes, fontsize=10)
                        ax_right.text(0.5, 0.5, 'Model Error', ha='center', va='center', 
                                    transform=ax_right.transAxes, fontsize=10)
                        ax_left.set_title(f'{self.ptt_combinations_en[sensor_pair]}\n{bp_label}')
                        continue
                    
                    # === 左侧：回归拟合图 + 误差带 ===
                    # 绘制误差带（改进的颜色设置：一层一层叠加）
                    bp_range = [min(bp_vals.min(), bp_pred.min()), max(bp_vals.max(), bp_pred.max())]
                    # 先绘制最大的误差带（15mmHg）
                    ax_left.fill_between(bp_range, 
                                       [b - 15 for b in bp_range], [b + 15 for b in bp_range],
                                       alpha=0.3, color='pink', label='±15 mmHg')
                    # 再绘制中等误差带（10mmHg）
                    ax_left.fill_between(bp_range, 
                                       [b - 10 for b in bp_range], [b + 10 for b in bp_range],
                                       alpha=0.4, color=(1.0, 1.0, 0.6), label='±10 mmHg')
                    # 最后绘制最小误差带（5mmHg）
                    ax_left.fill_between(bp_range, 
                                       [b - 5 for b in bp_range], [b + 5 for b in bp_range],
                                       alpha=0.5, color=(0.7, 1.0, 0.7), label='±5 mmHg')
                    
                    # 绘制理想线（y=x）
                    ax_left.plot(bp_range, bp_range, 'k--', alpha=0.5, linewidth=1, label='Perfect Match')
                    
                    # 最后绘制数据点（确保在最上层）
                    ax_left.scatter(bp_pred, bp_vals, alpha=0.6, s=20, color='blue')
                    
                    ax_left.set_xlabel('Predicted BP (mmHg)', fontsize=9)
                    ax_left.set_ylabel('Reference BP (mmHg)', fontsize=9)
                    ax_left.set_title(f'{self.ptt_combinations_en[sensor_pair]}\n{bp_label}', fontsize=10)
                    ax_left.grid(True, alpha=0.3)
                    ax_left.legend(fontsize=7, loc='upper left')
                    
                    # === 右侧：Bland-Altman图 ===
                    # Bland-Altman计算
                    mean_bp = (bp_vals + bp_pred) / 2
                    diff_bp = bp_pred - bp_vals
                    
                    # 计算统计量
                    mean_diff = np.mean(diff_bp)
                    std_diff = np.std(diff_bp)
                    limits_of_agreement_upper = mean_diff + 1.96 * std_diff
                    limits_of_agreement_lower = mean_diff - 1.96 * std_diff
                    
                    # 绘制Bland-Altman图
                    ax_right.scatter(mean_bp, diff_bp, alpha=0.6, s=20, color='blue')
                    
                    # 绘制均值线和一致性界限
                    ax_right.axhline(y=mean_diff, color='red', linestyle='-', linewidth=2, 
                                   label=f'Mean: {mean_diff:.2f}')
                    ax_right.axhline(y=limits_of_agreement_upper, color='red', linestyle='--', linewidth=1, 
                                   label=f'Upper LoA: {limits_of_agreement_upper:.2f}')
                    ax_right.axhline(y=limits_of_agreement_lower, color='red', linestyle='--', linewidth=1, 
                                   label=f'Lower LoA: {limits_of_agreement_lower:.2f}')
                    
                    ax_right.set_xlabel('Mean BP (mmHg)', fontsize=9)
                    ax_right.set_ylabel('Difference (Predicted - Reference) (mmHg)', fontsize=9)
                    ax_right.set_title(f'{self.ptt_combinations_en[sensor_pair]}\n{bp_label}', fontsize=10)
                    ax_right.grid(True, alpha=0.3)
                    ax_right.legend(fontsize=7, loc='upper right')
                    
                    # 计算误差带统计
                    abs_diff = np.abs(diff_bp)
                    within_5 = np.sum(abs_diff <= 5) / len(abs_diff) * 100
                    within_10 = np.sum(abs_diff <= 10) / len(abs_diff) * 100
                    within_15 = np.sum(abs_diff <= 15) / len(abs_diff) * 100
                    
                    error_band_stats.append({
                        'exp_id': exp_id,
                        'sensor_pair': sensor_pair,
                        'sensor_label': self.ptt_combinations_en[sensor_pair],
                        'bp_type': bp_label,
                        'n_samples': len(diff_bp),
                        'within_5_mmhg': within_5,
                        'within_10_mmhg': within_10,
                        'within_15_mmhg': within_15,
                        'mean_diff': mean_diff,
                        'std_diff': std_diff,
                        'loa_upper': limits_of_agreement_upper,
                        'loa_lower': limits_of_agreement_lower
                    })
                    
                    # 添加统计信息到图中
                    stats_text = f'n={len(diff_bp)}\n±5mmHg: {within_5:.1f}%\n±10mmHg: {within_10:.1f}%\n±15mmHg: {within_15:.1f}%'
                    ax_right.text(0.02, 0.98, stats_text, transform=ax_right.transAxes, 
                               verticalalignment='top', fontsize=7, 
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # 设置列标题
            for i, bp_label in enumerate(bp_labels):
                # 如果是Systolic BP，使用更大的字体和加粗
                if bp_label == 'Systolic BP':
                    fontsize = 14
                    fontweight = 'bold'
                else:
                    fontsize = 12
                    fontweight = 'bold'
                fig.text(0.167 + i * 0.333, 0.97, bp_label, ha='center', va='center', 
                        fontsize=fontsize, fontweight=fontweight)
            
            plt.tight_layout()
            
            # 保存图像到对应的实验文件夹
            exp_output_dir = None
            if exp_id is not None:
                # 创建实验特定的文件夹
                exp_output_dir = os.path.join(self.output_dir, f'exp_{exp_id}')
                os.makedirs(exp_output_dir, exist_ok=True)
                filename = f"{exp_output_dir}/bland_altman_bp_exp_{exp_id}.png"
            else:
                filename = f"{self.output_dir}/bland_altman_bp_overall.png"
            
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"💾 保存Bland-Altman图: {filename}")
            
            # 保存误差带统计到CSV
            if error_band_stats:
                stats_df = pd.DataFrame(error_band_stats)
                if exp_id is not None and exp_output_dir:
                    stats_filename = f"{exp_output_dir}/error_band_stats_exp_{exp_id}.csv"
                else:
                    stats_filename = f"{self.output_dir}/error_band_stats_overall.csv"
                stats_df.to_csv(stats_filename, index=False)
                print(f"💾 保存误差带统计: {stats_filename}")
            
            return fig
            
        except Exception as e:
            print(f"❌ Bland-Altman图创建失败: {e}")
            return None

def main():
    """主函数"""
    print("🩺 PTT-Cardiovascular Parameters Correlation Analysis")
    print("="*60)
    
    # # 创建分析器
    # analyzer = PTTBloodPressureAnalyzer()
    root_path = '/root/autodl-tmp/'
    # 获取所有受试者文件夹
    subject_list = sorted([d for d in os.listdir(root_path) 
                           if os.path.isdir(os.path.join(root_path, d)) and d.startswith('00')])
    print(f"📋 发现 {len(subject_list)} 个受试者")
    
    # print("\n📋 请选择分析方式:")
    # print("1. 综合分析 (单实验+跨实验)")
    # print("2. 单实验分析")
    # print("3. 跨实验分析")
    
    # try:
    #     choice = input("\n请输入选择 (1/2/3, 默认1): ").strip()
    #     if not choice:
    #         choice = "1"  # 默认综合分析
    # except:
    #     choice = "1"  # 默认选择
    
    # if choice == "1":
    #     print("\n🔬 运行综合分析...")
    #     # 运行综合分析
    for subject in subject_list:
        print(f"\n🔬 处理受试者: {subject}")
        # 为每个受试者设置输出目录（绝对路径）
        subject_output_dir = os.path.join(root_path, subject, 'ptt_bp_analysis')
        os.makedirs(subject_output_dir, exist_ok=True)
        
        # 创建分析器实例
        analyzer = PTTBloodPressureAnalyzer(output_dir=subject_output_dir)
        
        # 修改数据加载路径以包含subject（绝对路径）
        analyzer.ptt_output_dir = os.path.join(root_path, subject, 'ptt_output')
        analyzer.csv_output_dir = os.path.join(root_path, subject, 'csv_output')
        analyzer.subject = subject  # 添加subject属性用于文件名
        
        # 运行综合分析（模式1）
        results = analyzer.run_comprehensive_analysis()
        
        if results and results['overall']:
            overall_results = results['overall']
            
            # 显示最佳相关性
            # print(f"\n🏆 Top Significant Correlations (Overall Analysis):")
            print(f"\n🏆 Top Significant Correlations (Overall Analysis for {subject}):")
            all_corrs = []
            for sensor_pair, physio_data in overall_results['correlations'].items():
                for physio_col, stats_data in physio_data.items():
                    if stats_data['significant']:
                        all_corrs.append((abs(stats_data['correlation']), 
                                        analyzer._format_sensor_pair_label_en(sensor_pair),
                                        analyzer._format_physio_label_en(physio_col),
                                        stats_data['correlation'],
                                        stats_data['p_value'],
                                        stats_data['n_samples']))
            
            all_corrs.sort(reverse=True)
            for i, (abs_corr, sensor_label, physio_label, corr, p_val, n_samples) in enumerate(all_corrs[:10]):
                direction = "↑" if corr > 0 else "↓"
                print(f"   {i+1:2d}. {sensor_label} ←→ {physio_label}")
                print(f"       r={corr:+.3f} {direction}, p={p_val:.2e}, N={n_samples}")
    
    #     elif choice == "2":
    #     print("\n🎯 运行单独实验拟合分析...")
    #     # 运行单独实验拟合
    #     individual_models = analyzer.run_individual_regression_analysis()
        
    #     if individual_models:
    #         print(f"\n📊 单独实验拟合完成!")
    #         print(f"   • 成功分析实验数: {len(individual_models)}")
    #         print(f"   • 详细结果已保存: individual_experiment_models.csv")
    #         print(f"   • 性能对比图: individual_model_performance_comparison.png")
    
    # elif choice == "3":
    #      print("\n🎯 运行跨实验拟合分析...")
    #      # 运行跨实验拟合分析
    #      exp_sensor_models = analyzer.run_cross_experiments_analysis()
         
    #      if exp_sensor_models:
    #          print(f"\n✅ 跨实验拟合完成!")
    #          print(f"📁 结果保存在: {analyzer.output_dir}")
    # else:
    #     print("❌ 无效选择，默认运行综合分析")
    #     choice = "1"
    #     # 递归调用原始分析
    #     analyzer.run_comprehensive_analysis()
    
    # print(f"\n✅ 分析完成!")
    # print(f"📁 所有结果保存在: {analyzer.output_dir}")
    print(f"\n✅ 所有受试者分析完成!")
    print(f"📁 结果保存在每个受试者的 ptt_bp_analysis 目录中")

if __name__ == "__main__":
    main() 