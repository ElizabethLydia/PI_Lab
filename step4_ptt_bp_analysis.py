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
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class PTTBloodPressureAnalyzer:
    """PTT与血压相关性分析器"""
    
    def __init__(self, output_dir="ptt_bp_analysis"):
        self.output_dir = output_dir
        self.ptt_output_dir = "ptt_output2"  # 窗口化PTT数据目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 完整的生理指标映射（英文专业术语）
        self.physiological_indicators = {
            'systolic_bp': 'Systolic BP (mmHg)',
            'diastolic_bp': 'Diastolic BP (mmHg)', 
            'mean_bp': 'Mean Arterial Pressure (mmHg)',
            'bp': 'Continuous BP (mmHg)',
            'cardiac_output': 'Cardiac Output (L/min)',
            'cardiac_index': 'Cardiac Index (L/min/m²)',
            'hr': 'Heart Rate (bpm)',
            'systemic_vascular_resistance': 'Systemic Vascular Resistance (dyn·s/cm⁵)',
            'rsp': 'Respiration Rate (breaths/min)'
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
            # 加载CSV文件
            csv_file = f"output/csv_output/{exp_id}_biopac_aligned.csv"
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
            
            # 只保留有效窗口的PTT数据
            valid_ptt = ptt_df[ptt_df['window_id'].isin(valid_windows['window_id'])]
            
            print(f"📊 实验{exp_id}: 总窗口{len(window_df)}, 有效窗口{len(valid_windows)}, 有效PTT数据{len(valid_ptt)}")
            
            return {
                'window_info': valid_windows,
                'ptt_data': valid_ptt
            }
            
        except Exception as e:
            print(f"❌ 加载PTT数据失败: {e}")
            return None
    
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
            
            # 计算窗口内所有生理指标的统计量
            physio_values = {}
            for indicator in self.physiological_indicators.keys():
                if indicator in physio_data.columns:
                    physio_values[f'{indicator}_mean'] = window_physio[indicator].mean()
                    physio_values[f'{indicator}_std'] = window_physio[indicator].std()
                    physio_values[f'{indicator}_min'] = window_physio[indicator].min()
                    physio_values[f'{indicator}_max'] = window_physio[indicator].max()
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
        
        return sync_df
    
    def calculate_correlations(self, sync_df):
        """计算PTT与所有生理指标的相关性"""
        correlations = {}
        
        # 生理指标（扩展到所有可用指标）
        physio_metrics = []
        for indicator in self.physiological_indicators.keys():
            for stat in ['_mean', '_std', '_min', '_max']:
                col_name = f'{indicator}{stat}'
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
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # 绘制热图
        mask = np.isnan(corr_matrix)
        im = sns.heatmap(corr_matrix, 
                        xticklabels=[self._format_physio_label_en(col) for col in physio_cols],
                        yticklabels=[self._format_sensor_pair_label_en(pair) for pair in sensor_pairs],
                        annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                        mask=mask, square=False, linewidths=0.5,
                        cbar_kws={'label': 'Correlation Coefficient'},
                        annot_kws={'size': 8})
        
        # 添加显著性标记
        for i in range(len(sensor_pairs)):
            for j in range(len(physio_cols)):
                if not np.isnan(p_matrix[i, j]) and p_matrix[i, j] < 0.05:
                    ax.text(j + 0.5, i + 0.5, '*', ha='center', va='center', 
                           color='white', fontsize=8, fontweight='bold')
        
        plt.title(f'PTT-Cardiovascular Parameters Correlation Analysis{title_suffix}', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Physiological Parameters', fontsize=12, fontweight='bold')
        plt.ylabel('PTT Sensor Combinations', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # 保存图像
        filename = f"{self.output_dir}/ptt_cardiovascular_correlation_heatmap{title_suffix.replace(' ', '_')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"💾 保存相关性热图: {filename}")
        
        return fig
    
    def _format_ptt_label(self, ptt_col):
        """格式化PTT标签"""
        # ptt_sensor2-sensor3_ms -> nose→finger
        sensor_map = {'sensor2': 'nose', 'sensor3': 'finger', 'sensor4': 'wrist', 'sensor5': 'ear'}
        parts = ptt_col.replace('ptt_', '').replace('_ms', '').split('-')
        if len(parts) == 2:
            return f"{sensor_map.get(parts[0], parts[0])}→{sensor_map.get(parts[1], parts[1])}"
        return ptt_col
    
    def _format_bp_label(self, bp_col):
        """格式化血压标签"""
        label_map = {
            'systolic_bp': '收缩压',
            'diastolic_bp': '舒张压',
            'mean_bp': '平均动脉压',
            'bp': '连续血压'
        }
        
        for bp_type, label in label_map.items():
            if bp_col.startswith(bp_type):
                suffix = bp_col.replace(bp_type, '').replace('_', ' ')
                return f"{label}{suffix}"
        return bp_col
    
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
    
    def build_regression_models(self, sync_df):
        """构建PTT→生理指标的回归模型"""
        models = {}
        
        # 主要生理指标（均值）
        main_physio_cols = []
        for indicator in ['systolic_bp', 'diastolic_bp', 'mean_bp', 'cardiac_output', 'cardiac_index']:
            col_name = f'{indicator}_mean'
            if col_name in sync_df.columns:
                main_physio_cols.append(col_name)
        
        # 创建传感器对的透视表
        ptt_pivot = sync_df.pivot_table(
            index=['exp_id', 'window_id'], 
            columns='sensor_pair', 
            values='ptt_ms',
            aggfunc='mean'
        ).reset_index()
        
        # 合并生理数据（取平均值）
        physio_agg = sync_df.groupby(['exp_id', 'window_id']).agg({
            col: 'mean' for col in main_physio_cols if col in sync_df.columns
        }).reset_index()
        
        # 合并PTT和生理数据
        model_data = pd.merge(ptt_pivot, physio_agg, on=['exp_id', 'window_id'], how='inner')
        
        if len(model_data) < 20:
            print(f"⚠️ 模型数据不足: 只有{len(model_data)}个样本")
            return models
        
        # 获取PTT特征列
        ptt_cols = [col for col in model_data.columns if col not in ['exp_id', 'window_id'] + main_physio_cols]
        # 去除全空的PTT列
        ptt_cols = [col for col in ptt_cols if not model_data[col].isna().all()]
        
        if len(ptt_cols) == 0:
            print("❌ 没有有效的PTT特征")
            return models
        
        for physio_col in main_physio_cols:
            if physio_col not in model_data.columns:
                continue
                
            # 准备数据
            mask = ~model_data[physio_col].isna()
            for ptt_col in ptt_cols:
                mask &= ~model_data[ptt_col].isna()
            
            if mask.sum() < 10:  # 至少10个样本
                continue
            
            X = model_data.loc[mask, ptt_cols].values
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
            
            models[physio_col] = {
                'model': model,
                'scaler_X': scaler_X,
                'scaler_y': scaler_y,
                'feature_names': ptt_cols,
                'r2_score': r2,
                'mae': mae,
                'n_samples': len(y),
                'y_true': y,
                'y_pred': y_pred
            }
            
            print(f"📈 {self._format_physio_label_en(physio_col)}模型: R²={r2:.3f}, MAE={mae:.2f}, N={len(y)}")
        
        return models
    
    def create_regression_plots(self, models):
        """创建回归分析图（英文专业版）"""
        n_models = len(models)
        if n_models == 0:
            return
        
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
        if n_models == 1:
            axes = [axes]
        
        for idx, (physio_col, model_data) in enumerate(models.items()):
            ax = axes[idx]
            
            y_true = model_data['y_true']
            y_pred = model_data['y_pred']
            r2 = model_data['r2_score']
            mae = model_data['mae']
            
            # 散点图
            ax.scatter(y_true, y_pred, alpha=0.6, s=50, color='steelblue')
            
            # 理想线
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
            
            # 格式化
            physio_label = self._format_physio_label_en(physio_col)
            ax.set_xlabel(f'Actual {physio_label}', fontweight='bold')
            ax.set_ylabel(f'Predicted {physio_label}', fontweight='bold')
            ax.set_title(f'{physio_label}\nR²={r2:.3f}, MAE={mae:.2f}', fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.suptitle('PTT-Based Physiological Parameter Prediction Models', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # 保存图像
        filename = f"{self.output_dir}/ptt_cardiovascular_regression_analysis.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"💾 保存回归分析图: {filename}")
        
        return fig
    
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
        models = self.build_regression_models(sync_df)
        
        return {
            'sync_data': sync_df,
            'correlations': correlations,
            'models': models
        }
    
    def run_individual_experiment_analysis(self):
        """运行单个实验的分析"""
        print("🔬 开始单个实验分析...")
        
        individual_results = {}
        
        # 获取实验列表
        exp_list = [1, 2, 7, 9, 10]  # 只分析有PTT数据的实验
        
        for exp_id in exp_list:
            print(f"\n🔍 单独分析实验 {exp_id}")
            
            # 分析单个实验
            exp_result = self.analyze_experiment(exp_id)
            if exp_result:
                individual_results[exp_id] = exp_result['sync_data']
                
                # 计算相关性
                correlations = self.calculate_correlations(exp_result['sync_data'])
                
                # 创建单个实验的热图
                self.create_focused_correlation_heatmap(correlations, f"_实验{exp_id}")
                
                # 保存单个实验结果
                self.save_individual_experiment_results(exp_result['sync_data'], correlations, exp_id)
        
        return individual_results
    
    def create_focused_correlation_heatmap(self, correlations, title_suffix=""):
        """创建聚焦的相关性热图（只显示重要指标）"""
        # 选择重要的生理指标（减少图像大小）
        important_indicators = [
            'systolic_bp_mean', 'diastolic_bp_mean', 'mean_bp_mean',
            'cardiac_output_mean', 'cardiac_index_mean', 'hr_mean',
            'rsp_mean', 'systemic_vascular_resistance_mean'
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
        
        # 创建图形（更小更清晰）
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 绘制热图
        mask = np.isnan(corr_matrix)
        im = sns.heatmap(corr_matrix, 
                        xticklabels=[self._format_physio_label_en(col) for col in physio_cols],
                        yticklabels=[self._format_sensor_pair_label_en(pair) for pair in sensor_pairs],
                        annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                        mask=mask, square=False, linewidths=0.5,
                        cbar_kws={'label': 'Correlation Coefficient'},
                        annot_kws={'size': 10})
        
        # 添加显著性标记
        for i in range(len(sensor_pairs)):
            for j in range(len(physio_cols)):
                if not np.isnan(p_matrix[i, j]) and p_matrix[i, j] < 0.05:
                    ax.text(j + 0.5, i + 0.5, '*', ha='center', va='center', 
                           color='white', fontsize=14, fontweight='bold')
        
        plt.title(f'PTT-Cardiovascular Correlation Analysis (Key Parameters){title_suffix}', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Physiological Parameters', fontsize=12, fontweight='bold')
        plt.ylabel('PTT Sensor Combinations', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # 保存图像
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
        corr_file = f"{self.output_dir}/ptt_cardiovascular_correlations_exp_{exp_id}.csv"
        corr_df.to_csv(corr_file, index=False)
        print(f"💾 保存实验{exp_id}相关性: {corr_file}")
    
    def compare_experiments(self, individual_results):
        """比较不同实验的结果"""
        print("\n📊 实验间对比分析...")
        
        # 收集所有实验的相关性数据
        all_exp_correlations = []
        
        for exp_id, exp_data in individual_results.items():
            correlations = self.calculate_correlations(exp_data)
            
            for sensor_pair, physio_data in correlations.items():
                for physio_col, stats_data in physio_data.items():
                    if stats_data['significant']:
                        all_exp_correlations.append({
                            'experiment': exp_id,
                            'sensor_pair': sensor_pair,
                            'parameter': physio_col,
                            'correlation': stats_data['correlation'],
                            'n_samples': stats_data['n_samples']
                        })
        
        # 转换为DataFrame
        comp_df = pd.DataFrame(all_exp_correlations)
        
        if len(comp_df) == 0:
            print("⚠️ 没有足够的数据进行实验间比较")
            return
        
        # 保存比较结果
        comp_file = f"{self.output_dir}/experiment_comparison.csv"
        comp_df.to_csv(comp_file, index=False)
        print(f"💾 保存实验比较: {comp_file}")
        
        # 打印总结
        print(f"\n📋 实验间比较总结:")
        for exp_id in sorted(individual_results.keys()):
            exp_corr = comp_df[comp_df['experiment'] == exp_id]
            print(f"   实验{exp_id}: {len(exp_corr)}个显著相关性")
            
            if len(exp_corr) > 0:
                strongest = exp_corr.loc[exp_corr['correlation'].abs().idxmax()]
                print(f"     最强: {strongest['sensor_pair']}-{strongest['parameter']} (r={strongest['correlation']:.3f})")
    
    def run_comprehensive_analysis(self):
        """运行综合分析（整体+单个实验）"""
        print("🔬 开始PTT与生理参数综合分析")
        print("📋 分析实验列表: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]")
        
        # 1. 整体分析（原有功能）
        print("\n=== 第一部分：整体分析 ===")
        overall_results = self.run_overall_analysis()
        
        # 2. 单个实验分析（新增功能）
        print("\n=== 第二部分：单个实验分析 ===")
        individual_results = self.run_individual_experiment_analysis()
        
        # 3. 实验间比较（新增功能）
        print("\n=== 第三部分：实验间比较 ===")
        if individual_results:
            self.compare_experiments(individual_results)
        
        # 4. 创建聚焦热图（优化显示）
        print("\n=== 第四部分：创建聚焦热图 ===")
        if overall_results:
            self.create_focused_correlation_heatmap(overall_results['correlations'], "_整体分析_聚焦")
        
        return {
            'overall': overall_results,
            'individual': individual_results
        }
    
    def run_overall_analysis(self):
        """运行整体分析（原有功能重命名）"""
        # 原有的 run_comprehensive_analysis 内容
        all_experiments = []
        
        # 分析所有实验
        for exp_id in range(1, 12):  # 实验1-11
            print(f"\n🔍 分析实验 {exp_id}")
            exp_data = self.analyze_experiment(exp_id)
            
            if exp_data:
                # 提取sync_data（DataFrame）
                all_experiments.append(exp_data['sync_data'])
        
        if not all_experiments:
            print("❌ 没有有效的实验数据")
            return None
        
        # 合并所有实验的数据（修正：提取DataFrame）
        combined_df = pd.concat(all_experiments, ignore_index=True)
        print(f"\n📊 合并数据: {len(combined_df)}个样本，来自{len(all_experiments)}个实验")
        
        # 计算整体相关性
        print("\n📈 计算整体相关性...")
        correlations = self.calculate_correlations(combined_df)
        
        # 创建相关性热图
        self.create_correlation_heatmap(correlations, " (整体分析)")
        
        # 构建回归模型
        print("\n🎯 构建整体回归模型...")
        models = self.build_regression_models(combined_df)
        
        # 创建回归图
        if models:
            self.create_regression_plots(models)
        
        # 保存结果
        self.save_analysis_results(combined_df, correlations, models)
        
        return {
            'combined_data': combined_df,
            'correlations': correlations,
            'models': models
        }
    
    def run_individual_regression_analysis(self):
        """为每个实验单独构建回归模型"""
        print("\n🎯 开始单独实验回归分析...")
        individual_models = {}
        model_summary = []
        
        for exp_id in range(1, 12):  # 实验1-11
            print(f"\n📊 构建实验{exp_id}的回归模型")
            exp_data = self.analyze_experiment(exp_id)
            
            if not exp_data or len(exp_data['sync_data']) < 20:
                print(f"❌ 实验{exp_id}数据不足（<20样本）")
                continue
            
            # 为单个实验构建模型
            exp_models = self.build_regression_models(exp_data['sync_data'])
            
            if exp_models:
                individual_models[f'exp_{exp_id}'] = exp_models
                
                # 收集模型性能统计
                for physio_param, model_info in exp_models.items():
                    model_summary.append({
                        'experiment': exp_id,
                        'physiological_parameter': physio_param,
                        'parameter_label': self._format_physio_label_en(physio_param),
                        'r2_score': model_info['r2_score'],
                        'mae': model_info['mae'],
                        'n_samples': model_info['n_samples']
                    })
        
        # 保存单独实验的模型评估
        if model_summary:
            model_df = pd.DataFrame(model_summary)
            model_file = f"{self.output_dir}/individual_experiment_models.csv"
            model_df.to_csv(model_file, index=False)
            print(f"💾 保存单独实验模型评估: {model_file}")
            
            # 创建模型性能对比可视化
            self.create_individual_model_comparison(model_df)
        
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
        ax1.set_title('各实验MAE对比 (越低越好)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('实验编号')
        ax1.set_ylabel('生理参数')
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
        ax2.set_title('各实验R²对比 (越高越好)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('实验编号')
        ax2.set_ylabel('生理参数')
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
        plt.show()
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
        print(f"💾 保存相关性分析: {corr_file}")
        
        # 3. 保存模型评估结果
        model_results = []
        for physio_col, model_data in models.items():
            model_results.append({
                'physiological_parameter': physio_col,
                'parameter_label': self._format_physio_label_en(physio_col),
                'r2_score': model_data['r2_score'],
                'mae': model_data['mae'],
                'n_samples': model_data['n_samples']
            })
        
        model_df = pd.DataFrame(model_results)
        model_file = f"{self.output_dir}/ptt_cardiovascular_model_evaluation.csv"
        model_df.to_csv(model_file, index=False)
        print(f"💾 保存模型评估: {model_file}")


def main():
    """主函数"""
    print("🩺 PTT-Cardiovascular Parameters Correlation Analysis")
    print("="*60)
    
    # 创建分析器
    analyzer = PTTBloodPressureAnalyzer()
    
    print("\n📋 请选择分析方式:")
    print("1. 综合分析 (原始：池化所有实验数据拟合)")
    print("2. 单独实验拟合分析 (每个实验单独建模)")
    print("3. 完整对比分析 (包含1+2)")
    
    try:
        choice = input("\n请输入选择 (1/2/3, 默认1): ").strip()
        if not choice:
            choice = "1"
    except:
        choice = "1"  # 默认选择
    
    if choice == "1":
        print("\n🔬 运行综合分析（池化拟合）...")
        # 运行综合分析
        results = analyzer.run_comprehensive_analysis()
        
        if results and results['overall']:
            overall_results = results['overall']
            individual_results = results['individual']
            
            print("\n📋 Analysis Summary:")
            print(f"   • Total samples: {len(overall_results['combined_data'])}")
            physio_indicators = len([col for col in overall_results['combined_data'].columns 
                                   if any(indicator in col for indicator in analyzer.physiological_indicators.keys())])
            print(f"   • PTT combinations: {len(analyzer.ptt_combinations_en)}")
            print(f"   • Physiological parameters: {physio_indicators}")
            print(f"   • Regression models: {len(overall_results['models'])}")
            print(f"   • Individual experiments analyzed: {len(individual_results) if individual_results else 0}")
            
            # 显示最佳相关性
            print(f"\n🏆 Top Significant Correlations (Overall Analysis):")
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
            
            # 显示模型性能
            if overall_results['models']:
                print(f"\n📈 Best Prediction Models (池化拟合):")
                model_performance = [(model_data['r2_score'], physio_col, model_data) 
                                   for physio_col, model_data in overall_results['models'].items()]
                model_performance.sort(reverse=True)
                
                for i, (r2, physio_col, model_data) in enumerate(model_performance[:5]):
                    physio_label = analyzer._format_physio_label_en(physio_col)
                    print(f"   {i+1}. {physio_label}: R²={r2:.3f}, MAE={model_data['mae']:.2f}")
    
    elif choice == "2":
        print("\n🎯 运行单独实验拟合分析...")
        # 运行单独实验拟合
        individual_models = analyzer.run_individual_regression_analysis()
        
        if individual_models:
            print(f"\n📊 单独实验拟合完成!")
            print(f"   • 成功分析实验数: {len(individual_models)}")
            print(f"   • 详细结果已保存: individual_experiment_models.csv")
            print(f"   • 性能对比图: individual_model_performance_comparison.png")
    
    elif choice == "3":
        print("\n🔬 运行完整对比分析...")
        # 先运行综合分析
        results = analyzer.run_comprehensive_analysis()
        
        # 再运行单独实验拟合
        print("\n" + "="*60)
        individual_models = analyzer.run_individual_regression_analysis()
        
        # 对比总结
        if results and results['overall'] and individual_models:
            print("\n📈 拟合方式对比总结:")
            overall_models = results['overall']['models']
            
            print("\n🔍 池化拟合 vs 单独拟合对比:")
            for physio_param in ['systolic_bp_mean', 'diastolic_bp_mean', 'mean_bp_mean']:
                if physio_param in overall_models:
                    overall_mae = overall_models[physio_param]['mae']
                    overall_r2 = overall_models[physio_param]['r2_score']
                    overall_n = overall_models[physio_param]['n_samples']
                    
                    physio_label = analyzer._format_physio_label_en(physio_param)
                    print(f"\n📊 {physio_label}:")
                    print(f"   池化拟合: R²={overall_r2:.3f}, MAE={overall_mae:.2f}, N={overall_n}")
                    
                    # 统计单独实验的平均性能
                    individual_r2s = []
                    individual_maes = []
                    individual_ns = []
                    
                    for exp_key, exp_models in individual_models.items():
                        if physio_param in exp_models:
                            individual_r2s.append(exp_models[physio_param]['r2_score'])
                            individual_maes.append(exp_models[physio_param]['mae'])
                            individual_ns.append(exp_models[physio_param]['n_samples'])
                    
                    if individual_r2s:
                        avg_r2 = np.mean(individual_r2s)
                        avg_mae = np.mean(individual_maes)
                        total_n = sum(individual_ns)
                        print(f"   单独拟合: R²={avg_r2:.3f}±{np.std(individual_r2s):.3f}, MAE={avg_mae:.2f}±{np.std(individual_maes):.2f}, 总N={total_n}")
                        
                        # 性能比较
                        if overall_r2 > avg_r2:
                            print(f"   ✅ 池化拟合R²更优 (+{overall_r2-avg_r2:.3f})")
                        else:
                            print(f"   ⚠️  单独拟合R²更优 (+{avg_r2-overall_r2:.3f})")
                        
                        if overall_mae < avg_mae:
                            print(f"   ✅ 池化拟合MAE更优 (-{avg_mae-overall_mae:.2f})")
                        else:
                            print(f"   ⚠️  单独拟合MAE更优 (-{overall_mae-avg_mae:.2f})")
    else:
        print("❌ 无效选择，默认运行综合分析")
        choice = "1"
        # 递归调用原始分析
        analyzer.run_comprehensive_analysis()
    
    print(f"\n✅ 分析完成!")
    print(f"📁 所有结果保存在: {analyzer.output_dir}")

if __name__ == "__main__":
    main() 