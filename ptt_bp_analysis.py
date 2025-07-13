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
        
        # 血压指标映射
        self.bp_indicators = {
            'systolic_bp': '收缩压',
            'diastolic_bp': '舒张压', 
            'mean_bp': '平均动脉压',
            'bp': '连续血压'
        }
        
        # PTT传感器组合
        self.ptt_combinations = [
            'sensor2-sensor3', 'sensor2-sensor4', 'sensor2-sensor5',
            'sensor3-sensor4', 'sensor3-sensor5', 'sensor4-sensor5'
        ]
        
        print("🔬 PTT与血压相关性分析器初始化完成")
        print(f"📁 分析结果将保存到: {self.output_dir}")
    
    def load_ground_truth_bp(self, exp_id):
        """加载血压真标数据（从CSV文件）"""
        try:
            # 加载CSV文件
            csv_file = f"output/csv_output/{exp_id}_biopac_aligned.csv"
            if not os.path.exists(csv_file):
                print(f"❌ 血压数据文件不存在: {csv_file}")
                return None
            
            # 读取完整血压数据
            df = pd.read_csv(csv_file)
            print(f"✅ 加载血压数据: {len(df)}条记录")
            print(f"📊 血压指标: {[col for col in df.columns if col != 'timestamp']}")
            
            return df
            
        except Exception as e:
            print(f"❌ 加载血压数据失败: {e}")
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
    
    def synchronize_data(self, ptt_data, bp_data, exp_id):
        """时间同步PTT和血压数据"""
        synchronized_data = []
        
        for _, ptt_row in ptt_data['ptt_data'].iterrows():
            # PTT数据的时间信息（修正列名）
            start_time = ptt_row['window_start_s']
            end_time = ptt_row['window_end_s']
            window_center = (start_time + end_time) / 2
            
            # 转换为时间戳（假设血压数据的timestamp是绝对时间戳）
            # 需要找到血压数据时间戳的起始点
            bp_start_time = bp_data['timestamp'].iloc[0]
            start_timestamp = bp_start_time + start_time
            end_timestamp = bp_start_time + end_time
            
            # 找到时间窗口内的血压数据
            time_mask = (bp_data['timestamp'] >= start_timestamp) & (bp_data['timestamp'] <= end_timestamp)
            window_bp = bp_data[time_mask]
            
            if len(window_bp) == 0:
                continue  # 跳过没有血压数据的窗口
            
            # 计算窗口内血压的统计量
            bp_values = {}
            for bp_col in ['systolic_bp', 'diastolic_bp', 'mean_bp', 'bp']:
                if bp_col in bp_data.columns:
                    bp_values[f'{bp_col}_mean'] = window_bp[bp_col].mean()
                    bp_values[f'{bp_col}_std'] = window_bp[bp_col].std()
                    bp_values[f'{bp_col}_min'] = window_bp[bp_col].min()
                    bp_values[f'{bp_col}_max'] = window_bp[bp_col].max()
                    bp_values[f'{bp_col}_count'] = len(window_bp)
            
            # 构建同步数据行
            sync_row = {
                'exp_id': exp_id,
                'window_id': ptt_row['window_id'],
                'start_time': start_time,
                'end_time': end_time,
                'window_center': window_center,
                'sensor_pair': ptt_row['sensor_pair'],
                'ptt_ms': ptt_row['ptt_ms'],
                **bp_values
            }
            
            synchronized_data.append(sync_row)
        
        sync_df = pd.DataFrame(synchronized_data)
        print(f"📊 同步完成: {len(sync_df)}个有效窗口")
        
        return sync_df
    
    def calculate_correlations(self, sync_df):
        """计算PTT与血压的相关性"""
        correlations = {}
        
        # 血压指标
        bp_metrics = [col for col in sync_df.columns if any(bp in col for bp in ['systolic_bp', 'diastolic_bp', 'mean_bp', 'bp']) and col != 'bp_start_time']
        
        # 获取所有传感器对
        sensor_pairs = sync_df['sensor_pair'].unique()
        
        print(f"\n📊 计算相关性：{len(sensor_pairs)}个传感器对 × {len(bp_metrics)}个血压指标")
        
        for sensor_pair in sensor_pairs:
            correlations[sensor_pair] = {}
            
            # 提取该传感器对的数据
            pair_data = sync_df[sync_df['sensor_pair'] == sensor_pair]
            
            if len(pair_data) < 10:  # 至少10个数据点
                continue
            
            for bp_col in bp_metrics:
                # 提取有效数据
                mask = ~(pair_data['ptt_ms'].isna() | pair_data[bp_col].isna())
                if mask.sum() < 10:
                    continue
                
                ptt_vals = pair_data.loc[mask, 'ptt_ms']
                bp_vals = pair_data.loc[mask, bp_col]
                
                # 计算皮尔逊相关系数
                try:
                    corr_coef, p_value = stats.pearsonr(ptt_vals, bp_vals)
                    
                    correlations[sensor_pair][bp_col] = {
                        'correlation': corr_coef,
                        'p_value': p_value,
                        'n_samples': len(ptt_vals),
                        'significant': p_value < 0.05
                    }
                except Exception as e:
                    print(f"⚠️ 计算相关性失败 {sensor_pair}-{bp_col}: {e}")
                    continue
        
        return correlations
    
    def create_correlation_heatmap(self, correlations, title_suffix=""):
        """创建相关性热图"""
        # 准备数据
        sensor_pairs = list(correlations.keys())
        bp_cols = set()
        for pair_data in correlations.values():
            bp_cols.update(pair_data.keys())
        bp_cols = sorted(list(bp_cols))
        
        if len(sensor_pairs) == 0 or len(bp_cols) == 0:
            print("⚠️ 没有足够的相关性数据来创建热图")
            return None
        
        # 创建相关性矩阵
        corr_matrix = np.full((len(sensor_pairs), len(bp_cols)), np.nan)
        p_matrix = np.full((len(sensor_pairs), len(bp_cols)), np.nan)
        
        for i, sensor_pair in enumerate(sensor_pairs):
            for j, bp_col in enumerate(bp_cols):
                if bp_col in correlations[sensor_pair]:
                    corr_matrix[i, j] = correlations[sensor_pair][bp_col]['correlation']
                    p_matrix[i, j] = correlations[sensor_pair][bp_col]['p_value']
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 绘制热图
        mask = np.isnan(corr_matrix)
        sns.heatmap(corr_matrix, 
                    xticklabels=[self._format_bp_label(col) for col in bp_cols],
                    yticklabels=[self._format_sensor_pair_label(pair) for pair in sensor_pairs],
                    annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                    mask=mask, square=False, linewidths=0.5,
                    cbar_kws={'label': '相关系数'})
        
        # 添加显著性标记
        for i in range(len(sensor_pairs)):
            for j in range(len(bp_cols)):
                if not np.isnan(p_matrix[i, j]) and p_matrix[i, j] < 0.05:
                    ax.text(j + 0.5, i + 0.5, '*', ha='center', va='center', 
                           color='white', fontsize=16, fontweight='bold')
        
        plt.title(f'PTT与血压相关性分析{title_suffix}', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('血压指标', fontsize=12)
        plt.ylabel('PTT传感器组合', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # 保存图像
        filename = f"{self.output_dir}/ptt_bp_correlation_heatmap{title_suffix.replace(' ', '_')}.png"
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
    
    def build_regression_models(self, sync_df):
        """构建PTT→血压的回归模型"""
        models = {}
        
        # 主要血压指标
        main_bp_cols = ['systolic_bp_mean', 'diastolic_bp_mean', 'mean_bp_mean']
        
        # 创建传感器对的透视表
        ptt_pivot = sync_df.pivot_table(
            index=['exp_id', 'window_id'], 
            columns='sensor_pair', 
            values='ptt_ms',
            aggfunc='mean'
        ).reset_index()
        
        # 合并血压数据（取平均值）
        bp_agg = sync_df.groupby(['exp_id', 'window_id']).agg({
            col: 'mean' for col in main_bp_cols if col in sync_df.columns
        }).reset_index()
        
        # 合并PTT和血压数据
        model_data = pd.merge(ptt_pivot, bp_agg, on=['exp_id', 'window_id'], how='inner')
        
        if len(model_data) < 20:
            print(f"⚠️ 模型数据不足: 只有{len(model_data)}个样本")
            return models
        
        # 获取PTT特征列
        ptt_cols = [col for col in model_data.columns if col not in ['exp_id', 'window_id'] + main_bp_cols]
        # 去除全空的PTT列
        ptt_cols = [col for col in ptt_cols if not model_data[col].isna().all()]
        
        if len(ptt_cols) == 0:
            print("❌ 没有有效的PTT特征")
            return models
        
        for bp_col in main_bp_cols:
            if bp_col not in model_data.columns:
                continue
                
            # 准备数据
            mask = ~model_data[bp_col].isna()
            for ptt_col in ptt_cols:
                mask &= ~model_data[ptt_col].isna()
            
            if mask.sum() < 20:  # 至少20个样本
                continue
            
            X = model_data.loc[mask, ptt_cols].values
            y = model_data.loc[mask, bp_col].values
            
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
            
            models[bp_col] = {
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
            
            print(f"📈 {self._format_bp_label(bp_col)}模型: R²={r2:.3f}, MAE={mae:.2f}, N={len(y)}")
        
        return models
    
    def create_regression_plots(self, models):
        """创建回归分析图"""
        n_models = len(models)
        if n_models == 0:
            return
        
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
        if n_models == 1:
            axes = [axes]
        
        for idx, (bp_col, model_data) in enumerate(models.items()):
            ax = axes[idx]
            
            y_true = model_data['y_true']
            y_pred = model_data['y_pred']
            r2 = model_data['r2_score']
            mae = model_data['mae']
            
            # 散点图
            ax.scatter(y_true, y_pred, alpha=0.6, s=50)
            
            # 理想线
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='理想预测')
            
            # 格式化
            ax.set_xlabel(f'真实{self._format_bp_label(bp_col)} (mmHg)')
            ax.set_ylabel(f'预测{self._format_bp_label(bp_col)} (mmHg)')
            ax.set_title(f'{self._format_bp_label(bp_col)}\nR²={r2:.3f}, MAE={mae:.2f}mmHg')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        
        # 保存图像
        filename = f"{self.output_dir}/ptt_bp_regression_analysis.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"💾 保存回归分析图: {filename}")
        
        return fig
    
    def analyze_experiment(self, exp_id):
        """分析单个实验"""
        print(f"\n🔍 分析实验 {exp_id}")
        
        # 1. 加载数据
        bp_data = self.load_ground_truth_bp(exp_id)
        ptt_data = self.load_ptt_data(exp_id)
        
        if bp_data is None or ptt_data is None:
            print(f"❌ 实验 {exp_id} 数据加载失败")
            return None
        
        # 2. 时间同步
        sync_df = self.synchronize_data(ptt_data, bp_data, exp_id)
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
    
    def run_comprehensive_analysis(self, experiment_list=None):
        """运行综合分析"""
        print("🔬 开始PTT与血压相关性综合分析")
        
        if experiment_list is None:
            # 自动检测可用实验
            experiment_list = []
            for i in range(1, 12):
                if os.path.exists(f"{self.ptt_output_dir}/exp_{i}"):
                    experiment_list.append(i)
        
        print(f"📋 分析实验列表: {experiment_list}")
        
        all_results = {}
        all_sync_data = []
        
        # 分析每个实验
        for exp_id in experiment_list:
            result = self.analyze_experiment(exp_id)
            if result is not None:
                all_results[exp_id] = result
                all_sync_data.append(result['sync_data'])
        
        if not all_sync_data:
            print("❌ 没有有效的分析结果")
            return
        
        # 合并所有数据
        combined_df = pd.concat(all_sync_data, ignore_index=True)
        print(f"\n📊 合并数据: {len(combined_df)}个样本，来自{len(all_results)}个实验")
        
        # 整体相关性分析
        print("\n📈 计算整体相关性...")
        overall_correlations = self.calculate_correlations(combined_df)
        
        # 创建整体相关性热图
        self.create_correlation_heatmap(overall_correlations, " (整体分析)")
        
        # 整体回归分析
        print("\n🎯 构建整体回归模型...")
        overall_models = self.build_regression_models(combined_df)
        self.create_regression_plots(overall_models)
        
        # 保存详细结果
        self.save_analysis_results(combined_df, overall_correlations, overall_models)
        
        print(f"\n✅ PTT与血压相关性分析完成！")
        print(f"📁 结果保存在: {self.output_dir}")
        
        return {
            'combined_data': combined_df,
            'correlations': overall_correlations,
            'models': overall_models,
            'individual_results': all_results
        }
    
    def save_analysis_results(self, combined_df, correlations, models):
        """保存分析结果"""
        # 1. 保存同步数据
        sync_file = f"{self.output_dir}/synchronized_ptt_bp_data.csv"
        combined_df.to_csv(sync_file, index=False)
        print(f"💾 保存同步数据: {sync_file}")
        
        # 2. 保存相关性结果
        corr_results = []
        for sensor_pair, bp_data in correlations.items():
            for bp_col, stats_data in bp_data.items():
                corr_results.append({
                    'sensor_pair': sensor_pair,
                    'bp_metric': bp_col,
                    'correlation': stats_data['correlation'],
                    'p_value': stats_data['p_value'],
                    'n_samples': stats_data['n_samples'],
                    'significant': stats_data['significant']
                })
        
        corr_df = pd.DataFrame(corr_results)
        corr_file = f"{self.output_dir}/ptt_bp_correlations.csv"
        corr_df.to_csv(corr_file, index=False)
        print(f"💾 保存相关性分析: {corr_file}")
        
        # 3. 保存模型评估结果
        model_results = []
        for bp_col, model_data in models.items():
            model_results.append({
                'bp_metric': bp_col,
                'r2_score': model_data['r2_score'],
                'mae': model_data['mae'],
                'n_samples': model_data['n_samples']
            })
        
        model_df = pd.DataFrame(model_results)
        model_file = f"{self.output_dir}/ptt_bp_model_evaluation.csv"
        model_df.to_csv(model_file, index=False)
        print(f"💾 保存模型评估: {model_file}")


def main():
    """主函数"""
    print("🩺 PTT与血压相关性分析")
    print("="*60)
    
    # 创建分析器
    analyzer = PTTBloodPressureAnalyzer()
    
    # 运行综合分析
    results = analyzer.run_comprehensive_analysis()
    
    if results:
        print("\n📋 分析总结:")
        print(f"   • 总样本数: {len(results['combined_data'])}")
        print(f"   • PTT指标数: {len([col for col in results['combined_data'].columns if col.startswith('ptt_')])}")
        print(f"   • 血压指标数: {len([col for col in results['combined_data'].columns if any(bp in col for bp in ['systolic', 'diastolic', 'mean_bp', 'bp'])])}")
        print(f"   • 回归模型数: {len(results['models'])}")
        
        # 显示最佳相关性
        print(f"\n🏆 最强相关性 (前5名):")
        all_corrs = []
        for sensor_pair, bp_data in results['correlations'].items():
            for bp_col, stats_data in bp_data.items():
                if stats_data['significant']:
                    all_corrs.append((abs(stats_data['correlation']), 
                                    analyzer._format_sensor_pair_label(sensor_pair),
                                    analyzer._format_bp_label(bp_col),
                                    stats_data['correlation']))
        
        all_corrs.sort(reverse=True)
        for i, (abs_corr, ptt_label, bp_label, corr) in enumerate(all_corrs[:5]):
            print(f"   {i+1}. {ptt_label} ←→ {bp_label}: r={corr:.3f}")

if __name__ == "__main__":
    main() 