#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 4: Integrated PTT-BP Analysis
Uses results from step3 for further analysis: bar charts, integrated views, etc.
基于提供的目录结构（每个受试者如 /root/autodl-tmp/00003 有 ptt_bp_analysis/ 等子目录），
脚本从每个受试者的 ptt_bp_analysis/ 加载 step3 文件，进行跨受试者分析。
输出保存到 /root/autodl-tmp/integrated_analysis （全局目录）。
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.ioff()

class IntegratedPTTBloodPressureAnalyzer:
    def __init__(self, root_path="/root/autodl-tmp/", output_dir="integrated_analysis2"):
        self.root_path = root_path
        self.output_dir = os.path.join(root_path, output_dir)
        self.step3_dir = "ptt_bp_analysis2"  # step3 输出目录（每个受试者下）
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 只保留血压相关指标（与step3一致）
        self.physiological_indicators = {
            'systolic_bp': 'Systolic BP (mmHg)',
            'diastolic_bp': 'Diastolic BP (mmHg)', 
            'mean_bp': 'Mean Arterial Pressure (mmHg)'
        }
        
        self.ptt_combinations_en = {
            'sensor2-sensor3': 'Nose→Finger',
            'sensor2-sensor4': 'Nose→Wrist', 
            'sensor2-sensor5': 'Nose→Ear',
            'sensor3-sensor4': 'Finger→Wrist',
            'sensor3-sensor5': 'Finger→Ear',
            'sensor4-sensor5': 'Wrist→Ear'
        }
        
        print("🔬 Integrated PTT-Cardiovascular Parameters Correlation Analyzer")
        print(f"📁 Results will be saved to: {self.output_dir}")
        print(f"📂 Loading from each subject's {self.step3_dir}/")
    
    def load_subjects(self):
        """加载所有受试者（目录以 '00' 开头）"""
        return sorted([d for d in os.listdir(self.root_path) 
                       if os.path.isdir(os.path.join(self.root_path, d)) and d.startswith('00')])
    
    def load_step3_correlations(self, subject, exp_id=None):
        """从受试者的 ptt_bp_analysis2/ 加载 correlations CSV"""
        print(f"📂 加载 {subject} 的 correlations CSV")
        subject_dir = os.path.join(self.root_path, subject, self.step3_dir)
        if exp_id is not None:
            # 从exp_X文件夹中读取
            corr_file = os.path.join(subject_dir, f'exp_{exp_id}', f'ptt_cardiovascular_correlations_exp_{exp_id}.csv')
        else:
            # 整体相关性文件在根目录
            corr_file = os.path.join(subject_dir, 'ptt_cardiovascular_correlations.csv')
        if not os.path.exists(corr_file):
            print(f"⚠️ 文件不存在: {corr_file}")
            return None
        try:
            df = pd.read_csv(corr_file)
            if df.empty:
                print(f"⚠️ 空文件: {corr_file}")
                return None
            df['subject'] = subject
            return df
        except pd.errors.EmptyDataError:
            print(f"⚠️ 空数据错误: {corr_file}")
            return None
        except Exception as e:
            print(f"❌ 加载错误 {subject} exp_{exp_id}: {e}")
            return None
    
    def load_step3_sync_data(self, subject, exp_id=None):
        """从受试者的 ptt_bp_analysis/ 加载 sync 数据 CSV (always load overall file)"""
        print(f"📂 加载 {subject} 的 sync 数据")
        subject_dir = os.path.join(self.root_path, subject, self.step3_dir)
        sync_file = os.path.join(subject_dir, 'synchronized_ptt_cardiovascular_data.csv')
        if not os.path.exists(sync_file):
            print(f"⚠️ 文件不存在: {sync_file}")
            return None
        try:
            df = pd.read_csv(sync_file)
            if df.empty:
                print(f"⚠️ 空文件: {sync_file}")
                return None
            df['subject'] = subject
            return df
        except pd.errors.EmptyDataError:
            print(f"⚠️ 空数据错误: {sync_file}")
            return None
        except Exception as e:
            print(f"❌ 加载错误 {subject}: {e}")
            return None
    
    def create_correlation_bar_chart(self, corr_df, title_suffix, subjects, subdir, exp_id=None):
        """创建相关性柱状图（两种版本：multi-pair 一张图 + per pair 单独图）"""
        valid_sensors = list(self.ptt_combinations_en.values())
        for physio, physio_label in self.physiological_indicators.items():
            physio_col = f'{physio}_mean'
            data = []
            for subject in subjects:
                subj_df = corr_df[corr_df['subject'] == subject]
                for _, row in subj_df.iterrows():
                    sensor_label = row['sensor_combination']
                    if sensor_label in valid_sensors and row['physiological_parameter'] == physio_col:
                        data.append({
                            'subject': subject,
                            'sensor_pair': sensor_label,
                            'correlation': row['correlation_coefficient'],
                            'p_value': row.get('p_value', 1.0),  # Assume p_value in data
                            'significant': row.get('p_value', 1.0) < 0.05
                        })
            if not data:
                continue
            df = pd.DataFrame(data)
            n_subjects = len(df['subject'].unique())
            if n_subjects == 0:
                continue
            
            os.makedirs(subdir, exist_ok=True)
            
            # 版本1: 一张图所有6个 pair
            fig_width = max(12, n_subjects * 0.5)
            plt.figure(figsize=(fig_width, 8))
            ax = sns.barplot(data=df, x='subject', y='correlation', hue='sensor_pair', palette='tab10')
            lines = [(0.4, 'green'), (-0.4, 'green'), (0.5, 'blue'), (-0.5, 'blue'), (0.7, 'red'), (-0.7, 'red')]
            for val, color in lines:
                plt.axhline(val, color=color, linestyle='--')
            # Add significance asterisks
            for i, bar in enumerate(ax.patches):
                if i < len(df) and df.iloc[i]['significant']:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 if bar.get_height() > 0 else bar.get_height() - 0.05,
                            '*', ha='center', va='bottom' if bar.get_height() > 0 else 'top')
            plt.title(f'Correlation Bar Chart for {physio_label} (All Pairs) {title_suffix}')
            plt.ylim(-1, 1)
            plt.xlabel('Subject')
            plt.ylabel('Pearson Correlation')
            plt.xticks(rotation=90, ha='right')
            plt.legend(title='Sensor Pair', bbox_to_anchor=(1.05, 1), loc='upper left')
            filename_multi = f'correlation_bar_{physio}_multi{("_exp" + str(exp_id) if exp_id else "")}.png'
            plt.savefig(os.path.join(subdir, filename_multi), bbox_inches='tight')
            plt.close()
            print(f"💾 保存 multi-pair 柱状图: {os.path.join(subdir, filename_multi)}")
            
            # 版本2: 每个 pair 单独一张图
            for pair_label in valid_sensors:
                pair_data = [d for d in data if d['sensor_pair'] == pair_label]
                if not pair_data:
                    continue
                pair_df = pd.DataFrame(pair_data)
                fig_width = max(12, len(pair_df) * 0.5)
                plt.figure(figsize=(fig_width, 8))
                ax = sns.barplot(data=pair_df, x='subject', y='correlation', color='skyblue')
                for val, color in lines:
                    plt.axhline(val, color=color, linestyle='--')
                # Add significance asterisks
                for i, bar in enumerate(ax.patches):
                    if pair_df.iloc[i]['significant']:
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 if bar.get_height() > 0 else bar.get_height() - 0.05,
                                '*', ha='center', va='bottom' if bar.get_height() > 0 else 'top')
                plt.title(f'Correlation Bar Chart for {physio_label} - {pair_label} {title_suffix}')
                plt.ylim(-1, 1)
                plt.xlabel('Subject')
                plt.ylabel('Pearson Correlation')
                plt.xticks(rotation=90, ha='right')
                safe_pair = pair_label.replace('→', '-').replace(' ', '_')
                filename = f'correlation_bar_{physio}_{safe_pair}{("_exp" + str(exp_id) if exp_id else "")}.png'
                plt.savefig(os.path.join(subdir, filename), bbox_inches='tight')
                plt.close()
                print(f"💾 保存 per-pair 柱状图: {os.path.join(subdir, filename)}")
    
    def create_r_distribution_plot(self, corr_df, title_suffix, subjects, subdir, exp_id=None):
        """绘制 r 的分布图 (violin plot with gradient)"""
        valid_sensors = list(self.ptt_combinations_en.values())
        for physio, physio_label in self.physiological_indicators.items():
            physio_col = f'{physio}_mean'
            data = []
            for subject in subjects:
                subj_df = corr_df[corr_df['subject'] == subject]
                for _, row in subj_df.iterrows():
                    sensor_label = row['sensor_combination']
                    if sensor_label in valid_sensors and row['physiological_parameter'] == physio_col:
                        data.append({
                            'sensor_pair': sensor_label,
                            'correlation': row['correlation_coefficient']
                        })
            if not data:
                continue
            df = pd.DataFrame(data)
            os.makedirs(subdir, exist_ok=True)
            plt.figure(figsize=(12, 8))
            sns.violinplot(data=df, x='sensor_pair', y='correlation', palette='viridis', inner='box')
            
            # 添加水平参考线
            lines = [(0, 'black'), (0.4, 'green'), (-0.4, 'green'), (0.7, 'red'), (-0.7, 'red')]
            for val, color in lines:
                plt.axhline(val, color=color, linestyle='--', linewidth=1)
            
            # 计算并标注 Q1, median, Q3 和最宽点 (峰值)
            from scipy.stats import gaussian_kde
            quantiles = df.groupby('sensor_pair')['correlation'].quantile([0.25, 0.5, 0.75]).unstack()
            for i, pair in enumerate(df['sensor_pair'].unique()):
                pair_data = df[df['sensor_pair'] == pair]['correlation']
                if pair in quantiles.index and not pair_data.empty:
                    q1, median, q3 = quantiles.loc[pair, [0.25, 0.5, 0.75]]
                    # 标注 Q1 和 Q3 (调整位置以提高清晰度)
                    plt.text(i + 0.2, q1 - 0.05, f'Q1: {q1:.2f}', ha='left', va='top', fontsize=8, color='blue', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
                    plt.text(i + 0.2, q3 + 0.05, f'Q3: {q3:.2f}', ha='left', va='bottom', fontsize=8, color='blue', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
                    # 中位数用白色标注
                    plt.text(i, median + 0.05, f'Med: {median:.2f}', ha='center', va='bottom', fontsize=8, color='white', bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))
                    # 标注最宽点 (密度峰值)
                    if len(pair_data) > 1:
                        pair_data = pair_data.dropna()  # 移除 NaN 以避免 KDE 错误
                        if not pair_data.empty and np.isfinite(pair_data).all():
                            kde = gaussian_kde(pair_data)
                            y_vals = np.linspace(pair_data.min(), pair_data.max(), 100)
                            kde_vals = kde(y_vals)
                            peak_y = y_vals[np.argmax(kde_vals)]
                            plt.text(i - 0.2, peak_y, f'Peak: {peak_y:.2f}', ha='right', va='center', fontsize=8, color='red', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
            
            plt.title(f'Distribution of Correlations for {physio_label} {title_suffix}')
            plt.ylim(-1.1, 1.1)  # 略微扩展 y 轴以容纳标注
            plt.xlabel('Sensor Pair')
            plt.ylabel('Pearson Correlation')
            plt.xticks(rotation=45, ha='right')
            filename = f'r_distribution_{physio}{("_exp" + str(exp_id) if exp_id else "")}.png'
            plt.savefig(os.path.join(subdir, filename), bbox_inches='tight')
            plt.close()
            print(f"💾 保存 r 分布图: {os.path.join(subdir, filename)}")
    
    def remove_outliers_iqr(self, data_series):
        """使用IQR方法去除极值 (from step3)"""
        q1 = data_series.quantile(0.25)
        q3 = data_series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return data_series[(data_series >= lower_bound) & (data_series <= upper_bound)]

    def create_bland_altman_plots(self, sync_df, exp_id=None, title_suffix=""):
        """创建Bland-Altman图 - 六个传感器对的SBP、DBP和Mean BP"""
        try:
            # 血压指标 - 包括Mean BP（只处理血压相关指标）
            bp_indicators = ['systolic_bp_mean', 'diastolic_bp_mean', 'mean_bp_mean']
            bp_labels = ['Systolic BP', 'Diastolic BP', 'Mean BP']
            
            # 传感器对
            sensor_pairs = list(self.ptt_combinations_en.keys())
            
            # 创建6x6的子图布局 - 6行（传感器对）x 6列（每个生理指标左右两个图）
            fig, axes = plt.subplots(6, 6, figsize=(24, 20))
            fig.suptitle(f'PTT vs Reference BP Analysis{title_suffix}', 
                        fontsize=16, fontweight='bold', y=1)
            
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
            
            # 保存图像到当前工作目录
            if exp_id is not None:
                filename = f"bland_altman_bp_exp_{exp_id}{title_suffix.replace(' ', '_')}.png"
            else:
                filename = f"bland_altman_bp_overall{title_suffix.replace(' ', '_')}.png"
            
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"💾 保存Bland-Altman图: {filename}")
            
            # 保存误差带统计到CSV
            if error_band_stats:
                stats_df = pd.DataFrame(error_band_stats)
                if exp_id is not None:
                    stats_filename = f"error_band_stats_exp_{exp_id}{title_suffix.replace(' ', '_')}.csv"
                else:
                    stats_filename = f"error_band_stats_overall{title_suffix.replace(' ', '_')}.csv"
                stats_df.to_csv(stats_filename, index=False)
                print(f"💾 保存误差带统计: {stats_filename}")
            
            return fig
            
        except Exception as e:
            print(f"❌ Bland-Altman图创建失败: {e}")
            return None

    def run_individual_experiment_analysis(self, subjects):
        print("\n=== 单实验分析 ===")
        individual_dir = os.path.join(self.output_dir, 'individual_experiments_correlations')
        os.makedirs(individual_dir, exist_ok=True)
        for exp_id in range(1, 12):
            exp_corrs = pd.DataFrame()
            for subject in subjects:
                df = self.load_step3_correlations(subject, exp_id)
                if df is not None:
                    exp_corrs = pd.concat([exp_corrs, df])
            if not exp_corrs.empty:
                print(f"📊 生成实验 {exp_id} 的柱状图")
                subdir = os.path.join(individual_dir, f'exp_{exp_id}')
                os.makedirs(subdir, exist_ok=True)
                if 'subject' in exp_corrs.columns:
                    cols = ['subject'] + [col for col in exp_corrs.columns if col != 'subject']
                    exp_corrs = exp_corrs[cols]
                csv_path = os.path.join(subdir, f'exp_{exp_id}_correlations.csv')
                exp_corrs.to_csv(csv_path, index=False)
                print(f"💾 保存实验 {exp_id} 的 correlations CSV: {csv_path}")
                self.create_correlation_bar_chart(exp_corrs, f'(Exp {exp_id})', subjects, subdir, exp_id)
                self.create_r_distribution_plot(exp_corrs, f'(Exp {exp_id})', subjects, subdir, exp_id)
    
    def run_subject_overall_analysis(self, subjects):
        print("\n=== 每个受试者整体分析 ===")
        overall_corrs = pd.DataFrame()
        all_sync_df = pd.DataFrame()
        
        # 加载相关性和同步数据
        for subject in subjects:
            # 加载相关性数据
            df = self.load_step3_correlations(subject)
            if df is not None:
                overall_corrs = pd.concat([overall_corrs, df])
            
            # 加载同步数据
            sync_df = self.load_step3_sync_data(subject)
            if sync_df is not None:
                all_sync_df = pd.concat([all_sync_df, sync_df], ignore_index=True)
        
        if not overall_corrs.empty:
            print("📊 生成每个受试者整体柱状图")
            subdir = os.path.join(self.output_dir, 'overall_correlations')
            os.makedirs(subdir, exist_ok=True)
            if 'subject' in overall_corrs.columns:
                cols = ['subject'] + [col for col in overall_corrs.columns if col != 'subject']
                overall_corrs = overall_corrs[cols]
            csv_path = os.path.join(subdir, 'overall_correlations.csv')
            overall_corrs.to_csv(csv_path, index=False)
            print(f"💾 保存整体 correlations CSV: {csv_path}")
            self.create_correlation_bar_chart(overall_corrs, '(Overall per Subject)', subjects, subdir)
            self.create_r_distribution_plot(overall_corrs, '(Overall per Subject)', subjects, subdir)
        
        # 为整体数据创建Bland-Altman图
        if not all_sync_df.empty:
            print("📊 为整体数据创建Bland-Altman图")
            # 清理数据
            cleaned_overall = pd.DataFrame()
            for pair in all_sync_df['sensor_pair'].unique():
                pair_df = all_sync_df[all_sync_df['sensor_pair'] == pair]
                for col in [c for c in pair_df.columns if c.endswith('_mean') or c == 'ptt_ms']:
                    cleaned_series = self.remove_outliers_iqr(pair_df[col])
                    mask = pair_df[col].isin(cleaned_series)
                    pair_df = pair_df[mask]
                cleaned_overall = pd.concat([cleaned_overall, pair_df])
            
            if not cleaned_overall.empty:
                # 切换到正确的目录
                os.chdir(subdir)
                self.create_bland_altman_plots(cleaned_overall, None, " (Overall per Subject)")
                os.chdir(self.output_dir)  # 回到输出目录
                
                # 保存清理后的数据
                cleaned_path = os.path.join(subdir, 'overall_cleaned.csv')
                cleaned_overall.to_csv(cleaned_path, index=False)
                print(f"💾 保存整体清理数据: {cleaned_path}")
    
    def run_non_cross_experiment_fits(self, subjects):
        """不跨实验的线性拟合：每个实验单独拟合"""
        print("\n=== 不跨实验线性拟合分析 ===")
        non_cross_dir = os.path.join(self.output_dir, 'per_experiment_fits')
        os.makedirs(non_cross_dir, exist_ok=True)
        
        all_sync_df = pd.DataFrame()
        for subject in subjects:
            df = self.load_step3_sync_data(subject)
            if df is not None:
                all_sync_df = pd.concat([all_sync_df, df], ignore_index=True)
        
        if all_sync_df.empty:
            print("⚠️ 无可用 sync 数据")
            return
        
        if 'exp_id' not in all_sync_df.columns:
            print("⚠️ sync 数据缺少 'exp_id' 列，无法进行 per-exp 分析")
            return
        
        for exp_id in sorted(all_sync_df['exp_id'].unique()):
            exp_sync = all_sync_df[all_sync_df['exp_id'] == exp_id].reset_index(drop=True)
            if exp_sync.empty:
                continue
            
            # Clean outliers per group
            cleaned_exp = pd.DataFrame()
            for pair in exp_sync['sensor_pair'].unique():
                pair_df = exp_sync[exp_sync['sensor_pair'] == pair]
                for col in [c for c in pair_df.columns if c.endswith('_mean') or c == 'ptt_ms']:
                    cleaned_series = self.remove_outliers_iqr(pair_df[col])
                    mask = pair_df[col].isin(cleaned_series)
                    pair_df = pair_df[mask]
                cleaned_exp = pd.concat([cleaned_exp, pair_df])
            
            cleaned_path = os.path.join(non_cross_dir, f'per_exp_{exp_id}_cleaned.csv')
            cleaned_exp.to_csv(cleaned_path, index=False)
            print(f"💾 保存 per-exp cleaned CSV: {cleaned_path}")
            
            # 绘制散点图 + 线性拟合
            for physio, label in self.physiological_indicators.items():
                col = f'{physio}_mean'
                if col in cleaned_exp.columns:
                    for pair in cleaned_exp['sensor_pair'].unique():
                        pair_df = cleaned_exp[cleaned_exp['sensor_pair'] == pair].reset_index(drop=True)
                        if len(pair_df) < 10:
                            continue
                        plt.figure(figsize=(10, 8))
                        sns.scatterplot(data=pair_df, x='ptt_ms', y=col, hue='subject', palette='tab20', alpha=0.6)
                        
                        mask = ~(pair_df['ptt_ms'].isna() | pair_df[col].isna())
                        if mask.sum() >= 10:
                            X = pair_df.loc[mask, 'ptt_ms'].values.reshape(-1, 1)
                            y = pair_df.loc[mask, col].values
                            model = LinearRegression().fit(X, y)
                            pred = model.predict(X)
                            r, _ = stats.pearsonr(pair_df.loc[mask, 'ptt_ms'], y)
                            r2 = model.score(X, y)
                            mae = mean_absolute_error(y, pred)
                            std = np.std(y - pred)
                            x_sort = np.sort(X, axis=0)
                            plt.plot(x_sort, model.predict(x_sort), color='red', linewidth=2, label='Fit')
                            
                            stats_text = f'r = {r:.2f}\nR² = {r2:.2f}\nMAE = {mae:.2f}\nSTD = {std:.2f}'
                            plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
                                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'),
                                     verticalalignment='top')
                        
                        plt.title(f'{label} vs PTT ({self.ptt_combinations_en.get(pair, pair)}) - Per Exp {exp_id} (Cleaned)')
                        plt.xlabel('PTT (ms)')
                        plt.ylabel(label)
                        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                        filename = f'scatter_fit_{physio}_{pair}_per_exp{exp_id}_cleaned.png'
                        plt.savefig(os.path.join(non_cross_dir, filename), bbox_inches='tight')
                        plt.close()
                        print(f"💾 保存 per-exp 散点拟合图: {os.path.join(non_cross_dir, filename)}")
            
            # 为每个实验创建Bland-Altman图
            if not cleaned_exp.empty:
                print(f"📊 为实验 {exp_id} 创建Bland-Altman图")
                # 切换到正确的目录
                os.chdir(non_cross_dir)
                self.create_bland_altman_plots(cleaned_exp, exp_id, f" (Per Exp {exp_id})")
                os.chdir(self.output_dir)  # 回到输出目录

    def run_cross_experiment_fits(self, subjects):
        """跨实验的线性拟合：综合所有实验"""
        print("\n=== 跨实验线性拟合分析 ===")
        cross_dir = os.path.join(self.output_dir, 'cross_experiment_fits')
        os.makedirs(cross_dir, exist_ok=True)
        
        all_sync_df = pd.DataFrame()
        for subject in subjects:
            df = self.load_step3_sync_data(subject)
            if df is not None:
                all_sync_df = pd.concat([all_sync_df, df], ignore_index=True)
        
        if all_sync_df.empty:
            print("⚠️ 无可用 sync 数据")
            return
        
        # Clean comprehensive
        cleaned_comp = pd.DataFrame()
        for pair in all_sync_df['sensor_pair'].unique():
            pair_df = all_sync_df[all_sync_df['sensor_pair'] == pair]
            for col in [c for c in pair_df.columns if c.endswith('_mean') or c == 'ptt_ms']:
                cleaned_series = self.remove_outliers_iqr(pair_df[col])
                mask = pair_df[col].isin(cleaned_series)
                pair_df = pair_df[mask]
            cleaned_comp = pd.concat([cleaned_comp, pair_df])
        
        cleaned_comp_path = os.path.join(cross_dir, 'cross_experiments_cleaned.csv')
        cleaned_comp.to_csv(cleaned_comp_path, index=False)
        print(f"💾 保存 cross-exp cleaned CSV: {cleaned_comp_path}")
        
        # 综合散点 + 拟合
        for physio, label in self.physiological_indicators.items():
            col = f'{physio}_mean'
            if col in cleaned_comp.columns:
                for pair in cleaned_comp['sensor_pair'].unique():
                    pair_df = cleaned_comp[cleaned_comp['sensor_pair'] == pair].reset_index(drop=True)
                    if len(pair_df) < 10:
                        continue
                    plt.figure(figsize=(10, 8))
                    sns.scatterplot(data=pair_df, x='ptt_ms', y=col, hue='subject', palette='tab20', alpha=0.6)
                    
                    mask = ~(pair_df['ptt_ms'].isna() | pair_df[col].isna())
                    if mask.sum() >= 10:
                        X = pair_df.loc[mask, 'ptt_ms'].values.reshape(-1, 1)
                        y = pair_df.loc[mask, col].values
                        model = LinearRegression().fit(X, y)
                        pred = model.predict(X)
                        r, _ = stats.pearsonr(pair_df.loc[mask, 'ptt_ms'], y)
                        r2 = model.score(X, y)
                        mae = mean_absolute_error(y, pred)
                        std = np.std(y - pred)
                        x_sort = np.sort(X, axis=0)
                        plt.plot(x_sort, model.predict(x_sort), color='red', linewidth=2, label='Overall Fit')
                        
                        stats_text = f'r = {r:.2f}\nR² = {r2:.2f}\nMAE = {mae:.2f}\nSTD = {std:.2f}'
                        plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
                                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'),
                                 verticalalignment='top')
                    
                    plt.title(f'{label} vs PTT ({self.ptt_combinations_en.get(pair, pair)}) - Cross Experiments (Cleaned)')
                    plt.xlabel('PTT (ms)')
                    plt.ylabel(label)
                    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    filename = f'scatter_fit_{physio}_{pair}_cross_experiments_cleaned.png'
                    plt.savefig(os.path.join(cross_dir, filename), bbox_inches='tight')
                    plt.close()
                    print(f"💾 保存 cross-exp 散点拟合图: {os.path.join(cross_dir, filename)}")
            
            # 为跨实验数据创建Bland-Altman图
            if not cleaned_comp.empty:
                print(f"📊 为跨实验数据创建Bland-Altman图")
                # 切换到正确的目录
                os.chdir(cross_dir)
                self.create_bland_altman_plots(cleaned_comp, None, " (Cross Experiments)")
                os.chdir(self.output_dir)  # 回到输出目录

    def run_integrated_analysis(self, subjects):
        """融合分析：跨受试者整合 (现在只保存CSV，不生成拟合图)"""
        print("\n=== 融合分析 ===")
        integrated_dir = os.path.join(self.output_dir, 'integrated_experiments')
        os.makedirs(integrated_dir, exist_ok=True)
        
        all_sync_df = pd.DataFrame()
        for subject in subjects:
            df = self.load_step3_sync_data(subject)
            if df is not None:
                all_sync_df = pd.concat([all_sync_df, df], ignore_index=True)
        
        if all_sync_df.empty:
            print("⚠️ 无可用 sync 数据")
            return
        
        if 'subject' in all_sync_df.columns:
            cols = ['subject'] + [col for col in all_sync_df.columns if col != 'subject']
            all_sync_df = all_sync_df[cols]
        
        if 'exp_id' in all_sync_df.columns:
            for exp_id in sorted(all_sync_df['exp_id'].unique()):
                exp_sync = all_sync_df[all_sync_df['exp_id'] == exp_id].reset_index(drop=True)
                if not exp_sync.empty:
                    csv_path = os.path.join(integrated_dir, f'integrated_exp_{exp_id}.csv')
                    exp_sync.to_csv(csv_path, index=False)
                    print(f"💾 保存整合 CSV: {csv_path}")
        else:
            print("⚠️ sync 数据缺少 'exp_id' 列，无法 per-exp 拆分")
        
        comprehensive_path = os.path.join(self.output_dir, 'comprehensive_integrated.csv')
        all_sync_df.to_csv(comprehensive_path, index=False)
        print(f"💾 保存综合 CSV: {comprehensive_path}")

def main():
    analyzer = IntegratedPTTBloodPressureAnalyzer()
    subjects = analyzer.load_subjects()
    print(f"📋 发现 {len(subjects)} 个受试者")
    
    print("\n📋 请选择分析方式:")
    print("1. 综合分析 (所有)")
    print("2. 单实验相关性分析 (每个实验的柱状图)")
    print("3. 综合实验相关性分析 (所有实验的柱状图)")
    print("4. 不跨实验的线性拟合")
    print("5. 跨实验的线性拟合")
    
    choice = input("\n请输入选择 (1/2/3/4/5, 默认1): ").strip()
    if not choice:
        choice = "1"
    
    if choice == "1":
        analyzer.run_individual_experiment_analysis(subjects)
        analyzer.run_subject_overall_analysis(subjects)
        analyzer.run_integrated_analysis(subjects)
        analyzer.run_non_cross_experiment_fits(subjects)
        analyzer.run_cross_experiment_fits(subjects)
    elif choice == "2":
        analyzer.run_individual_experiment_analysis(subjects)
    elif choice == "3":
        analyzer.run_subject_overall_analysis(subjects)
    elif choice == "4":
        analyzer.run_non_cross_experiment_fits(subjects)
    elif choice == "5":
        analyzer.run_cross_experiment_fits(subjects)
    else:
        print("❌ 无效选择，默认运行综合分析")
        analyzer.run_individual_experiment_analysis(subjects)
        analyzer.run_subject_overall_analysis(subjects)
        analyzer.run_integrated_analysis(subjects)
        analyzer.run_non_cross_experiment_fits(subjects)
        analyzer.run_cross_experiment_fits(subjects)
    
    print("\n✅ 分析完成！")
    print(f"📁 结果保存在: {analyzer.output_dir}")

if __name__ == "__main__":
    main()