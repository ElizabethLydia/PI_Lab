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

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.ioff()

class IntegratedPTTBloodPressureAnalyzer:
    def __init__(self, root_path="/root/autodl-tmp/", output_dir="integrated_analysis"):
        self.root_path = root_path
        self.output_dir = os.path.join(root_path, output_dir)
        self.step3_dir = "ptt_bp_analysis"  # step3 输出目录（每个受试者下）
        os.makedirs(self.output_dir, exist_ok=True)
        
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
        """从受试者的 ptt_bp_analysis/ 加载 correlations CSV"""
        print(f"📂 加载 {subject} 的 correlations CSV")
        subject_dir = os.path.join(self.root_path, subject, self.step3_dir)
        if exp_id is not None:
            corr_file = os.path.join(subject_dir, f'ptt_cardiovascular_correlations_exp_{exp_id}.csv')
        else:
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
    
    def create_correlation_bar_chart(self, corr_df, title_suffix, subjects, exp_id=None):
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
            
            subdir = os.path.join(self.output_dir, f'exp_{exp_id}' if exp_id else 'overall')
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
    
    def create_r_distribution_plot(self, corr_df, title_suffix, subjects, exp_id=None):
        """绘制 r 的分布图 (violin plot with gradient)"""
        valid_sensors = list(self.ptt_combinations_en.values())
        for physio, physio_label in self.physiological_indicators.items():
            data = []
            for subject in subjects:
                subj_df = corr_df[corr_df['subject'] == subject]
                for _, row in subj_df.iterrows():
                    sensor_label = row['sensor_combination']
                    if sensor_label in valid_sensors:
                        data.append({
                            'sensor_pair': sensor_label,
                            'correlation': row['correlation_coefficient']
                        })
            if not data:
                continue
            df = pd.DataFrame(data)
            subdir = os.path.join(self.output_dir, f'exp_{exp_id}' if exp_id else 'overall')
            os.makedirs(subdir, exist_ok=True)
            plt.figure(figsize=(12, 8))
            sns.violinplot(data=df, x='sensor_pair', y='correlation', palette='viridis', inner='box')
            plt.title(f'Distribution of Correlations for {physio_label} {title_suffix}')
            plt.ylim(-1, 1)
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

    def run_individual_experiment_analysis(self, subjects):
        print("\n=== 单实验分析 ===")
        for exp_id in range(1, 12):
            exp_corrs = pd.DataFrame()
            for subject in subjects:
                df = self.load_step3_correlations(subject, exp_id)
                if df is not None:
                    exp_corrs = pd.concat([exp_corrs, df])
            if not exp_corrs.empty:
                print(f"📊 生成实验 {exp_id} 的柱状图")
                subdir = os.path.join(self.output_dir, f'exp_{exp_id}')
                os.makedirs(subdir, exist_ok=True)
                if 'subject' in exp_corrs.columns:
                    cols = ['subject'] + [col for col in exp_corrs.columns if col != 'subject']
                    exp_corrs = exp_corrs[cols]
                csv_path = os.path.join(subdir, f'exp_{exp_id}_correlations.csv')
                exp_corrs.to_csv(csv_path, index=False)
                print(f"💾 保存实验 {exp_id} 的 correlations CSV: {csv_path}")
                self.create_correlation_bar_chart(exp_corrs, f'(Exp {exp_id})', subjects, exp_id)
                self.create_r_distribution_plot(exp_corrs, f'(Exp {exp_id})', subjects, exp_id)
    
    def run_subject_overall_analysis(self, subjects):
        print("\n=== 每个受试者整体分析 ===")
        overall_corrs = pd.DataFrame()
        for subject in subjects:
            df = self.load_step3_correlations(subject)
            if df is not None:
                overall_corrs = pd.concat([overall_corrs, df])
        if not overall_corrs.empty:
            print("📊 生成每个受试者整体柱状图")
            subdir = os.path.join(self.output_dir, 'overall')
            os.makedirs(subdir, exist_ok=True)
            if 'subject' in overall_corrs.columns:
                cols = ['subject'] + [col for col in overall_corrs.columns if col != 'subject']
                overall_corrs = overall_corrs[cols]
            csv_path = os.path.join(subdir, 'overall_correlations.csv')
            overall_corrs.to_csv(csv_path, index=False)
            print(f"💾 保存整体 correlations CSV: {csv_path}")
            self.create_correlation_bar_chart(overall_corrs, '(Overall per Subject)', subjects)
            self.create_r_distribution_plot(overall_corrs, '(Overall per Subject)', subjects)
    
    def run_integrated_analysis(self, subjects):
        """融合分析：跨受试者整合"""
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
        
        # Reorder with subject first
        if 'subject' in all_sync_df.columns:
            cols = ['subject'] + [col for col in all_sync_df.columns if col != 'subject']
            all_sync_df = all_sync_df[cols]
        
        # Per exp_id (assuming exp_id column exists)
        if 'exp_id' in all_sync_df.columns:
            for exp_id in sorted(all_sync_df['exp_id'].unique()):
                exp_sync = all_sync_df[all_sync_df['exp_id'] == exp_id].reset_index(drop=True)
                if not exp_sync.empty:
                    csv_path = os.path.join(integrated_dir, f'integrated_exp_{exp_id}.csv')
                    exp_sync.to_csv(csv_path, index=False)
                    print(f"💾 保存整合 CSV: {csv_path}")
                    
                    # Clean outliers per group (sensor_pair)
                    cleaned_exp = pd.DataFrame()
                    for pair in exp_sync['sensor_pair'].unique():
                        pair_df = exp_sync[exp_sync['sensor_pair'] == pair]
                        for col in [c for c in pair_df.columns if c.endswith('_mean') or c == 'ptt_ms']:
                            cleaned_series = self.remove_outliers_iqr(pair_df[col])
                            mask = pair_df[col].isin(cleaned_series)
                            pair_df = pair_df[mask]
                        cleaned_exp = pd.concat([cleaned_exp, pair_df])
                    cleaned_path = os.path.join(integrated_dir, f'integrated_exp_{exp_id}_cleaned.csv')
                    cleaned_exp.to_csv(cleaned_path, index=False)
                    print(f"💾 保存 cleaned 整合 CSV: {cleaned_path}")
                    
                    # Use cleaned for plots
                    exp_sync = cleaned_exp
                    
                    # 绘制散点图 + 线性拟合
                    for physio, label in self.physiological_indicators.items():
                        col = f'{physio}_mean'
                        if col in exp_sync.columns:
                            for pair in exp_sync['sensor_pair'].unique():
                                pair_df = exp_sync[exp_sync['sensor_pair'] == pair].reset_index(drop=True)
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
                                    
                                    # 添加标注
                                    stats_text = f'r = {r:.2f}\nR² = {r2:.2f}\nMAE = {mae:.2f}\nSTD = {std:.2f}'
                                    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
                                             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'),
                                             verticalalignment='top')
                                
                                plt.title(f'{label} vs PTT ({self.ptt_combinations_en.get(pair, pair)}) - Integrated Exp {exp_id} (Cleaned)')
                                plt.xlabel('PTT (ms)')
                                plt.ylabel(label)
                                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                                filename = f'scatter_fit_{physio}_{pair}_exp{exp_id}_cleaned.png'
                                plt.savefig(os.path.join(integrated_dir, filename), bbox_inches='tight')
                                plt.close()
                                print(f"💾 保存散点拟合图 (cleaned): {os.path.join(integrated_dir, filename)}")
        else:
            print("⚠️ sync 数据缺少 'exp_id' 列，无法 per-exp 拆分")
        
        # 综合所有实验
        comprehensive_path = os.path.join(self.output_dir, 'comprehensive_integrated.csv')
        all_sync_df.to_csv(comprehensive_path, index=False)
        print(f"💾 保存综合 CSV: {comprehensive_path}")
        
        # Clean comprehensive
        cleaned_comp = pd.DataFrame()
        for pair in all_sync_df['sensor_pair'].unique():
            pair_df = all_sync_df[all_sync_df['sensor_pair'] == pair]
            for col in [c for c in pair_df.columns if c.endswith('_mean') or c == 'ptt_ms']:
                cleaned_series = self.remove_outliers_iqr(pair_df[col])
                mask = pair_df[col].isin(cleaned_series)
                pair_df = pair_df[mask]
            cleaned_comp = pd.concat([cleaned_comp, pair_df])
        cleaned_comp_path = os.path.join(self.output_dir, 'comprehensive_integrated_cleaned.csv')
        cleaned_comp.to_csv(cleaned_comp_path, index=False)
        print(f"💾 保存 cleaned 综合 CSV: {cleaned_comp_path}")
        
        # Use cleaned for comprehensive plots
        all_sync_df = cleaned_comp
        
        # 综合散点 + 拟合
        for physio, label in self.physiological_indicators.items():
            col = f'{physio}_mean'
            if col in all_sync_df.columns:
                for pair in all_sync_df['sensor_pair'].unique():
                    pair_df = all_sync_df[all_sync_df['sensor_pair'] == pair].reset_index(drop=True)
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
                        
                        # 添加标注
                        stats_text = f'r = {r:.2f}\nR² = {r2:.2f}\nMAE = {mae:.2f}\nSTD = {std:.2f}'
                        plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
                                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'),
                                 verticalalignment='top')
                    
                    plt.title(f'{label} vs PTT ({self.ptt_combinations_en.get(pair, pair)}) - Comprehensive (Cleaned)')
                    plt.xlabel('PTT (ms)')
                    plt.ylabel(label)
                    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    filename = f'scatter_fit_{physio}_{pair}_comprehensive_cleaned.png'
                    plt.savefig(os.path.join(self.output_dir, filename), bbox_inches='tight')
                    plt.close()
                    print(f"💾 保存综合散点拟合图 (cleaned): {os.path.join(self.output_dir, filename)}")

def main():
    analyzer = IntegratedPTTBloodPressureAnalyzer()
    subjects = analyzer.load_subjects()
    print(f"📋 发现 {len(subjects)} 个受试者")
    
    print("\n📋 请选择分析方式:")
    print("1. 综合分析 (所有)")
    print("2. 单实验分析 (每个实验的柱状图)")
    print("3. 融合分析 (跨受试者整合 + 散点拟合)")
    print("4. 每个受试者整体分析 (所有实验的柱状图)")
    
    choice = input("\n请输入选择 (1/2/3/4, 默认1): ").strip()
    if not choice:
        choice = "1"
    
    if choice == "1":
        analyzer.run_individual_experiment_analysis(subjects)
        analyzer.run_subject_overall_analysis(subjects)
        analyzer.run_integrated_analysis(subjects)
    elif choice == "2":
        analyzer.run_individual_experiment_analysis(subjects)
    elif choice == "3":
        analyzer.run_integrated_analysis(subjects)
    elif choice == "4":
        analyzer.run_subject_overall_analysis(subjects)
    else:
        print("❌ 无效选择，默认运行综合分析")
        analyzer.run_individual_experiment_analysis(subjects)
        analyzer.run_subject_overall_analysis(subjects)
        analyzer.run_integrated_analysis(subjects)
    
    print("\n✅ 分析完成！")
    print(f"📁 结果保存在: {analyzer.output_dir}")

if __name__ == "__main__":
    main()