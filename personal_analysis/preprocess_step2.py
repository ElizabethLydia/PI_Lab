# 将00017_aligned.pkl中的数据整合为单个csv文件，具体体现为
# 1. 将biopac数据整合为单个csv文件，文件名为00017_biopac_aligned.csv
# 2. 将hub数据整合为单个csv文件，文件名为00017_hub_sensor{i}_aligned.csv

# 导入必要的库
import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# 设置路径
output_dir = 'D:\code\Python\PI_Lab\output'
pkl_files = [f for f in os.listdir(output_dir) if f.endswith('_aligned.pkl')]
print(f'找到 {len(pkl_files)} 个.pkl文件: {pkl_files}')

# 加载并检查数据
def load_and_verify_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    experiment_name = list(data.keys())[0]
    print(f'加载实验: {experiment_name}')
    
    biopac_data = data[experiment_name]['biopac']
    hub_data = data[experiment_name]['hub']

    print(f'  Biopac文件数: {len(biopac_data)}, HUB文件数: {len(hub_data)}')
    for key, df in {**biopac_data, **hub_data}.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            print(f'  {key}: {len(df)} 行, 列: {list(df.columns)}')

    return data

selected_file = pkl_files[0]  # 取第一个文件，可手动修改索引
data = load_and_verify_pkl(os.path.join(output_dir, selected_file))

# 可视化数据对齐情况
experiment_name = list(data.keys())[0]
biopac_df = data[experiment_name]['biopac'].get('bp', pd.DataFrame())
hub_df = data[experiment_name]['hub'].get('sensor2', pd.DataFrame())

if not biopac_df.empty and not hub_df.empty:
    plt.figure(figsize=(12, 6))
    plt.plot(biopac_df['timestamp'], biopac_df.get('bp', np.zeros(len(biopac_df))), label='Biopac BP')
    plt.plot(hub_df['timestamp'], hub_df.get('red', np.zeros(len(hub_df))), label='HUB Sensor2 Red')
    plt.xlabel('Timestamp')
    plt.ylabel('Value')
    plt.title(f'Alignment Check - {experiment_name}')
    plt.legend()
    plt.show()
else:
    print('警告: 缺少有效数据，无法绘制')

# 导入必要的库
import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
# %matplotlib inline

# 设置路径
output_dir = 'D:\code\Python\PI_Lab\output'
csv_dir = os.path.join(output_dir, 'csv_output')
os.makedirs(csv_dir, exist_ok=True)

# 查找所有文件
pkl_files = [f for f in os.listdir(output_dir) if f.endswith('_aligned.pkl')]
npy_files = [f for f in os.listdir(output_dir) if f.endswith('_aligned.npy')]
all_files = pkl_files + npy_files
print(f'找到 {len(all_files)} 个文件: {all_files}')

# 加载并检查数据
def load_and_verify_file(file_path):
    if file_path.endswith('.pkl'):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    elif file_path.endswith('.npy'):
        data = np.load(file_path, allow_pickle=True).item()
    else:
        return None
    
    experiment_name = list(data.keys())[0]
    print(f'\n加载实验: {experiment_name}')
    
    biopac_data = data[experiment_name]['biopac']
    hub_data = data[experiment_name]['hub']

    print(f'  Biopac文件数: {len(biopac_data)}, HUB文件数: {len(hub_data)}')
    for key, df in {**biopac_data, **hub_data}.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            print(f'  {key}: {len(df)} 行, 列: {list(df.columns)}')
            non_null = df.notna().sum()
            print(f'    非空值: {non_null.to_dict()}')
            if len(df) > 0:
                print(f'    均值: {df.mean().round(2).to_dict()}')
                print(f'    标准差: {df.std().round(2).to_dict()}')
    
    return data

# 批量处理所有文件
for file in all_files:
    file_path = os.path.join(output_dir, file)
    data = load_and_verify_file(file_path)
    if data is None:
        print(f'警告: {file} 格式不支持，跳过')
        continue
    
    # 可视化数据对齐情况
    experiment_name = list(data.keys())[0]
    biopac_df = data[experiment_name]['biopac'].get('bp', pd.DataFrame())
    hub_df = data[experiment_name]['hub'].get('sensor2', pd.DataFrame())

    if not biopac_df.empty and not hub_df.empty:
        plt.figure(figsize=(12, 6))
        plt.plot(biopac_df['timestamp'], biopac_df.get('bp', np.zeros(len(biopac_df))), label='Biopac BP')
        plt.plot(hub_df['timestamp'], hub_df.get('red', np.zeros(len(hub_df))), label='HUB Sensor2 Red')
        plt.xlabel('Timestamp')
        plt.ylabel('Value')
        plt.title(f'Alignment Check - {experiment_name}')
        plt.legend()
        plt.show()
    
    # 整合Biopac数据为单文件CSV
    biopac_data = data[experiment_name]['biopac']
    if biopac_data:
        ref_timestamps = data[experiment_name]['hub'].get('sensor2', pd.DataFrame())['timestamp'].values
        if len(ref_timestamps) == 0:
            ref_timestamps = biopac_data[next(iter(biopac_data))]['timestamp'].values
        
        merged_biopac = pd.DataFrame({'timestamp': ref_timestamps})
        
        for key, df in biopac_data.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                merged_biopac = merged_biopac.merge(df[['timestamp', key]], on='timestamp', how='left')
        
        merged_biopac = merged_biopac.fillna(method='ffill').fillna(method='bfill')
        biopac_csv_path = os.path.join(csv_dir, f'{experiment_name}_biopac_aligned.csv')
        merged_biopac.to_csv(biopac_csv_path, index=False)
        print(f'  保存整合Biopac CSV: {biopac_csv_path}')

    # 保存HUB数据为独立CSV文件，时间戳第一列
    for key, df in data[experiment_name]['hub'].items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            # 提取时间戳列并移到第一列
            columns = ['timestamp'] + [col for col in df.columns if col != 'timestamp']
            df_reordered = df[columns]
            hub_csv_path = os.path.join(csv_dir, f'{experiment_name}_hub_{key}_aligned.csv')
            df_reordered.to_csv(hub_csv_path, index=False)
            print(f'  保存HUB CSV: {hub_csv_path}')

print('✅ 数据查看和保存完成！')