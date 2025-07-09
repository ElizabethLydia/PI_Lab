# raw的数据处理
# 对所有重复时间戳进行去重复的操作，时间复杂度为O(n^2)，非常慢
import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from tqdm import tqdm

# 定义数据结构
sample_data = {
    'biopac': {
        'bp': [],
        'systolic_bp': [],
        'diastolic_bp': [],
        'mean_bp': [],
        'hr': [],
        'cardiac_output': [],
        'cardiac_index': [],
        'systemic_vascular_resistance': [],
        'rsp': []
    },
    'hub': {
        'sensor2': [],
        'sensor3': [],
        'sensor4': [],
        'sensor5': []
    }
}

def interpolate_duplicate_timestamps(df, time_col='timestamp'):
    if df.empty or time_col not in df.columns:
        return df
    
    # 转换为 NumPy 数组
    timestamps = df[time_col].to_numpy()
    unique_times, counts = np.unique(timestamps, return_counts=True)
    new_timestamps = np.zeros_like(timestamps)
    
    # 处理重复时间戳
    for t, count in zip(unique_times, counts):
        if count > 1:
            idx = np.where(timestamps == t)[0]
            current_idx = np.where(unique_times == t)[0][0]
            if current_idx < len(unique_times) - 1:
                delta = (unique_times[current_idx + 1] - t) / count
            else:
                delta = 0.001
            new_timestamps[idx] = np.linspace(t, t + (count - 1) * delta, count)
        else:
            new_timestamps[timestamps == t] = t
    
    df = df.copy()
    df[time_col] = new_timestamps
    return df.sort_values(by=time_col).reset_index(drop=True)

def interpolate_with_reftime(time, data, reftime):
    """
    将数据插值对齐到参考时间戳。

    参数:
        time: 待插值数据的时间戳（Series）
        data: 待插值的数据（DataFrame）
        reftime: 参考时间戳（Series）
    返回:
        对齐到reftime的插值DataFrame
    """
    if len(time) < 2 or len(data) < 2:
        return pd.DataFrame(columns=data.columns)

    # 使用安全的插值参数
    min_time, max_time = time.min(), time.max()
    clipped_reftime = np.clip(reftime, min_time, max_time)
    
    interp_func = interp1d(time, data, axis=0, bounds_error=False, fill_value=np.nan)
    interpolated_data = interp_func(clipped_reftime)
    return pd.DataFrame(interpolated_data, columns=data.columns, index=reftime.index)

def downsample_data(df, target_freq=100, time_col='timestamp'):
    """
    对数据进行降采样，目标频率为100Hz（保守安全的降采样方法）

    参数:
        df: 输入DataFrame
        target_freq: 目标采样率（默认100Hz）
        time_col: 时间戳列名（默认'timestamp'）
    返回:
        降采样后的DataFrame
    """
    if df.empty or time_col not in df.columns:
        return df

    original_len = len(df)
    
    try:
        # 估算当前采样率
        timestamps = df[time_col].values
        if len(timestamps) > 1:
            time_range = timestamps.max() - timestamps.min()
            current_freq = len(timestamps) / time_range if time_range > 0 else target_freq
            
            print(f'    估算频率: {current_freq:.1f}Hz -> 目标: {target_freq}Hz')
            
            # 只有当当前频率明显高于目标频率时才降采样
            if current_freq > target_freq * 1.5:  # 1.5倍安全系数
                step = max(1, int(current_freq / target_freq))
                # 使用步长采样，保持时间顺序
                result = df.iloc[::step].copy()
                print(f'    降采样: {original_len} -> {len(result)} 行 (步长: {step})')
                return result
            else:
                print(f'    频率已足够低，跳过降采样')
                return df
        else:
            return df
            
    except Exception as e:
        print(f'    降采样失败: {e}，返回原数据')
        return df

def load_pi_lab_data(folder_path):
    """
    从PI-Lab数据集加载数据，忽略非数字文件夹并按实验编号排序。

    参数:
        folder_path: PI_Lab/00017目录路径
    返回:
        data_dict: 包含所有实验数据的字典
    """
    # 获取所有文件夹并过滤非数字文件夹
    all_folders = os.listdir(folder_path)
    experiment_folders = [f for f in all_folders if f.isdigit() and os.path.isdir(os.path.join(folder_path, f))]
    experiment_folders.sort(key=lambda x: int(x))
    print(f'从 {folder_path} 加载PI-Lab数据...')
    print(f'找到 {len(experiment_folders)} 个实验文件夹: {experiment_folders}')

    data_dict = {}
    for experiment in tqdm(experiment_folders, desc="加载实验"):
        experiment_path = os.path.join(folder_path, experiment)
        print(f'\n开始加载实验 {experiment}...')
        data_dict[experiment] = {
            'biopac': {
                'bp': [],
                'systolic_bp': [],
                'diastolic_bp': [],
                'mean_bp': [],
                'hr': [],
                'cardiac_output': [],
                'cardiac_index': [],
                'systemic_vascular_resistance': [],
                'rsp': []
            },
            'hub': {
                'sensor2': [],
                'sensor3': [],
                'sensor4': [],
                'sensor5': []
            }
        }

        # 加载Biopac数据
        biopac_path = os.path.join(experiment_path, 'Biopac')
        if os.path.isdir(biopac_path):
            biopac_files = [f for f in os.listdir(biopac_path) if f.endswith('.csv')]
            print(f'找到 {len(biopac_files)} 个Biopac文件: {biopac_files}')
            for file in tqdm(biopac_files, desc=f"加载Biopac文件 (实验 {experiment})"):
                file_path = os.path.join(biopac_path, file)
                print(f'处理Biopac文件: {file_path}')
                try:
                    data = pd.read_csv(file_path)
                    print(f'原始行数: {len(data)}')
                    if data.empty:
                        print(f"警告: {file_path} 为空，跳过。")
                        continue
                    data = interpolate_duplicate_timestamps(data)
                    print(f'处理重复时间戳后行数: {len(data)}')
                    # 降采样Biopac数据
                    data = downsample_data(data, target_freq=100)
                    print(f'降采样后行数: {len(data)}')
                    file_key = file.split('-')[0]
                    if file_key in data_dict[experiment]['biopac']:
                        data_dict[experiment]['biopac'][file_key] = data
                        print(f'成功加载 {file_key} 数据')
                except Exception as e:
                    print(f"读取 {file_path} 出错: {e}，跳过。")

        # 加载HUB数据
        hub_path = os.path.join(experiment_path, 'HUB')
        if os.path.isdir(hub_path):
            hub_files = [f for f in os.listdir(hub_path) if f.endswith('.csv')]
            print(f'找到 {len(hub_files)} 个HUB文件: {hub_files}')
            for file in tqdm(hub_files, desc=f"加载HUB文件 (实验 {experiment})"):
                file_path = os.path.join(hub_path, file)
                print(f'处理HUB文件: {file_path}')
                try:
                    data = pd.read_csv(file_path)
                    print(f'原始行数: {len(data)}')
                    if data.empty:
                        print(f"警告: {file_path} 为空，跳过。")
                        continue
                    data = interpolate_duplicate_timestamps(data)
                    print(f'处理重复时间戳后行数: {len(data)}')
                    file_key = file.replace('.csv', '')
                    if file_key in data_dict[experiment]['hub']:
                        data_dict[experiment]['hub'][file_key] = data
                        print(f'成功加载 {file_key} 数据')
                except Exception as e:
                    print(f"读取 {file_path} 出错: {e}，跳过。")

        # 按时间戳排序
        for key in data_dict[experiment]['biopac']:
            if isinstance(data_dict[experiment]['biopac'][key], pd.DataFrame) and not data_dict[experiment]['biopac'][key].empty:
                data_dict[experiment]['biopac'][key] = data_dict[experiment]['biopac'][key].sort_values(by='timestamp').reset_index(drop=True)
                print(f'Biopac {key} 数据按时间戳排序完成，行数: {len(data_dict[experiment]["biopac"][key])}')

        for key in data_dict[experiment]['hub']:
            if isinstance(data_dict[experiment]['hub'][key], pd.DataFrame) and not data_dict[experiment]['hub'][key].empty:
                data_dict[experiment]['hub'][key] = data_dict[experiment]['hub'][key].sort_values(by='timestamp').reset_index(drop=True)
                print(f'HUB {key} 数据按时间戳排序完成，行数: {len(data_dict[experiment]["hub"][key])}')

    return data_dict

def align_pi_lab_data(data_dict):
    """
    将所有信号对齐到HUB sensor2.csv时间戳，并为每个实验单独保存。

    参数:
        data_dict: 包含原始PI-Lab数据的字典
    返回:
        aligned_data: 对齐后的数据字典
    """
    aligned_data = {}
    output_dir = '/root/PI_Lab/output'
    os.makedirs(output_dir, exist_ok=True)

    for experiment in tqdm(data_dict.keys(), desc="对齐实验"):
        print(f'\n开始为实验 {experiment} 对齐数据...')
        aligned_data[experiment] = {
            'biopac': {
                'bp': [],
                'systolic_bp': [],
                'diastolic_bp': [],
                'mean_bp': [],
                'hr': [],
                'cardiac_output': [],
                'cardiac_index': [],
                'systemic_vascular_resistance': [],
                'rsp': []
            },
            'hub': {
                'sensor2': [],
                'sensor3': [],
                'sensor4': [],
                'sensor5': []
            }
        }

        # 使用HUB sensor2时间戳作为参考
        ref_data = data_dict[experiment]['hub'].get('sensor2', pd.DataFrame())
        if ref_data.empty or 'timestamp' not in ref_data.columns:
            print(f"警告: 实验 {experiment} 无有效sensor2数据，跳过对齐。")
            continue
        reftime = ref_data['timestamp']
        print(f'使用 sensor2 作为参考时间轴，长度: {len(reftime)}')

        # 对齐Biopac信号
        print('开始对齐Biopac信号...')
        for key in tqdm(data_dict[experiment]['biopac'], desc=f"对齐Biopac信号 (实验 {experiment})"):
            data = data_dict[experiment]['biopac'][key]
            print(f'处理Biopac信号: {key}, 原始行数: {len(data) if isinstance(data, pd.DataFrame) else 0}')
            if isinstance(data, pd.DataFrame) and not data.empty and 'timestamp' in data.columns:
                columns_to_interpolate = [col for col in data.columns if col != 'timestamp']
                if columns_to_interpolate:
                    interpolated_data = interpolate_with_reftime(data['timestamp'], data[columns_to_interpolate], reftime)
                    interpolated_data['timestamp'] = reftime
                    aligned_data[experiment]['biopac'][key] = interpolated_data
                    print(f'对齐后 {key} 行数: {len(interpolated_data)}')
                else:
                    aligned_data[experiment]['biopac'][key] = pd.DataFrame(columns=data.columns)
                    print(f'{key} 无有效列可插值')
            else:
                aligned_data[experiment]['biopac'][key] = pd.DataFrame(columns=['timestamp'] + ([key] if key != 'bp' else ['bp']))
                print(f'{key} 数据为空或无时间戳')

        # 对齐HUB信号
        print('开始对齐HUB信号...')
        for key in tqdm(data_dict[experiment]['hub'], desc=f"对齐HUB信号 (实验 {experiment})"):
            data = data_dict[experiment]['hub'][key]
            print(f'处理HUB信号: {key}, 原始行数: {len(data) if isinstance(data, pd.DataFrame) else 0}')
            if isinstance(data, pd.DataFrame) and not data.empty and 'timestamp' in data.columns:
                columns_to_interpolate = [col for col in data.columns if col != 'timestamp' and col != 'time']
                if columns_to_interpolate:
                    interpolated_data = interpolate_with_reftime(data['timestamp'], data[columns_to_interpolate], reftime)
                    interpolated_data['timestamp'] = reftime
                    aligned_data[experiment]['hub'][key] = interpolated_data
                    print(f'对齐后 {key} 行数: {len(interpolated_data)}')
                else:
                    aligned_data[experiment]['hub'][key] = pd.DataFrame(columns=data.columns)
                    print(f'{key} 无有效列可插值')
            else:
                aligned_data[experiment]['hub'][key] = pd.DataFrame(columns=['timestamp', 'red', 'ir', 'green', 'ax', 'ay', 'az', 'rx', 'ry', 'rz', 'mx', 'my', 'mz', 'temp'])
                print(f'{key} 数据为空或无时间戳')

        # 为当前实验单独保存对齐数据
        output_path = os.path.join(output_dir, f'experiment_{experiment}_aligned.pkl')
        import pickle
        with open(output_path, 'wb') as f:
            pickle.dump({experiment: aligned_data[experiment]}, f)
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # 转换为MB
        print(f'已将实验 {experiment} 的对齐数据保存到 {output_path}, 文件大小: {file_size:.2f} MB')

    return aligned_data

def main():
    # 定义路径
    pi_lab_folder = '/root/PI_Lab/00017'

    # 加载数据
    print('开始加载PI-Lab数据...')
    data_dict = load_pi_lab_data(pi_lab_folder)

    # 对齐数据并逐个实验保存
    print('开始对齐数据...')
    aligned_data = align_pi_lab_data(data_dict)

    # 验证对齐数据
    print('\n验证对齐数据...')
    print('对齐数据实验编号:', list(aligned_data.keys()), len(aligned_data.keys()))
    for experiment in aligned_data:
        print(f'\n实验 {experiment} 数据详情:')
        for key in aligned_data[experiment]['biopac']:
            if isinstance(aligned_data[experiment]['biopac'][key], pd.DataFrame) and not aligned_data[experiment]['biopac'][key].empty:
                print(f"{experiment} biopac {key}: {len(aligned_data[experiment]['biopac'][key])} 行")
        for key in aligned_data[experiment]['hub']:
            if isinstance(aligned_data[experiment]['hub'][key], pd.DataFrame) and not aligned_data[experiment]['hub'][key].empty:
                print(f"{experiment} hub {key}: {len(aligned_data[experiment]['hub'][key])} 行")

if __name__ == "__main__":
    main()