import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

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
    """
    处理DataFrame中的重复时间戳，通过插值分配微小间隔。

    参数:
        df: 输入DataFrame
        time_col: 时间戳列名（默认: 'timestamp'）
    返回:
        具有唯一时间戳的DataFrame
    """
    df = df.copy()
    unique_times = df[time_col].unique()

    new_timestamps = []
    for t in unique_times:
        indices = df[df[time_col] == t].index
        n_points = len(indices)

        if n_points > 1:
            current_idx = np.where(unique_times == t)[0][0]
            if current_idx == len(unique_times) - 1:
                delta = 0.001  # 最后一个时间戳使用默认小间隔
            else:
                next_t = unique_times[current_idx + 1]
                delta = (next_t - t) / n_points

            for i, idx in enumerate(indices):
                new_timestamps.append((t + i * delta, idx))
        else:
            new_timestamps.append((t, indices[0]))

    new_timestamps.sort(key=lambda x: x[1])
    df[time_col] = [t for t, _ in new_timestamps]
    return df

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

    interp_func = interp1d(time, data, axis=0, fill_value='extrapolate', bounds_error=False)
    interpolated_data = interp_func(reftime)
    return pd.DataFrame(interpolated_data, columns=data.columns, index=reftime.index)

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
    # 按数字顺序排序
    experiment_folders.sort(key=lambda x: int(x))
    print(f'从 {folder_path} 加载PI-Lab数据...')
    print(f'实验文件夹: {experiment_folders}')

    data_dict = {}
    for experiment in experiment_folders:
        experiment_path = os.path.join(folder_path, experiment)
        print(f'加载实验 {experiment}...')
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
            for file in os.listdir(biopac_path):
                file_path = os.path.join(biopac_path, file)
                if not os.path.isfile(file_path):
                    continue
                print(f'加载Biopac文件 {file}...')
                try:
                    data = pd.read_csv(file_path)
                    if data.empty:
                        print(f"警告: {file_path} 为空，跳过。")
                        continue
                    data = interpolate_duplicate_timestamps(data)
                    # 假设Biopac文件名格式为<signal>-<exp_id>.csv
                    file_key = file.split('-')[0]
                    if file_key in data_dict[experiment]['biopac']:
                        data_dict[experiment]['biopac'][file_key] = data
                except Exception as e:
                    print(f"读取 {file_path} 出错: {e}，跳过。")

        # 加载HUB数据
        hub_path = os.path.join(experiment_path, 'HUB')
        if os.path.isdir(hub_path):
            for file in os.listdir(hub_path):
                file_path = os.path.join(hub_path, file)
                if not os.path.isfile(file_path):
                    continue
                print(f'加载HUB文件 {file}...')
                try:
                    data = pd.read_csv(file_path)
                    if data.empty:
                        print(f"警告: {file_path} 为空，跳过。")
                        continue
                    data = interpolate_duplicate_timestamps(data)
                    file_key = file.replace('.csv', '')
                    if file_key in data_dict[experiment]['hub']:
                        data_dict[experiment]['hub'][file_key] = data
                except Exception as e:
                    print(f"读取 {file_path} 出错: {e}，跳过。")

        # 按时间戳排序
        for key in data_dict[experiment]['biopac']:
            if isinstance(data_dict[experiment]['biopac'][key], pd.DataFrame) and not data_dict[experiment]['biopac'][key].empty:
                data_dict[experiment]['biopac'][key] = data_dict[experiment]['biopac'][key].sort_values(by='timestamp').reset_index(drop=True)

        for key in data_dict[experiment]['hub']:
            if isinstance(data_dict[experiment]['hub'][key], pd.DataFrame) and not data_dict[experiment]['hub'][key].empty:
                data_dict[experiment]['hub'][key] = data_dict[experiment]['hub'][key].sort_values(by='timestamp').reset_index(drop=True)

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

    for experiment in data_dict:
        print(f'为实验 {experiment} 对齐数据...')
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

        # 对齐Biopac信号
        for key in data_dict[experiment]['biopac']:
            data = data_dict[experiment]['biopac'][key]
            if isinstance(data, pd.DataFrame) and not data.empty and 'timestamp' in data.columns:
                columns_to_interpolate = [col for col in data.columns if col != 'timestamp']
                if columns_to_interpolate:
                    interpolated_data = interpolate_with_reftime(data['timestamp'], data[columns_to_interpolate], reftime)
                    interpolated_data['timestamp'] = reftime
                    aligned_data[experiment]['biopac'][key] = interpolated_data
                else:
                    aligned_data[experiment]['biopac'][key] = pd.DataFrame(columns=data.columns)
            else:
                aligned_data[experiment]['biopac'][key] = pd.DataFrame(columns=['timestamp'] + ([key] if key != 'bp' else ['bp']))

        # 对齐HUB信号
        for key in data_dict[experiment]['hub']:
            data = data_dict[experiment]['hub'][key]
            if isinstance(data, pd.DataFrame) and not data.empty and 'timestamp' in data.columns:
                columns_to_interpolate = [col for col in data.columns if col != 'timestamp' and col != 'time']
                if columns_to_interpolate:
                    interpolated_data = interpolate_with_reftime(data['timestamp'], data[columns_to_interpolate], reftime)
                    interpolated_data['timestamp'] = reftime
                    aligned_data[experiment]['hub'][key] = interpolated_data
                else:
                    aligned_data[experiment]['hub'][key] = pd.DataFrame(columns=data.columns)
            else:
                aligned_data[experiment]['hub'][key] = pd.DataFrame(columns=['timestamp', 'red', 'ir', 'green', 'ax', 'ay', 'az', 'rx', 'ry', 'rz', 'mx', 'my', 'mz', 'temp'])

        # 为当前实验单独保存对齐数据
        output_path = os.path.join(output_dir, f'experiment_{experiment}_aligned.npy')
        np.save(output_path, {experiment: aligned_data[experiment]})
        print(f'已将实验 {experiment} 的对齐数据保存到 {output_path}')

    return aligned_data

def main():
    # 定义路径
    pi_lab_folder = '/root/PI_Lab/00017'

    # 加载数据
    data_dict = load_pi_lab_data(pi_lab_folder)

    # 对齐数据并逐个实验保存
    aligned_data = align_pi_lab_data(data_dict)

    # 验证对齐数据
    print('对齐数据实验编号:', list(aligned_data.keys()), len(aligned_data.keys()))
    for experiment in aligned_data:
        for key in aligned_data[experiment]['biopac']:
            if isinstance(aligned_data[experiment]['biopac'][key], pd.DataFrame) and not aligned_data[experiment]['biopac'][key].empty:
                print(f"{experiment} biopac {key}: {len(aligned_data[experiment]['biopac'][key])} 行")
        for key in aligned_data[experiment]['hub']:
            if isinstance(aligned_data[experiment]['hub'][key], pd.DataFrame) and not aligned_data[experiment]['hub'][key].empty:
                print(f"{experiment} hub {key}: {len(aligned_data[experiment]['hub'][key])} 行")

if __name__ == "__main__":
    main()