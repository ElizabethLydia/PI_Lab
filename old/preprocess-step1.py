# 数据预处理目标：将所有数据统一时间戳，以Ring1的作为参考值，并且分段存储到npy文件中，一级标签包括：Health, Daily, Sport,二级标签包括：Sit, SpO2, DeepSquat, Talk, HeadRotate, Stand, Walk, Hiit, Recover
# 数据预处理思路：
'''
1. 根据不同的实验来源赋予一级标签，根据文件夹下的编号和Label赋予二级标签
2. 根据ring1 signals统一ring1，ring2，bvp，resp，ecg，hr，spo2时间戳，确定对比的竞品心率和额外的血压标签时间
3. 统一为字典的数据格式，一级key是用户S1-S57，二级key是ring1 signals统一ring1，ring2，bvp，hr，spo2，resp，ecg，ecg_hr,ecg_rr, Samsung，oura，BP，Experiment，Label，Label下三级Key是二级标签，包括每个任务的开始和结束时间的列表，单个标签可能会有多段数据，取决于中间是否有暂停
4. Health：01 Sit, 02 SpO2, 03 DeepSquat, 04 DeepSquat
Daily: 0 Sit, 1 Talk, HeadRotate, Stand, Walk
Sport: TODO
5. 保存为npy文件，每个文件包含一个字典，key是用户编号，value是上述的字典
6. ring1: timestamp,green,ir,red,ax,ay,az
ring2: timestamp,green,ir,red,ax,ay,az
bvp: timestamp,bvp
hr: timestamp,hr
spo2: timestamp,spo2
resp: timestamp,resp
ecg: timestamp,ecg
ecg_hr: timestamp,ecg_hr
ecg_rr: timestamp,ecg_rr
samsung: timestamp,hr
oura: timestamp,hr
BP: timestamp,sys,dia
Experiment: Health, Daily, Sport
Labels: timestamp,edit_timestamp,action (start，end, label)
7. Daily
- subject
    - experiment (0/1:0 Sit, 1 Talk, HeadRotate, Stand, Walk)
        - Oximeter
            - bvp.csv
            - hr.csv
            - spo2.csv
        - Respiration
            - resp.csv
        - Ring1
            - signals.csv
        - Ring2
            - signals.csv
        - labels.csv (timestamp,edit_timestamp,action:start_sync, end_sync, pause, continue, start_sitting, end_sitting, start_talking, end_talking, start_shaking_head, end_shaking_head, start_standing, end_standing, start_striding, end_striding)
8. Health
- subject
    - experiment (01/02/03/04: 01 Sit, 02 SpO2, 03 DeepSquat, 04 DeepSquat)
        - Oximeter
            - bvp.csv
            - hr.csv
            - spo2.csv
        - Respiration
            - resp.csv
        - Ring1
            - signals.csv
        - Ring2
            - signals.csv
        - labels.csv (timestamp,edit_timestamp,action:BP sys dia, oura hr, samsung hr)
9. 进一步处理，分Ring1和Ring2，根据任务选取10s/30s/60s时间戳-对应Label-HRV计算-RR变异性计算，PPG单通道滤波+差分+质量检测，保留仅有部分数据的Subject
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import interp1d
import time
import datetime

# 示例目标数据
sample_data = {
    'ring1': [],
    'ring2': [],
    'bvp': [],
    'hr': [],
    'spo2': [],
    'resp': [],
    'ecg': [],
    'ecg_hr': [],
    'ecg_rr': [],
    'ecg_rri': [],
    'samsung': [],
    'oura': [],
    'BP': [],
    'Experiment': [],
    'Labels': []
}

subject_csv = pd.DataFrame(columns=['S','Label','Folder'])
# 读取Daily数据,从S1开始编号，保留S和文件夹名称
daily_folder = '/home/disk2/disk/3/tjk/RingData/Daily'
daily_folders = os.listdir(daily_folder)

def interpolate_duplicate_timestamps(df, time_col='timestamp'):
    """
    对具有重复时间戳的 dataframe 自动插值，打散相同时间戳的行。
    
    参数:
        df: 原始 DataFrame
        time_col: 时间戳列名，默认是 'timestamp'
        
    返回:
        处理后的 DataFrame，时间戳不再重复
    """
    df = df.copy()
    unique_times = df[time_col].unique()

    new_timestamps = []
    for t in unique_times:
        # 当前时间戳对应的行索引
        indices = df[df[time_col] == t].index
        n_points = len(indices)

        if n_points > 1:
            # 多个点共享同一时间戳
            current_idx = np.where(unique_times == t)[0][0]
            if current_idx == len(unique_times) - 1:
                delta = 0.001  # 最后一个时间点使用默认小间隔
            else:
                next_t = unique_times[current_idx + 1]
                delta = (next_t - t) / n_points

            for i, idx in enumerate(indices):
                new_timestamps.append((t + i * delta, idx))
        else:
            new_timestamps.append((t, indices[0]))

    # 按原始顺序重新排列时间戳
    new_timestamps.sort(key=lambda x: x[1])
    df[time_col] = [t for t, _ in new_timestamps]

    return df

def interpolate_with_reftime(time, data, reftime):
    """
    使用参考数据的时间戳对 data 进行插值。
    
    参数:
        time: 待插值的时间戳，Series 格式
        data: 待插值的数据，DataFrame 格式
        reftime: 参考时间戳，Series 格式
    返回:
        插值后的数据，DataFrame 格式
    """
    # 插值函数
    interp_func = interp1d(time, data, axis=0, fill_value='extrapolate')
    return interp_func(reftime)

def load_daily(folder_path):
    """
    从 Daily 数据文件夹中加载数据。
    
    参数:
        folder_path: Daily 数据文件夹路径
    返回:
        data_dict: 数据字典，包含 Daily 数据
    """
    # 检索到文件夹下所有的subject文件夹
    subject_folders = os.listdir(folder_path)
    data_dict = {}
    print(f'Loading Daily data from {folder_path}...')
    print(subject_folders)
    for subject in subject_folders:
        subject_path = os.path.join(folder_path, subject)
        if not os.path.isdir(subject_path):
            continue
        print(f'Loading subject {subject}...')
        data_dict[subject] = {
            'ring1': [],
            'ring2': [],
            'bvp': [],
            'hr': [],
            'spo2': [],
            'resp': [],
            'ecg': [],
            'ecg_hr': [],
            'ecg_rr': [],
            'ecg_rri': [],
            'samsung': [],
            'oura': [],
            'BP': [],
            'Experiment': [],
            'Labels': []
        }
        # 读取0,1文件夹
        for experiment in os.listdir(subject_path):
            experiment_path = os.path.join(subject_path, experiment)
            if not os.path.isdir(experiment_path):
                continue
            print(f'Loading experiment {experiment}...')
            # 读取Oximeter, Respiration, Ring1, Ring2
            for device in os.listdir(experiment_path):
                device_path = os.path.join(experiment_path, device)
                if not os.path.isdir(device_path):
                    continue
                print(f'Loading device {device}...')
                # 读取bvp, hr, spo2, resp, signals
                for file in os.listdir(device_path):
                    file_path = os.path.join(device_path, file)
                    if not os.path.isfile(file_path):
                        continue
                    print(f'Loading file {file}...')
                    # 读取数据
                    if file == 'signals.csv':
                        data = pd.read_csv(file_path)
                        data = interpolate_duplicate_timestamps(data)
                        # device转为小写
                        data_dict[subject][device.lower()].append(data)
                    elif file == 'bvp.csv':
                        data = pd.read_csv(file_path)
                        data_dict[subject]['bvp'].append(data)
                    elif file == 'hr.csv':
                        data = pd.read_csv(file_path)
                        data_dict[subject]['hr'].append(data)
                    elif file == 'spo2.csv':
                        data = pd.read_csv(file_path)
                        data_dict[subject]['spo2'].append(data)
                    elif file == 'resp.csv':
                        data = pd.read_csv(file_path)
                        data_dict[subject]['resp']. append(data)

            # 读取labels.csv
            labels_file_path = os.path.join(experiment_path, 'labels.csv')
            if os.path.isfile(labels_file_path):
                data_raw = pd.read_csv(labels_file_path)
                # 处理 action 列，将其转换为 start, end, label 格式
                processed_labels = []
                current_label = None
                start_time = None
                paused = False

                for _, row in data_raw.iterrows():
                    action = row['action']
                    timestamp = row['timestamp']
                    # print(action)
                    if action.startswith('start_'):
                        current_label = action.replace('start_', '')
                        start_time = timestamp
                        paused = False
                    elif action == 'pause' and current_label:
                        # 如果当前有活动，记录暂停前的段落
                        processed_labels.append({
                            'start': start_time,
                            'end': timestamp,
                            'label': current_label
                        })
                        paused = True
                    elif action == 'continue' and paused:
                        # 如果继续，则开始新的段落
                        start_time = timestamp
                        paused = False
                    elif action.startswith('end_') and current_label == action.replace('end_', ''):
                        # 如果有暂停，记录最后一段
                        processed_labels.append({
                            'start': start_time,
                            'end': timestamp,
                            'label': current_label
                        })
                        current_label = None
                        start_time = None
                        paused = False

                # 转换为 DataFrame 并存储
                data_dict[subject]['Labels'].append(pd.DataFrame(processed_labels))
            # 对于每个信号，将列表中的数据拼接并按照timestamp排序
        for key in ['ring1', 'ring2', 'bvp', 'hr', 'spo2', 'resp']:
            if data_dict[subject][key]:
                data_dict[subject][key] = pd.concat(data_dict[subject][key], ignore_index=True)
                data_dict[subject][key] = data_dict[subject][key].sort_values(by='timestamp').reset_index(drop=True)
        
        # 对Labels也进行拼接（如果有多个实验的标签）
        if data_dict[subject]['Labels']:
            data_dict[subject]['Labels'] = pd.concat(data_dict[subject]['Labels'], ignore_index=True)
            data_dict[subject]['Labels'] = data_dict[subject]['Labels'].sort_values(by='start').reset_index(drop=True)
    
    return data_dict


# 读取Health数据
health_folder = '/home/disk2/disk/3/tjk/RingData/Health'
health_folders = os.listdir(health_folder)

def load_health(folder_path):
    """
    从 Health 数据文件夹中加载数据。
    
    参数:
        folder_path: Health 数据文件夹路径
    返回:
        data_dict: 数据字典，包含 Health 数据
    """
    # 检索到文件夹下所有的subject文件夹
    subject_folders = os.listdir(folder_path)
    data_dict = {}
    print(f'Loading Health data from {folder_path}...')
    print(subject_folders)
    label_map = {'01': 'sitting', '02': 'spo2', '03': 'deepsquat', '04': 'deepsquat','1': 'sitting', '2': 'spo2', '3': 'deepsquat', '4': 'deepsquat'}
    for subject in subject_folders:
        subject_path = os.path.join(folder_path, subject)
        if not os.path.isdir(subject_path):
            continue
        print(f'Loading subject {subject}...')
        data_dict[subject] = {
            'ring1': [],
            'ring2': [],
            'bvp': [],
            'hr': [],
            'spo2': [],
            'resp': [],
            'ecg': [],
            'ecg_hr': [],
            'ecg_rr': [],
            'ecg_rri': [],
            'samsung': [],
            'oura': [],
            'BP': [],
            'Experiment': [],
            'Labels': []
        }
        # 读取0,1文件夹
        for experiment in os.listdir(subject_path):
            # 根据experiment确定标签和开始结束的时间，并写入到Labels中'start': start_time,'end': timestamp, 'label': current_label
            label_df = pd.DataFrame(columns=['start','end','label'])
        
            experiment_path = os.path.join(subject_path, experiment)
            if not os.path.isdir(experiment_path):
                continue
            print(f'Loading experiment {experiment}...')
            # 读取Oximeter, Respiration, Ring1, Ring2, Labels
            for device in os.listdir(experiment_path):
                device_path = os.path.join(experiment_path, device)
                if not os.path.isdir(device_path):
                    continue
                print(f'Loading device {device}...')
                # 读取bvp, hr, spo2, resp, signals, labels
                for file in os.listdir(device_path):
                    file_path = os.path.join(device_path, file)
                    if not os.path.isfile(file_path):
                        continue
                    print(f'Loading file {file}...')
                    # 读取数据
                    if file == 'signals.csv':
                        try:
                            data = pd.read_csv(file_path)
                            if data.empty:
                                print(f"Warning: {file_path} is empty. Skipping.")
                                continue
                            data = interpolate_duplicate_timestamps(data)
                            data_dict[subject][device.lower()].append(data)
                        except Exception as e:
                            print(f"Error reading {file_path}: {e}. Skipping.")
                        
                        start_timestamp = data['timestamp'][0]
                        end_timestamp = data['timestamp'][len(data)-1]
                        
                        
                        
                    elif file == 'bvp.csv':
                        data = pd.read_csv(file_path)
                        data_dict[subject]['bvp'].append(data)
                    elif file == 'hr.csv':
                        data = pd.read_csv(file_path)
                        data_dict[subject]['hr'].append(data)
                    elif file == 'spo2.csv':
                        data = pd.read_csv(file_path)
                        data_dict[subject]['spo2'].append(data)
                    elif file == 'resp.csv':
                        data = pd.read_csv(file_path)
                        data_dict[subject]['resp'].append(data)
            # if experiment starts with a key in label_map, then match the label
            for key in label_map.keys():
                if experiment.startswith(key):
                    action_label = label_map[key]
                    new_row = pd.DataFrame({
                        'start': [start_timestamp],
                        'end': [end_timestamp],
                        'label': [action_label]
                    })
                    label_df = pd.concat([label_df, new_row], ignore_index=True)
                    print('Loading label:', label_map[key])
                    break
            
            data_dict[subject]['Labels'].append(label_df)
            # 读取labels.csv
            labels_file_path = os.path.join(experiment_path, 'labels.csv')
            if os.path.isfile(labels_file_path):
                try:
                    data_raw = pd.read_csv(labels_file_path)
                    print(data_raw)
                    # 如果data_raw是空的或者只有表头，或者没有'action'列，退出跳过
                    if data_raw.empty or len(data_raw.columns) == 0 or 'action' not in data_raw.columns:
                        print(f"Skipping {labels_file_path} as it contains only headers, is empty, or lacks the 'action' column.")
                        continue
                    
                    # 检查'action'列是否全为空值
                    if data_raw['action'].isnull().all():
                        print(f"Skipping {labels_file_path} as 'action' column contains only null values.")
                        continue
                    
                    # 读取BP sys dia, oura hr, samsung hr，存储为字典
                    # BP data: start,end,sys,dia, sys和dia需要从action中提取，空格分割
                    try:
                        bp_df = data_raw[data_raw['action'].str.contains('BP', na=False)]
                        # print(bp_df)
                        if not bp_df.empty:
                            # 确保data_dict[subject]['BP']是一个列表
                            if not isinstance(data_dict[subject]['BP'], list):
                                data_dict[subject]['BP'] = []
                                
                            for _, row in bp_df.iterrows():
                                try:
                                    action = row['action']
                                    start_timestamp = row['timestamp']
                                    end_timestamp = row['edit_timestamp']
                                    # 从action中提取sys和dia
                                    parts = action.split()
                                    if len(parts) >= 3:
                                        sys = parts[1]
                                        dia = parts[2]
                                        data_dict[subject]['BP'].append({
                                            'start': start_timestamp,
                                            'end': end_timestamp,
                                            'sys': sys,
                                            'dia': dia
                                        })
                                except ValueError:
                                    print(f"Skipping malformed BP action: {action}")
                    except Exception as e:
                        print(f"Error processing BP data: {e}")
                
                    # oura data: start,end,hr
                    try:
                        oura_df = data_raw[data_raw['action'].str.contains('oura', na=False)]
                        # print(oura_df)
                        if not oura_df.empty:
                            # 确保data_dict[subject]['oura']是一个列表
                            if not isinstance(data_dict[subject]['oura'], list):
                                data_dict[subject]['oura'] = []
                                
                            for _, row in oura_df.iterrows():
                                try:
                                    action = row['action']
                                    start_timestamp = row['timestamp']
                                    end_timestamp = row['edit_timestamp']
                                    # 从action中提取hr
                                    parts = action.split()
                                    if len(parts) >= 2:
                                        hr = parts[1]
                                        data_dict[subject]['oura'].append({
                                            'start': start_timestamp,
                                            'end': end_timestamp,
                                            'hr': hr
                                        })
                                except ValueError:
                                    print(f"Skipping malformed oura action: {action}")
                    except Exception as e:
                        print(f"Error processing oura data: {e}")
                
                    # samsung data: start,end,hr
                    try:
                        samsung_df = data_raw[data_raw['action'].str.contains('samsung', na=False)]
                        # print(samsung_df)
                        if not samsung_df.empty:
                            # 确保data_dict[subject]['samsung']是一个列表
                            if not isinstance(data_dict[subject]['samsung'], list):
                                data_dict[subject]['samsung'] = []
                                
                            for _, row in samsung_df.iterrows():
                                try:
                                    action = row['action']
                                    start_timestamp = row['timestamp']
                                    end_timestamp = row['edit_timestamp']
                                    # 从action中提取hr
                                    parts = action.split()
                                    if len(parts) >= 2:
                                        hr = parts[1]
                                        data_dict[subject]['samsung'].append({
                                            'start': start_timestamp,
                                            'end': end_timestamp,
                                            'hr': hr
                                        })
                                except ValueError:
                                    print(f"Skipping malformed samsung action: {action}")
                    except Exception as e:
                        print(f"Error processing samsung data: {e}")
                except Exception as e:
                    print(f"Error reading or processing {labels_file_path}: {e}")
        # 对于每个信号，将列表中的数据拼接并按照timestamp排序
        for key in ['ring1', 'ring2', 'bvp', 'hr', 'spo2', 'resp']:
            if data_dict[subject][key]:
                data_dict[subject][key] = pd.concat(data_dict[subject][key], ignore_index=True)
                data_dict[subject][key] = data_dict[subject][key].sort_values(by='timestamp').reset_index(drop=True)
        
        # 对于可能包含字典的数据结构
        for key in ['Labels','oura','samsung','BP']:
            if data_dict[subject][key]:
                # 检查第一个元素是否为字典类型
                if isinstance(data_dict[subject][key][0], dict):
                    # 将字典列表转换为DataFrame
                    data_dict[subject][key] = pd.DataFrame(data_dict[subject][key])
                else:
                    # 如果是DataFrame列表，则正常拼接
                    data_dict[subject][key] = pd.concat(data_dict[subject][key], ignore_index=True)
                
                # 确保DataFrame有'start'列后再排序
                if 'start' in data_dict[subject][key].columns:
                    data_dict[subject][key] = data_dict[subject][key].sort_values(by='start').reset_index(drop=True)
    return data_dict



# 不预处理，改从文件中读取
data_daily = np.load('/home/disk2/disk/3/tjk/RingData/data_daily.npy', allow_pickle=True).item()
data_health = np.load('/home/disk2/disk/3/tjk/RingData/data_health.npy', allow_pickle=True).item()
# 检查数据中的subject是否完整齐全
print('Daily data subject:',data_daily.keys(),len(data_daily.keys()))
print('Health data subject:',data_health.keys(),len(data_health.keys()))
# 统计每个subject ring1和ring2的行数
for key in data_daily.keys():
    if 'ring1' in data_daily[key]:
        print(key, 'Daily ring1:', len(data_daily[key]['ring1']))
    if 'ring2' in data_daily[key]:
        print(key, 'Daily ring2:', len(data_daily[key]['ring2']))
for key in data_health.keys():
    if 'ring1' in data_health[key]:
        print(key, 'Health ring1:', len(data_health[key]['ring1']))
    if 'ring2' in data_health[key]:
        print(key, 'Health ring2:', len(data_health[key]['ring2']))
# 统计总数
print('Total subject:',set(data_daily.keys()).union(set(data_health.keys())),len(set(data_daily.keys()).union(set(data_health.keys()))))

# 合并data_daily和data_health,对于相同的用户、键值直接拼接，根据时间戳排列，对于Experiment键值start,end,experiment, 进行拼接,start为所有键值最早的时间，end为所有键值最晚的时间
def merge_data(data_daily, data_health):
    """
    合并 Daily 和 Health 数据，处理空数据和没有时间戳的情况。
    
    参数:
        data_daily: Daily 数据字典
        data_health: Health 数据字典
    返回:
        merged_data: 合并后的数据字典
    """
    merged_data = {}
    
    # 获取所有唯一的用户ID
    all_subjects = set(list(data_daily.keys()) + list(data_health.keys()))
    
    for key in all_subjects:
        merged_data[key] = {}
        
        # 获取每个数据集中的数据（如果存在）
        daily_data = data_daily.get(key, {})
        health_data = data_health.get(key, {})
        
        # 处理每种数据类型
        for sub_key in sample_data.keys():
            # 特殊处理 Experiment 数据
            if sub_key == 'Experiment':
                experiment_data = []
                
                # 处理 Daily 实验数据
                if key in data_daily:
                    daily_timestamps = []
                    for data_key in ['ring1', 'ring2', 'bvp', 'hr', 'spo2', 'resp']:
                        if (data_key in daily_data and 
                            isinstance(daily_data[data_key], pd.DataFrame) and 
                            not daily_data[data_key].empty and 
                            'timestamp' in daily_data[data_key].columns):
                            daily_timestamps.extend(daily_data[data_key]['timestamp'].tolist())
                    
                    if daily_timestamps:
                        experiment_data.append({
                            'start': min(daily_timestamps),
                            'end': max(daily_timestamps),
                            'experiment': 'Daily'
                        })
                
                # 处理 Health 实验数据
                if key in data_health:
                    health_timestamps = []
                    for data_key in ['ring1', 'ring2', 'bvp', 'hr', 'spo2', 'resp']:
                        if (data_key in health_data and 
                            isinstance(health_data[data_key], pd.DataFrame) and 
                            not health_data[data_key].empty and 
                            'timestamp' in health_data[data_key].columns):
                            health_timestamps.extend(health_data[data_key]['timestamp'].tolist())
                    
                    if health_timestamps:
                        experiment_data.append({
                            'start': min(health_timestamps),
                            'end': max(health_timestamps),
                            'experiment': 'Health'
                        })
                
                # 创建 Experiment DataFrame
                if experiment_data:
                    merged_data[key]['Experiment'] = pd.DataFrame(experiment_data)
                    merged_data[key]['Experiment'] = merged_data[key]['Experiment'].sort_values(by='start').reset_index(drop=True)
                else:
                    merged_data[key]['Experiment'] = pd.DataFrame(columns=['start', 'end', 'experiment'])
            
            # 处理其他数据类型
            else:
                daily_item = daily_data.get(sub_key, [])
                health_item = health_data.get(sub_key, [])
                
                # 检查两者是否都是 DataFrame
                if isinstance(daily_item, pd.DataFrame) and isinstance(health_item, pd.DataFrame):
                    # 只在两者都有数据的情况下合并
                    daily_has_data = not daily_item.empty
                    health_has_data = not health_item.empty
                    
                    # 检查是否有合适的排序列
                    daily_has_timestamp = daily_has_data and 'timestamp' in daily_item.columns
                    daily_has_start = daily_has_data and 'start' in daily_item.columns
                    health_has_timestamp = health_has_data and 'timestamp' in health_item.columns
                    health_has_start = health_has_data and 'start' in health_item.columns
                    
                    # 合并两个 DataFrame，只要有一个有数据就合并
                    if daily_has_data or health_has_data:
                        merged_df = pd.concat([daily_item, health_item], ignore_index=True)
                        
                        # 根据可用列进行排序
                        if 'timestamp' in merged_df.columns:
                            merged_df = merged_df.sort_values(by='timestamp').reset_index(drop=True)
                        elif 'start' in merged_df.columns:
                            merged_df = merged_df.sort_values(by='start').reset_index(drop=True)
                        
                        merged_data[key][sub_key] = merged_df
                    else:
                        # 两者都为空，保持为 DataFrame
                        merged_data[key][sub_key] = pd.DataFrame()
                
                # 处理两者都是列表的情况
                elif isinstance(daily_item, list) and isinstance(health_item, list):
                    merged_data[key][sub_key] = daily_item + health_item
                
                # 处理混合类型，保留原始类型
                elif isinstance(daily_item, pd.DataFrame):
                    merged_data[key][sub_key] = daily_item
                elif isinstance(health_item, pd.DataFrame):
                    merged_data[key][sub_key] = health_item
                elif isinstance(daily_item, list):
                    merged_data[key][sub_key] = daily_item
                elif isinstance(health_item, list):
                    merged_data[key][sub_key] = health_item
                else:
                    # 未知类型，默认为空列表
                    merged_data[key][sub_key] = []
    
    return merged_data

#merge data
merged_data = merge_data(data_daily, data_health)
# 保存数据
np.save('/home/disk2/disk/3/tjk/RingData/ring_data.npy', merged_data)

ring_data = np.load('/home/disk2/disk/3/tjk/RingData/ring_data.npy', allow_pickle=True).item()
print('Merged data subject:',ring_data.keys(),len(ring_data.keys()))
# 统计每个subject ring1和ring2的行数
for key in ring_data.keys():
    if 'ring1' in ring_data[key]:
        print(key, 'ring1:', len(ring_data[key]['ring1']))
        # 统计Labels的分布
        if 'Labels' in ring_data[key]:
            print(key, 'Labels:', len(ring_data[key]['Labels']))
            # 统计每个标签的数量
            for label in ring_data[key]['Labels']['label'].unique():
                count = len(ring_data[key]['Labels'][ring_data[key]['Labels']['label'] == label])
                print(key, 'Label:', label, 'Count:', count)
    if 'ring2' in ring_data[key]:
        print(key, 'ring2:', len(ring_data[key]['ring2']))