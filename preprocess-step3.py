'''重构数据结构，统一转化到一个DataFrame里，目标格式，如果该数据不存在，显示为None
-id,start,end,fs,ir-raw,ir-standardized,ir-filtered,ir-difference,ir-welch,red-raw,red-standardized,red-filtered,red-difference,red-welch,ax-raw,ax-standardized,ay-raw,ay-standardized,az-raw,az-standardized,ir-quality,red-quality,hr,bvp_hr,bvp_sdnn,bvp_rmssd,bvp_nn50,bvp_pnn50,resp_rr,spo2,samsung_hr,oura_hr,BP_sys,BP_dia,Experiment,Label
其中为ir-raw,ir-standardized,ir-filtered,ir-difference,ir-welch,red-raw,red-standardized,red-filtered,red-difference,red-welch,ax-raw,ax-standardized,ay-raw,ay-standardized,az-raw,az-standardized为np.array，Experiment,Label为str，其他均为float
'''
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, welch, butter, filtfilt
from scipy.interpolate import interp1d
from neurokit2 import ppg_peaks, ppg_quality
from tqdm import tqdm
import pdb

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


def calculate_sampling_rate(timestamps):
    """Calculate the sampling rate based on the time difference between consecutive timestamps."""
    if len(timestamps) < 2:
        return None
    time_diff = np.diff(timestamps)
    # Filter out any negative or zero values that would cause division by zero
    valid_diffs = time_diff[time_diff > 0]
    if len(valid_diffs) == 0:
        return None
    return 1 / np.mean(valid_diffs)

def bandpass_filter(data, lowcut=0.5, highcut=3, fs=30, order=3):
    """Apply a bandpass filter to the data."""
    if fs is None or fs <= 0:
        return np.zeros_like(data)
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def diff_normalize_label(label):
    """Calculate discrete difference in labels along the time-axis and normalize by its standard deviation."""
    if len(label) <= 1:
        return np.zeros_like(label)
    diff_label = np.diff(label, axis=0)
    std_val = np.std(diff_label)
    if std_val == 0:
        diffnormalized_label = np.zeros_like(diff_label)
    else:
        diffnormalized_label = diff_label / std_val
    diffnormalized_label = np.append(diffnormalized_label, np.zeros(1), axis=0)
    diffnormalized_label[np.isnan(diffnormalized_label)] = 0
    return diffnormalized_label

def get_hr(y, fs=30, min=30, max=180):
    p, q = welch(y, fs, nfft=int(1e5/fs), nperseg=np.min((len(y)-1, 512)))
    return p[(p>min/60)&(p<max/60)][np.argmax(q[(p>min/60)&(p<max/60)])]*60 # Using welch method to caculate PSD and find the peak of it.

def welch_spectrum(x, fs, window='hann', nperseg=None, noverlap=None, nfft=None, min_value=30, max_value=180):
    """Calculate the Welch power spectrum."""
    if fs is None or fs <= 0 or len(x) < 2:
        return np.array([]), np.array([])
    
    if nperseg is not None and len(x) < nperseg:
        nperseg = 2 ** int(np.log2(len(x)))
        if nperseg < 2:
            nperseg = len(x)
        
        # Adjust noverlap if it's too large
        if noverlap is not None and noverlap >= nperseg:
            noverlap = nperseg // 2
    
    # Use appropriate nfft
    if nfft is None:
        nfft = max(256, 2 ** int(np.log2(len(x))))
    
    try:
        f, Pxx = welch(x, fs, window=window, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
        
        # Filter to only include 0.5-3Hz range (common PPG frequency band)
        mask = (f >= min_value/60) & (f <= max_value/60)
        f_filtered = f[mask]
        Pxx_filtered = Pxx[mask]
        
        # Interpolate to match the length of the input signal
        if len(Pxx_filtered) > 0:
            # Create interpolation function
            interp_func = interp1d(
                np.linspace(0, 1, len(Pxx_filtered)),
                Pxx_filtered,
                kind='linear',
                bounds_error=False,
                fill_value=(Pxx_filtered[0], Pxx_filtered[-1])
            )
            
            # Generate interpolated values with the same length as x
            x_points = np.linspace(0, 1, len(x))
            Pxx_interpolated = interp_func(x_points)
            
            return f_filtered, Pxx_interpolated
        else:
            # If no data in the filtered range, return zeros
            return f_filtered, np.zeros(len(x))
    
    except Exception as e:
        print(f"Error in welch spectrum calculation: {e}")
        return np.array([]), np.zeros(len(x))

def single_signal_quality_assessment(signal, fs, method_quality='templatematch', method_peaks='elgendi'):
    assert method_quality in ['templatematch', 'dissimilarity'], "method_quality must be one of ['templatematch', 'dissimilarity']"
    

    signal_filtered = signal
    
    # Check if the signal is too short or has no variation
    if len(signal_filtered) < 10 or np.all(signal_filtered == signal_filtered[0]):
        print(f"Warning: Signal is too short or constant. Skipping quality assessment.")
        return 0 # Return a high value indicating poor quality

    if method_quality in ['templatematch', 'dissimilarity']:
        method_quality = 'dissimilarity' if method_quality == 'dissimilarity' else method_quality
        
        try:
            # Attempt peak detection on the filtered signal
            _, peak_info = ppg_peaks(
                signal_filtered,
                sampling_rate=fs,
                method=method_peaks
            )
            
            # If no peaks were detected, return a high quality value
            if peak_info["PPG_Peaks"].size == 0:
                print("No peaks detected in the signal. Skipping quality assessment.")
                return 0

            quality = ppg_quality(
                signal_filtered,
                ppg_pw_peaks=peak_info["PPG_Peaks"],
                sampling_rate=fs,
                method=method_quality
            )
            
            # Calculate mean quality excluding NaN values
            quality = np.nanmean(quality)
        
        except ValueError as e:
            print(f"Error in ppg_quality function: {e}")
            quality = 0
        
        return quality

def compute_time_domain_hrv(ppg_signal, fs):
    """
    Calculate time domain HRV metrics from PPG signal.
    
    Args:
        ppg_signal: 1D array of PPG signal data
        fs: Sampling frequency (Hz)
    
    Returns:
        Dictionary containing HRV metrics: mean_rr, sdnn, rmssd, nn50, pnn50
    """
    if fs is None or fs <= 0 or len(ppg_signal) < 2 * fs:  # Need at least 2 seconds of data
        return {'mean_rr': None, 'sdnn': None, 'rmssd': None, 'nn50': None, 'pnn50': None}
    
    try:
        # Detect peaks with minimum distance based on max possible heart rate
        min_distance = int(fs * 60 / 200)  # Minimum distance for 200 bpm
        peaks, _ = find_peaks(ppg_signal, distance=min_distance)
        
        if len(peaks) < 2:
            return {'mean_rr': None, 'sdnn': None, 'rmssd': None, 'nn50': None, 'pnn50': None}
        
        # Calculate RR intervals in seconds
        rr_intervals = np.diff(peaks) / fs
        
        # Filter for physiologically plausible RR intervals (250ms to 1500ms)
        valid_rr = rr_intervals[(rr_intervals >= 0.25) & (rr_intervals <= 1.5)]
        
        if len(valid_rr) < 2:
            return {'mean_rr': None, 'sdnn': None, 'rmssd': None, 'nn50': None, 'pnn50': None}
        
        # Calculate HRV metrics
        mean_rr = np.mean(valid_rr)
        sdnn = np.std(valid_rr, ddof=1)
        rmssd = np.sqrt(np.mean(np.diff(valid_rr)**2))
        nn50 = np.sum(np.abs(np.diff(valid_rr)) > 0.05)
        pnn50 = (nn50 / len(valid_rr)) * 100 if len(valid_rr) > 0 else 0
        
        return {
            'mean_rr': mean_rr,
            'sdnn': sdnn,
            'rmssd': rmssd,
            'nn50': nn50,
            'pnn50': pnn50
        }
    
    except Exception as e:
        print(f"Error computing HRV metrics: {e}")
        return {'mean_rr': None, 'sdnn': None, 'rmssd': None, 'nn50': None, 'pnn50': None}

def preprocess_data(data, fs=100,min_value=30, max_value=180):
    """
    Process raw signal data to extract standardized, filtered, and other features.
    
    Args:
        data: Input signal
        fs: Sampling frequency
    
    Returns:
        Tuple containing (raw, standardized, filtered, difference, welch_spectrum) data
    """
    if data is None or len(data) < 2 or fs is None or fs <= 0:
        return (np.array([]), np.array([]), np.array([]), np.array([]), (np.array([]), np.array([])))
    
    # try:
    if 1:
        # Standardize data
        mean_val = np.mean(data)
        std_val = np.std(data)
        
        if std_val == 0:
            standardize_data = np.zeros_like(data)
        else:
            standardize_data = (data - mean_val) / std_val
        
        # Bandpass filter
        filtered_data = bandpass_filter(standardize_data, lowcut=min_value/60, highcut=max_value/60, fs=fs)
        
        # Calculate difference
        difference_data = diff_normalize_label(filtered_data)
        
        # Calculate Welch spectrum
        nperseg_value = 512 if len(filtered_data) > 513 else len(filtered_data)-1
        f, welch_data = welch_spectrum(filtered_data, fs=fs, window='hann', nperseg=nperseg_value, min_value=min_value, max_value=max_value)
        
        return data, standardize_data, filtered_data, difference_data, (f, welch_data)
    
    # except Exception as e:
    #     print(f"Error in data preprocessing: {e}")
    #     return (np.array([]), np.array([]), np.array([]), np.array([]), (np.array([]), np.array([])))

def extract_signal_segments(df, channels=['ir', 'red', 'ax', 'ay', 'az'], interval=30, overlap=0):
    """
    Extract signal segments from DataFrame with specified time interval and overlap.
    
    Args:
        df: DataFrame containing timestamp and signal channels
        channels: List of channel names to process
        interval: Time interval in seconds for each segment
        overlap: Overlap time in seconds between segments
    
    Returns:
        List of segments with metadata
    """
    if not isinstance(df, pd.DataFrame) or 'timestamp' not in df.columns:
        return []
    
    # Verify channels exist in the dataframe
    valid_channels = [ch for ch in channels if ch in df.columns]
    if not valid_channels:
        return []
    
    timestamps = df['timestamp'].values
    if len(timestamps) < 2:
        return []
    
    # Extract segments based on time intervals
    segments = []
    segment_id = 0
    
    # Define time points for segment starts
    start_time = timestamps[0]
    end_time = timestamps[-1]
    step_time = interval - overlap
    
    current_time = start_time
    while current_time + interval <= end_time:
        # Find closest indices for start and end of this segment
        start_idx = np.argmin(np.abs(timestamps - current_time))
        segment_end_time = current_time + interval
        end_idx = np.argmin(np.abs(timestamps - segment_end_time))
        
        # Ensure we have enough data points
        actual_duration = timestamps[end_idx] - timestamps[start_idx]
        if actual_duration >= 0.8 * interval and end_idx > start_idx:
            segment_data = {
                'id': segment_id,
                'start': timestamps[start_idx],
                'end': timestamps[end_idx],
                'data': {}
            }
            
            # Calculate sampling rate for this segment
            segment_ts = timestamps[start_idx:end_idx+1]
            fs = calculate_sampling_rate(segment_ts)
            segment_data['fs'] = fs
            
            # Skip if sampling rate is too low
            if fs is not None and fs >= 10:
                # Extract data for each channel
                for channel in valid_channels:
                    channel_data = df[channel].values[start_idx:end_idx+1]
                    segment_data['data'][channel] = channel_data
                
                segments.append(segment_data)
                segment_id += 1
        
        # Move to next segment start time
        current_time += step_time
    
    return segments

def process_segments(segments):
    """
    Process extracted segments to create unified DataFrame with specified format.
    
    Args:
        segments: List of segment dictionaries
    
    Returns:
        DataFrame with processed data for all segments
    """
    if not segments:
        return pd.DataFrame()
    
    processed_data = []
    
    for segment in segments:
        segment_id = segment['id']
        start_time = segment['start']
        end_time = segment['end']
        fs = segment['fs']
        
        # Initialize row with required columns
        row = {
            'id': segment_id,
            'start': start_time,
            'end': end_time,
            'fs': fs,
        }
        
        # Process IR channel
        if 'ir' in segment['data']:
            ir_data = segment['data']['ir']
            ir_raw, ir_std, ir_filtered, ir_diff, (_, ir_welch) = preprocess_data(ir_data, fs)
            _,_,ir_filtered_rr, ir_diff_rr, (_, ir_welch_rr) = preprocess_data(ir_data, fs, min_value=6, max_value=30)
            # print(f"IR data length: {len(ir_raw)}, {len(ir_std)}, {len(ir_filtered)}, {len(ir_diff)}, {len(ir_welch)}")
            row['ir-raw'] = ir_raw
            row['ir-standardized'] = ir_std
            row['ir-filtered'] = ir_filtered
            row['ir-difference'] = ir_diff
            row['ir-welch'] = ir_welch
            row['ir-filtered-rr'] = ir_filtered_rr
            row['ir-difference-rr'] = ir_diff_rr
            row['ir-welch-rr'] = ir_welch_rr
            
            # Calculate signal quality
            row['ir-quality'] = single_signal_quality_assessment(ir_filtered, fs) if len(ir_filtered) > 0 else 0
        else:
            row['ir-raw'] = None
            row['ir-standardized'] = None
            row['ir-filtered'] = None
            row['ir-difference'] = None
            row['ir-welch'] = None
            row['ir-quality'] = None
        
        # Process RED channel
        if 'red' in segment['data']:
            red_data = segment['data']['red']
            red_raw, red_std, red_filtered, red_diff, (_, red_welch) = preprocess_data(red_data, fs)
            _,_,red_filtered_rr, red_diff_rr, (_, red_welch_rr) = preprocess_data(red_data, fs, min_value=6, max_value=30)
            row['red-raw'] = red_raw
            row['red-standardized'] = red_std
            row['red-filtered'] = red_filtered
            row['red-difference'] = red_diff
            row['red-welch'] = red_welch
            row['red-filtered-rr'] = red_filtered_rr
            row['red-difference-rr'] = red_diff_rr
            row['red-welch-rr'] = red_welch_rr
            
            # Calculate signal quality
            row['red-quality'] = single_signal_quality_assessment(red_filtered, fs) if len(red_filtered) > 0 else 0
        else:
            row['red-raw'] = None
            row['red-standardized'] = None
            row['red-filtered'] = None
            row['red-difference'] = None
            row['red-welch'] = None
            row['red-quality'] = None
        
        # Process accelerometer channels
        for axis in ['ax', 'ay', 'az']:
            if axis in segment['data']:
                axis_data = segment['data'][axis]
                axis_raw, axis_std, axis_filter,axis_diff, (_, axis_welch) = preprocess_data(axis_data, fs)
                _,_,axis_filtered_rr, axis_diff_rr, (_, axis_welch_rr) = preprocess_data(axis_data, fs, min_value=6, max_value=30)
                row[f'{axis}-raw'] = axis_raw
                row[f'{axis}-standardized'] = axis_std
                row[f'{axis}-filtered'] = axis_filter
                row[f'{axis}-difference'] = axis_diff
                row[f'{axis}-welch'] = axis_welch
                row[f'{axis}-filtered-rr'] = axis_filtered_rr
                row[f'{axis}-difference-rr'] = axis_diff_rr
                row[f'{axis}-welch-rr'] = axis_welch_rr
            else:
                row[f'{axis}-raw'] = None
                row[f'{axis}-standardized'] = None
                row[f'{axis}-filtered'] = None
                row[f'{axis}-difference'] = None
                row[f'{axis}-welch'] = None
                row[f'{axis}-filtered-rr'] = None
                row[f'{axis}-difference-rr'] = None
                row[f'{axis}-welch-rr'] = None
        
        # Initialize other columns with None
        row.update({'bvp':None, 'resp':None,
            'hr': None, 'bvp_hr': None,
            'bvp_sdnn': None, 'bvp_rmssd': None, 'bvp_nn50': None, 'bvp_pnn50': None,
            'resp_rr': None, 'spo2': None, 'samsung_hr': None, 'oura_hr': None,
            'BP_sys': None, 'BP_dia': None, 'Experiment': None, 'Label': None
        })
        
        processed_data.append(row)
    
    return pd.DataFrame(processed_data)

def get_value_for_time_period(data_df, start_time, end_time, value_col):
    """
    Extract values from a DataFrame for a specific time period.
    
    Args:
        data_df: DataFrame with 'timestamp' and value columns
        start_time: Start timestamp
        end_time: End timestamp
        value_col: Column name for the value to extract
    
    Returns:
        Mean value within the time period or None if no data
    """
    if data_df is None or len(data_df) == 0 or 'timestamp' not in data_df.columns or value_col not in data_df.columns:
        return None
    
    try:
        mask = (data_df['timestamp'] >= start_time) & (data_df['timestamp'] <= end_time)
        period_data = data_df.loc[mask, value_col]
        
        if len(period_data) == 0:
            return None
        
        # Filter out non-numeric values and calculate mean
        numeric_data = pd.to_numeric(period_data, errors='coerce')
        valid_data = numeric_data.dropna()
        
        if len(valid_data) == 0:
            return None
        
        return float(valid_data.mean())
    
    except Exception as e:
        print(f"Error extracting {value_col}: {e}")
        return None

def add_physiological_data(segment_df, physiological_data):
    """
    Add physiological data (HR, SPO2, etc.) to segment DataFrame.
    
    Args:
        segment_df: DataFrame with processed segments
        physiological_data: Dictionary containing physiological data DataFrames
    
    Returns:
        Updated segment DataFrame with physiological data
    """
    if segment_df.empty or not physiological_data:
        return segment_df
    
    result_df = segment_df.copy()
    
    for idx, row in result_df.iterrows():
        start_time = row['start']
        end_time = row['end']
        
        # Get heart rate data if available
        try:
            if 'hr' in physiological_data and physiological_data['hr'] is not None:
                if not (isinstance(physiological_data['hr'], dict) and len(physiological_data['hr']) == 0):
                    result_df.at[idx, 'hr'] = get_value_for_time_period(
                    physiological_data['hr'], start_time, end_time, 'hr')
        except Exception as e:
            print(f"Error processing heart rate data: {e}")
            result_df.at[idx, 'hr'] = None
        
        # Get BVP data and calculate HR and HRV
        try:
            bvp_fs = None
            bvp_values = []
            
            if 'bvp' in physiological_data and physiological_data['bvp'] is not None:
                if not (isinstance(physiological_data['bvp'], dict) and len(physiological_data['bvp']) == 0):
                    bvp_data = physiological_data['bvp']
                    bvp_in_period = bvp_data[(bvp_data['timestamp'] >= start_time) & 
                        (bvp_data['timestamp'] <= end_time)]
                    
                    if len(bvp_in_period) > 0:
                        bvp_fs = calculate_sampling_rate(bvp_in_period['timestamp'].values)
                        bvp_values = bvp_in_period['bvp'].values
                
                    # Calculate BVP-derived heart rate
                    if bvp_fs is not None and len(bvp_values) > bvp_fs:
                        result_df.at[idx, 'bvp'] = np.array(bvp_values)
                        result_df.at[idx, 'bvp_hr'] = get_hr(bvp_values, fs=bvp_fs)
                        
                        # Calculate HRV metrics
                        hrv_metrics = compute_time_domain_hrv(bvp_values, fs=bvp_fs)
                        result_df.at[idx, 'bvp_sdnn'] = hrv_metrics['sdnn']
                        result_df.at[idx, 'bvp_rmssd'] = hrv_metrics['rmssd']
                        result_df.at[idx, 'bvp_nn50'] = hrv_metrics['nn50']
                        result_df.at[idx, 'bvp_pnn50'] = hrv_metrics['pnn50']
        except Exception as e:
            print(f"Error processing BVP data: {e}")
            result_df.at[idx, 'bvp_hr'] = None
            result_df.at[idx, 'bvp_sdnn'] = None
            result_df.at[idx, 'bvp_rmssd'] = None
            result_df.at[idx, 'bvp_nn50'] = None
            result_df.at[idx, 'bvp_pnn50'] = None
            result_df.at[idx, 'bvp'] = None
        
        # Get respiratory rate
        try:
            resp_fs = None
            resp_values = []
            
            if 'resp' in physiological_data and physiological_data['resp'] is not None:
                if not (isinstance(physiological_data['resp'], dict) and len(physiological_data['resp']) == 0):
                    resp_data = physiological_data['resp']
                    resp_in_period = resp_data[(resp_data['timestamp'] >= start_time) & 
                        (resp_data['timestamp'] <= end_time)]
                    
                    if len(resp_in_period) > 0:
                        resp_fs = calculate_sampling_rate(resp_in_period['timestamp'].values)
                    resp_values = resp_in_period['resp'].values
                    
                    # Filter valid respiratory values
                    valid_resp = resp_values[(resp_values > -1) & (resp_values < 256)]
                    
                    # Calculate respiratory rate
                    if resp_fs is not None and len(valid_resp) > resp_fs:
                        result_df.at[idx, 'resp'] = np.array(valid_resp)
                        result_df.at[idx, 'resp_rr'] = get_hr(valid_resp, fs=resp_fs, min=6, max=30)
        except Exception as e:
            print(f"Error processing respiratory data: {e}")
            result_df.at[idx, 'resp_rr'] = None
            result_df.at[idx, 'resp'] = None
        
        # Get SpO2
        try:
            if 'spo2' in physiological_data and physiological_data['spo2'] is not None:
                if not (isinstance(physiological_data['spo2'], dict) and len(physiological_data['spo2']) == 0):
                    result_df.at[idx, 'spo2'] = get_value_for_time_period(
                    physiological_data['spo2'], start_time, end_time, 'spo2')
        except Exception as e:
            print(f"Error processing SpO2 data: {e}")
            result_df.at[idx, 'spo2'] = None
        
        # Get Samsung HR
        try:
            if 'samsung' in physiological_data and physiological_data['samsung'] is not None:
                samsung_data = physiological_data['samsung']
            
            # Convert to DataFrame if it's a list
            if isinstance(samsung_data, list):
                if samsung_data:  # Check if list is not empty
                    try:
                        samsung_data = pd.DataFrame(samsung_data)
                    except Exception as e:
                        print(f"Error converting samsung data to DataFrame: {e}")
                        samsung_data = None
                else:
                    samsung_data = None
                
            # Process DataFrame
            if isinstance(samsung_data, pd.DataFrame) and 'end' in samsung_data.columns and 'start' in samsung_data.columns:
                samsung_in_period = samsung_data[(samsung_data['end'] >= start_time) & 
                   (samsung_data['start'] <= end_time)]
                
                if len(samsung_in_period) > 0 and 'hr' in samsung_in_period.columns:
                # Convert hr to numeric, handling potential non-numeric values
                    samsung_in_period['hr'] = pd.to_numeric(samsung_in_period['hr'], errors='coerce')
                    result_df.at[idx, 'samsung_hr'] = samsung_in_period['hr'].mean()
        except Exception as e:
            print(f"Error processing samsung HR data: {e}")
            result_df.at[idx, 'samsung_hr'] = None
        
        # Get Oura HR
        try:
            if 'oura' in physiological_data and physiological_data['oura'] is not None:
                oura_data = physiological_data['oura']
            
            # Convert to DataFrame if it's a list
            if isinstance(oura_data, list):
                if oura_data:  # Check if list is not empty
                    try:
                        oura_data = pd.DataFrame(oura_data)
                    except Exception as e:
                        print(f"Error converting oura data to DataFrame: {e}")
                        oura_data = None
                else:
                    oura_data = None
            
            # Process DataFrame
            if isinstance(oura_data, pd.DataFrame) and 'end' in oura_data.columns and 'start' in oura_data.columns:
                oura_in_period = oura_data[(oura_data['end'] >= start_time) & 
                 (oura_data['start'] <= end_time)]
                
                if len(oura_in_period) > 0 and 'hr' in oura_in_period.columns:
                # Convert hr to numeric, handling potential non-numeric values
                    oura_in_period['hr'] = pd.to_numeric(oura_in_period['hr'], errors='coerce')
                    result_df.at[idx, 'oura_hr'] = oura_in_period['hr'].mean()
        except Exception as e:
            print(f"Error processing oura HR data: {e}")
            result_df.at[idx, 'oura_hr'] = None
        
        # Get blood pressure
        try:
            if 'BP' in physiological_data and physiological_data['BP'] is not None:
                bp_data = physiological_data['BP']
            
            # Convert to DataFrame if it's a list
            if isinstance(bp_data, list):
                if bp_data:  # Check if list is not empty
                    try:
                        bp_data = pd.DataFrame(bp_data)
                    except Exception as e:
                        print(f"Error converting BP data to DataFrame: {e}")
                        bp_data = None
                else:
                    bp_data = None
            
            # Process DataFrame
            if isinstance(bp_data, pd.DataFrame) and 'end' in bp_data.columns and 'start' in bp_data.columns:
                # Get BP measurements from current period plus 1 minute before and after
                extended_start_time = start_time - 60  # 1 minute before
                extended_end_time = end_time + 60     # 1 minute after
                
                bp_in_period = bp_data[
                ((bp_data['end'] >= start_time) & (bp_data['start'] <= end_time)) |  # Current period
                ((bp_data['end'] >= extended_start_time) & (bp_data['start'] <= start_time)) |  # 1 min before
                ((bp_data['end'] >= end_time) & (bp_data['start'] <= extended_end_time))  # 1 min after
                ]
                
                if len(bp_in_period) > 0:
                # Convert sys and dia to numeric, handling potential non-numeric values
                    if 'sys' in bp_in_period.columns:
                        bp_in_period['sys'] = pd.to_numeric(bp_in_period['sys'], errors='coerce')
                        result_df.at[idx, 'BP_sys'] = bp_in_period['sys'].mean()
                    
                    if 'dia' in bp_in_period.columns:
                        bp_in_period['dia'] = pd.to_numeric(bp_in_period['dia'], errors='coerce')
                        result_df.at[idx, 'BP_dia'] = bp_in_period['dia'].mean()
        except Exception as e:
            print(f"Error processing BP data: {e}")
            result_df.at[idx, 'BP_sys'] = None
            result_df.at[idx, 'BP_dia'] = None
            
        # Get experiment type
        try:
            if 'Experiment' in physiological_data and physiological_data['Experiment'] is not None:
                exp_data = physiological_data['Experiment']
            
            # Convert to DataFrame if it's a list
            if isinstance(exp_data, list):
                if exp_data:  # Check if list is not empty
                    try:
                        exp_data = pd.DataFrame(exp_data)
                    except Exception as e:
                        print(f"Error converting Experiment data to DataFrame: {e}")
                        exp_data = None
                else:
                    exp_data = None
            
            # Process DataFrame
            if isinstance(exp_data, pd.DataFrame) and 'end' in exp_data.columns and 'start' in exp_data.columns:
                exp_in_period = exp_data[(exp_data['end'] >= start_time) & 
                (exp_data['start'] <= end_time)]
                
                if len(exp_in_period) > 0 and 'experiment' in exp_in_period.columns:
                # Get most common experiment type in this period
                    result_df.at[idx, 'Experiment'] = exp_in_period['experiment'].mode()[0] if not exp_in_period['experiment'].empty else None
        except Exception as e:
            print(f"Error processing Experiment data: {e}")
            result_df.at[idx, 'Experiment'] = None
        
        # Get labels
        try:
            if 'Labels' in physiological_data and physiological_data['Labels'] is not None:
                label_data = physiological_data['Labels']
                
            # Convert to DataFrame if it's a list
            if isinstance(label_data, list):
                if label_data:  # Check if list is not empty
                    try:
                        label_data = pd.DataFrame(label_data)
                    except Exception as e:
                        print(f"Error converting Labels data to DataFrame: {e}")
                        label_data = None
                else:
                    label_data = None
            
            # Process DataFrame
            if isinstance(label_data, pd.DataFrame) and 'end' in label_data.columns and 'start' in label_data.columns:
                label_in_period = label_data[(label_data['end'] >= start_time) & 
                       (label_data['start'] <= end_time)]
                
                if len(label_in_period) > 0 and 'label' in label_in_period.columns:
                # Get most common label in this period
                    result_df.at[idx, 'Label'] = label_in_period['label'].mode()[0] if not label_in_period['label'].empty else None
        except Exception as e:
            print(f"Error processing Labels data: {e}")
            result_df.at[idx, 'Label'] = None
                
        print(f'samsung, oura, BP, Experiment, Label: {result_df.at[idx, "samsung_hr"]}, {result_df.at[idx, "oura_hr"]}, {result_df.at[idx, "BP_sys"]}, {result_df.at[idx, "Experiment"]}, {result_df.at[idx, "Label"]}')
    return result_df

def visualize_segment(subject, segment_row, output_dir, ring_type):
    """
    Visualize a segment with all its data channels.
    
    Args:
        subject: Subject ID
        segment_row: Row from the DataFrame for this segment
        output_dir: Directory to save the visualization
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    start_time = segment_row['start']
    end_time = segment_row['end']
    segment_id = segment_row['id']
    
    # Create figure with subplots
    fig, axes = plt.subplots(5, 5, figsize=(20, 10))
    fig.suptitle(f"Subject: {subject} | ID: {segment_id} | Time: {start_time:.2f}-{end_time:.2f}", fontsize=16)
    
    # Flatten axes for easier indexing
    ax = axes.ravel()
    
    # Add physiological data info to title if available
    title_parts = []
    for metric in ['hr', 'bvp_hr', 'bvp_sdnn' 'resp_rr', 'spo2', 'samsung_hr', 'oura_hr']:
        if pd.notna(segment_row.get(metric)):
            title_parts.append(f"{metric}: {segment_row[metric]:.1f}")
    
    if title_parts:
        plt.figtext(0.5, 0.92, " | ".join(title_parts), ha='center', fontsize=12)
    
    # BP, Experiment, Label info
    subtitle_parts = []
    if pd.notna(segment_row.get('BP_sys')) and pd.notna(segment_row.get('BP_dia')):
        subtitle_parts.append(f"BP: {segment_row['BP_sys']:.0f}/{segment_row['BP_dia']:.0f}")
    if pd.notna(segment_row.get('Experiment')):
        subtitle_parts.append(f"Exp: {segment_row['Experiment']}")
    if pd.notna(segment_row.get('Label')):
        subtitle_parts.append(f"Label: {segment_row['Label']}")
    
    if subtitle_parts:
        plt.figtext(0.5, 0.89, " | ".join(subtitle_parts), ha='center', fontsize=10)
    
    # Plot IR data
    plot_types = ['raw', 'standardized', 'filtered', 'difference', 'welch']
    for i, plot_type in enumerate(plot_types):
        ir_key = f'ir-{plot_type}'
        if ir_key in segment_row and segment_row[ir_key] is not None and len(segment_row[ir_key]) > 0:
            ax[i].plot(segment_row[ir_key])
            ax[i].set_title(f"IR {plot_type.capitalize()}")
            if plot_type == 'raw' and 'ir-quality' in segment_row and segment_row['ir-quality'] is not None:
                ax[i].set_xlabel(f"Quality: {segment_row['ir-quality']:.2f}")
        else:
            ax[i].set_visible(False)
    
    # Plot RED data
    for i, plot_type in enumerate(plot_types):
        red_key = f'red-{plot_type}'
        if red_key in segment_row and segment_row[red_key] is not None and len(segment_row[red_key]) > 0:
            ax[i+5].plot(segment_row[red_key], color='red')
            ax[i+5].set_title(f"Red {plot_type.capitalize()}")
            if plot_type == 'raw' and 'red-quality' in segment_row and segment_row['red-quality'] is not None:
                ax[i+5].set_xlabel(f"Quality: {segment_row['red-quality']:.2f}")
        else:
            ax[i+5].set_visible(False)
    
    # Plot accelerometer data
    for i, axis in enumerate(['ax', 'ay', 'az']):
        for j, plot_type in enumerate(['raw', 'standardized', 'filtered', 'difference', 'welch']):
            key = f'{axis}-{plot_type}'
            plot_idx = 10 + i * 5 + j
            if plot_idx < len(ax) and key in segment_row and segment_row[key] is not None and len(segment_row[key]) > 0:
                ax[plot_idx].plot(segment_row[key], color='green')
                ax[plot_idx].set_title(f"{axis.upper()} {plot_type.capitalize()}")
            elif plot_idx < len(ax):
                ax[plot_idx].set_visible(False)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    # Save the figure
    output_filename = f"{subject}_{ring_type}_segment{segment_id}_time{start_time:.0f}-{end_time:.0f}.png"
    plt.savefig(os.path.join(output_dir, output_filename))
    plt.close(fig)

def process_ring_data(base_folder, output_folder):
    """
    Process all ring data from the base folder and save unified DataFrames.
    
    Args:
        base_folder: Folder containing ring data files
        output_folder: Folder to save processed data and visualizations
    """
    # Create output folders if they don't exist
    os.makedirs(output_folder, exist_ok=True)
    rings_output = os.path.join(output_folder, "rings")
    vis_output = os.path.join(output_folder, "visualizations")
    os.makedirs(rings_output, exist_ok=True)
    os.makedirs(vis_output, exist_ok=True)
    
    # Load ring data
    ring_data_file = os.path.join(base_folder, 'ring_data.npy')
    try:
        ring_data = np.load(ring_data_file, allow_pickle=True).item()
        print(f'Loaded data for {len(ring_data)} subjects')
    except Exception as e:
        print(f"Error loading ring data: {e}")
        return
    
    # Process each subject
    for subject in tqdm(ring_data.keys(), desc="Processing subjects"):
        print(f"\nProcessing subject: {subject}")
        subject_data = ring_data[subject]
        
        # Process each ring
        for ring_type in ['ring1', 'ring2']:
            if ring_type not in subject_data:
                print(f"No {ring_type} data for subject {subject}")
                continue
            
            # try:
            if 1:
                ring_df = subject_data[ring_type]
                print(f"Processing {ring_type} data for subject {subject}...length: {len(ring_df)}")
                
                # Skip if no data
                if ring_df is None or len(ring_df) == 0 or not isinstance(ring_df, pd.DataFrame):
                    print(f"Empty {ring_type} data for subject {subject}")
                    continue
                
                # Extract segments
                print(f"Extracting segments for {subject} {ring_type}...")
                segments = extract_signal_segments(ring_df, channels=['ir', 'red', 'ax', 'ay', 'az'], interval=30, overlap=0)
                
                if len(segments) == 0:
                    print(f"No valid segments extracted for {subject} {ring_type}")
                    continue
                
                # Process segments
                print(f"Processing {len(segments)} segments...")
                processed_df = process_segments(segments)
                
                if processed_df.empty:
                    print(f"No segments processed successfully for {subject} {ring_type}")
                    continue
                # Add physiological data
                physiological_data = {
                    'hr': subject_data.get('hr'),
                    'bvp': subject_data.get('bvp'),
                    'resp': subject_data.get('resp'),
                    'spo2': subject_data.get('spo2'),
                    'samsung': subject_data.get('samsung'),
                    'oura': subject_data.get('oura'),
                    'BP': subject_data.get('BP'),
                    'Experiment': subject_data.get('Experiment'),
                    'Labels': subject_data.get('Labels')
                }
                processed_df = add_physiological_data(processed_df, physiological_data)
                # Save processed DataFrame
                output_filename = os.path.join(rings_output, f"{subject}_{ring_type}_processed.csv")
                processed_df.to_csv(output_filename, index=False)
                # pkl
                output_filename_pkl = os.path.join(rings_output, f"{subject}_{ring_type}_processed.pkl")
                processed_df.to_pickle(output_filename_pkl)
                
                print(f"Saved processed data for {subject} {ring_type} to {output_filename}")
            # except Exception as e:
            #     print(f"Error processing {subject} {ring_type}: {e}")
            #     continue
            # Visualize segments
            # Visualize segments with tqdm progress bar
            # for _, segment_row in tqdm(processed_df.iterrows(), 
            #                            desc=f"Visualizing segments for {subject} {ring_type}", 
            #                            total=len(processed_df)):
            #     visualize_segment(subject, segment_row, vis_output, ring_type)
                # print(f"Visualized segment {segment_row['id']} for {subject} {ring_type}")
            # pdb.set_trace()
    print("Processing complete.")
    
if __name__ == "__main__":
    base_folder = '/home/disk2/disk/3/tjk/RingData'  # Replace with actual path
    output_folder = '/home/disk2/disk/3/tjk/RingData/PreprocessedV3'  # Replace with actual path
    process_ring_data(base_folder, output_folder)
    print("All done!")