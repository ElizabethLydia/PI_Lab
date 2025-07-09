# 批量分割PPG片段
import os
import numpy as np
import matplotlib.pyplot as plt
from neurokit2 import ppg_peaks, ppg_quality
from tqdm import tqdm
from scipy.signal import find_peaks,welch, butter, filtfilt
from scipy.sparse import spdiags, diags, eye
from scipy.sparse.linalg import spsolve 
from scipy.interpolate import interp1d
import pdb
import pandas as pd

def extract_and_process_segments(df, channels=['ir', 'red', 'ax', 'ay', 'az'], interval=30, overlap=0):
    """
    Extracts signal segments from multiple channels in a DataFrame and processes each segment.
    
    Args:
        df: DataFrame with 'timestamp' and signal channels (like 'ir', 'red', etc.)
        channels: List of channel names to process (default: ['ir', 'red'])
        interval: Time interval in seconds for each segment (default: 10 seconds)
        overlap: Overlap time in seconds between segments (default: 0 seconds)
    
    Returns:
        Dictionary with structure: {channel: {process_type: [(start_time, end_time, processed_data), ...], ...}, ...}
        where process_type is one of: 'raw', 'standardized', 'filtered', 'difference', 'welch'
    """
    # 如果不是DataFrame，抛出异常返回空字典
    if not isinstance(df, pd.DataFrame):
        print("Input data is not a DataFrame. Returning empty dictionary.")
        return {}
        
        
    if 'timestamp' not in df.columns:
        raise ValueError("DataFrame must contain 'timestamp' column")
    
    for channel in channels:
        if channel not in df.columns:
            raise ValueError(f"DataFrame must contain '{channel}' column")
    
    timestamps = df['timestamp'].values
    
    result = {}
    for channel in channels:
        channel_data = df[channel].values
        
        # Extract segments using the original function
        segments = extract_signal_segments(channel_data, timestamps, interval, overlap)
        
        # Initialize dictionary to store processed data for this channel
        if channel in ['ir', 'red']:
            result[channel] = {
                'raw': [],
                'standardized': [],
                'filtered': [],
                'difference': [],
                'welch': [],
                'quality': []
            }
        else:
            result[channel] = {
                'raw': [],
                'standardized': []
            }
        
        for id, (start_time, end_time, segment) in enumerate(segments):
            # Calculate sampling rate for this segment
            fs = calculate_sampling_rate(timestamps[(timestamps >= start_time) & (timestamps <= end_time)])
            # print(f"Processing segment from {start_time} to {end_time} with fs={fs}")
            # if fs<=10, skip:
            if fs <= 10:
                print(f"Skipping segment due to low sampling rate: {fs} Hz")
                continue
            
            # Process the segment
            raw, standardized, filtered, difference, welch = preprocess_data(segment, fs=fs)
            
            if channel in ['ir', 'red']:
                # Store the processed segments
                result[channel]['raw'].append((id, start_time, end_time, raw))
                result[channel]['standardized'].append((id, start_time, end_time, standardized))
                result[channel]['filtered'].append((id, start_time, end_time, filtered))
                result[channel]['difference'].append((id, start_time, end_time, difference))
                result[channel]['welch'].append((id, start_time, end_time, welch))
                try:
                    quality = single_signal_quality_assessment(filtered, fs=fs)
                except:
                    quality = 0
                result[channel]['quality'].append((id, start_time, end_time, quality))
            else:
                # Store the processed segments for other channels
                result[channel]['raw'].append((id, start_time, end_time, raw))
                result[channel]['standardized'].append((id, start_time, end_time, standardized))
        # 转化为DataFrame，加上列名id,start,end,segment:raw/standardized/filtered/difference/welch
        for process_type in result[channel]:
            # Create a DataFrame where each row contains id, start time, end time, and signal data
            # Note: The signal data (in the last column) will be stored as arrays in DataFrame cells
            result[channel][process_type] = pd.DataFrame(
            result[channel][process_type], 
            columns=['id', 'start', 'end', process_type]
            )
            
    return result

def extract_signal_segments(data, timestamps, interval=10, overlap=0):
    """
    Extracts signal segments based on the specified time interval with optional overlap.
    
    Args:
        data: The signal data array
        timestamps: Array of timestamps
        interval: Time interval in seconds (default: 10 seconds)
        overlap: Overlap time in seconds between segments (default: 0 seconds)
    
    Returns:
        List of tuples with (start_timestamp, end_timestamp, segment_data)
    """
    segments = []
    if len(timestamps) != len(data):
        raise ValueError("Data and timestamps must have the same length")
    
    i = 0
    while i < len(timestamps):
        start_idx = i
        start_time = timestamps[start_idx]
        
        # Find the end index for this segment
        end_idx = start_idx
        while end_idx < len(timestamps) and timestamps[end_idx] - start_time < interval:
            end_idx += 1
        
        # Ensure we don't go out of bounds
        if end_idx >= len(timestamps):
            end_idx = len(timestamps) - 1
        
        end_time = timestamps[end_idx]
        
        # Only add the segment if it covers at least 80% of the desired interval
        if end_time - start_time >= 0.8 * interval:
            segment = data[start_idx:end_idx+1]
            segments.append((start_time, end_time, segment))
        
        # Move to the next starting position, considering overlap
        next_start_time = start_time + interval - overlap
        i = start_idx
        while i < len(timestamps) and timestamps[i] < next_start_time:
            i += 1
        
        # If we couldn't advance, prevent infinite loop
        if i == start_idx:
            i += 1
    
    return segments



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
    
def calculate_sampling_rate(timestamps):
    """Calculate the sampling rate based on the time difference between consecutive timestamps."""
    time_diff = np.diff(timestamps)
    # Assuming the signal is regularly sampled, the mean of time differences gives the sampling interval
    return 1 / np.mean(time_diff)

def diff_normalize_label(label):
    """Calculate discrete difference in labels along the time-axis and normalize by its standard deviation."""
    diff_label = np.diff(label, axis=0)
    diffnormalized_label = diff_label / np.std(diff_label)
    diffnormalized_label = np.append(diffnormalized_label, np.zeros(1), axis=0)
    diffnormalized_label[np.isnan(diffnormalized_label)] = 0
    return diffnormalized_label

def bandpass_filter(data, lowcut=0.5, highcut=3, fs=30, order=3):
    b, a = butter(order, [lowcut, highcut], fs=fs, btype='band')  # Using Butterworth filter to filt wave frequency between 0.5, 3 Hz (30 ~ 180 BPM).
    return filtfilt(b, a, data)

def get_hr(y, fs=30, min=30, max=180):
    p, q = welch(y, fs, nfft=int(1e5/fs), nperseg=np.min((len(y)-1, 512)))
    return p[(p>min/60)&(p<max/60)][np.argmax(q[(p>min/60)&(p<max/60)])]*60 # Using welch method to caculate PSD and find the peak of it.


def welch_(x, fs, window='hann', nperseg=None, noverlap=None, nfft=None):
    if nperseg is not None and len(x) < nperseg:
        nperseg = 2 ** int(np.log2(len(x)))
        print(f'Warning: nperseg is too large, set to {nperseg}')
        
        # Also adjust noverlap if it's too large
        if noverlap is not None and noverlap >= nperseg:
            noverlap = nperseg // 2  # Set noverlap to half of nperseg
            print(f'Warning: noverlap adjusted to {noverlap}')
    
    # Match output length to input signal length
    if nfft is None:
        nfft = len(x)
    
    f, Pxx = welch(x, fs, window=window, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
    
    # Filter to only include 0.5-3Hz range
    mask = (f >= 0.5) & (f <= 3)
    f_filtered = f[mask]
    Pxx_filtered = Pxx[mask]
    
    # Interpolate to match the input signal length if needed
    if len(f_filtered) != len(x):
        f_full = np.linspace(f_filtered.min(), f_filtered.max(), len(x))
        interp_func = interp1d(f_filtered, Pxx_filtered, kind='linear', bounds_error=False, fill_value=0)
        Pxx_filtered = interp_func(f_full)
        f_filtered = f_full
    
    return f_filtered, Pxx_filtered


def compute_time_domain_hrv(ppg_signal, fs):
    """
    根据 PPG 信号计算时间域 HRV 指标
    参数:
        ppg_signal: 1D 数组，PPG 信号数据
        fs: 采样频率（Hz）
    返回:
        包含时间域指标的字典：
            - mean_rr: RR 间期均值（秒）
            - sdnn: RR 间期标准差（秒）
            - rmssd: 邻接 RR 间期差值均方根（秒）
            - nn50: 相邻 RR 间期差值>50ms的个数
            - pnn50: NN50 占比（%）
    """
    # 检测 PPG 信号中的峰值，设定一个合理的最小峰间隔
    # 例如：如果心率不可能超过 200 bpm，则峰间距至少 60/200 秒
    min_distance = int(fs * 60 / 200)
    peaks, _ = find_peaks(ppg_signal, distance=min_distance)
    
    if len(peaks) < 2:
        raise ValueError("检测到的峰值过少，无法计算 HRV 指标。")
    
    # 计算相邻峰值之间的时间间隔（RR 间期），单位为秒
    rr_intervals = np.diff(peaks) / fs
    
    # 时间域指标计算
    mean_rr = np.mean(rr_intervals)  # 单位：秒，表示心跳间隔的平均时长
    sdnn = np.std(rr_intervals, ddof=1)  # 单位：秒，表示RR间期的标准差，反映总体HRV
    rmssd = np.sqrt(np.mean(np.diff(rr_intervals)**2))  # 单位：秒，反映短期HRV变异性
    nn50 = np.sum(np.abs(np.diff(rr_intervals)) > 0.05)  # 单位：计数，相邻心跳间期差值>50ms的数量
    pnn50 = (nn50 / len(rr_intervals)) * 100  # 单位：百分比，NN50占总RR间期数的百分比
    
    hrv_time = {
        'mean_rr': mean_rr,  # 单位: 秒
        'sdnn': sdnn,        # 单位: 秒
        'rmssd': rmssd,      # 单位: 秒
        'nn50': nn50,        # 单位: 计数
        'pnn50': pnn50       # 单位: %
    }
    return hrv_time

def preprocess_data(data, fs=100):
    # 0. standardize
    standardize_data = (data - np.mean(data)) / np.std(data)
    
    # 1. Bandpass filter
    filtered_data = bandpass_filter(standardize_data, lowcut=0.5, highcut=3, fs=fs)
    
    # 2. difference
    difference_data = diff_normalize_label(filtered_data)
    
    # 3. Welch method
    f, welch_data = welch_(filtered_data, fs=fs, window='hann', nperseg=512, noverlap=256)
    
    return data, standardize_data, filtered_data, difference_data, welch_data

def get_value_from_timestamp(data, start_time, end_time):
    # data: timestamp, value， 从数据中获取start_time和end_time之间的值
    # start_time: 开始时间
    # end_time: 结束时间
    # 返回值：start_time和end_time之间的值
    if data is None or len(data) == 0:
        return None
    
    # Convert start_time and end_time to float for comparison with timestamp
    try:
        start_time_float = float(start_time)
        end_time_float = float(end_time)
        
        mask = (data['timestamp'] >= start_time_float) & (data['timestamp'] <= end_time_float)
        duration_data = data[mask].copy()
        
        if len(duration_data) == 0:
            return None
        
        return duration_data
    except (ValueError, TypeError) as e:
        print(f"Error converting timestamps: {e}")
        return None
    

def preprocess_label(data, segments):
    '''
    already have label:
    bvp: timestamp,bvp
    hr: timestamp,hr
    spo2: timestamp,spo2
    resp: timestamp,resp
    ecg: timestamp,ecg
    ecg_hr: timestamp,ecg_hr
    ecg_rr: timestamp,ecg_rr
    BP: timestamp,sys,dia
    
    adding new label:
    bvp_hr: timestamp,hr
    bvp_hrv: timestamp,mean_rr,sdnn,rmssd,nn50,pnn50
    resp_rr: timestamp,rr
    
    根据segments里面的start_time和end_time来分割数据，生成对应的标签：
    hr: start,end,hr
    bvp_hr: start,end,hr
    bvp_hrv: start,end,mean_rr,sdnn,rmssd,nn50,pnn50
    resp_rr: start,end,rr
    spo2: start,end,spo2
    '''
    
    hr, bvp_hr, bvp_hrv, resp_rr, spo2 = [], [], [], [], []
    print(segments)
    pdb.set_trace()
    for segment_data in segments:
        # Each segment data contains (id, start_time, end_time, segment_values)
        id, start_time, end_time, segment = segment_data
        try:
            
            # 计算平均心率
            hr_data = get_value_from_timestamp(data.get('hr'), start_time, end_time)
            mean_hr = np.mean(hr_data['hr']) if hr_data is not None and len(hr_data) > 0 else None
            
            # 计算平均血氧饱和度
            spo2_data = get_value_from_timestamp(data.get('spo2'), start_time, end_time)
            mean_spo2 = np.mean(spo2_data['spo2']) if spo2_data is not None and len(spo2_data) > 0 else None
            
            # bvp计算平均心率
            bvp_data = get_value_from_timestamp(data.get('bvp'), start_time, end_time)
            mean_bvp_hr = None
            bvp_hrv_values = None
            
            if bvp_data is not None and len(bvp_data) > 0:
                bvp_fs = calculate_sampling_rate(bvp_data['timestamp'])
                bvp_value = bvp_data['bvp']
                if len(bvp_value) > 1:  # Ensure there's enough data for HR calculation
                    mean_bvp_hr = get_hr(bvp_value, fs=bvp_fs)
                    # 计算bvp_hrv
                    try:
                        bvp_hrv_values = compute_time_domain_hrv(bvp_value, fs=bvp_fs)
                    except ValueError as e:
                        print(f"Error computing HRV: {e}")
                        bvp_hrv_values = None
            
            # 计算resp_rr
            mean_resp_rr = None
            resp_data = get_value_from_timestamp(data.get('resp'), start_time, end_time)
            if resp_data is not None and len(resp_data) > 0:
                resp_fs = calculate_sampling_rate(resp_data['timestamp'])
                resp_values = resp_data['resp']
                # 对于不在0-255的异常resp_rr过滤出去
                valid_resp = resp_values[(resp_values > -1) & (resp_values < 256)]
                if len(valid_resp) > 1:  # Ensure there's enough data
                    mean_resp_rr = get_hr(valid_resp, fs=resp_fs, min=6, max=30)
            
            # Append the results
            hr.append((id, start_time, end_time, mean_hr))
            bvp_hr.append((id, start_time, end_time, mean_bvp_hr))
            bvp_hrv.append((id, start_time, end_time, bvp_hrv_values))
            resp_rr.append((id, start_time, end_time, mean_resp_rr))
            spo2.append((id, start_time, end_time, mean_spo2))
            
        except Exception as e:
            print(f"Error processing segment {start_time}-{end_time}: {e}")
            # Continue to next segment instead of failing
            continue
        
    # 转化为DataFrame，加上列名id,start,end,segment:hr/bvp_hr/bvp_hrv/resp_rr/spo2
    hr = pd.DataFrame(hr, columns=['id', 'start', 'end', 'hr'])
    bvp_hr = pd.DataFrame(bvp_hr, columns=['id', 'start', 'end', 'bvp_hr'])
    # Expand HRV dictionary values into separate columns
    bvp_hrv_expanded = []
    for id_val, start_val, end_val, hrv_dict in bvp_hrv:
        if hrv_dict is None:
            # Handle None values
            bvp_hrv_expanded.append([id_val, start_val, end_val, None, None, None, None, None])
        else:
            # Extract individual metrics from the dictionary
            bvp_hrv_expanded.append([
                id_val, 
                start_val, 
                end_val, 
                hrv_dict.get('mean_rr'), 
                hrv_dict.get('sdnn'), 
                hrv_dict.get('rmssd'), 
                hrv_dict.get('nn50'), 
                hrv_dict.get('pnn50')
            ])
    
    # Create DataFrame with separate columns for all HRV metrics
    bvp_hrv = pd.DataFrame(
        bvp_hrv_expanded, 
        columns=['id', 'start', 'end', 'mean_rr', 'sdnn', 'rmssd', 'nn50', 'pnn50']
    )
    resp_rr = pd.DataFrame(resp_rr, columns=['id', 'start', 'end', 'resp_rr'])
    spo2 = pd.DataFrame(spo2, columns=['id', 'start', 'end', 'spo2'])
            
    return hr, bvp_hr, bvp_hrv, resp_rr, spo2


    '''
    subject
            -ring
            -ir/red
                -raw/standardized/filtered/difference/welch
                    -segments
                        -id,start,end,segment
                -quality
            -ax/ay/az
                -raw/standardized
        -hr/bvp_hr/bvp_hrv/resp_rr/spo2
            -segements
                -id,start,end,segment:hr/bvp_hr/bvp_hrv/resp_rr/spo2
        -samsung/oura
            -start,end,hr
        -BP
            -start,end, sys,dia
        -Experiments
            -start,end,Experiment:Health, Daily, Sport
        -Labels
            -start,end,label 
    每个Segment画一张有2*8=16个子图的大图，包含2通道ir/red的raw,standilized,filtered,difference,welch和3通道的ax,ay,az的raw,standilized
    在大标题上显示hr/bvp_hr/bvp_hrv/resp_rr/spo2，如果此时有samsung/oura/BP/Experiment/Labels，显示在小标题上
    有可能出现某些元素缺省，使用空图代表或者空值代表，但需要保证每个segment都有图
    '''

def filter_by_time(data, start_time, end_time):
    """ 根据时间范围筛选数据 """
    return [entry for entry in data if is_time_overlap(entry['start'], entry['end'], start_time, end_time)]

def is_time_overlap(start1, end1, start2, end2):
    """ 判断两个时间段是否有交集 """
    return not (end1 <= start2 or start1 >= end2)

def visualize_data_gt(subject, start_time, end_time, ring_data, ring_type='ring1'):
    '''
    subject: 用户ID，如S1-S57
    start_time, end_time: 时间段，用于过滤数据
    ring_data: 数据字典，包含各种信息，包括ring1, ring2, hr, bvp_hr, bvp_hrv等
    '''
    
    # 获取该用户的数据
    subject_data = ring_data

    # 定义可能的图形标题
    fig, axes = plt.subplots(2, 8, figsize=(20, 10))  # 2行8列的子图
    fig.suptitle(f"Subject: {subject} | HR/BVP/HRV/Resp/SPO2", fontsize=16)
    
    # 获取子图中的每个坐标
    ax = axes.ravel()

    # 为每个Segment画图
    # try:
    if 1:
        # 处理IR/RED信号的raw, standardized, filtered, difference, welch
        ir_data = subject_data.get('ring', {}).get('ir', {})
        red_data = subject_data.get('ring', {}).get('red', {})
        # print(ir_data)

        plot_types = ['raw', 'standardized', 'filtered', 'difference', 'welch']
        for i, plot_type in enumerate(plot_types):
            # 获取IR/RED数据
            ir_data_filtered = filter_by_time(ir_data.get(plot_type, []), start_time, end_time)
            red_data_filtered = filter_by_time(red_data.get(plot_type, []), start_time, end_time)
            print(ir_data_filtered)
            # 如果IR/RED数据存在，绘制
            if ir_data_filtered:
                ax[i].plot(ir_data_filtered[0]['start'], ir_data_filtered[0]['end'])  # Example plotting
                ax[i].set_title(f"IR {plot_type.capitalize()}")
            else:
                ax[i].axis('off')  # 如果没有数据，则关闭子图
                
            if red_data_filtered:
                ax[i+5].plot(red_data_filtered[0]['start'], red_data_filtered[0]['end'])
                ax[i+5].set_title(f"Red {plot_type.capitalize()}")
            else:
                ax[i+5].axis('off')  # 如果没有数据，则关闭子图
                
        # 处理加速度数据
        ax_offset = 10  # 初始加速度图的偏移量
        acceleration_types = ['raw', 'standardized']
        for i, acc_type in enumerate(acceleration_types):
            # 获取加速度数据
            ax[i+ax_offset].plot(subject_data.get('ring', {}).get('ax', {}).get(acc_type, []))
            ax[i+ax_offset].set_title(f"Ax {acc_type.capitalize()}")
            ax[i+ax_offset+1].plot(subject_data.get('ring', {}).get('ay', {}).get(acc_type, []))
            ax[i+ax_offset+1].set_title(f"Ay {acc_type.capitalize()}")
            ax[i+ax_offset+2].plot(subject_data.get('ring', {}).get('az', {}).get(acc_type, []))
            ax[i+ax_offset+2].set_title(f"Az {acc_type.capitalize()}")
            
        # 处理HR/BVP/HRV等标签数据
        label_types = ['hr', 'bvp_hr', 'bvp_hrv', 'resp_rr', 'spo2']
        for i, label in enumerate(label_types):
            label_data = subject_data.get(label, [])
            # 筛选时间段内的标签片段
            label_data_filtered = filter_by_time(label_data, start_time, end_time)
            print(label_data_filtered)
            
            if label_data_filtered:
                ax[i+ax_offset+3].plot(label_data_filtered[0]['start'], label_data_filtered[0]['end'])
                ax[i+ax_offset+3].set_title(f"{label.upper()}")
            

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        # Make sure the output directory exists
        os.makedirs(f"/home/disk2/disk/3/tjk/OutputImages/{ring_type}", exist_ok=True)
        plt.savefig(f"/home/disk2/disk/3/tjk/OutputImages/{ring_type}/{subject}_{start_time}_{end_time}.png")
        plt.close(fig)  # Close the figure to save memory
    # except Exception as e:
    #     print(f"Error visualizing data for subject {subject}: {e}")
    #     plt.close(fig)

def save_to_file(filename):
    # 存储为两个Ring1.npy和Ring2.npy文件，统一为字典的数据格式，一级key是用户S1-S57，二级key是raw, standilized, filtered, difference, welch，hr, bvp_hr, bvp_hrv, resp_rr, spo2,Samsung，oura，BP，Experiment，Label，Label下三级Key是二级标签
    '''
    6. ring1: timestamp,green,ir,red,ax,ay,az
    ring2: timestamp,green,ir,red,ax,ay,az
    bvp: timestamp,bvp
    hr: timestamp,hr
    spo2: timestamp,spo2
    resp: timestamp,resp
    ecg: timestamp,ecg
    ecg_hr: timestamp,ecg_hr
    ecg_rr: timestamp,ecg_rr
    Samsung: timestamp,hr
    oura: timestamp,hr
    BP: timestamp,sys,dia
    Experiment: Health, Daily, Sport
    Labels: timestamp,edit_timestamp,action (start，end, label)
    '''
    ring_data = np.load(filename+'ring_data.npy', allow_pickle=True).item()
    print('Merged data subjects:', list(ring_data.keys()), 'Total:', len(ring_data.keys()))
    ring1_data = {}
    ring2_data = {}
    
    # Using tqdm to show processing progress
    for subject in tqdm(ring_data.keys(), desc="Processing subjects"):
        print(f"Processing subject: {subject}")
        ring_subject = ring_data[subject]
        
        ring1_data[subject] = {}
        ring2_data[subject] = {}
        
        # Process Ring1 data if it exists
        # try:
        ring1_raw_data = ring_subject['ring1']
        ring1_data[subject]['ring'] = extract_and_process_segments(ring1_raw_data, channels=['ir', 'red', 'ax', 'ay', 'az'], interval=30, overlap=0)
        ring1_data[subject]['hr'], ring1_data[subject]['bvp_hr'], ring1_data[subject]['bvp_hrv'], ring1_data[subject]['resp_rr'], ring1_data[subject]['spo2'] = preprocess_label(ring_subject, ring1_data[subject]['ring']['ir']['raw'])
        # except:
            # print(f"No Ring1 data found for subject {subject}")
            # ring1_data[subject] = {}
            
        # Process Ring2 data if it exists
        try:
            ring2_raw_data = ring_subject['ring2']
            ring2_data[subject]['ring'] = extract_and_process_segments(ring2_raw_data, channels=['ir', 'red', 'ax', 'ay', 'az'], interval=30, overlap=0)
            ring2_data[subject]['hr'], ring2_data[subject]['bvp_hr'], ring2_data[subject]['bvp_hrv'], ring2_data[subject]['resp_rr'], ring2_data[subject]['spo2'] = preprocess_label(ring_subject, ring2_data[subject]['ring']['ir']['raw'])
        except:
            print(f"No Ring2 data found for subject {subject}")
            ring2_data[subject] = {}
        
        # Add additional data to non-empty subject dictionaries
        if ring1_data[subject]:
            ring1_data[subject]['samsung'] = ring_subject.get('samsung')
            ring1_data[subject]['oura'] = ring_subject.get('oura')
            ring1_data[subject]['BP'] = ring_subject.get('BP')
            ring1_data[subject]['Experiment'] = ring_subject.get('Experiment')
            ring1_data[subject]['Labels'] = ring_subject.get('Labels')
            
        if ring2_data[subject]:
            ring2_data[subject]['samsung'] = ring_subject.get('samsung')
            ring2_data[subject]['oura'] = ring_subject.get('oura')
            ring2_data[subject]['BP'] = ring_subject.get('BP')
            ring2_data[subject]['Experiment'] = ring_subject.get('Experiment')
            ring2_data[subject]['Labels'] = ring_subject.get('Labels')
        # 寻找时间戳画图可视化
        for id,start_time,end_time,segment in ring1_data[subject]['ring']['ir']['raw']:
            visualize_data_gt(subject, start_time, end_time, ring1_data[subject], ring_type='ring1')
        for id,start_time,end_time,segment in ring2_data[subject]['ring']['ir']['raw']:
            visualize_data_gt(subject, start_time, end_time, ring2_data[subject], ring_type='ring2')
        

    
    # save to file
    ring1_filename = filename + 'ring1.npy'
    ring2_filename = filename + 'ring2.npy'
    print(f"Saving processed data to {ring1_filename} and {ring2_filename}")
    np.save(ring1_filename, ring1_data)
    np.save(ring2_filename, ring2_data)
    return ring1_filename, ring2_filename
        
file_folder = '/home/disk2/disk/3/tjk/RingData/'
save_to_file(file_folder)



    
    
    
    