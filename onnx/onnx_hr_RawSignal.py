import os
import pandas as pd
import numpy as np
import onnxruntime as ort
from scipy.signal import butter, filtfilt

# ONNX Runtime session
onnx_model = "/root/RingTool/export_onnx/resnet_hr_fold1.onnx"
sess = ort.InferenceSession(
    onnx_model,
    providers=["CPUExecutionProvider"]
)

def bandpass_filter(data, lowcut=0.5, highcut=3, fs=100, order=3):
    """Apply a bandpass filter to the data with specified frequency range."""
    if fs is None or fs <= 0:
        return np.zeros_like(data)
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def standardize(data):
    """Standardize data to zero mean and unit variance."""
    mean_val = np.mean(data)
    std_val = np.std(data)
    return np.zeros_like(data) if std_val == 0 else (data - mean_val) / std_val

def process_and_infer(csv_file):
    """Process raw CSV data and perform ONNX inference to predict heart rate."""
    # Load data
    df = pd.read_csv(csv_file)
    ir_data = df['ir'].values
    timestamps = df['timestamp'].values

    # Force sampling rate to 100 Hz as per your modification
    fs = 100
    print(f"Using fixed sampling rate: {fs:.2f} Hz")

    # Extract segments (30 seconds, target 3000 points)
    target_size = 3000
    window_size = int(30 * fs)  # 30 seconds * 100 Hz = 3000 points
    segments = []
    for i in range(0, len(ir_data) - window_size + 1, window_size):
        segment = ir_data[i:i + window_size]
        if len(segment) == window_size:  # Ensure segment length is correct
            # Standardize the segment
            standardized_segment = standardize(segment)
            # Apply bandpass filter (0.5-3 Hz for heart rate range)
            filtered_segment = bandpass_filter(standardized_segment, lowcut=0.5, highcut=3, fs=fs, order=3)
            # Pad or truncate to 3000 points
            if len(filtered_segment) > target_size:
                padded_segment = filtered_segment[:target_size]
            else:
                padded_segment = np.pad(filtered_segment, (0, target_size - len(filtered_segment)), 'constant')
            segments.append(padded_segment)

    if not segments:
        print(f"No valid segments found in {csv_file}")
        return

    # Adjust to model input format (N, 3000, 1)
    X = np.stack(segments).astype(np.float32)
    print(f"Input shape: {X.shape}")
    X = X[..., None]  # (N, 3000, 1)

    # ONNX inference
    y_pred = sess.run(None, {"ir_ppg": X})[0]  # Get output
    print(f"Output shape before squeeze: {y_pred.shape}")

    # Dynamic handle output shape
    if y_pred.shape[-1] == 1:
        y_pred = y_pred.squeeze(-1)  # If last dim is 1, remove
    elif len(y_pred.shape) == 1:
        y_pred = y_pred  # If already (N,), no change
    else:
        raise ValueError(f"Unexpected output shape: {y_pred.shape}")

    # Output results
    print(f"{os.path.basename(csv_file)}: Total windows = {len(y_pred)}")
    for i, pred in enumerate(y_pred[:5]):  # Print first 5 windows' predictions
        print(f"Window {i+1}: Predicted HR = {pred:.2f} bpm")

if __name__ == "__main__":
    # Example path
    csv_file = "/root/RingTool/RingDataRaw/Daily/bwy/0/Ring1/signals.csv"
    if os.path.exists(csv_file):
        process_and_infer(csv_file)
    else:
        print(f"File not found: {csv_file}")