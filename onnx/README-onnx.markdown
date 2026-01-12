# Heart Rate Prediction with ONNX Model

This project provides a Python script for running heart rate predictions using a pre-trained ResNet1D model exported to ONNX format. The script processes input data from processed ring sensor pickle files and generates heart rate predictions.

## Prerequisites

- Required packages:

  ```bash
  pip install numpy pandas onnxruntime
  ```

## Files

- `onnx_resnet.py`: Core inference script containing the `ONNXInference` class and functions to process input data and generate heart rate predictions.
- `resnet_hr_fold1.onnx`: Pre-trained ONNX model file (currently using Fold-1; future work may involve averaging predictions across five folds for improved results).
- Input data: Processed pickle files containing an 'ir-filtered' column with ring sensor data.

## Setup

1. Ensure the ONNX model file (`resnet_hr_fold1.onnx`) is existed.
2. Place input pickle files in `/root/RingTool/RingDataProcessed/rings/`. For example, use `bwy_ring1_processed.pkl` as the input file.
3. Install required dependencies:

   ```bash
   pip install numpy pandas onnxruntime
   ```
4. Activate the appropriate conda environment (if used):

   ```bash
   conda activate RingTool
   ```

## Usage

1. **Run inference on a single file**:

   ```bash
   python onnx_resnet.py
   ```

   This script processes the input file specified in the script (e.g., `bwy_ring1_processed.pkl`) and saves predictions to `/root/RingTool/RingDataProcessed/rings/bwy_hr_predictions.pkl`.

2. **Using the inference module in your code**:

   ```python
   from onnx_resnet import process_single_file
   from pathlib import Path
   
   model_path = Path("/root/RingTool/export_onnx/resnet_hr_fold1.onnx")
   data_path = Path("/root/RingTool/RingDataProcessed/rings/bwy_ring1_processed.pkl")
   output_path = Path("/root/RingTool/RingDataProcessed/rings/bwy_hr_predictions.pkl")
   
   result = process_single_file(data_path, model_path, output_path)
   print(result["hr"].head().round(1))
   ```

## Input Data Format

- Input files must be pickle files containing a pandas DataFrame.
- The DataFrame must include an 'ir-filtered' column with 1D numpy arrays of IR sensor data.
- Input data is automatically padded or truncated to 3000 samples to match the model's expected input shape.

## Output

- Predictions are saved as a new pickle file with an additional 'hr' column containing heart rate predictions.
- The output DataFrame retains all original columns plus the new 'hr' column.

## Notes

- The model expects input data with shape (N, 3000, 1), where N is the number of samples.
- Predictions are returned as a 1D array of heart rate values.
- The script uses the CPU execution provider for ONNX Runtime. To use CUDA, modify the `providers` argument in `onnx_resnet.py` to include `"CUDAExecutionProvider"`.
- If you encounter errors (e.g., reshape errors), verify the input file exists at the specified path and contains a valid 'ir-filtered' column with 1D numpy arrays. You can debug by adding `print([a.shape for a in df["ir-filtered"]])` in the `process_single_file` function.
- Future improvements may include averaging predictions across five folds of the ResNet1D model for enhanced accuracy.