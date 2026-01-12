# %%
# export_onnx/export_resnet_hr.py
import json, torch, pathlib, sys

ROOT = pathlib.Path("/root/RingTool").resolve()
sys.path.insert(0, str(ROOT))                 # 让 Python 找到 nets/

from nets.resnet import ResNet1D

# ---------- 路径 ----------
CFG  = ROOT / "config/supervised/ring1/hr/ir/resnet-ring1-hr-all-ir.json"
CKPT = ROOT / "models/resnet-ring1-hr-all-ir/hr/Fold-1" / \
        "resnet-ring1-hr-all-ir_hr_Fold-1_best.pt"
OUT  = ROOT / "export_onnx/resnet_hr_fold1.onnx"
OUT.parent.mkdir(exist_ok=True)

# ---------- 1) 读取 JSON ----------
cfg  = json.load(open(CFG))
args = cfg["method"]["params"]               # 全量超参 → ResNet1D

# ---------- 2) 构造模型 ----------
model = ResNet1D(**args)

# ---------- 3) 读取权重 ----------
raw   = torch.load(CKPT, map_location="cpu")     # 顶层含 model_state_dict
state = raw["model_state_dict"]
state = {k.replace("module.", ""): v for k, v in state.items()}
model.load_state_dict(state, strict=True)        # 不再报缺键
model.eval()

# ---------- 4) 计算 dummy ----------
fs       = cfg["dataset"]["target_fs"]           # 100
win_sec  = cfg["dataset"]["window_duration"]     # 30
seq_len  = fs * win_sec                          # 3000
dummy = torch.randn(1, seq_len, args["in_channels"])  # (batch, time, channels)

# ---------- 5) 导出 ONNX ----------
torch.onnx.export(
    model, dummy, OUT.as_posix(),
    input_names=["ir_ppg"], output_names=["pred_hr"],
    opset_version=17,
    do_constant_folding=True,
    # time 轴现在是 dim-1
    dynamic_axes={
        "ir_ppg":  {0: "batch", 1: "time"},
        "pred_hr": {0: "batch"}
    }
)
print(f"✅  ONNX saved to {OUT}")


# %% [markdown]
# ## 模型合法性检查

# %%
import onnx
onnx_model = onnx.load("/root/RingTool/export_onnx/resnet_hr_fold1.onnx")
onnx.checker.check_model(onnx_model)   # 无异常即通过
print("ONNX graph is valid")


# %% [markdown]
# ## ONNX Runtime 快速自测

# %%
import onnxruntime as ort, numpy as np
sess = ort.InferenceSession("/root/RingTool/export_onnx/resnet_hr_fold1.onnx",
                            providers=["CUDAExecutionProvider","CPUExecutionProvider"])

dummy = np.random.randn(1, 3000, 1).astype("float32")   # (batch, time, channel)
pred  = sess.run(None, {"ir_ppg": dummy})[0]
print(pred.shape)   # 应为 (1, 1) 或 (1,) 取决于模型最后一层

# %% [markdown]
# ## 只推理不评分

# %%
from pathlib import Path
import numpy as np, pandas as pd, onnxruntime as ort

DATA_DIR   = Path("/root/RingTool/RingDataProcessed/rings")
ONNX_MODEL = Path("/root/RingTool/export_onnx/resnet_hr_fold1.onnx")
TARGET_LEN = 3000

sess = ort.InferenceSession(ONNX_MODEL.as_posix(),
                            providers=["CPUExecutionProvider"])

def to_3000(arr):
    arr = np.asarray(arr, np.float32)
    return arr[:TARGET_LEN] if arr.size>=TARGET_LEN else \
           np.pad(arr, (0, TARGET_LEN-arr.size))

for p in sorted(DATA_DIR.glob("*_ring1_processed.pkl")):
    df = pd.read_pickle(p)
    df.columns = df.columns.str.strip()
    X = np.stack([to_3000(a) for a in df["ir-filtered"]])[..., None]  # (N,3000,1)
    pred = sess.run(None, {"ir_ppg": X})[0].squeeze()
    print(f"{p.name:<30}  windows={len(pred):3}  sample preds={pred[:5].round(1)}")

# 保存预测结果到onnx文件夹，存储为csv格式
    df["hr"] = pred
    df.to_pickle(p.with_name("hr.pkl"))
    print(f"✅  Saved HR predictions to {p.with_name('hr.pkl')}")




