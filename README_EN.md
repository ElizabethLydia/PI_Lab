# 🩺 PI Lab - PPG-based Blood Pressure Estimation

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Research](https://img.shields.io/badge/Research-HCI-orange.svg)]()
[![Status](https://img.shields.io/badge/Status-Completed-green.svg)]()
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC_BY--NC--SA_4.0-lightgrey.svg)](LICENSE)

> Tsinghua University PI Lab internship project: continuous non-invasive blood pressure estimation via Pulse Transit Time (PTT). **Non-commercial academic use only (CC BY-NC-SA 4.0).**

## 📖 Overview

This repo explores multi-sensor photoplethysmography (PPG) to measure Pulse Transit Time (PTT) and model its relationship with blood pressure for wearable, non-invasive continuous monitoring. Methods include physical modeling (Moens–Korteweg) and machine/deep learning (regression, LSTM, CNN-LSTM) with subject-specific calibration.

## 🗂️ Project Layout

```
PI_Lab/
├── step1_preprocess.py              # Preprocess: filtering, quality check
├── step2_ir_ptt_peak_detector.py    # Peak detection & PTT
├── step3_ptt_bp_analysis.py         # PTT–BP relation analysis
├── step4_integrated_ptt_bp_analysis.py  # Integrated modeling
├── data_processor.py                # Core data processing
├── check_results.py                 # Result validation
├── results_summary.py               # Batch result summary
│
├── readme/                          # Module docs
│   ├── step1_preprocess.md
│   ├── step2_ir_ptt_peak_detector.md
│   ├── step3_ptt_bp_analysis.md
│   └── step4_integrated_ptt_bp_analysis.md
│
├── blood_pressure_reconstruction/   # BP reconstruction algorithms
├── personal_analysis/               # Subject-level analyses
├── step*_calibrated_check_results/  # Calibration checks
│
├── 研究方案详解.md / 数据说明.txt        # Study plan & data note (Chinese)
└── *.pdf                            # References
```

## 🚀 Quick Start

### Environment

```bash
Python >= 3.8
numpy
scipy
pandas
matplotlib
scikit-learn
tensorflow / pytorch  # deep learning models
wfdb                   # physiological signal processing
```

### Pipeline

```bash
# Step 1: preprocess
python step1_preprocess.py

# Step 2: peak detection & PTT
python step2_ir_ptt_peak_detector.py

# Step 3: PTT–BP analysis
python step3_ptt_bp_analysis.py

# Step 4: integrated modeling
python step4_integrated_ptt_bp_analysis.py
```

### Batch (multiprocessing)

```bash
python step2_ir_ptt_peak_detector_mulPro.py
python step3_ptt_bp_analysis_mulPro.py
```

## 📊 Data

- Multi-sensor PPG signals under various physiological states.
- Reference arterial blood pressure (ABP) recorded synchronously.
- Data description: see 数据说明.txt (Chinese).

> If raw data cannot be released publicly, please provide synthetic/sample data or a data request process before open release.

## 📈 Findings (brief)

- Multi-sensor PTT outperforms single-site measurements; six PTT combinations provide complementary information.
- Strong subject variability → individual calibration is important.
- Physical models degrade under complex physiological states; dataset size limits deep models.

## 📄 License

- License: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
- You may share/adapt with attribution, **non-commercial only**, and must relicense under the same terms.
- Commercial use or closed-source redistribution is not allowed.

Full text in [LICENSE](LICENSE).

## 🙏 Acknowledgments

Thanks to the Tsinghua University PI Lab for platform, data support, and guidance.
