# Turbofan-rul-prediction Predictive Maintenance Project

Predicting the **Remaining Useful Life (RUL)** of turbofan engines from multivariate sensor data, comparing classical machine learning models against an LSTM deep learning model, evaluated with both RMSE and NASA's safety-weighted asymmetric scoring function.

> Academic research project (Scientific Approach Application) — CESI École d'Ingénieurs, Toulouse. February 2026.

## Overview

Unplanned turbofan engine failures are safety-critical and costly, while fixed-schedule preventive maintenance often replaces components before the end of their useful life. This project investigates whether machine learning can estimate RUL accurately enough from raw sensor data to support a shift toward predictive maintenance.

Using the **FD001 subset of NASA's C-MAPSS** dataset (100 training engines, 100 test engines, single operating condition, single fault mode — High-Pressure Compressor degradation), four model families are implemented and compared:

- Linear Regression (interpretable baseline)
- Random Forest
- Gradient Boosting
- LSTM (recurrent deep learning, 30-cycle sliding windows)

## Key Results

| Model | Test RMSE (cycles) | Test NASA Score | Training Time (s) |
| --- | --- | --- | --- |
| Linear Regression | 21.88 | 1307.6 | 0.01 |
| Gradient Boosting | 18.37 | 1047.7 | 13.72 |
| Random Forest | 18.08 | 912.6 | 28.68 |
| **LSTM** | **15.41** | **423.3** | 219.37 |

- The **LSTM outperforms every tabular baseline**: 14.7% lower RMSE and 53.6% lower NASA score than the best tabular model (Random Forest).
- Its advantage is far larger in the **operationally critical zone** (true RUL < 50 cycles): RMSE of 6.14 cycles vs 16.47–16.49 cycles for the tabular models — roughly **2.7x more accurate exactly where maintenance decisions matter most**.
- Feature importance is highly concentrated: **sensor 11** (HPC outlet static pressure) alone accounts for ~65% of Random Forest importance, consistent with published C-MAPSS analyses.
- Evaluating with RMSE alone can be misleading in a safety-critical context — the asymmetric NASA scoring function (which penalizes late/optimistic predictions more heavily) changes the practical ranking of "best" model.

## Methodology

1. **Feature selection** — 6 of 21 raw sensors excluded as near-constant under FD001's single operating condition, leaving 15 informative sensors + 3 operational settings.
2. **RUL capping** — training labels capped at 125 cycles (piecewise-linear target), a standard practice since early-life degradation isn't observable in the sensors.
3. **Normalization** — min-max scaling, fit on training data only, to avoid leakage.
4. **Train/validation split** — performed at the **engine level** (not row level) to prevent leakage across a single engine's trajectory.
5. **Sequence construction** — 30-cycle sliding windows per engine for the LSTM input.
6. **Evaluation** — RMSE (symmetric) and the **NASA PHM08 asymmetric scoring function** (penalizes late predictions more heavily than early ones), computed on the last cycle of each of the 100 test engines.

## Repo Structure

```
turbofan-rul-prediction/
├── data/                        # C-MAPSS FD001 raw files (not included — see Data section)
├── notebooks/
│   ├── 01_eda.ipynb             # Exploratory data analysis, sensor variance/correlation
│   ├── 02_baseline_models.ipynb # Linear Regression, Random Forest, Gradient Boosting
│   └── 03_lstm_model.ipynb      # LSTM sequence model
├── src/
│   ├── data_prep.py             # RUL label construction, capping, normalization
│   ├── scoring.py                # NASA asymmetric scoring function
│   ├── train_baselines.py        # Baseline model training/evaluation
│   └── train_lstm.py             # Sliding-window construction + LSTM training
├── figures/                      # Generated plots (sensor trends, RMSE/score comparisons, etc.)
├── report/
│   └── SAA_Report_Predictive_Maintenance.pdf
├── requirements.txt
└── README.md
```

## Dataset

- **Source:** [NASA C-MAPSS Turbofan Engine Degradation Simulation Dataset](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) (FD001 subset)
- Public, open-access — no proprietary or confidential data used.
- 20,631 training rows across 100 engines (lifetimes: 128–362 cycles, mean ≈ 206); 13,096 test rows across 100 engines.

## Tech Stack

- **Python 3**, pandas, numpy
- **scikit-learn** — Linear Regression, Random Forest, Gradient Boosting
- **TensorFlow / Keras** — LSTM model
- **matplotlib** — visualizations

## Getting Started

```bash
git clone https://github.com/<your-username>/turbofan-rul-prediction.git
cd turbofan-rul-prediction
pip install -r requirements.txt
```

Download the FD001 files from the [NASA PCoE data repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) into `data/`, then run the notebooks in order (`01` → `02` → `03`).

## Limitations & Future Work

- Trained on a **simulation**, not real fleet telemetry — a real-world validation gap remains.
- Limited to a **single operating condition and fault mode** (FD001); FD002/FD004 (multi-condition) are natural next steps.
- No uncertainty quantification — current models output a point estimate only.
- Hyperparameters set from literature defaults + light manual tuning rather than a systematic search.
- Future directions: 1D-CNN / Transformer architectures, Monte Carlo Dropout for uncertainty, multi-condition generalization, real fleet data validation.

## Author

**Manil Doudou** — CS Engineering student (Data Science & AI), CESI Toulouse.

## License

Academic project — data used is public (NASA C-MAPSS). Code released for educational/portfolio purposes.
