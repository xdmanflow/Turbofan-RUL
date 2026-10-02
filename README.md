# Turbofan RUL Predictive Maintenance Scientific Research

Predicting the **Remaining Useful Life (RUL)** of turbofan engines from multivariate sensor data, comparing classical machine learning models against an LSTM deep learning model, evaluated with both RMSE and NASA's safety-weighted asymmetric scoring function.

## Overview

Unplanned turbofan engine failures are safety-critical and costly, while fixed-schedule preventive maintenance often replaces components before the end of their useful life. This project investigates whether machine learning can estimate RUL accurately enough from raw sensor data to support a shift toward predictive maintenance.

Using the **FD001** subset of NASA's C-MAPSS dataset (100 training engines, 100 test engines, single operating condition, single fault mode — High-Pressure Compressor degradation), four model families are implemented and compared:

- **Linear Regression** — interpretable baseline
- **Random Forest**
- **Gradient Boosting**
- **LSTM** — recurrent deep learning, 30-cycle sliding windows

## Key Results

| Model | Test RMSE (cycles) | Test NASA Score | Training Time (s) |
|---|---|---|---|
| Linear Regression | 21.88 | 1307.6 | 0.01 |
| Gradient Boosting | 18.37 | 1047.7 | 13.72 |
| Random Forest | 18.08 | 912.6 | 28.68 |
| **LSTM** | **15.41** | **423.3** | 219.37 |

- The LSTM outperforms every tabular baseline: **14.7% lower RMSE** and **53.6% lower NASA score** than the best tabular model (Random Forest).
- Its advantage is far larger in the operationally critical zone (true RUL < 50 cycles): **RMSE of 6.14 cycles vs 16.47–16.49 cycles** for the tabular models — roughly **2.7× more accurate** exactly where maintenance decisions matter.
- Feature importance is highly concentrated: **sensor 11 (HPC outlet static pressure)** alone accounts for ~65% of Random Forest importance, consistent with published C-MAPSS analyses.
- Evaluating with RMSE alone can be misleading in a safety-critical context — the asymmetric NASA scoring function, which penalizes late (optimistic) predictions more heavily, changes the practical ranking of the "best" model.

## Methodology

- **Feature selection** — 6 of 21 raw sensors excluded as near-constant under FD001's single operating condition, leaving 15 informative sensors + 3 operational settings.
- **RUL capping** — training labels capped at 125 cycles (piecewise-linear target), a standard practice since early-life degradation isn't observable in the sensors.
- **Normalization** — min-max scaling, fit on training data only, to avoid leakage.
- **Train/validation split** — performed at the engine level to prevent leakage across a single engine's trajectory.
- **Sequence construction** — 30-cycle sliding windows per engine for the LSTM input.
- **Evaluation** — RMSE (symmetric) and the NASA PHM08 asymmetric scoring function, computed on the last cycle of each of the 100 test engines.

**NASA scoring function** — with $d_i = \text{RUL}_i^{\text{pred}} - \text{RUL}_i^{\text{true}}$:

$$
s = \sum_{i=1}^{n}
\begin{cases}
e^{-d_i/13} - 1 & \text{if } d_i < 0 \ \text{(early prediction)} \\
e^{d_i/10} - 1 & \text{if } d_i \geq 0 \ \text{(late prediction)}
\end{cases}
$$

Late predictions are penalized more heavily, because overestimating remaining life is the dangerous error.

## Repo Structure

```
Turbofan-RUL/
├── data/
│   ├── README.md                     # File format description
│   ├── train_FD001.txt               # Training run-to-failure trajectories (100 engines)
│   ├── test_FD001.txt                # Test trajectories, truncated before failure (100 engines)
│   └── RUL_FD001.txt                 # Ground-truth RUL for the test engines
├── notebooks/
│   ├── 01_eda.ipynb                  # Exploratory data analysis, sensor variance/correlation
│   ├── 02_baseline_models.ipynb      # Linear Regression, Random Forest, Gradient Boosting
│   └── 03_lstm_model.ipynb           # LSTM sequence model + final comparison
├── src/
│   ├── 01_load_data.py               # Builds RUL labels -> data/*_rul.parquet
│   ├── 02_eda.py                     # EDA + sensor selection (fig1–3)
│   ├── 03_features.py                # Normalization, train/validation split
│   ├── 04_baselines.py               # Linear Regression, Random Forest, Gradient Boosting
│   ├── 05_lstm.py                    # Sliding-window construction + LSTM training
│   ├── 06_results_figures.py         # Model comparison figures (fig4–8)
│   ├── 07_extra_analysis.py          # Error-by-zone analysis
│   ├── 08_zone_figure.py             # Error-by-zone figure (fig9)
│   ├── 09_baseline_scatter.py        # True vs predicted RUL for baselines (fig10)
│   └── main.js, part2–7.js,          # Node.js scripts (docx package) that assemble
│       assemble.js, build_report.js  # the Word report from the text + figures
├── model/
│   ├── lstm_model.keras              # Trained LSTM model
│   └── scaler.pkl                    # Fitted MinMaxScaler — load both to run inference without retraining
├── results/
│   ├── all_results_table.csv         # Final comparison table (all models)
│   ├── baseline_results.json         # Baseline metrics (RMSE, NASA score, training time)
│   ├── baseline_test_predictions.csv # Baseline predictions on the 100 test engines
│   ├── lstm_results.json             # LSTM metrics
│   ├── lstm_test_predictions.csv     # LSTM predictions on the 100 test engines
│   ├── lstm_history.csv              # LSTM training/validation loss per epoch
│   ├── error_by_zone.csv             # RMSE by true-RUL zone
│   ├── rf_feature_importance.csv     # Random Forest feature importances
│   ├── feature_cols.txt              # Final feature list used by the models
│   └── informative_sensors.txt       # The 15 retained sensors
├── figures/
│   ├── fig1_sensor_trajectories.png
│   ├── fig2_lifetime_distribution.png
│   ├── fig3_sensor_rul_correlation.png
│   ├── fig4_rmse_comparison.png
│   ├── fig5_nasa_score_comparison.png
│   ├── fig6_true_vs_pred_lstm.png
│   ├── fig7_feature_importance.png
│   ├── fig8_lstm_training_curve.png
│   ├── fig9_error_by_zone.png
│   └── fig10_baseline_true_vs_pred.png
├── report/
│   └── SAA_Report_Predictive_Maintenance.pdf
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Dataset

- **Source:** NASA C-MAPSS Turbofan Engine Degradation Simulation Dataset (FD001 subset)
- **Public, open-access** — no proprietary or confidential data used
- 20,631 training rows across 100 engines (lifetimes: 128–362 cycles, mean ≈ 206); 13,096 test rows across 100 engines

The FD001 files are included in `data/` — see `data/README.md` for the file format.

## Tech Stack

- **Python 3**, pandas, numpy, pyarrow (parquet I/O), joblib
- **scikit-learn** — Linear Regression, Random Forest, Gradient Boosting
- **TensorFlow / Keras** — LSTM model
- **matplotlib** — visualizations
- **Node.js** + `docx` npm package — Word report generation

## Getting Started

```bash
git clone https://github.com/xdmanflow/Turbofan-RUL.git
cd Turbofan-RUL
pip install -r requirements.txt
```

The FD001 data is already in `data/`, and every figure, metric and the trained model used in the report are already stored in `figures/`, `results/` and `model/` — no need to re-run anything to inspect them. To reuse the trained LSTM on new data, load `model/lstm_model.keras` together with `model/scaler.pkl`.

**Reproduce the results from scratch** — either run the notebooks in order (`01 → 02 → 03`), or run the scripts in numeric order. The scripts use relative paths: they read from a local `data/` folder and write figures to a local `figs/` folder.

```bash
mkdir figs
python src/01_load_data.py        # builds RUL labels -> data/*_rul.parquet
python src/02_eda.py              # EDA + sensor selection -> figs/fig1-3
python src/03_features.py         # normalization, train/val split
python src/04_baselines.py        # Linear Regression, RF, GB
python src/05_lstm.py             # LSTM model (takes a few minutes on CPU)
python src/06_results_figures.py  # comparison figures -> figs/fig4-8
python src/07_extra_analysis.py   # error-by-zone analysis
python src/08_zone_figure.py      # -> figs/fig9
python src/09_baseline_scatter.py # -> figs/fig10
```

**Rebuild the Word report**

```bash
cd src
npm install docx
node assemble.js                  # -> SAA_Report_Predictive_Maintenance.docx
```

## Reproducibility

- Random seed fixed at **42** throughout (train/validation split, model initialization).
- Library versions used: pandas 3.0.2, numpy 2.4.4, scikit-learn 1.8.0, TensorFlow/Keras 2.21.0.
- The FD002/FD003/FD004 subsets of C-MAPSS are not used in this study (see Section 6.3 of the report, "Perspectives for future work").

## Limitations & Future Work

**Limitations**

- Trained on a simulation, not real fleet telemetry — a real-world validation gap remains.
- Limited to a single operating condition and fault mode (FD001); FD002/FD004 (multi-condition) are natural next steps.
- No uncertainty quantification — current models output a point estimate only.
- Hyperparameters set from literature defaults + light manual tuning rather than a systematic search.

**Future directions**

- 1D-CNN / Transformer architectures
- Monte Carlo Dropout for uncertainty estimation
- Multi-condition generalization (FD002–FD004)
- Validation on real fleet data

## Author

**Manil DOUDOU** — CS Engineering student (Data Science & AI), CESI Toulouse
[GitHub](https://github.com/xdmanflow) · [LinkedIn](https://www.linkedin.com/in/manil-doudou-4745923a0)

## License

Academic project — the data used is public (NASA C-MAPSS). Code released under the MIT License for educational and portfolio purposes.

## References

- Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation*. PHM08.
