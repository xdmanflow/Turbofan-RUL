# Predictive Maintenance of Turbofan Engines — Source Code Package

This package contains the complete, executable pipeline used to produce every
number, figure, and table in `SAA_Report_Predictive_Maintenance.docx`.

## Folder structure

- `code/01_load_data.py` ... `code/09_baseline_scatter.py` — Python scripts, run in
  numeric order. Each is self-contained and reads/writes to a local `data/` and
  `figs/` folder (create these next to the scripts, or point the paths to the
  `data/` and `figures/` folders included here).
- `code/*.js` — Node.js scripts (using the `docx` npm package) that assemble the
  final Word report from pre-written text + the generated figures.
  Run with `node assemble.js` after `npm install docx`.
- `data/` — the raw NASA C-MAPSS FD001 files (`train_FD001.txt`, `test_FD001.txt`,
  `RUL_FD001.txt`) plus the original dataset readme, exactly as published.
- `figures/` — all 10 PNG figures used in the report, at full resolution.
- `results/` — every intermediate result file: JSON metrics for each model,
  CSV predictions per test engine, feature importance, training history, etc.
- `model/` — the trained LSTM model (`lstm_model.keras`) and the fitted
  `MinMaxScaler` (`scaler.pkl`) used for normalization, so you can load them
  and run inference on new data without retraining.

## How to reproduce the results from scratch

```bash
pip install pandas numpy scikit-learn tensorflow-cpu matplotlib joblib pyarrow

mkdir data figs
cp <this-package>/data/*.txt data/    # raw NASA files

python3 01_load_data.py        # builds RUL labels -> data/*_rul.parquet
python3 02_eda.py               # EDA + sensor selection -> figs/fig1-3
python3 03_features.py          # normalization, train/val split
python3 04_baselines.py         # Linear Regression, RF, GB
python3 05_lstm.py              # LSTM model (takes a few minutes on CPU)
python3 06_results_figures.py   # comparison figures -> figs/fig4-8
python3 07_extra_analysis.py    # error-by-zone analysis
python3 08_zone_figure.py       # -> figs/fig9
python3 09_baseline_scatter.py  # -> figs/fig10
```

## How to rebuild the Word report

```bash
npm install docx
node assemble.js
# -> produces SAA_Report_Predictive_Maintenance.docx
```

## Key results (for quick reference)

| Model              | Test RMSE | NASA Score | Train time (s) |
|---------------------|-----------|------------|-----------------|
| Linear Regression    | 21.88     | 1307.6     | 0.01            |
| Gradient Boosting     | 18.37     | 1047.7     | 13.72           |
| Random Forest         | 18.08     | 912.6      | 28.68           |
| **LSTM (best)**       | **15.41** | **423.3**  | 219.37          |

Critical-zone (true RUL < 50 cycles) RMSE: Random Forest 16.47, Gradient
Boosting 16.49, **LSTM 6.14** — see Section 4.6 of the report.

## Notes

- Random seed fixed at 42 throughout (train/validation split, model
  initialization) for reproducibility.
- Library versions used: pandas 3.0.2, numpy 2.4.4, scikit-learn 1.8.0,
  TensorFlow/Keras 2.21.0.
- The FD002/FD003/FD004 subsets of C-MAPSS are not used in this study
  (see Section 6.3 of the report, "Perspectives for future work").
