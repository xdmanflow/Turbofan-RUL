# Data

The C-MAPSS dataset is **not included** in this repository. Download it from the NASA Prognostics Center of Excellence (PCoE) data repository ("Turbofan Engine Degradation Simulation Data Set").

Place these three files in this folder:

```
data/
├── train_FD001.txt   # 100 engines run to failure (training)
├── test_FD001.txt    # 100 engines stopped before failure (test)
└── RUL_FD001.txt     # True RUL at the last cycle of each test engine
```

## File format

Space-separated, no header, 26 columns:

| Columns | Content |
|---|---|
| 1 | Engine unit number |
| 2 | Time (cycles) |
| 3–5 | Operational settings 1–3 |
| 6–26 | Sensor measurements 1–21 |

## Reference

Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation.* International Conference on Prognostics and Health Management (PHM08).
