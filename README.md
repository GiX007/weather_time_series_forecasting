# Weather Long-term Time Series Forecasting

This project implements and compares different forecasting models (AR, RNN, Attention, Transformer) on the 
[Weather Long-term Time Series Forecasting dataset](https://www.kaggle.com/datasets/alistairking/weather-long-term-time-series-forecasting).
The goal is to evaluate short and long-term prediction performance across multiple time horizons (1h, 6h, 24h).

## Contents

- `src/` - Core Python code (eda, preprocessing, models)
- `results` - Saved metrics, logs, figures and predictions
- `requirements.txt` - Dependencies
- `auxiliary` - Exploratory analysis and model demos with custom and modular implementations
- `README.md` - Project documentation

## Installation

Clone the repo and install dependencies:

```
git clone https://github.com/YOUR_USERNAME/weather-forecasting.git
cd weather-forecasting
pip install -r requirements.txt
```

## Quickstart

Run the entire project (EDA, preprocessing, training and evaluation):

```
python main.py
```

## Models Implemented

- **AR (autoregressive baseline)**
- **RNN (vanilla recurrent network)**
- **ATT (lightweight self-attention network)**
- **TRANS (transformer-encoder)**

We report MAE/RMSE/MAPE across resolutions (1h, 6h, 24h) and horizons (t+1, t+4, t+6).

## Results

We evaluate AR(p), RNN, ATT, and TRANS across short vs. longer context windows. All metrics/logs are under `results/`.

---
### 1h - Short sequence lengths (`{1h:72, 6h:28, 24h:21}`) 

**Basic configs used:** 
- **RNN:** `hidden=32`, `num_layers=1` 
- **ATT:** `hidden=32`, `heads=2`, `num_layers=1`, `dropout=0.1`, `pool=mean`
- **TRANS:** `d_model=32`, `heads=2`, `ffn_dim=64`, `num_layers=1`, `dropout=0.1`, `pool=mean`

| Model    | Resolution   | Horizon   | Target Variable   | Input Type   |  MAE | RMSE | MAPE (%) |   Fit Time |   # Params |
|----------|--------------|-----------|-------------------|--------------|------|------|----------|------------|------------|
| AR(1)    | 1h           | t+1       | T                 | Univariate   | 0.87 | 1.28 |   145.78 |     0.0030 |          2 |
| AR(1)    | 1h           | t+6       | T                 | Univariate   | 2.25 | 3.05 |   311.78 |     0.0030 |          2 |
| AR(8)    | 1h           | t+1       | T                 | Univariate   | 0.66 | 0.96 |   122.74 |     0.0050 |          9 |
| AR(8)    | 1h           | t+6       | T                 | Univariate   | 1.58 | 2.07 |   307.90 |     0.0050 |          9 |
| RNN      | 1h           | t+1       | T                 | Univariate   | 0.41 | 0.53 |    82.72 |   197.7429 |       1153 |
| RNN      | 1h           | t+6       | T                 | Univariate   | 2.70 | 3.13 |   715.44 |   197.7429 |       1153 |
| RNN_MV   | 1h           | t+1       | T                 | Multivariate | 0.36 | 0.47 |    71.00 |   306.0387 |       1217 |
| RNN_MV   | 1h           | t+6       | T                 | Multivariate | 2.59 | 3.20 |   614.57 |   306.0387 |       1217 |
| ATT      | 1h           | t+1       | T                 | Univariate   | 0.43 | 0.56 |    92.28 |   281.0838 |      10945 |
| ATT      | 1h           | t+6       | T                 | Univariate   | 2.24 | 2.63 |   572.59 |   281.0838 |      10945 |
| ATT_MV   | 1h           | t+1       | T                 | Multivariate | 0.44 | 0.55 |    97.81 |   142.6808 |      11009 |
| ATT_MV   | 1h           | t+6       | T                 | Multivariate | 2.26 | 2.75 |   457.99 |   142.6808 |      11009 |
| TRANS    | 1h           | t+1       | T                 | Univariate   | 0.42 | 0.56 |    82.25 |   250.8536 |       8641 |
| TRANS    | 1h           | t+6       | T                 | Univariate   | 1.92 | 2.37 |   458.63 |   250.8536 |       8641 |
| TRANS_MV | 1h           | t+1       | T                 | Multivariate | 0.46 | 0.58 |   104.68 |   202.7085 |       8705 |
| TRANS_MV | 1h           | t+6       | T                 | Multivariate | 2.11 | 2.54 |   458.05 |   202.7085 |       8705 |

---
### 1h - Longer sequence lengths (`{1h:120, 6h:56, 24h:28}`)

**Configs used:** 
- **RNN:** `hidden=32`, `num_layers=1` 
- **ATT:** `hidden=32`, `heads=2`, `num_layers=2`, `dropout=0.1`, `pool=mean` 
- **TRANS:** `d_model=48`, `heads=3`, `ffn_dim=96`, `num_layers=2`, `dropout=0.1`, `pool=mean`

| Model    | Resolution   | Horizon   | Target Variable   | Input Type   |  MAE | RMSE | MAPE (%) |   Fit Time |   # Params |
|----------|--------------|-----------|-------------------|--------------|------|------|----------|------------|------------|
| AR(1)    | 1h           | t+1       | T                 | Univariate   | 0.87 | 1.28 |   145.78 |     0.0031 |          2 |
| AR(1)    | 1h           | t+6       | T                 | Univariate   | 2.25 | 3.05 |   311.78 |     0.0031 |          2 |
| AR(8)    | 1h           | t+1       | T                 | Univariate   | 0.66 | 0.96 |   122.74 |     0.0058 |          9 |
| AR(8)    | 1h           | t+6       | T                 | Univariate   | 1.58 | 2.07 |   307.90 |     0.0058 |          9 |
| RNN      | 1h           | t+1       | T                 | Univariate   | 0.41 | 0.53 |    83.09 |   225.6078 |       1153 |
| RNN      | 1h           | t+6       | T                 | Univariate   | 2.73 | 3.17 |   706.32 |   225.6078 |       1153 |
| RNN_MV   | 1h           | t+1       | T                 | Multivariate | 0.31 | 0.42 |    39.73 |   238.5690 |       1217 |
| RNN_MV   | 1h           | t+6       | T                 | Multivariate | 2.04 | 2.70 |   324.10 |   238.5690 |       1217 |
| ATT      | 1h           | t+1       | T                 | Univariate   | 0.40 | 0.54 |    76.05 |   667.3014 |      21025 |
| ATT      | 1h           | t+6       | T                 | Univariate   | 2.48 | 3.20 |   463.14 |   667.3014 |      21025 |
| ATT_MV   | 1h           | t+1       | T                 | Multivariate | 0.43 | 0.57 |    88.93 |   246.8591 |      21089 |
| ATT_MV   | 1h           | t+6       | T                 | Multivariate | 2.78 | 3.59 |   532.47 |   246.8591 |      21089 |
| TRANS    | 1h           | t+1       | T                 | Univariate   | 0.36 | 0.50 |    48.87 |   933.6536 |      38065 |
| TRANS    | 1h           | t+6       | T                 | Univariate   | 1.59 | 2.10 |   245.95 |   933.6536 |      38065 |
| TRANS_MV | 1h           | t+1       | T                 | Multivariate | 0.39 | 0.53 |    84.55 |   585.6651 |      38161 |
| TRANS_MV | 1h           | t+6       | T                 | Multivariate | 2.36 | 3.22 |   444.91 |   585.6651 |      38161 |

---
### Key findings
- **Short windows:** the 1-layer **RNN** is strongest and **ATT/TRANS** underperform with limited context.
- **Longer windows:** **ATT improves** and **TRANS** often leads.
- **Depth without context doesn’t help:** increasing layers on short windows did **not** improve ATT/TRANS (see full tables in `results/`).

## Notes

- The code is for educational purposes.
- Tuning scope: a light and shared tuning of structural hyperparameters (e.g., `num_layers`, `hidden_size`/`d_model` and `ffn_dim`) has been applied across all resolutions and horizons (no per-setting specialization).
- Results can vary slightly from run to run.

## Contributing

Contributions and feedback are welcome! Feel free to open issues or submit pull requests.
