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

| Model    | Resolution   | Horizon   | Target Variable   | Input Type   |    MAE |   RMSE |   MAPE (%) |   Fit Time |   # Params |
|----------|--------------|-----------|-------------------|--------------|--------|--------|------------|------------|------------|
| AR(1)    | 1h           | t+1       | T                 | Univariate   | 0.8700 | 1.2800 |   145.7800 |     0.0030 |          2 |
| AR(1)    | 1h           | t+6       | T                 | Univariate   | 2.2500 | 3.0500 |   311.7800 |     0.0030 |          2 |
| AR(8)    | 1h           | t+1       | T                 | Univariate   | 0.6600 | 0.9600 |   122.7400 |     0.0050 |          9 |
| AR(8)    | 1h           | t+6       | T                 | Univariate   | 1.5800 | 2.0700 |   307.9000 |     0.0050 |          9 |
| RNN      | 1h           | t+1       | T                 | Univariate   | 0.4100 | 0.5300 |    82.7200 |   197.7429 |       1153 |
| RNN      | 1h           | t+6       | T                 | Univariate   | 2.7000 | 3.1300 |   715.4400 |   197.7429 |       1153 |
| RNN_MV   | 1h           | t+1       | T                 | Multivariate | 0.3600 | 0.4700 |    71.0000 |   306.0387 |       1217 |
| RNN_MV   | 1h           | t+6       | T                 | Multivariate | 2.5900 | 3.2000 |   614.5700 |   306.0387 |       1217 |
| ATT      | 1h           | t+1       | T                 | Univariate   | 0.4300 | 0.5600 |    92.2800 |   281.0838 |      10945 |
| ATT      | 1h           | t+6       | T                 | Univariate   | 2.2400 | 2.6300 |   572.5900 |   281.0838 |      10945 |
| ATT_MV   | 1h           | t+1       | T                 | Multivariate | 0.4400 | 0.5500 |    97.8100 |   142.6808 |      11009 |
| ATT_MV   | 1h           | t+6       | T                 | Multivariate | 2.2600 | 2.7500 |   457.9900 |   142.6808 |      11009 |
| TRANS    | 1h           | t+1       | T                 | Univariate   | 0.4200 | 0.5600 |    82.2500 |   250.8536 |       8641 |
| TRANS    | 1h           | t+6       | T                 | Univariate   | 1.9200 | 2.3700 |   458.6300 |   250.8536 |       8641 |
| TRANS_MV | 1h           | t+1       | T                 | Multivariate | 0.4600 | 0.5800 |   104.6800 |   202.7085 |       8705 |
| TRANS_MV | 1h           | t+6       | T                 | Multivariate | 2.1100 | 2.5400 |   458.0500 |   202.7085 |       8705 |

---
### 1h - Longer sequence lengths (`{1h:120, 6h:56, 24h:28}`)

**Configs used:** 
- **RNN:** `hidden=32`, `num_layers=1` 
- **ATT:** `hidden=32`, `heads=2`, `num_layers=2`, `dropout=0.1`, `pool=mean` 
- **TRANS:** `d_model=48`, `heads=3`, `ffn_dim=96`, `num_layers=2`, `dropout=0.1`, `pool=mean`

| Model    | Resolution   | Horizon   | Target Variable   | Input Type   |    MAE |   RMSE |   MAPE (%) |   Fit Time |   # Params |
|----------|--------------|-----------|-------------------|--------------|--------|--------|------------|------------|------------|
| AR(1)    | 1h           | t+1       | T                 | Univariate   | 0.8700 | 1.2800 |   145.7800 |     0.0031 |          2 |
| AR(1)    | 1h           | t+6       | T                 | Univariate   | 2.2500 | 3.0500 |   311.7800 |     0.0031 |          2 |
| AR(8)    | 1h           | t+1       | T                 | Univariate   | 0.6600 | 0.9600 |   122.7400 |     0.0058 |          9 |
| AR(8)    | 1h           | t+6       | T                 | Univariate   | 1.5800 | 2.0700 |   307.9000 |     0.0058 |          9 |
| RNN      | 1h           | t+1       | T                 | Univariate   | 0.4100 | 0.5300 |    83.0900 |   225.6078 |       1153 |
| RNN      | 1h           | t+6       | T                 | Univariate   | 2.7300 | 3.1700 |   706.3200 |   225.6078 |       1153 |
| RNN_MV   | 1h           | t+1       | T                 | Multivariate | 0.3100 | 0.4200 |    39.7300 |   238.5690 |       1217 |
| RNN_MV   | 1h           | t+6       | T                 | Multivariate | 2.0400 | 2.7000 |   324.1000 |   238.5690 |       1217 |
| ATT      | 1h           | t+1       | T                 | Univariate   | 0.4000 | 0.5400 |    76.0500 |   667.3014 |      21025 |
| ATT      | 1h           | t+6       | T                 | Univariate   | 2.4800 | 3.2000 |   463.1400 |   667.3014 |      21025 |
| ATT_MV   | 1h           | t+1       | T                 | Multivariate | 0.4300 | 0.5700 |    88.9300 |   246.8591 |      21089 |
| ATT_MV   | 1h           | t+6       | T                 | Multivariate | 2.7800 | 3.5900 |   532.4700 |   246.8591 |      21089 |
| TRANS    | 1h           | t+1       | T                 | Univariate   | 0.3600 | 0.5000 |    48.8700 |   933.6536 |      38065 |
| TRANS    | 1h           | t+6       | T                 | Univariate   | 1.5900 | 2.1000 |   245.9500 |   933.6536 |      38065 |
| TRANS_MV | 1h           | t+1       | T                 | Multivariate | 0.3900 | 0.5300 |    84.5500 |   585.6651 |      38161 |
| TRANS_MV | 1h           | t+6       | T                 | Multivariate | 2.3600 | 3.2200 |   444.9100 |   585.6651 |      38161 |

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
