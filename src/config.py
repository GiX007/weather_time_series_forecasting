# src/config.py
#
# Central configuration for the project.
#
CONFIG = {
    "features": ["SWDR", "rh", "T"],
    "target": "T",
    "horizons": [1, 4, 6], # steps ahead to predict
    "resolutions": ["1h", "6h", "24h"], # resampling ranges

    # context/history window candidates per resolution for ARs
    "max_lags": {
        "1h": 168, # 1 week
        "6h": 56, # ~2 week
        "24h": 30 # ~1 month
        },

    # initial single-run defaults
    "nn_defaults": {
        "batch_size": 32,
        "epochs": 200,
        "ffn_dim": 64, # or 2 * hidden_size
        "num_layers": 2,
        "num_heads": 2,
        "hidden_size": 32,
        "learning_rate": 1e-3,
        "dropout": 0.1,
        "weight_decay": 1e-4,
        "optimizer": "adamw",
        "seq_len":
            {"1h": 72, "6h": 28, "24h": 21} # L0: shortest sequence lengths
            # {"1h": 120, "6h": 56, "24h": 28} # L1: short sequence lengths
            # {"1h": 168, "6h": 84, "24h": 35} # L2: longer sequence lengths
            # {"1h": 336, "6h": 112, "24h": 50} # L3: even longer sequence lengths
        },

    "split": {"train": 0.7, "val": 0.15, "test": 0.15},
    }