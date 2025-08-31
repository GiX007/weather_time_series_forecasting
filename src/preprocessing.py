# src.preprocessing.py
#
# Helper functions for Preprocessing the weather dataset.
#
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader

from src.eda import summary_info, correlation_analysis, plot_dataset, plot_df_avg_line_trends, plot_series_overview, test_stationarity, plot_nonstat_vs_diff
from src.utils import write_log

# paths
eda_base = "results/figures/eda/"
preprocessing = os.path.join(eda_base, "preprocessing")


def clean_dataframe(df, target, features):
    """
    Cleans the dataset by: sorting index, dropping duplicates and missing values and keeping only the selected features and target.

    Args:
        df (pandas dataframe): dataframe containing weather data
        target (str): target variable
        features (list): list of features to keep
    """
    df_len_before = df.shape[0]
    df = df.sort_index()    # ensure sorted by time
    df.drop_duplicates(inplace=True)   # drop duplicate timestamps
    df.dropna(inplace=True) # drop NaNs
    kept_cols = [target] + [f for f in features if f != target]  # keep only target and selected features columns
    df = df[kept_cols]

    # eda for the processed dataset
    summary_info(df, "Processed Dataset")
    write_log(f"\nDropped {df_len_before - len(df)} rows due to NaNs/duplicates.\n")
    correlation_analysis(df, "Processed Dataset")

    return df


def resample_frames(df, resolutions, agg="mean"):
    """
    Resamples the dataframe into different resolutions.

    Args:
        df (pandas dataframe): cleaned dataframe with datetime index
        resolutions (list): resolutions to resample, e.g. ["1h", "6h", "24h"]
        agg (str): aggregation function, e.g., mean
    """
    resampled = {}
    write_log("==== Resampled Datasets ====")
    for res in resolutions:
        df_res = getattr(df.resample(res), agg)()   # apply aggregation by name
        nans = df_res.isna().sum().sum()    # total number of NaN cells across all columns (features)
        df_res.dropna(inplace=True)
        resampled[res] = df_res

        # eda for the current resampled dataset
        summary_info(resampled[res], f"{res} Dataset")
        write_log(f"\nResampling introduced {nans} missing values across all columns (all rows with NaNs were removed).")
        correlation_analysis(resampled[res], f"{res} Dataset")

    return resampled


def preprocess_dataset(weather_df, target, features, resolutions):
    """
    Cleans the dataset, resamples it per resolution and makes some overview plots.

    Args:
        weather_df (pandas dataframe): dataframe containing weather data
        target (str): target variable
        features (list): list of features to keep
        resolutions (list): list of resolutions to resample

    Returns:
        dictionary of resampled frames per resolution
    """
    df = clean_dataframe(weather_df, target, features)
    resampled = resample_frames(df, resolutions)  # dict: {"1H": df_1h, "6H": df_6h, "24H": df_24h}

    # single-df plots
    for name, df_res in resampled.items():
        plot_dataset(df_res, name)

    # multi-df overview plots
    plot_df_avg_line_trends(resampled["1h"], resampled["6h"], resampled["24h"])
    plot_series_overview(resampled, resolutions, features)

    return resampled


def difference_series(series, lag=1):
  """
  Applies first-order or seasonal differencing to a time series.

  Args:
    series (array-like): The time series values
    lag (int): The lag to use for differencing (default=1):
        - lag = 1 -> first-order differencing (diff_t = x_t - x_{t-1})
        - lag > 1 -> seasonal differencing (subtracts the value from `lag` steps earlier: diff_t = x_t - x_{t-L})

  Returns
    pdSeries: Differenced series (NaNs removed)
  """
  diff2 = pd.Series(series).diff(lag).dropna()
  return diff2


def stationarity_and_diff(resampled, resolutions):
    """
    Runs stationarity tests, differences non-stationary series, re-tests.

    Args:
        resampled (dict): dictionary of resampled frames per resolution
        resolutions (list): list of resolutions to resample

    Returns:
        final_stationary: list[(res, feat)] that are stationary (original or differenced)
        stationary_series_data: {(res, feat): Series_to_use}
        provenance: {(res, feat): "original"|"differenced"}
    """
    write_log("==== Summary for Stationarity Tests ====", filename="results/stationarity_summary.txt")

    all_needs_diff, all_stationary = [], []
    for res in resolutions:
        df_res = resampled[res]
        needs_diff, is_stationary = test_stationarity(df_res, res, feat="*")  # "*" as a placeholder
        all_needs_diff += needs_diff
        all_stationary += is_stationary

    summary_msg = (f"\nNon-stationarity series that need to be differenced for applying AR models: {all_needs_diff}.\n"
                   f"\nStationary series that do not need to be differenced for applying AR models: {all_stationary}.\n")
    write_log(summary_msg, filename="results/stationarity_summary.txt")

    # originals that were already stationary
    stationary_series_data = {}
    provenance = {}
    for (res, feat) in all_stationary:
        stationary_series_data[(res, feat)] = resampled[res][feat]
        provenance[(res, feat)] = "original"

    # difference non-stationary series for ARs
    write_log("==== Differencing Non-Stationary Series and Re-Testing ====",
              filename="results/stationarity_summary.txt")

    all_needs_diff_after, all_stationary_after = [], []
    for (res, feat) in all_needs_diff:
        df_res = resampled[res]
        s_orig = df_res[feat]
        s_diff = difference_series(s_orig)  # apply differencing

        # plot original vs. differenced side by side
        plot_nonstat_vs_diff(s_orig, s_diff, res, feat)

        # re-test stationarity on the differenced series
        needs_diff2, stationary2 = test_stationarity(s_diff, res, feat)

        if len(stationary2) > 0:    # differenced version is now stationary -> store differenced series
            stationary_series_data[(res, feat)] = s_diff
            all_stationary_after += stationary2
        else:
            all_needs_diff_after += needs_diff2

    summary_msg2 = (f"\nAfter differencing:\n"
                    f"\nStill non-stationary: {all_needs_diff_after}.\n"
                    f"Now stationary (all): {all_stationary} + {all_stationary_after}.\n")
    write_log(summary_msg2, filename="results/stationarity_summary.txt")

    # union of originals + newly stationary (diff)
    final_stationary = all_stationary + all_stationary_after # list[(res, feat)], e.g., # ([('1h','T'), ('1h','SWDR'), ('1h','rh'), ('6h','rh'), ('24h','rh') + ('6h','T'), ('6h','SWDR'), ('24h','T'), ('24h','SWDR')])

    return final_stationary, stationary_series_data, provenance


def train_val_test_split(series, train_ratio=0.7, val_ratio=0.15):
    """
    Chronologically split a time series into train, validation, and test sets. If `freq` is given (e.g., "1h", "6h", "24h"), keep that frequency on each split
    so statsmodels can forecast without warnings.

    Args:
        series (pd.Series or pd.DataFrame): Input time series
        train_ratio (float): Fraction of samples for training (default 0.7)
        val_ratio (float): Fraction of samples for validation (default 0.15)

    Returns:
        train, val, test (same type as input)
    """
    n = len(series)
    split_idx_train = int(n * train_ratio)
    split_idx_val   = int(n * (train_ratio + val_ratio))

    train = series.iloc[:split_idx_train]
    val   = series.iloc[split_idx_train:split_idx_val]
    test  = series.iloc[split_idx_val:]

    return train, val, test


def plot_splits_for_feature(splits_by_res, resolutions, feat, save_dir=preprocessing):
    """
    Plots train/val/test for one feature across multiple resolutions side-by-side.

    Args:
        splits_by_res (dict): {"1h": (train_df, val_df, test_df), "6h": (...), "24h": (...)}. Each df must contain the column `feat` and a datetime index
        resolutions (list): list of resolutions to plot
        feat (str): feature/column name to plot (e.g., "T")
        save_dir (str): directory to save the figure
    """
    os.makedirs(save_dir, exist_ok=True)
    n = len(resolutions)

    fig, axes = plt.subplots(1, n, figsize=(10*n, 8))
    fig.suptitle(f"{feat} — Train/Val/Test Across Resolutions", fontsize=16, fontweight="bold")

    for i, res in enumerate(resolutions):
        train_df, val_df, test_df = splits_by_res[res]

        ax = axes[i]
        ax.plot(train_df.index, train_df[feat], label="train")
        ax.plot(val_df.index, val_df[feat], label="val")
        ax.plot(test_df.index, test_df[feat], label="test")
        ax.set_title(f"{feat} [{res}] - Splits")
        ax.set_xlabel("Time")
        ax.set_ylabel(feat)
        ax.grid(linestyle="--", alpha=0.4)
        ax.legend(loc="upper left")


    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(save_dir, f"{feat}_splits.png")
    plt.savefig(save_path)
    plt.close(fig)


def normalize(series):
  """
  Normalizes a time series using z-score normalization: x_norm = (x - mean) / std.

  Args:
      series (pd.Series or np.ndarray): Input time series

  Returns:
      pd.Series: Normalized series (mean=0, std=1)
      float: Mean of the original series (for inverse transform)
      float: Std of the original series (for inverse transform)
  """
  mean = series.mean()
  std = series.std()

  norm_series = (series - mean) / (std + 1e-8)  # avoid dividing by zero
  return norm_series, mean, std


def denormalize(x_norm, mu, std):
  """
  Inverse of normalize_series.

  Args:
      x_norm (pd.Series or np.ndarray): Normalized time series
      mu (float): Mean of the original series
      std (float): Std of the original series

  Returns:
      pd.Series: Denormalized series
  """
  return x_norm * std + 1e-12 if std == 0 else x_norm * std + mu


def plot_rescaling_is_harmless(resampled, feature="T", save_dir=preprocessing):
    """
    Plots a feature in all resolutions, before (row 1) and after (row 2) normalization.

    Args:
        resampled (dict): Dictionary of DataFrames per resolution, e.g. {"1h": df_1h, "6h": df_6h, "24h": df_24h}
        feature (str): Feature/column name to plot (default "T")
        save_dir (str): Directory to save the figure
    """
    os.makedirs(save_dir, exist_ok=True)

    resolutions = list(resampled.keys())  # e.g. ["1h", "6h", "24h"]
    n_res = len(resolutions)

    fig, axes = plt.subplots(2, n_res, figsize=(25, 8))
    fig.suptitle(f"Rescaling is Harmless — {feature}", fontsize=16, fontweight="bold")

    for j, res in enumerate(resolutions):
        df = resampled[res]
        s = df[feature]

        # normalize series
        norm, _, _ = normalize(s)

        # row 1 → original series
        axes[0, j].plot(s.index, s.values)
        axes[0, j].set_title(f"{feature} [{res}] — Original Series")
        axes[0, j].set_xlabel("Time")
        axes[0, j].set_ylabel("Value")
        axes[0, j].grid(linestyle="--", alpha=0.4)

        # row 2 → normalized
        axes[1, j].plot(norm.index, norm.values, color="orange")
        axes[1, j].set_title(f"{feature} [{res}] — Normalized Series (z-score)")
        axes[1, j].set_xlabel("Time")
        axes[1, j].set_ylabel("Z-score")
        axes[1, j].grid(linestyle="--", alpha=0.4)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    save_path = os.path.join(save_dir, f"rescaling_is_harmless_{feature}.png")
    plt.savefig(save_path)
    plt.close(fig)


def split_and_normalize(stationary_series_data, resampled, resolutions):
    """
    Splits each stationary series into (train, val, test), then normalizes on the train set. Also builds LEVEL splits (and their
    normalized versions) for the same (res, feat) keys.

    Args:
        stationary_series_data (dict): {(res, feat): Series_to_use_in_model_space}
        resampled (dict): Dictionary of DataFrames per resolution for level series
        resolutions (list): list of resolutions, (e.g., ["1h","6h","24h"])

     Returns:
        level_splits_norm (dict): {feat: {res: (train_df_n, val_df_n, test_df_n)}}  # LEVEL series, normalized per-LEVEL train mean and std
        splits_norm (dict): {feat: {res: (train_df_n, val_df_n, test_df_n)}}  # MODEL space (LEVEL or DIFF), normalized per-model train mean and std
        scalers (dict): {(res, feat): {"mean": mu_model, "std": sd_model}} # from MODEL train set (used to denormalize preds/targets)
    """
    write_log("==== Splits Summary ====\n"
              "\nSeries for AR Baseline models:\n", filename="results/input_data_summary.txt")

    # containers
    splits = {} # {feat: {res: (train_df, val_df, test_df)}} in MODEL space
    level_splits = {} # {feat: {res: (train_df, val_df, test_df)}} in LEVEL space
    splits_norm = {} # {feat: {res: (train_df_n, val_df_n, test_df_n)}} MODEL space normalized
    level_splits_norm = {} # {feat: {res: (train_df_n, val_df_n, test_df_n)}} LEVEL space normalized
    scalers = {} # {(res, feat): {"mean": mu, "std": sd}}

    # split
    for (res, feat), series in stationary_series_data.items():  # tuple-unpack
        tr_s, val_s, te_s = train_val_test_split(series)    # MODEL space series (LEVEL or DIFF)

        # also keep LEVEL splits for inversion during prediction
        level_series = resampled[res][feat] # LEVEL space series (use of resampled)
        tr_ls, val_ls, te_ls = train_val_test_split(level_series)

        # ensure nested dicts exist (because python complains!)
        if feat not in splits:
            splits[feat] = {}
            level_splits[feat] = {}

        # convert Series -> DataFrames with column name = feature (keeps splits consistent and easier to merge later)
        splits[feat][res] = (tr_s.to_frame(feat), val_s.to_frame(feat), te_s.to_frame(feat))
        level_splits[feat][res] = (tr_ls.to_frame(feat), val_ls.to_frame(feat), te_ls.to_frame(feat))

        # log basic info for MODEL series
        write_log(
            f"{res} {feat} Time Series\n"
            f"Length: {len(series)} | Nulls: {series.isna().sum()} | dtype: {series.dtype}\n"
            f"Train size: {len(tr_s)} | Val size: {len(val_s)} | Test size: {len(te_s)}\n\n",
            filename="results/input_data_summary.txt")

    # plot splits per feature (MODEL space)
    for feat, res_dict in splits.items():
        plot_splits_for_feature(res_dict, resolutions, feat)

    # normalize
    for feat, res_dict in splits.items():
        if feat not in splits_norm:
            splits_norm[feat] = {}
            level_splits_norm[feat] = {}

        for res, (tr_df, va_df, te_df) in res_dict.items(): # loop over we actually split

            # MODEL space normalization (used for AR training/prediction and final denorm)
            tr_norm, mu, sd = normalize(tr_df[feat]) # fit on the MODEL train set
            val_norm = (va_df[feat] - mu) / (sd + 1e-8) # apply to the val set
            te_norm = (te_df[feat] - mu) / (sd + 1e-8) # apply to the test set

            splits_norm[feat][res] = (tr_norm.to_frame(feat), val_norm.to_frame(feat), te_norm.to_frame(feat))

            # store MODEL-space scaler (the one you'll use to denormalize preds/targets)
            scalers[(res, feat)] = {"mean": float(mu), "std": float(sd)}

            # LEVEL space normalization (usually not needed for AR inversion)
            # lvl_tr_df, lvl_va_df, lvl_te_df = level_splits[feat][res]
            # lvl_tr_n, mu_level, sd_level = normalize(lvl_tr_df[feat])  # fit on LEVEL train
            # lvl_va_n = (lvl_va_df[feat] - mu_level) / (sd_level + 1e-8)
            # lvl_te_n = (lvl_te_df[feat] - mu_level) / (sd_level + 1e-8)
            #
            # level_splits_norm[feat][res] = (lvl_tr_n.to_frame(feat), lvl_va_n.to_frame(feat), lvl_te_n.to_frame(feat))

            # compute summary stats for normalized train+val+test
            combined = pd.concat([tr_norm, val_norm, te_norm])
            min_val, max_val = combined.min(), combined.max()
            mean_val, std_val = combined.mean(), combined.std()

            # log normalization summary (MODEL space)
            write_log(
                f"Normalization summary (MODEL space) for [{res} | {feat}] series:\n"
                f"  min={min_val:.4f}, max={max_val:.4f}\n"
                f"  mean={mean_val:.4f}, std={std_val:.4f}\n",
                filename="results/input_data_summary.txt")

    # an intermediate check
    # for feat, res_dict in splits_norm.items():
    #     for res, (tr, va, te) in res_dict.items():
    #         print(f"{feat}-{res}: train={tr.shape}, val={va.shape}, test={te.shape}")

    return level_splits, splits_norm, scalers


def create_sequences(data, window_size, target_col, horizon = 1, step = 1):
    """
    Converts a (uni-/multi-variate) time series into supervised windows for NNs.

    Examples:
        - Univariate case (Series, horizon=1):
            # series = pd.Series([1,2,3,4,5,6,7,8,9,10])
            # X, y = create_sequences(series, window_size=3, target_col=None, horizon=1)
            # X.shape -> (7, 3, 1), y.shape -> (7, )
            # First window: X[0] = [[1],[2],[3]], y[0] = 4

        - Multivariate case (DataFrame, horizon=2):
            # df = pd.DataFrame({ "temp": [10,11,12,13,14,15,16,17], "humidity": [30,32,34,36,38,40,42,44] })
            # X, y = create_sequences(df, window_size=3, target_col="temp", horizon=2)
            # X.shape -> (4, 3, 2), y.shape -> (4, 2)   # notice it's 4 because of horizon = 2!
            # First window: X[0] = [[10,30],[11,32],[12,34]], y[0] = [13,14]

    Args:
        data (pd.Series or pd.DataFrame): Can be:
            - Series -> univariate (target is the series itself)
            - DataFrame -> uni- or multi-variate. If multi-col, set `target_col`
        window_size (int): Number of past steps in each input window
        target_col (str): Column to predict when `data` is a DataFrame
            - If None and DataFrame has exactly 1 column, that column is used
            - If None and DataFrame has >1 column -> error
        horizon (int): Steps ahead to predict. 1 = next step; >1 = multi-step vector
        step (int): Stride of the sliding window (default 1)

    Returns:
        X (np.ndarray): (N, window_size, F) if ensure_3d, else (N, window_size) for univariate
        y (np.ndarray): (N,) if horizon==1 else (N, horizon)
    """
    if isinstance(data, pd.DataFrame):
        arr = data.values  # (Target, Features)
        tgt_idx = data.columns.get_loc(target_col)
    elif isinstance(data, pd.Series):
        arr = data.values.reshape(-1, 1)  # (Target, 1)
        tgt_idx = 0
    else:
        raise ValueError("data must be a pd.Series or pd.DataFrame")

    X_list, y_list = [], []
    T = arr.shape[0]
    max_start = T - (window_size + horizon)

    for i in range(0, max_start + 1, step):
        past = arr[i: i + window_size]  # past window as input of (window_size, F)
        future = arr[i + window_size: i + window_size + horizon, tgt_idx]  # future targets of (horizon,)
        X_list.append(past)
        if horizon == 1:
            y_list.append(future[0])
        else:
            y_list.append(future)

    # RNNs expect 3D input shaped (N, window_size, features), so we keep the feature dim even for univariate (F=1) to avoid unsqueezing later
    # - X_list holds N windows, each `past` is (window_size, F) with F=1 for univariate.
    # - np.stack(X_list, axis=0) inserts a new leading axis and packs the N windows, [(window_size, F)] * N  -->  (N, window_size, F)
    X = np.stack(X_list).astype(np.float32)  # (N, window_size, F)
    y = np.stack(y_list).astype(np.float32) if horizon > 1 else np.array(y_list, dtype=np.float32)  # (N,) for horizon=1, (N, horizon) elsewhere

    return X, y


def make_univariate_sequences(splits_norm, config):
    """
    Creates {feat: {res: (Xtr, ytr, Xval, yval, Xte, yte)}}, e.g., (X = T → y = T, X = rh → y = rh, X = SWDR → y = SWDR) for horizon = 1, and extra test sequences for specific horizons.

    Examples: Assume we have a simple list of T = [1,2,3,4,5,6,7,8,9,10]
        - For a window size = 3 and a horizon = 1:
            X[0] = [[1],[2],[3]], y[0] = 4 or X[1] = [[2],[3],[4]], y[1] = 5, ...
        - For a window size = 3 horizon = 4:
            X[0] = [[1],[2],[3]], y[0] = [4,5,6,7] or X[1] = [[2],[3],[4]], y4[1] = [5,6,7,8], ... so y is of shape (N, horizon)

    Args:
        splits_norm (dict): Dictionary with the normalized splits of DataFrames per resolution
        config (dict): Configuration dictionary for extracting sequence lengths (window_sizes) and horizons (how many steps ahead to predict)

    Returns:
        seqs (dict): Dictionary with the splits of DataFrames as window sized sequences per resolution
        seqs_test (dict): Dictionary with the test splits of DataFrames as window sized sequences per resolution for when horizon > 1
    """
    seqs = {}   # {feat: {res: (Xtr, ytr, Xval, yval, Xte, yte)}}
    seqs_test = {}  # {(feat, res, horizon): (Xte, yte)}  # extra test-only horizons

    h_win = config["nn_defaults"]["seq_len"]["1h"]
    s_win = config["nn_defaults"]["seq_len"]["6h"]
    d_win = config["nn_defaults"]["seq_len"]["24h"]
    horizon = config["horizons"][0]

    write_log("\n==== Series for NNs (Sequence Generation) ====\n\nUnivariate Sequences\n", filename="results/input_data_summary.txt")
    for feat, res_dict in splits_norm.items():
        if feat not in seqs:
            seqs[feat] = {}

        for res, (tr_df, va_df, te_df) in res_dict.items():
            if res == "1h":
                window_size = h_win
            elif res == "6h":
                window_size = s_win
            elif res == "24h":
                window_size = d_win
            else:
                window_size = 1

            # dataframes are 1-col, pass target_col=feat
            Xtr, ytr = create_sequences(tr_df, window_size, target_col=feat, horizon=horizon)
            Xval, yval = create_sequences(va_df, window_size, target_col=feat, horizon=horizon)
            Xte, yte = create_sequences(te_df, window_size, target_col=feat, horizon=horizon)

            seqs[feat][res] = (Xtr, ytr, Xval, yval, Xte, yte)

            # log concise info
            write_log(
                f"[{res} | {feat}] window={window_size}, horizon={horizon}:\n"
                f"  train: X{tuple(Xtr.shape)}, y{tuple(ytr.shape)}  "
                f"val: X{tuple(Xval.shape)}, y{tuple(yval.shape)}  "
                f"test: X{tuple(Xte.shape)}, y{tuple(yte.shape)}\n",
                filename="results/input_data_summary.txt")

            # create test sequences for res="1h", horizon=CONFIG["horizons"][1]=4 and res="6h", horizon=CONFIG["horizons"][2]=6
            if res == "1h":
                h2 = config["horizons"][2]  # e.g., 6
                Xte_h2, yte_h2 = create_sequences(te_df, window_size, target_col=feat, horizon=h2)
                seqs_test[(feat, res, h2)] = (Xte_h2, yte_h2)
                write_log(
                    f"[extra test] [{res} | {feat}] window={window_size}, horizon={h2}:  "
                    f"X{tuple(Xte_h2.shape)}, y{tuple(yte_h2.shape)}\n",
                    filename="results/input_data_summary.txt")
            if res == "6h":
                h1 = config["horizons"][1]  # e.g., 4
                Xte_h1, yte_h1 = create_sequences(te_df, window_size, target_col=feat, horizon=h1)
                seqs_test[(feat, res, h1)] = (Xte_h1, yte_h1)
                write_log(
                    f"[extra test] [{res} | {feat}] window={window_size}, horizon={h1}:  "
                    f"X{tuple(Xte_h1.shape)}, y{tuple(yte_h1.shape)}\n",
                    filename="results/input_data_summary.txt")

                # log first sample for horizon=1 and horizon=h1=4
                x0 = Xte[0].squeeze().tolist()  # (window_size, 1) -> (window_size,)
                y0 = float(yte[0])  # scalar (horizon=1)
                write_log(
                    f"[example] first sample of [{res} | {feat}] with horizon=1 and window_size={window_size}:\n"
                    f"\nX[0]={x0}\n"
                    f"y[0]={y0}\n",
                    filename="results/input_data_summary.txt")

                x0_h = Xte_h1[0].squeeze().tolist() # (window_size, 1) -> (window_size,)
                y0_h = yte_h1[0].tolist()   # vector (length = h1)
                write_log(
                    f"[example] first sample of [{res} | {feat}] with horizon={h1} and window_size={window_size}:\n"
                    f"\nX[0]={x0_h}\n"
                    f"y[0]={y0_h}\n",
                    filename="results/input_data_summary.txt")

    return seqs, seqs_test


def make_multivariate_sequences(splits_norm, features, target, config, resolutions):
    """
    Creates multivariate sequences {res: (Xtr, ytr, Xval, yval, Xte, yte)}, e.g., (X = T, rh, SWDR → y = T), and extra test sequences.

    Examples: Assume we have a tiny DataFrame with three features:
        T = [10,11,12,13,14,15,16,17]
        RH = [30,32,34,36,38,40,42,44]
        SWDR = [100,101,102,103,104,105,106,107], so features = ["T", "RH", "SWDR"], target = "T"

        - For window_size = 3 and horizon = 1:
            X[0] = [[10,30,100], [11,32,101], [12,34,102]], y[0] = 13 # scalar (next T)
            Shapes: X → (N, 3, 3), y → (N,)

        - For window_size = 3 and horizon = 4
            X[0] = [[10,30,100], [11,32,101], [12,34,102]], y[0] = [13,14,15,16] # vector of next 4 T values
            Shapes: X → (N, 3, 3), y → (N, 4)

    Args:
        splits_norm (dict): Dictionary with the normalized splits of DataFrames per resolution
        features (list): All the features that will be used as X
        target (str): Target column name
        config (dict): Configuration dictionary for extracting sequence lengths and horizons
        resolutions (list): Resolutions

    Returns:
        seqs_mv (dict): Dictionary with sequence splits per resolution for a horizon = 1
        seqs_mv_test (dict): Dictionary with test splits per resolution for a horizon > 1
    """
    seqs_mv = {}
    seqs_mv_test = {}

    h_win = config["nn_defaults"]["seq_len"]["1h"]
    s_win = config["nn_defaults"]["seq_len"]["6h"]
    d_win = config["nn_defaults"]["seq_len"]["24h"]
    horizon = config["horizons"][0]

    write_log("\nMultivariate Sequences\n", filename="results/input_data_summary.txt")
    for res in resolutions:
        # choose window_size per resolution
        if res == "1h":
            window_size = h_win
        elif res == "6h":
            window_size = s_win
        elif res == "24h":
            window_size = d_win
        else:
            window_size = 1

        tr_parts, va_parts, te_parts = [], [], []
        for f in features: # features == ["SWDR", "rh", "T"]
            if f in splits_norm and res in splits_norm[f]:
                tr_df, va_df, te_df = splits_norm[f][res]
                tr_parts.append(tr_df)
                va_parts.append(va_df)
                te_parts.append(te_df)

        # mv_tr = pd.concat(tr_parts, axis=1)
        # mv_va = pd.concat(va_parts, axis=1)
        # mv_te = pd.concat(te_parts, axis=1)

        # clean NaNs per column to avoid poisoned samples
        mv_tr = (pd.concat(tr_parts, axis=1)
                 .interpolate(limit_direction="both")  # fill gaps between valid values
                 .bfill()  # backfill if still missing at start
                 .ffill())  # forward fill if still missing at the end

        mv_va = (pd.concat(va_parts, axis=1).interpolate(limit_direction="both").bfill())
        mv_te = (pd.concat(te_parts, axis=1).interpolate(limit_direction="both").ffill())

        Xtr, ytr = create_sequences(mv_tr, window_size, target_col=target, horizon=horizon)
        Xval, yval = create_sequences(mv_va, window_size, target_col=target, horizon=horizon)
        Xte, yte = create_sequences(mv_te, window_size, target_col=target, horizon=horizon)

        seqs_mv[res] = (Xtr, ytr, Xval, yval, Xte, yte)

        # # log concise info (use target in the tag)
        write_log(
            f"[{res} | features X={features} | target y={target}] window={window_size}, horizon={horizon}:\n"
            f"  train: X{tuple(Xtr.shape)}, y{tuple(ytr.shape)}  "
            f"val: X{tuple(Xval.shape)}, y{tuple(yval.shape)}  "
            f"test: X{tuple(Xte.shape)}, y{tuple(yte.shape)}\n",
            filename="results/input_data_summary.txt")

        # create test sequences for res="1h", horizon=CONFIG["horizons"][2]=6 and res="6h", horizon=CONFIG["horizons"][1]=4
        if res == "1h":
            h2 = config["horizons"][2]  # e.g., 6
            Xte_h2, yte_h2 = create_sequences(mv_te, window_size, target_col=target, horizon=h2)
            seqs_mv_test[(res, h2)] = (Xte_h2, yte_h2)
            write_log(
                f"[extra test MV] [{res}] window={window_size}, horizon={h2}:  "
                f"X{tuple(Xte_h2.shape)}, y{tuple(yte_h2.shape)}\n",
                filename="results/input_data_summary.txt")
        if res == "6h":
            h1 = config["horizons"][1]  # e.g., 4
            Xte_h1, yte_h1 = create_sequences(mv_te, window_size, target_col=target, horizon=h1)
            seqs_mv_test[(res, h1)] = (Xte_h1, yte_h1)
            write_log(
                f"[extra test MV] [{res}] window={window_size}, horizon={h1}:  "
                f"X{tuple(Xte_h1.shape)}, y{tuple(yte_h1.shape)}\n",
                filename="results/input_data_summary.txt")

            # log first sample for horizon=1 and horizon=4
            x0_mv = Xte[0].tolist()  # (window_size, F)
            y0_mv = float(yte[0])
            write_log(
                f"[example] first sample of [{res}] MV with horizon=1 and window_size={window_size}:\n"
                f"\nX[0]={x0_mv}\n"
                f"y[0]={y0_mv}\n",
                filename="results/input_data_summary.txt")

            x0_h_mv = Xte_h1[0].tolist()  # (window_size, F)
            y0_h_mv = yte_h1[0].tolist()  # vector length = h1
            write_log(
                f"[example] first sample of [{res}] MV with horizon={h1} and window_size={window_size}:\n"
                f"\nX[0]={x0_h_mv}\n"
                f"y[0]={y0_h_mv}\n",
                filename="results/input_data_summary.txt")

    return seqs_mv, seqs_mv_test


def processed_data_summary(processed_data):
    """
    Summarizes the structure of processed_data and writes it to input_data_summary.txt.
    """
    write_log("\n\n==== processed_data Summary ====", filename="results/input_data_summary.txt")

    # 1. resampled: {res: DataFrame}
    # Example: {"1h": DataFrame(shape=(8784, 3), cols=["T","SWDR","rh"]),
    #           "6h": DataFrame(shape=(1465, 3), cols=["T","SWDR","rh"]),
    #           "24h": DataFrame(shape=(367, 3), cols=["T","SWDR","rh"])}
    resampled = processed_data["resampled"]
    lines = [
        "\n-- resampled: {res: DataFrame} --\n",
        f"container type: {type(resampled)}",
        f"num entries: {len(resampled)}",
        f"keys (resolutions): {list(resampled.keys())}",
        "values (per resolution):",
    ]
    for res, df in resampled.items():
        cols = list(df.columns)
        lines.append(f" [{res}] type={type(df)}, shape={df.shape}, cols={cols}")
    write_log("\n".join(lines), filename="results/input_data_summary.txt")

    # 2. final_stationary: list[(res, feat)]
    # Example: [("1h","T"), ("1h","SWDR"), ("6h","rh"), ...]
    final_stationary = processed_data["final_stationary"]
    lines = [
        "\n-- final_stationary: list[(res, feat)] --\n",
        f"container type: {type(final_stationary)}",
        f"length: {len(final_stationary)}",
        "values:"
    ]
    for item in final_stationary[:10]:
        lines.append(f" {item}")
    write_log("\n".join(lines), filename="results/input_data_summary.txt")

    # 3. stationary_series_data: {(res, feat): pandas.Series}
    # Example keys: [("1h","T"), ("1h","SWDR"), ("1h","rh"), ("6h","rh"), ("24h","rh"), ...]
    # Each value is a stationary series indexed by datetime (per resolution/feature).
    stationary_series_data = processed_data["stationary_series_data"]
    lines = [
        "\n-- stationary_series_data: {(res, feat): pandas.Series} --\n",
        f"container type: {type(stationary_series_data)}",
        f"num entries: {len(stationary_series_data)}",
        f"keys: {list(stationary_series_data.keys())}",
        "values:"
    ]
    for (res, feat), series in list(stationary_series_data.items()):
        lines.append(f" ({res},{feat}): type={type(series)}, shape={series.shape}, head=\n{series.head()}\n")
    write_log("\n".join(lines), filename="results/input_data_summary.txt")

    # 4. level_splits: {feat: {res: (train_idx, val_idx, test_idx)}}
    # Example: {"T": {"1h": (df_tr, df_val, df_te), "6h": (...), "24h": (...)}, "SWDR": {...}, "rh": {...}}
    level_splits = processed_data["level_splits"]
    lines = [
        "\n-- level_splits: {feat: {res: (train_idx, val_idx, test_idx)}} --\n",
        f"container type: {type(level_splits)}",
        f"num entries: {len(level_splits)}",
        f"keys (features): {list(level_splits.keys())}",
        "values (per feature) is another dictionary:"
    ]
    for feat, res_dict in level_splits.items():
        res_keys = list(res_dict.keys())
        lines.append(f"num entries: {len(res_keys)}")
        lines.append(f"keys: {res_keys}")
        lines.append(f" [{feat}] resolutions: {res_keys}")
        # show up to 3 resolution examples per feature
        for res, (tr_df, val_df, te_df) in list(res_dict.items())[:3]:
            lines.append(
                f"    ({res}) train={tr_df.shape}, val={val_df.shape}, test={te_df.shape} ")
                #f"cols(train)={list(tr_df.columns)}")
    write_log("\n".join(lines), filename="results/input_data_summary.txt")

    # 5. splits_norm: {feat: {res: (train_df, val_df, test_df)}}
    # Example: {"T": {"1h": (df_tr, df_val, df_te), "6h": (...), "24h": (...)}, "SWDR": {...}, "rh": {...}}
    splits_norm = processed_data["splits_norm"]
    lines = [
        "\n-- splits_norm: {feat: {res: (train_df, val_df, test_df)}} --\n",
        f"container type: {type(splits_norm)}",
        f"num entries: {len(splits_norm)}",
        f"keys (features): {list(splits_norm.keys())}",
        "values (per feature) is another dictionary:"
    ]
    for feat, res_dict in splits_norm.items():
        res_keys = list(res_dict.keys())
        lines.append(f"num entries: {len(res_keys)}")
        lines.append(f"keys: {res_keys}")
        lines.append(f" [{feat}] resolutions: {res_keys}")
        # show up to 3 resolution examples per feature
        for res, (tr_df, val_df, te_df) in list(res_dict.items())[:3]:
            lines.append(
                f"    ({res}) train={tr_df.shape}, val={val_df.shape}, test={te_df.shape} ")
                #f"cols(train)={list(tr_df.columns)}")
    write_log("\n".join(lines), filename="results/input_data_summary.txt")

    # 6. scalers: {(res, feat): {"mean": mu, "std": sd}}
    # Example: {("1h","T"): {"mean": 15.23, "std": 7.81}, ("6h","SWDR"): {...}, ...}
    scalers = processed_data["scalers"]
    lines = [
        "\n-- scalers: {(res, feat): {'mean': mu, 'std': sd}} --\n",
        f"container type: {type(scalers)}",
        f"num entries: {len(scalers)}",
        f"keys: {list(scalers.keys())}",
    ]
    resolutions = sorted({res for (res, feat) in scalers.keys()})
    features = sorted({feat for (res, feat) in scalers.keys()})
    lines.append(f"resolutions: {resolutions}")
    lines.append(f"features: {features}")
    lines.append("values:")
    for (res, feat), s in list(scalers.items()):
        lines.append(f" ({res},{feat}): mean={s['mean']:.6f}, std={s['std']:.6f}")
    write_log("\n".join(lines), filename="results/input_data_summary.txt")

    # 7. seqs_uni: {feat: {res: (Xtr, ytr, Xval, yval, Xte, yte)}}
    # Example: {"T": {"1h": (Xtr,ytr,Xval,yval,Xte,yte), "6h": (...), "24h": (...)}, "SWDR": {...}, "rh": {...}}
    seqs_uni = processed_data["seqs_uni"]
    lines = [
        "\n-- seqs_uni: {feat: {res: (Xtr, ytr, Xval, yval, Xte, yte)}} --\n",
        f"container type: {type(seqs_uni)}",
        "num entries: {len(seqs_uni)}",
        f"keys (features): {list(seqs_uni.keys())}",
        "values (per feature) is another dictionary:"
    ]
    for feat, res_dict in seqs_uni.items():
        res_keys = list(res_dict.keys())
        lines.append(f"num entries: {len(res_keys)}")
        lines.append(f"keys: {res_keys}")
        lines.append(f" [{feat}] resolutions: {res_keys}")
        for res, (Xtr, ytr, Xval, yval, Xte, yte) in list(res_dict.items()):
            lines.append(
                f"    ({res}) Xtr{Xtr.shape}, ytr{ytr.shape}, "
                f"Xval{Xval.shape}, yval{yval.shape}, Xte{Xte.shape}, yte{yte.shape}")
    write_log("\n".join(lines), filename="results/input_data_summary.txt")

    # 8. seqs_uni_test: {(feat, res, horizon): (Xte, yte)}
    # Example: {("T","1h",6): (Xte,yte), ("T","6h",4): (Xte,yte), ...}
    seqs_uni_test = processed_data["seqs_uni_test"]
    lines = [
        "\n-- seqs_uni_test: {(feat, res, horizon): (Xte, yte)} --\n",
        f"container type: {type(seqs_uni_test)}",
        f"num entries: {len(seqs_uni_test)}",
        f"keys: {list(seqs_uni_test.keys())}"
    ]
    feats_set = sorted({k[0] for k in seqs_uni_test.keys()})
    res_set = sorted({k[1] for k in seqs_uni_test.keys()})
    horizons_set = sorted({k[2] for k in seqs_uni_test.keys()})
    lines.append(f"features: {feats_set}")
    lines.append(f"resolutions: {res_set}")
    lines.append(f"horizons: {horizons_set}")
    lines.append("values:")
    for (feat, res, h), (Xte, yte) in list(seqs_uni_test.items()):
        lines.append(f" ({feat},{res},h={h}) -> Xte{Xte.shape}, yte{yte.shape}")
    write_log("\n".join(lines), filename="results/input_data_summary.txt")

    # 9. seqs_mv: {res: (Xtr, ytr, Xval, yval, Xte, yte)}
    # Example: {"1h": (Xtr,ytr,Xval,yval,Xte,yte), "6h": (...), "24h": (...)}
    seqs_mv = processed_data["seqs_mv"]
    lines = [
        "\n-- seqs_mv: {res: (Xtr, ytr, Xval, yval, Xte, yte)} --\n",
        f"container type: {type(seqs_mv)}",
        f"num entries: {len(seqs_mv)}",
        f"keys (resolutions): {list(seqs_mv.keys())}",
        "values (per resolution):"
    ]
    for res, (Xtr, ytr, Xval, yval, Xte, yte) in list(seqs_mv.items()):
        lines.append(
            f" [{res}] Xtr{Xtr.shape}, ytr{ytr.shape}, "
            f"Xval{Xval.shape}, yval{yval.shape}, "
            f"Xte{Xte.shape}, yte{yte.shape}")
    write_log("\n".join(lines), filename="results/input_data_summary.txt")

    # 10. seqs_mv_test: {(res, horizon): (Xte, yte)}
    # Example: {("1h",6): (Xte,yte), ("6h",4): (Xte,yte), ...}
    seqs_mv_test = processed_data["seqs_mv_test"]
    lines = [
        "\n-- seqs_mv_test: {(res, horizon): (Xte, yte)} --\n",
        f"container type: {type(seqs_mv_test)}",
        f"num entries: {len(seqs_mv_test)}",
        f"keys: {seqs_mv_test.keys()}",
        "values:"
    ]
    for (res, h), (Xte, yte) in list(seqs_mv_test.items()):
        lines.append(f" ({res}, h={h}) -> Xte{Xte.shape}, yte{yte.shape}")
    write_log("\n".join(lines), filename="results/input_data_summary.txt")

    # 11. uni_loaders: {feat: {res: {"train": DL, "val": DL, "test": DL}}}
    # Example: {"T": {"1h": {"train": DL, "val": DL, "test": DL}, "6h": {...}, "24h": {...}}, "SWDR": {...}, "rh": {...}}
    uni_loaders = processed_data["uni_loaders"]
    lines = [
        "\n-- uni_loaders: {feat: {res: {'train': DL, 'val': DL, 'test': DL}}} --\n",
        f"container type: {type(uni_loaders)}",
        f"num entries: len(uni_loaders)",
        f"keys (features): {list(uni_loaders.keys())}",
        "values (per feature → resolution → split):"
    ]
    for feat, res_dict in uni_loaders.items():
        lines.append(f" [{feat}] resolutions: {list(res_dict.keys())}")
        for res, split_dict in res_dict.items():  # split_dict = {"train": DL, "val": DL, "test": DL}
            for split_name, dl in split_dict.items():
                xb, yb = next(iter(dl))
                lines.append(f"    ({res}) {split_name}: DataLoader batch X{xb.shape}, y{yb.shape}")
    write_log("\n".join(lines), filename="results/input_data_summary.txt")

    # 12. uni_test_loaders: {(feat, res, horizon): DataLoader}
    # Example: {("T","1h",6): DL, ("T","6h",4): DL, ...}
    uni_test_loaders = processed_data["uni_test_loaders"]
    lines = [
        "\n-- uni_test_loaders: {(feat, res, horizon): DataLoader} --\n",
        f"container type: {type(uni_test_loaders)}",
        f"num entries: {len(uni_test_loaders)}",
        f"keys (features): {list(uni_test_loaders.keys())}",
        "values:"
    ]
    for (feat, res, h), dl in list(uni_test_loaders.items()):
        xb, yb = next(iter(dl))
        lines.append(f" ({feat}, {res}, h={h}) -> DataLoader batch X{xb.shape}, y{yb.shape}")
    write_log("\n".join(lines), filename="results/input_data_summary.txt")

    # 13. mv_loaders: {res: {"train": DL, "val": DL, "test": DL}}
    # Example: {"1h": {"train": DL, "val": DL, "test": DL},
    #           "6h": {"train": DL, "val": DL, "test": DL},
    #           "24h": {"train": DL, "val": DL, "test": DL}}
    mv_loaders = processed_data["mv_loaders"]
    lines = [
        "\n-- mv_loaders: {res: {'train': DL, 'val': DL, 'test': DL}} --\n",
        f"container type: {type(mv_loaders)}",
        f"num entries: {len(mv_loaders)}",
        f"keys (resolutions): {list(mv_loaders.keys())}",
        "values (per resolution → split):"
    ]
    for res, split_dict in mv_loaders.items():  # split_dict = {"train":DL,"val":DL,"test":DL}
        lines.append(f" [{res}] splits: {list(split_dict.keys())}")
        for split_name, dl in split_dict.items():
            xb, yb = next(iter(dl))
            lines.append(f"    {split_name}: DataLoader batch X{xb.shape}, y{yb.shape}")
    write_log("\n".join(lines), filename="results/input_data_summary.txt")

    # 14. mv_test_loaders: {(res, horizon): DataLoader}
    # Example: {("1h",6): DL, ("6h",4): DL, ...}
    mv_test_loaders = processed_data["mv_test_loaders"]
    lines = [
        "\n-- mv_test_loaders: {(res, horizon): DataLoader} --\n",
        f"container type: {type(mv_test_loaders)}",
        f"num entries: {len(mv_test_loaders)}",
        f"keys (features): {list(mv_test_loaders.keys())}",
        "values:"
    ]
    for (res, h), dl in list(mv_test_loaders.items()):
        xb, yb = next(iter(dl))
        lines.append(f" ({res}, h={h}) -> DataLoader batch X{xb.shape}, y{yb.shape}")
    write_log("\n".join(lines), filename="results/input_data_summary.txt")


def to_loader(X, y, batch_size=32, shuffle=False):
    """
    Creates a DataLoader from NumPy arrays, X: (N, L, F), y: (N,) or (N, H).
    """
    tx = torch.from_numpy(X).float()
    ty = torch.from_numpy(y).float()
    if ty.ndim == 1:
        ty = ty.view(-1, 1)  # (N, 1)
    ds = TensorDataset(tx, ty)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def build_dataloaders_from_seqs(seqs_uni, seqs_uni_test, seqs_mv, seqs_mv_test, batch_size_by_res=32, log_file="results/input_data_summary.txt"):
    """
    Converts sequence dicts to DataLoaders and logs their shapes.

    Args:
        seqs_uni (dict): Univariate sequence splits {feat: {res: (Xtr, ytr, Xval, yval, Xte, yte)}}
        seqs_uni_test (dict): Extra univariate test-only sequences {(feat, res, horizon): (Xte, yte)}
        seqs_mv (dict): Multivariate sequence splits {res: (Xtr, ytr, Xval, yval, Xte, yte)}
        seqs_mv_test (dict): Extra multivariate test-only sequences {(res, horizon): (Xte, yte)}
        batch_size_by_res (dict or int, optional): Batch size per resolution (e.g. {"1h": 64, "6h": 64}) or int for all
        log_file (str, optional): Path to log file

    Returns:
        tuple: (uni_loaders, uni_test_loaders, mv_loaders, mv_test_loaders)
            - uni_loaders (dict): {feat: {res: {"train","val","test": DataLoader}}}
            - uni_test_loaders (dict): {(feat, res, horizon): DataLoader}
            - mv_loaders (dict): {res: {"train","val","test": DataLoader}}
            - mv_test_loaders (dict): {(res, horizon): DataLoader}
    """
    uni_loaders, uni_test_loaders, mv_loaders, mv_test_loaders = {}, {}, {}, {}

    with open(log_file, "a", encoding="utf-8") as f:
        f.write("\n==== Univariate Dataloaders ====\n\n")

        # univariate loaders
        for feat, res_dict in seqs_uni.items():
            uni_loaders[feat] = {}
            for res, (Xtr, ytr, Xval, yval, Xte, yte) in res_dict.items():
                uni_loaders[feat][res] = {
                    "train": to_loader(Xtr, ytr, batch_size=batch_size_by_res),
                    "val":   to_loader(Xval, yval, batch_size=batch_size_by_res),
                    "test":  to_loader(Xte, yte, batch_size=batch_size_by_res),
                }
                f.write(f"[{res} | {feat}] train: X{Xtr.shape}, y{ytr.shape} "
                        f"val: X{Xval.shape}, y{yval.shape} "
                        f"test: X{Xte.shape}, y{yte.shape}\n")

        for (feat, res, h), (Xte, yte) in seqs_uni_test.items():
            uni_test_loaders[(feat, res, h)] = to_loader(Xte, yte, batch_size=batch_size_by_res)
            f.write(f"[extra test] [{res} | {feat}] horizon={h}: "
                    f"X{Xte.shape}, y{yte.shape}\n")

        f.write("\n==== Multivariate Dataloaders ====\n\n")

        # multivariate loaders
        for res, (Xtr, ytr, Xval, yval, Xte, yte) in seqs_mv.items():
            mv_loaders[res] = {
                "train": to_loader(Xtr, ytr, batch_size=batch_size_by_res),
                "val":   to_loader(Xval, yval, batch_size=batch_size_by_res),
                "test":  to_loader(Xte, yte, batch_size=batch_size_by_res),
            }
            f.write(f"[{res} MV] train: X{Xtr.shape}, y{ytr.shape} "
                    f"val: X{Xval.shape}, y{yval.shape} "
                    f"test: X{Xte.shape}, y{yte.shape}\n")

        for (res, h), (Xte, yte) in (seqs_mv_test or {}).items():
            mv_test_loaders[(res, h)] = to_loader(Xte, yte, batch_size=batch_size_by_res)
            f.write(f"[extra test MV] [{res}] horizon={h}: "
                    f"X{Xte.shape}, y{yte.shape}\n")

        # inspect one loader as example (here: 1h MV train)
        if "1h" in mv_loaders and "train" in mv_loaders["1h"]:
            dl = mv_loaders["1h"]["train"]
            dataset = dl.dataset
            num_batches = len(dl)
            dataset_len = len(dataset)

            x0, y0 = dataset[0]
            xb, yb = next(iter(dl))

            f.write("\n---- Inspecting 1h MV train DataLoader ----\n\n")
            f.write(f"Num batches: {num_batches} of size 32\n")
            f.write(f"Dataset length (samples): {dataset_len}\n\n")

            f.write(f"Dataset sample X shape: {x0.shape}\n")
            f.write(f"Dataset sample y shape: {y0.shape}\n")
            f.write(f"Example x^(t) (dataset sample X[0]) shape: {x0[0].shape}\n")
            f.write(f"Example x^(t) (dataset sample X[0]) value: {x0[0]}\n")
            f.write(f"Example y value: {y0}\n\n")

            f.write(f"First batch X shape: {xb.shape}\n")
            f.write(f"First batch y shape: {yb.shape}\n")
            f.write(f"Example x^(t) (batch sample X[0,0]) shape: {xb[0, 0].shape}\n")
            f.write(f"Example x^(t) (batch sample X[0,0]) value: {xb[0, 0]}\n")
            f.write(f"Example y value: {yb[0]}\n")
            f.write(f"Tensor dtypes -> X: {xb.dtype} | y: {yb.dtype}\n")

    return uni_loaders, uni_test_loaders, mv_loaders, mv_test_loaders
