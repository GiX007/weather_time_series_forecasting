# src.eda.py
#
# Helper functions for Exploratory Data Analysis.
#
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_pacf
from src.utils import write_log

# paths
eda_base = "results/figures/eda/"
overview = os.path.join(eda_base, "overview")
roll_stats = os.path.join(eda_base, "rolling_stats")
differencing = os.path.join(eda_base, "differencing")
pacfs = os.path.join(eda_base, "pacfs")


def summary_info(df, name="Dataset"):
    """
    Prints summary information for the given DataFrame

    Args:
        df (pandas.DataFrame): The dataset to summarize
        name (str): The name of the dataset
    """
    write_log(f"\n==== Summary for the {name} ====")
    write_log(f"Shape: {df.shape}")
    write_log(f"Columns: {df.columns}")
    write_log(f"Index type: {type(df.index)}")
    write_log(f"Index range: index={df.index.min()} → {df.index.max()}")
    write_log(f"Monotonic index: {df.index.is_monotonic_increasing}")
    write_log(f"Duplicate index values: {df.index.duplicated().sum()}")

    write_log("\nColumn dtypes:")
    write_log(df.dtypes)

    write_log("\nMissing values:")
    write_log(df.isna().sum())
    # for col in df.columns:
    #     n_missing = df[col].isna().sum()
    #     perc_missing = (n_missing / len(df)) * 100
    #     print(f" {col}: {n_missing} ({perc_missing:.2f}%)")

    write_log("\nDescriptive statistics:")
    write_log(df.describe(include="all"))

    write_log(f"\nFirst 5 rows:\n {df.head()}")


def correlation_analysis(df, name="Dataset", save_dir=overview):
    """
    Computes and plots correlation matrix for the given DataFrame, shows correlation heatmap

    Args:
        df (pandas.DataFrame): The dataset to compute correlation matrix for
        name (str): The name of the dataset
        save_dir (str): The directory to save the figure to
    """
    corr = df.corr()

    # create save dir, if not exists
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize = (15,10))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(f"{name} - Correlation Heatmap", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    save_path = os.path.join(save_dir, f"{name}_correlation_heatmap.png")
    plt.savefig(save_path)
    plt.close()


def scatter_vs_target(df, feature, target="T", save_dir=overview):
    """
    Creates and saves scatterplot of target vs. a given feature

    Args:
        df (pandas.DataFrame): The dataframe with features and target
        feature (str): name of feature column to plot
        target (str): name of target variable (default = "T")
        save_dir (str): directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize = (14, 8))
    sns.scatterplot(data=df, x=feature, y=target)
    plt.title(f"{target} vs. {feature}", fontsize=16, fontweight="bold")
    plt.xlabel(feature, fontsize=14)
    plt.ylabel(target, fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.4)  # light dashed grid
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    save_path = os.path.join(save_dir, f"scatter_{target}_vs_{feature}.png")
    plt.savefig(save_path)
    plt.close()


def run_eda(weather_df):
    """
    Runs a lightweight EDA and logs results

    Args:
        weather_df (pandas.DataFrame): The dataframe with features and target
    """
    write_log("==== Weather Forecasting Project ====", mode="w")    # all info to a txt in ./results
    write_log("All basic EDA information is logged here.")
    summary_info(weather_df, "Original Dataset")
    correlation_analysis(weather_df, "Original Dataset")
    scatter_vs_target(weather_df, "rh", target="T")
    scatter_vs_target(weather_df, "SWDR", target="T")


def plot_dataset(df, res="1h", save_dir=overview):
    """
    Plots per-time-group summaries (box, line, bar) for all columns/features

    Args:
        df (pandas.DataFrame): The dataframe with features
        res (str, optional): Resolution of the dataset. Controls grouping and titles
        save_dir (str): directory to save plots
    """
    # extract grouping var from datetime index
    res = str(res).upper()
    if res == "1H":
        group_values = df.index.hour    # "1H" -> for grouping by hour (0..23)
    elif res == "6H":
        group_values = df.index.hour // 6   # "6H" -> for grouping by 6-hour slot (0..3)
    elif res ==  "24H":
        group_values = df.index.date    # "24H" -> for grouping by date
    else:
        group_values = []

    features = list(df.columns)
    n_rows = len(features)

    fig, axes = plt.subplots(nrows=n_rows, ncols=3, figsize = (20, 4 * n_rows))
    fig.suptitle(f"{res} Feature Distributions and Trends", fontsize=16, fontweight="bold")

    for i, feat in enumerate(features):
        grouped = df.groupby(group_values)[feat].mean()

        if res == "24H":
            # Only line chart makes sense
            sns.lineplot(ax=axes[i, 1], x=grouped.index, y=grouped.values)
            axes[i, 1].set_title(f"{feat} — Line Chart")
            axes[i, 0].axis("off")
            axes[i, 2].axis("off")

        else:
            # box plot
            sns.boxplot(ax=axes[i, 0], x=group_values, y=df[feat])
            axes[i, 0].set_title(f"Average {feat} - Box Plot")

            # line chart
            sns.lineplot(ax=axes[i, 1], x=grouped.index, y=grouped.values)
            axes[i, 1].set_title(f"Average {feat} - Line Chart")

            # bar chart
            sns.barplot(ax=axes[i, 2], x=grouped.index, y=grouped.values)
            axes[i, 2].set_title(f"Average {feat} - Bar Chart")

        # common settings
        for ax in axes[i]:
            ax.set_xlabel(f"Time in {res} Resolution")
            ax.set_ylabel(feat)
            ax.grid(True, linestyle="--", alpha=0.4)  # light dashed grid

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(save_dir, f"{res}_dataset.png")
    plt.savefig(save_path)
    plt.close()


def plot_df_avg_line_trends(df_1h, df_6h, df_24h, first_days=5, save_dir=overview):
    """
    Plots line charts for each feature across resolutions in a 4-column grid:
        - first `first_days` days from the 1H dataframe,
        - 1H average by hour-of-day (0..23),
        - average by 6-hour slot-of-day (00–06, 06–12, 12–18, 18–24),
        - 24H timeline

    Assumes all DataFrames share the same columns and have a DatetimeIndex

    Args:
        df_1h (pandas.DataFrame): The first dataframe with features
        df_6h (pandas.DataFrame): The second dataframe with features
        df_24h (pandas.DataFrame): The third dataframe with features
        first_days (int, optional): The first day of the 1H dataframe
        save_dir (str): directory to save plots
    """
    cols = df_1h.columns
    n_rows, n_cols = len(cols), 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3 * n_rows))
    fig.suptitle("Line Trends: First Days, 1H Avg (Hour), 6H Avg (6-Hour Slot) and 24H", fontsize=16, fontweight="bold")

    # window for the first `first_days` days in 1H
    start = df_1h.index.min()
    end = start + pd.Timedelta(days=first_days)
    df_1h_first = df_1h.loc[start:end]

    # group df_1h by hour of the day (we plot the mean of all hours through the year)
    hod = df_1h.index.hour  # 0..23

    for i, col in enumerate(cols):

        # col 1: first days (1H)
        ax = axes[i, 0]
        ax.plot(df_1h_first.index, df_1h_first[col].values)
        ax.set_title(f"{col} - 1H (first {first_days} days)")
        ax.set_ylabel(f"{col}")
        ax.set_xlabel(f"Datetime (first {first_days} days)")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        for t in pd.date_range(start=start, end=end, freq='1D'):
            ax.axvline(t, color='red', linestyle='--', linewidth=0.8, alpha=0.4)

        # col 2: 1H avg by hour-of-day
        ax = axes[i, 1]
        hourly_avg = df_1h.groupby(hod)[col].mean()
        ax.plot(hourly_avg.index, hourly_avg.values, marker='o')
        ax.set_title(f"{col} - 1H avg by hour")
        ax.set_ylabel(f"{col}")
        ax.set_xlabel("Hour of the Day (0–23)")
        ax.set_xticks(range(0, 24, 2))
        ax.grid(True, linestyle="--", alpha=0.4)

        # col 3: 6H timeline
        ax = axes[i, 2]
        avg_6h = df_6h.groupby(df_6h.index.hour)[col].mean()
        ax.plot(avg_6h.index, avg_6h.values)
        ax.set_title(f"{col} - 6H avg by slot")
        ax.set_ylabel(f"{col}")
        ax.set_xlabel("6-Hour Slot (00, 06, 12, 18)")
        ax.set_xticks([0, 6, 12, 18])
        ax.grid(True, linestyle="--", alpha=0.4)

        # col 4: 24H timeline
        ax = axes[i, 3]
        ax.plot(df_24h.index, df_24h[col].values)
        ax.set_title(f"{col} - 24H")
        ax.set_ylabel(f"{col}")
        ax.set_xlabel("Date")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(save_dir, "avg_1h_avg_6h_24h_line_trends.png")
    plt.savefig(save_path)
    plt.close()


def plot_series_overview(resampled, resolutions, features, save_dir=overview):
    """
    Quick visual stationarity check for every (resolution, feature) pair:
    full series, last 1 day, last 1 week (or 30 days for daily)

    Args:
        resampled (dict): The resampled dataframes (of same features and DatetimeIndex)
        resolutions (lists): Sampling frequencies like "1H", "6H", "24H"
        features (lists): Feature names for titles/labels
        save_dir (str): Directory to save the figure to
    """
    n_rows = len(resolutions) * len(features)
    n_cols = 3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 3 * n_rows))
    fig.suptitle(f"Visual Series Overview (All Features × Resolutions)", fontsize=16, fontweight="bold")

    row = 0
    for freq in resolutions:    # "1h","6h","24h"
        df_res = resampled[freq]
        for feat in features:
            series = df_res[feat]

            # windows per resolution
            if freq == "1h":
                windows = [("Full series", len(series)), ("Last 1 day", 24), ("Last 1 week", 7 * 24)]
            elif freq == "6h":
                windows = [("Full series", len(series)), ("Last 1 day", 4), ("Last 1 week", 7 * 4)]
            elif freq == "24h":
                windows = [("Full series", len(series)), ("Last 1 week", 7), ("Last 30 days", 30)]
            else:
                windows = [("Full series", len(series))]

            for j, (label, npts) in enumerate(windows):
                ax = axes[row, j]   # plot index
                view = series.iloc[-npts:]  # use this to be able to retrieve the last 1 or 7 days
                ax.plot(view.index, view.values)
                ax.set_title(f"{feat} @ {freq} — {label}")
                ax.set_xlabel("Time")
                ax.set_ylabel(feat)
                ax.grid(True, linestyle="--", alpha=0.4)
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())    # for nicer date ticks
                ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))

                # quick mean/std annotation box
                mu = view.mean()
                sd = view.std()
                ax.text(0.02, 0.90, f"mean={mu:.2f}\nstd={sd:.2f}",
                        transform=ax.transAxes, va="top", ha="left",
                        bbox=dict(boxstyle="round", alpha=0.15, lw=0))

            row += 1

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(save_dir, "ts_visual_overview.png")
    plt.savefig(save_path)
    plt.close(fig)


def plot_rolling_stats(series, res, feat, save_dir=roll_stats):
    """
    Plots rolling statistics over a series

    Args:
        series (DataFrame): The series to plot
        res (list): Resampling frequencies
        feat (str): Feature name
        save_dir (str): Directory to save the figure to
    """
    # create the save dir
    os.makedirs(save_dir, exist_ok=True)

    # choose a reasonable default window size (just for plotting)
    if res == "1h":
        window = 24
    elif res == "6h":
        window = 28
    elif res == "24h":
        window = 7
    else:
        window = 30

    ts_mean = series.rolling(window=window, min_periods=1).mean()
    ts_std = series.rolling(window=window, min_periods=1).std()

    plt.figure(figsize=(14, 8))
    plt.plot(series.index, series.values, label='Original')
    plt.plot(ts_mean.index, ts_mean.values, label=f'Rolling Mean (w={window})')
    plt.plot(ts_std.index, ts_std.values, label=f'Rolling Std (w={window})')
    plt.legend(loc="upper left")
    plt.title(f'Rolling Mean & Standard Deviation for {res} for {feat}', fontsize=16, fontweight="bold")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    save_path = os.path.join(save_dir, f"{res}_{feat}_rolling_stats_for_stationarity.png")
    plt.savefig(save_path)
    plt.close()


def test_stationarity(series, res, feat, log_file="results/stationarity_summary.txt"):
    """
    Plots rolling stats and applies the ADF test for stationarity

    Args:
        series (pd.Series or pd.DataFrame):
         - Series: univariate (single time series)
         - DataFrame: multivariate (each numeric column is a feature time series)
        res (str): Resolution label (e.g., '1h', '6h', '24h')
        feat (str): Feature label for titles/filenames
        log_file (str): Where figures are saved
    """
    # collectors (lists of (res, feature))
    stationary = []
    needs_diff = []

    # if it's a Dataframe, pass its columns recursively
    if isinstance(series, pd.DataFrame):
        for col in series.columns:
            nd, st = test_stationarity(series[col], res, str(col), log_file)
            needs_diff += nd
            stationary += st
        return needs_diff, stationary

    # if it's not a DataFrame (series), plot rolling stats and apply adf test
    plot_rolling_stats(series, res, feat)

    write_log(f"\nResults of Dickey-Fuller Test for {res} {feat}:", log_file)

    adf_stat, pval, usedlag, nobs, crit_vals, icbest = adfuller(series, autolag="AIC")
    if pval <= 0.05:
        msg = "The time series is stationary"
        stationary.append((res, str(feat)))
    else:
        msg = "The time series is non-stationary"
        needs_diff.append((res, str(feat)))
    write_log(msg, log_file)

    return needs_diff, stationary


def plot_nonstat_vs_diff(original_series, diff_series, res, feat, save_dir=differencing):
    """
    Side-by-side plot: (left) original non-stationary series, (right) differenced stationary series

    Args:
        original_series (pd.Series): The original (non-stationary) series
        diff_series (pd.Series): The corresponding differenced (stationary) series
        res (str): Resolution label (e.g., '1h', '6h', '24h')
        feat (str): Feature name
        save_dir (str): Where to save the figure
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    fig.suptitle(f"{feat} [{res}] - Non-stationary vs Differenced", fontsize=16, fontweight="bold")

    axes[0].plot(original_series.index, original_series.values)
    axes[0].set_title(f"{feat} [{res}] — Original Series")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Value")
    plt.grid(True, linestyle="--", alpha=0.4)

    axes[1].plot(diff_series.index, diff_series.values)
    axes[1].set_title(f"{feat} [{res}] - Differenced Series (lag=1)")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Δ Value")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    save_path = os.path.join(save_dir, f"{res}_{feat}_nonstat_vs_diff.png")
    plt.savefig(save_path)
    plt.close()


def pacf_plot(series, res_feat, max_lags=40, save_dir=pacfs):
    """
    Plots the Partial Autocorrelation Function (PACF) for a time series

    Args:
        series (pd.Series): The input time series (should be stationary)
        res_feat (tuple): (resolution, feature) pair
        max_lags (int): Maximum number of lags to display
        save_dir (str): Where to save the figure
    """
    os.makedirs(save_dir, exist_ok=True)

    res, feat = res_feat

    plt.figure(figsize=(14, 8))
    plot_pacf(series, lags=max_lags)
    plt.title(f"{feat} [{res}] - PACF", fontsize=16, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    save_path = os.path.join(save_dir, f"{res}_{feat}_pacf.png")
    plt.savefig(save_path)
    plt.close()
