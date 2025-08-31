# src/utils.py
#
# Helper functions for common operations.
#
import os, shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import kagglehub
import torch


def load_dataset():
    """
    Ensures the weather dataset is available in ./data. If .CSV files are already there, reuse them. Otherwise, download it.
    """
    dataset_dir = "data"
    os.makedirs(dataset_dir, exist_ok=True)

    #  if a CSV already exists, return it (skip downloading)
    for f in os.listdir(dataset_dir):
        if f.endswith(".csv"):
            csv_path = os.path.join(dataset_dir, f)
            print("Using existing dataset:", csv_path)
            return csv_path

    # otherwise, download from kaggle via kagglehub
    path = kagglehub.dataset_download("alistairking/weather-long-term-time-series-forecasting")
    # print("Path to dataset files:", path)

    # copy the CSV to ./data
    for f in os.listdir(path):
        if f.endswith(".csv"):
            local_path = os.path.join(dataset_dir, f)
            shutil.copy(os.path.join(path, f), local_path)
            print("Downloaded dataset copied to:", local_path)
            return local_path

    # if no csv found at all (shouldn’t happen, but just in case)
    raise FileNotFoundError("No CSV file found in dataset.")


def write_log(message, filename="results/data_summary.txt", mode="a"):
    """
    Appends a message to a log file (creates a file if not exists).

    Args:
        message (str): The message to append
        filename (str): The filename to save the log to
        mode (str): The mode to save the log. It can be 'a' for appending or 'w' for writing to a clear file
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # convert all to strings
    if not isinstance(message, str):
        message = str(message)

    with open(filename, mode, encoding="utf-8") as f:
        f.write(message + "\n")


def plot_residuals_and_predictions(test, preds, res, feat, horizon, model_type, date_fmt="%Y-%m-%d %H:%M", save_dir="results/figures/predictions"):
    """
    Plots residuals (test - preds) and Test vs. Predictions side-by-side.

    Args:
        test (pd.Series or 1D array-like): Ground truth values
        preds (pd.Series or 1D array-like): Predictions
        res (str): Resolution (e.g. "1h", "6h", "24h") for title/filename
        feat (str): Feature name (e.g. "T", "rh", "SWDR") for title/filename
        horizon (int): Horizon for title/filename
        model_type (str): Type of model, e.g., AR, RNN, etc.
        date_fmt (str): Datetime tick format when index is datetime-like
        save_dir (str): Output directory to save plots
    """
    # convert to series, if needed
    if not isinstance(test, pd.Series):
        test = pd.Series(test)
    if not isinstance(preds, pd.Series):
        preds = pd.Series(preds, index=test.index if test.index.size == len(preds) else None)

    # align on common index
    common_index = test.index.intersection(preds.index)
    test_al = test.loc[common_index].sort_index()
    preds_al = preds.loc[common_index].sort_index()

    # compute residuals
    residuals = test_al - preds_al

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(f"{feat} [{res}] for {model_type} (t+{horizon}) - Residuals vs. Predictions", fontsize=16, fontweight="bold")

    # plot residuals
    ax0 = axes[0]
    ax0.plot(residuals, label="Residuals")
    ax0.axhline(0, linestyle="--", color='red', linewidth=1)
    ax0.set_title(f"Residuals ({res}, {feat})", fontsize=12)
    ax0.set_ylabel("Error")
    ax0.set_xlabel("Time")
    ax0.legend()

    # plot test vs. predictions
    ax1 = axes[1]
    ax1.plot(test_al, label="Test Data")
    ax1.plot(preds_al, label="Predictions")
    ax1.set_title(f"Test vs Predictions ({res}, {feat})", fontsize=12)
    ax1.set_ylabel(feat)
    ax1.set_xlabel("Time")
    ax1.legend()

    # datetime formatting
    if pd.api.types.is_datetime64_any_dtype(test_al.index):
        for ax in axes:
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter(date_fmt))
            for label in ax.get_xticklabels():
                label.set_rotation(45)
                label.set_ha("right")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # save plot
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{res}_{feat}_{model_type}_(t+{horizon})_residuals_preds.png")
    plt.savefig(save_path)
    plt.close()


def evaluate_model(y_true, y_pred):
  """
  Computes MAE, RMSE, and MAPE (%) and return them as formatted strings.

  Args:
      y_true (array-like): Ground truth values
      y_pred (array-like): Model predictions
  """
  # ensure y_true and y_pred are of the same lengths
  if len(y_true) != len(y_pred):
    raise ValueError("Input arrays must have the same length.")

  # MAE: mean absolute error
  mae = np.mean(np.abs(y_true - y_pred))

  # RMSE: root mean squared error
  rmse = np.sqrt(np.mean((y_true - y_pred)**2))

  # MAPE: mean absolute percentage error
  denom = np.where(y_true != 0, y_true, 1e-8)  # if 0 -> use 1e-8 (avoid division by zero)
  mape = np.mean(np.abs((y_true - y_pred) / denom)) * 100

  return f"{mae:.2f}", f"{rmse:.2f}", f"{mape:.2f}"


def count_trainable_params(params):
  """
  Counts the total number of trainable params. This will be used for capacity comparisons between simple and more complex deep neural networks.

  Args:
    - params: list of torch.nn.Parameter objects

  Returns:
    - total number of trainable params
  """
  return sum(p.numel() for p in params if p.requires_grad)


def r2_fn(y_trues, y_preds):
  """
  Calculates the R² score for regression ("How well is the model explaining the variance of the target?").

    R² measures how well the predicted values approximate the actual values

    R² = 1 - (SS_residual / SS_total), where:

      SS_residual = Σ(yᵢ - ŷᵢ)² → the sum of squared errors (model residuals)
      SS_total = Σ(yᵢ - ȳ)² → the total variance in the data (relative to mean)

  Interpretation:
    - R² = 1 → perfect prediction
    - R² = 0 → model does no better than mean
    - R² < 0 → model is worse than just predicting the mean

  Args:
    - y_trues: list or tensor of true target values (e.g., actual temperatures)
    - y_preds: list or tensor of predicted values from the model

  Returns:
    - R² score: a float in (-∞, 1), where 1.0 is perfect prediction
  """
  if not isinstance(y_trues, torch.Tensor):
    y_trues = torch.tensor(y_trues, dtype=torch.float32)
  if not isinstance(y_preds, torch.Tensor):
    y_preds = torch.tensor(y_preds, dtype=torch.float32)

  # flatten tensors to work for (N,), (N, 1), multi-horizon (N, h)
  y_trues = y_trues.reshape(-1)
  y_preds = y_preds.reshape(-1)

  ss_total = torch.sum((y_trues - y_trues.mean()) ** 2)
  ss_residual = torch.sum((y_trues - y_preds) ** 2)

  r2 = 1.0 - ss_residual / (ss_total - 1e-8) # results a tensor
  return float(r2)  # return a float number


def save_history_plots(history, out_path):
    """
    Saves Loss and R² curves to out_path.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
    fig.suptitle('Training & Validation Curves', fontsize=16, fontweight="bold")

    # Loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'],   label='Val Loss')
    ax1.set_title('Loss (MSE)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.4)

    # R²
    ax2.plot(history['train_r2'], label='Train')
    ax2.plot(history['val_r2'],   label='Val')
    ax2.set_title('R²')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('R²')
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)


def plot_learning_curves_all(histories, resolutions, model_tag):
    """
    Plots train/val MSE loss and R² for multiple time resolutions.

    Args:
        histories: list[dict] with keys: train_loss, val_loss, train_r2, val_r2
        resolutions: list[str], e.g. ["1h","6h","24h"]
        model_tag: str, for naming the plot
    """
    assert len(histories) == len(resolutions), "histories and labels must align"
    n = len(histories)

    fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(14, 3*n))
    fig.suptitle("Training & Validation Curves per Time Resolution", fontsize=16, fontweight="bold")

    for i, (history, label) in enumerate(zip(histories, resolutions)):
        # Loss
        axes[i, 0].plot(history['train_loss'], label='Train Loss')
        axes[i, 0].plot(history['val_loss'], label='Val Loss')
        axes[i, 0].set_ylabel(f'{label} Loss (MSE)')
        axes[i, 0].set_title(f'{label}')
        axes[i, 0].legend()
        axes[i, 0].grid(True, linestyle="--", alpha=0.4)

        # R²
        axes[i, 1].plot(history['train_r2'], label='Train R²')
        axes[i, 1].plot(history['val_r2'], label='Val R²')
        axes[i, 1].set_ylabel(f'{label} R²')
        axes[i, 1].set_title(f'{label}')
        axes[i, 1].legend()
        axes[i, 1].grid(True, linestyle="--", alpha=0.4)

    axes[-1, 0].set_xlabel("Epoch")
    axes[-1, 1].set_xlabel("Epoch")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    save_dir="results/figures/NNs_training"
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"{model_tag}_learning_curves.png")
    plt.savefig(save_path)
    plt.close(fig)
