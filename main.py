# src/main.py
#
# Main script for running Weather Time Series Forecasting experiments.
# Runs data preparation, model training, evaluation, and saves results.
#
import os
import pandas as pd
# from tabulate import tabulate

from src.config import CONFIG
from src.eda import run_eda, pacf_plot
from src.preprocessing import preprocess_dataset, stationarity_and_diff, plot_rescaling_is_harmless, split_and_normalize, make_univariate_sequences, make_multivariate_sequences, processed_data_summary, build_dataloaders_from_seqs, denormalize
from src.utils import load_dataset, write_log, plot_residuals_and_predictions, evaluate_model, plot_learning_curves_all
from src.ar_models import run_ar_one, select_significant_lags, ar_forecast_fixed_lags, select_var_order_bic, var_forecast
from src.nn_models import RNN, AttNN, TransformerEncoder, trainer, predict_nn_mod, predict_nn_mod_mv


def eda_and_preprocessing(weather_df, config):
    """
    Full EDA + preprocessing pipeline.

    Args:
        weather_df (pd.DataFrame): Dataframe of weather data
        config (dict): Configuration dictionary

    Returns:
        a dict with everything for experiments
    """
    # ensure datetime index
    weather_df = weather_df.copy()
    weather_df["date"] = pd.to_datetime(weather_df["date"])
    weather_df = weather_df.set_index("date")

    features = config["features"]
    target = config["target"]
    resolutions = config["resolutions"]

    # eda
    run_eda(weather_df)

    # clean, resample and overview plots
    resampled = preprocess_dataset(weather_df, target, features, resolutions)

    # stationarity and differencing
    final_stationary, stationary_series_data, provenance = stationarity_and_diff(resampled, resolutions)

    # PACF and harmless scaling plot
    for res_feat in final_stationary:  # each item is (res, feat)
        res, feat = res_feat
        df_res = resampled[res]
        s = df_res[feat]
        pacf_plot(s, res_feat)

    plot_rescaling_is_harmless(resampled, feature="T")

    # splits and normalization
    level_splits, splits_norm, scalers = split_and_normalize(stationary_series_data, resampled, resolutions)

    # sequences generation for NNs
    seqs, seqs_test = make_univariate_sequences(splits_norm, CONFIG)
    seqs_mv, seqs_mv_test = make_multivariate_sequences(splits_norm, features, target, CONFIG, resolutions)

    # convert sequences to dataloaders
    uni_loaders, uni_test_loaders, mv_loaders, mv_test_loaders = build_dataloaders_from_seqs(seqs, seqs_test, seqs_mv, seqs_mv_test)

    # results
    results = {
        "resampled": resampled,
        "final_stationary": final_stationary,
        "stationary_series_data": stationary_series_data,
        "level_splits": level_splits,
        "splits_norm": splits_norm,
        "scalers": scalers,
        "seqs_uni": seqs,
        "seqs_uni_test": seqs_test,
        "seqs_mv": seqs_mv,
        "seqs_mv_test": seqs_mv_test,
        "uni_loaders": uni_loaders,
        "uni_test_loaders": uni_test_loaders,
        "mv_loaders": mv_loaders,
        "mv_test_loaders": mv_test_loaders
    }

    # log the summary of results
    processed_data_summary(results)

    return results


def run_ar(splits_norm, scalers, config, level_splits):
    """
    AR baseline across resolutions for a single target config["target"]:
     - handles LEVEL/DIFF modeling spaces, for horizons h=1 and h>1, and DIFF inversion
     - locks significant lags once on (train+val), and reuses them in walk-forward
     - sets RangeIndex inside statsmodels, predictions aligned to timestamps

    Args:
        splits_norm: {feat: {res: (train_df, val_df, test_df)}} for both LEVEL and DIFF sets
        scalers: {(res, feat): {"mean": mu, "std": sd}}
        config: dict that must contain "target" and "horizons"
        level_splits: {feat: {res: (level_train, level_val, level_test)}} only LEVEL sets

    Returns:
        list[dict]: one row per (resolution,horizon) with keys:
            "Model", "Resolution", "Horizon", "Target Variable", "Input Type",
            "MAE", "RMSE", "MAPE (%)", "Fit Time", "# Params"
    """
    write_log("==== Training Results ====\n", filename="results/training_results.txt")

    feat = config["target"] # e.g. "T"
    resolutions = config["resolutions"] # e.g., ["1h","6h","24h"]
    horizon, h1, h2 = config["horizons"] # e.g., [1,4,6]
    diff_res = {"6h", "24h"} # this is for feature "T" (hardcoded)

    # include AR(1) first, for all resolutions
    results = run_ar_one(splits_norm, scalers, config, level_splits)

    for res in resolutions:
        train_n, val_n, test_n = splits_norm[feat][res]
        mu = scalers[(res, feat)]["mean"]
        std = scalers[(res, feat)]["std"]
        max_lags_res = config["max_lags"][res]

        # fit on (train+val) for lags selection
        init_history = pd.concat([train_n.squeeze(), val_n.squeeze()])
        meta = select_significant_lags(init_history, max_lags=max_lags_res, ic="bic", signif=0.05)
        fixed_lags = meta["fixed_lags"]
        selected_ic = meta["selected_lags_ic"]
        model_summary = meta["model"].summary()
        p_order = len(fixed_lags) or len(selected_ic)
        fit_time = meta["total_training_time"]
        n_params = meta["model"].params.size

        # logging
        write_log(
            f"[{feat} | {res}] - AR({p_order}) model\n"
            f"- Train size: {len(init_history)} | IC: bic | max_lags: {max_lags_res}\n"
            f"- Selected lags (IC): {selected_ic}\n"
            f"- {len(fixed_lags)} significant lags are kept (p<0.05): {fixed_lags} (We reuse these lags during walk-forward, we could re-search each step, but we choose speed!)\n"
            f"- Trainable params: {n_params}\n"
            f"- Training time (s): {fit_time}\n"
            f"- Model's Summary:\n{model_summary}\n\n",
            filename="results/training_results.txt")

        # horizons per resolution
        if res == "1h":
            horizons_for_res = [horizon, h2]
        elif res == "6h":
            horizons_for_res = [horizon, h1]
        elif res == "24h":
            horizons_for_res = [horizon]
        else:
            horizons_for_res = [horizon]

        # get LEVEL series for inversion if DIFF
        _, _, level_test_df = level_splits[feat][res]
        level_test_series = level_test_df.squeeze()

        is_diff = (res in diff_res)

        # predict for all horizons
        for h in horizons_for_res:

            if (not is_diff) or (h == 1):
                # LEVEL, or DIFF with 1-step
                y_pred_level, y_true_level = ar_forecast_fixed_lags(
                    train_n=train_n,
                    val_n=val_n,
                    test_n=test_n,
                    res=res, feat=feat, horizon=h,
                    fixed_lags=fixed_lags,
                    mu=mu, std=std,
                    is_diff=is_diff,
                    level_series_raw=level_test_series,  # only used if is_diff=True
                    return_full_path=False
                )
            else:
                # DIFF and multi-step => keep the full path for correct inversion
                y_pred_level, y_true_level, _full_path_level = ar_forecast_fixed_lags(
                    train_n=train_n,
                    val_n=val_n,
                    test_n=test_n,
                    res=res, feat=feat, horizon=h,
                    fixed_lags=fixed_lags,
                    mu=mu, std=std,
                    is_diff=True,
                    level_series_raw=level_test_series,
                    return_full_path=True
                )

            # save preds vs. trues to CSV
            os.makedirs("results/predictions", exist_ok=True)
            save_path = f"results/predictions/{res}_{feat}_AR_(t+{h})_prediction_results.csv"
            pd.DataFrame({"true": y_true_level, "pred": y_pred_level}).to_csv(save_path, index=False)

            # plot and evaluate (LEVEL space)
            plot_residuals_and_predictions(y_true_level, y_pred_level, res, feat, h, model_type=f"AR({p_order})")
            mae, rmse, mape = evaluate_model(y_true_level.values, y_pred_level.values)

            results.append({
                "Model": f"AR({p_order})",
                "Resolution": res,
                "Horizon": f"t+{h}",
                "Target Variable": feat,
                "Input Type": "Univariate",
                "MAE": mae,
                "RMSE": rmse,
                "MAPE (%)": mape,
                "Fit Time": fit_time,
                "# Params": n_params,
            })

    return results


def run_var(splits_norm, scalers, config, level_splits):
    """
    VAR baseline using multiple features to predict `config["target"]`
    - Selects order p by **BIC** once on (train+val) per resolution
    - Walk-forward forecasting is then done with fixed p
    - Target is denormalized & DIFF-inverted to LEVEL space for evaluation

    Args:
        splits_norm (dict): {feat: {res: (train_df_n, val_df_n, test_df_n)}} normalized splits in MODEL space.
        scalers (dict): {(res, feat): {"mean": mu, "std": sd}} mean/std scalers from MODEL train.
        config (dict): must contain keys of:
            "features" (list[str]) input features for VAR,
            "target" (str) target variable name,
            "resolutions" (list[str]) e.g. ["1h","6h","24h"],
            "horizons" (list[int]) e.g. [1, h1, h2],
            "max_lags" (dict[str,int]) per-resolution max VAR lag.
        level_splits (dict): {feat: {res: (train_df_level, val_df_level, test_df_level)}} LEVEL splits for inversion.

    Returns:
        list[dict]: one row per (resolution,horizon) with keys:
            "Model", "Resolution", "Horizon", "Target Variable", "Input Type",
            "MAE", "RMSE", "MAPE (%)", "Fit Time", "# Params"
    """
    write_log("==== VAR Training Results ====\n", filename="results/training_results.txt")

    feats = list(config["features"]) # e.g., ["SWDR","rh","T"]
    target = config["target"] # e.g., "T"
    resolutions = config["resolutions"] # e.g., ["1h","6h","24h"]
    horizon, h1, h2 = config["horizons"] # e.g., [1,4,6]
    results = []

    # hardcode DIFF policy per feature!
    diff_res_map = {
        "T": {"6h", "24h"}, # target T → differenced at 6h & 24h
        "SWDR": {"6h", "24h"}, # SWDR → differenced at 6h, 24h
        "rh": set(), # rh → stationary everywhere (LEVEL always)
    }

    for res in resolutions:
        # defaults values so variables always exist
        fit_time = "N/A"
        n_params = "N/A"
        p = 1

        # build multivariate splits (normalized MODEL space)
        train_mv = pd.concat([splits_norm[f][res][0].squeeze().rename(f) for f in feats], axis=1)
        val_mv = pd.concat([splits_norm[f][res][1].squeeze().rename(f) for f in feats], axis=1)
        test_mv = pd.concat([splits_norm[f][res][2].squeeze().rename(f) for f in feats], axis=1)

        # target scaler (for denorm)
        mu_t = scalers[(res, target)]["mean"]
        std_t = scalers[(res, target)]["std"]

        # set resolution-specific max_lags for VAR
        #   - 1h has many samples → up to 24 lags
        #   - 6h has fewer samples → up to 2 lags for select_var_order → SVD errors
        #   - 24h (daily) has very few samples → SVD errors
        # max_lags_res = 24 if res == "1h" else 2 if res == "6h" else 3
        # # fixed orders due to instability of BIC-based VAR order selection (small samples in 6h/24h cause SVD/MKL errors).
        # p_res = {"1h": 8, "6h": 2, "24h": 1}
        # p = p_res[res]

        if res == "24h": # skip 24h for VAR to avoid SVD failures due to very small sample size
            write_log(
                f"[T | {res}] - SKIP VAR: daily sample too small and SVD failures arise in fit/select_order.\n",
                filename="results/training_results.txt")
            results.append({
                "Model": "VAR",
                "Resolution": res,
                "Horizon": f"t+{horizon}",
                "Target Variable": target,
                "Input Type": "Multivariate",
                "MAE": None,
                "RMSE": None,
                "MAPE (%)": None,
                "Fit Time": "N/A",
                "# Params": "N/A",
            })
            continue

        elif res == "6h":
            p = 2 # fixed small order to avoid SVD failures (skip select_var_order_bic)

            # compute params for the summary
            init_history = pd.concat([train_mv, val_mv]).sort_index().reset_index(drop=True)
            k = init_history.shape[1]

            # total number of coefficients in VAR(p) = k * (k*p + 1):
            #   - k equations (one per variable)
            #   - each equation has k*p lag coefficients
            #   - plus k intercepts (one per equation)
            n_params = k * k * p + k  # total params for VAR(p)
            fit_time = "N/A"

            write_log(
                f"[{target} | {res}] - VAR({p}) (fixed order)\n"
                f"- Skip IC selection to avoid SVD failures\n"
                f"- Variables (k={len(feats)}): {feats}\n"
                f"- Train size: {len(init_history)}\n"
                f"- Trainable params: {n_params}\n"
                f"- Training time: {fit_time}\n",
                filename="results/training_results.txt")

        elif res == "1h":
            # safe to use BIC-based selection
            max_lags_res = config["max_lags"][res]

            # select order p by BIC on (train+val)
            init_history = pd.concat([train_mv, val_mv]).sort_index()
            meta = select_var_order_bic(init_history, max_lags=max_lags_res)
            p = meta["p"]
            var_fit = meta["model"]
            n_params = meta["n_params"]
            fit_time = meta["fit_time"]

            write_log(
                f"[{target} | {res}] - VAR({p}) (BIC order)\n"
                f"- Variables (k={len(feats)}): {feats}\n"
                f"- Train size: {len(init_history)}\n"
                f"- Trainable params: {n_params}\n"
                f"- Training time (s): {fit_time}\n"
                f"- Model's Summary:\n{var_fit.summary()}\n",
                filename="results/training_results.txt")

        # horizons per resolution (same policy as AR)
        if res == "1h":
            horizons_for_res = [horizon, h2]
        elif res == "6h":
            horizons_for_res = [horizon, h1]
        elif res == "24h":
            horizons_for_res = [horizon]
        else:
            horizons_for_res = [horizon]

        # Level series for inversion (target only)
        _, _, level_test_df = level_splits[target][res]
        level_test_series = level_test_df.squeeze()
        is_diff_target = (res in diff_res_map.get(target, set()))

        # walk-forward per horizon (same policy as ar)
        for h in horizons_for_res:

            if (not is_diff_target) or (h == 1):
                y_pred_level, y_true_level = var_forecast(
                    train_mv=train_mv, val_mv=val_mv, test_mv=test_mv,
                    res=res, target=target, horizon=h, p=p,
                    mu_t=mu_t, std_t=std_t,
                    is_diff_target=is_diff_target,
                    level_series_raw=level_test_series,
                    return_full_path=False
                )
            else:
                y_pred_level, y_true_level, _full_path_level = var_forecast(
                    train_mv=train_mv, val_mv=val_mv, test_mv=test_mv,
                    res=res, target=target, horizon=h, p=p,
                    mu_t=mu_t, std_t=std_t,
                    is_diff_target=True,
                    level_series_raw=level_test_series,
                    return_full_path=True
                )

            # save preds vs. trues to CSV
            save_path = f"results/predictions/{res}_{target}_VAR_(t+{h})_prediction_results.csv"
            pd.DataFrame({"true": y_true_level, "pred": y_pred_level}).to_csv(save_path, index=False)

            # plot (+ horizon in title) & evaluate
            plot_residuals_and_predictions(y_true_level, y_pred_level, res, target, h, model_type=f"VAR({p})")
            mae, rmse, mape = evaluate_model(y_true_level.values, y_pred_level.values)

            results.append({
                "Model": f"VAR({p})",
                "Resolution": res,
                "Horizon": f"t+{h}",
                "Target Variable": target,
                "Input Type": "Multivariate",
                "MAE": mae,
                "RMSE": rmse,
                "MAPE (%)": mape,
                "Fit Time": fit_time,
                "# Params": n_params,
            })

    return results


def build_model(model_kind, seq_len, input_dim=1, nn_cfg=None):
    """
    Builds any of: "rnn_uni", "rnn_mv", "att_uni", "att_mv", "trans_uni", "trans_mv".

    Args:
      model_kind (str): one of the above strings
      seq_len (int): input window length S for the model
      input_dim (int): #features per timestep (1 for uni, 3 for mv)
      nn_cfg (dict): CONFIG['nn_defaults'] subset from which we read: hidden_size (int), num_layers (int), num_heads (int), dropout (float), ffn_dim (int), e.g., default 2*hidden_size

    Returns:
      nn.Module: an initialized model placed on CPU.
    """
    nn_cfg = nn_cfg or {}
    hidden_size = nn_cfg["hidden_size"]
    num_layers = nn_cfg["num_layers"]
    num_heads = nn_cfg["num_heads"]
    dropout = nn_cfg["dropout"]
    ffn_dim = nn_cfg["ffn_dim"]

    if model_kind.startswith("RNN"):
        return RNN(input_dim=input_dim, hidden_dim=hidden_size, output_dim=1, num_layers=1) # num_layers=1 for RNN, 2 for ATT and TRANS

    if model_kind.startswith("ATT"):
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads for ATT"
        return AttNN(input_dim=input_dim, model_dim=hidden_size, num_heads=num_heads, num_layers=num_layers, seq_len=seq_len, dropout=dropout, pool="mean")

    if model_kind.startswith("TRANS"):
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads for TRANS"
        return TransformerEncoder(seq_len=seq_len, input_dim=input_dim, d_model=48, num_heads=3, ffn_dim=96, num_layers=1, dropout=dropout, pool="mean")

    raise ValueError(f"Unknown model_kind: {model_kind}")


def run_nn_model_pipeline(model_kind, config, loaders_by_res, scalers, input_type="Univariate", extra_test_loaders=None):
    """
    Trains, predicts, logs and plots for one model kind across all resolutions.

    Args:
      model_kind (str): "RNN", "RNN_MV",  "ATT", "ATT_MV", "TRANS", "TRANS_MV"
      config (dict): expects CONFIG["nn_defaults"] with: "batch_size", "epochs", "learning_rate", "hidden_size", "dropout", "weight_decay", "optimizer", "seq_len": {"1h": int, "6h": int, "24h": int}, "num_layers", "num_heads", "ffn_dim"
      loaders_by_res (dict): {"1h": {"train":..., "val":..., "test":...}, ...}
      scalers: mean and std for target feature (e.g., T)
      input_type (str): "Univariate" or "Multivariate"
      extra_test_loaders (dict): loaders when the horizon > 1

    Returns:
      results (list[dict]): evaluation results
    """
    target_feat = config["target"]
    nn_defaults = config["nn_defaults"]
    epochs = nn_defaults["epochs"]
    lr = nn_defaults["learning_rate"]
    weight_decay = nn_defaults["weight_decay"]
    histories_per_res, results, results_extra = {}, [], []

    for res, loaders in loaders_by_res.items():
        train_loader = loaders["train"]
        val_loader = loaders["val"]
        test_loader = loaders["test"]

        # seq_len from CONFIG mapping
        seq_len = nn_defaults["seq_len"][res]

        # input_dim from data
        xb, _ = next(iter(train_loader))  # take one batch: xb shape is (B, L, F)
        input_dim = 1 if input_type == "Univariate" else int(xb.shape[2])

        # build model
        model = build_model(model_kind, seq_len=seq_len, input_dim=input_dim, nn_cfg=nn_defaults)

        # train
        history = trainer(model, f"{res}_{target_feat}_{model_kind}", train_loader, val_loader, epochs=epochs, lr=lr, weight_decay=weight_decay)
        fit_time = history["total_training_time"]
        n_params = history["total_params"]
        histories_per_res[res] = history

        # # infer horizon from test_loader batch: y_test_sample shape = (B,1) or (B,) for 1-step OR (B, H) for multi-step
        _, y_test_sample = next(iter(test_loader))
        n_steps = 1 if y_test_sample.ndim == 1 or y_test_sample.shape[-1] == 1 else int(y_test_sample.shape[-1])

        # predictions (normalized space)
        if input_type == "Univariate":
            preds_last, trues_last, preds_roll, trues_roll = predict_nn_mod(model, test_loader, n_steps=n_steps)
        else:
            preds_last, trues_last, preds_roll, trues_roll = predict_nn_mod_mv(model, test_loader, n_steps=n_steps)

        # denormalize
        s = scalers[(res, target_feat)] # dict: {"mean": mu, "std": sd}
        mu, std = s["mean"], s["std"]
        preds_last_den = denormalize(preds_last, mu, std)
        trues_last_den = denormalize(trues_last, mu, std)

        # save preds vs. trues to CSV
        save_path = f"results/predictions/{res}_{target_feat}_{model_kind}_(t+{n_steps})_prediction_results.csv"
        pd.DataFrame({"trues": trues_last_den, "preds": preds_last_den}).to_csv(save_path, index=False)

        # metrics on denormalized arrays
        mae, rmse, mape = evaluate_model(trues_last_den, preds_last_den)

        # plot predictions (denormalized)
        plot_residuals_and_predictions(trues_last_den, preds_last_den, res, target_feat, n_steps, f"{model_kind}")

        # extra test loaders (for predicting n_steps ahead)
        if extra_test_loaders is not None:
            for key, dl in extra_test_loaders.items():
                # Univariate case
                if input_type == "Univariate":
                    feat_key, res_key, horizon_key = key    # uni_test_loaders are of {(feat, res, horizon): DataLoader}
                    if feat_key != target_feat or res_key != res:
                        continue
                else:  # Multivariate case
                    res_key, horizon_key = key  # mv_test_loaders is of {(res, horizon): DataLoader}
                    if res_key != res:
                        continue

                # infer the horizon from a batch of this extra loader (handles (B,), (B,1), (B,H))
                _, y_extra = next(iter(dl))
                if y_extra.ndim == 1 or (y_extra.ndim == 2 and y_extra.shape[-1] == 1):
                    n_steps_extra = 1
                else:
                    n_steps_extra = int(y_extra.shape[-1])

                # predict (normalized space)
                if input_type == "Univariate":
                    preds_extra, trues_extra, _, _ = predict_nn_mod(model, dl, n_steps=n_steps_extra)
                else:
                    preds_extra, trues_extra, _, _ = predict_nn_mod_mv(model, dl, n_steps=n_steps_extra)

                # denormalize (use the scaler of the current resolution + target)
                preds_extra_den = denormalize(preds_extra, mu, std)
                trues_extra_den = denormalize(trues_extra, mu, std)

                # metrics
                mae_e, rmse_e, mape_e = evaluate_model(trues_extra_den, preds_extra_den)

                # save, plot and log
                save_path = f"results/predictions/{res}_{target_feat}_{model_kind}_(t+{n_steps_extra})_prediction_results.csv"
                pd.DataFrame({"trues": trues_extra_den, "preds": preds_extra_den}).to_csv(save_path, index=False)

                plot_residuals_and_predictions(trues_extra_den, preds_extra_den, res, target_feat, n_steps_extra, f"{model_kind}")

                results_extra.append({
                    "Model": model_kind,
                    "Resolution": res,
                    "Horizon": f"t+{n_steps_extra}",
                    "Target Variable": target_feat,
                    "Input Type": input_type,
                    "MAE": mae_e,
                    "RMSE": rmse_e,
                    "MAPE (%)": mape_e,
                    "Fit Time": fit_time,
                    "# Params": n_params,
                })

        # write log for training
        train_size = len(train_loader.dataset)
        batch_size = train_loader.batch_size
        with open("results/training_results.txt", "a", encoding="utf-8") as f:
            f.write(
                f"[{target_feat} | {res}] - {model_kind} model {input_type}\n"
                f"- Train size: {train_size} | Batch size: {batch_size}\n"
                f"- Training time (s): {fit_time}\n"
                f"- Trainable params: {n_params}\n"
                f"- Best Val Loss (MSE): {history['best_val_loss']:.6f}\n"
                f"- Stopped epoch: {history['stopped_epoch']}\n"
                f"- Model's Summary:\n{model}\n\n")

        # results
        results.append({
            "Model": model_kind,
            "Resolution": res,
            "Horizon": f"t+{n_steps}",
            "Target Variable": target_feat,
            "Input Type": input_type,
            "MAE": mae,
            "RMSE": rmse,
            "MAPE (%)": mape,
            "Fit Time": fit_time,
            "# Params": n_params,
        })

    # save combined learning curves for all resolutions
    histories_list = [histories_per_res[k] for k in histories_per_res.keys()]
    resolutions = [k for k in histories_per_res.keys()]
    plot_learning_curves_all(histories_list, resolutions, model_kind)

    return results + results_extra


def main():
    """
    Main driver function for the forecasting pipeline.
    """

    # get the data and load it into a DataFrame
    weather_path = load_dataset()
    weather_df = pd.read_csv(weather_path)

    # container for all results
    all_results = []

    # 1. EDA and preprocessing
    print("\n>>> Running EDA and Preprocessing ...")
    processed_data = eda_and_preprocessing(weather_df, CONFIG)

    # 2. Run AR baseline for feature and target "T"
    print("\n>>> Training and Evaluating AR (univariate: T) ... ")
    ar_T_results = run_ar(splits_norm = processed_data["splits_norm"], scalers = processed_data["scalers"], config = CONFIG, level_splits = processed_data["level_splits"])
    all_results.extend(ar_T_results)

    # 3. Run VAR on features ["SWDR", "rh", "T"] and target "T"
    # print("\n>>> Training and Evaluating VAR (joint: SWDR, rh, T → T) ... ")
    # var_T_results = run_var(splits_norm = processed_data["splits_norm"], scalers = processed_data["scalers"], config = CONFIG, level_splits = processed_data["level_splits"])
    # all_results.extend(var_T_results)

    # 4. Run vanilla RNN for feature and target "T"
    print("\n>>> Training and Evaluating RNN (univariate: T) ... ")
    rnn_T_results = run_nn_model_pipeline(
        model_kind="RNN",
        config=CONFIG,
        loaders_by_res=processed_data["uni_loaders"]["T"], # {"1h":{"train":DL,"val":DL,"test":DL}, ...}
        scalers=processed_data["scalers"],
        input_type="Univariate",
        extra_test_loaders=processed_data["uni_test_loaders"]
    )
    all_results.extend(rnn_T_results)

    # 5. Run vanilla RNN for features ["SWDR", "rh", "T"] and target "T"
    print("\n>>> Training and Evaluating RNN (multivariate: SWDR, rh, T → T) ... ")
    rnn_T_mv_results = run_nn_model_pipeline(
        model_kind="RNN_MV",
        config=CONFIG,
        loaders_by_res=processed_data["mv_loaders"], # {"1h": {"train":DL,"val":DL,"test":DL}, ...}
        scalers=processed_data["scalers"],
        input_type="Multivariate",
        extra_test_loaders=processed_data["mv_test_loaders"]
    )
    all_results.extend(rnn_T_mv_results)

    # 6. Run Self-Attention for feature and target "T"
    print("\n>>> Training and Evaluating Self-Attention (univariate: T) ... ")
    att_T_results = run_nn_model_pipeline(
        model_kind="ATT",
        config=CONFIG,
        loaders_by_res=processed_data["uni_loaders"]["T"],
        scalers=processed_data["scalers"],
        input_type="Univariate",
        extra_test_loaders = processed_data["uni_test_loaders"]
        )
    all_results.extend(att_T_results)

    # 7. Run Attention for features ["SWDR", "rh", "T"] and target "T"
    print("\n>>> Training and Evaluating Self-Attention (multivariate: SWDR, rh, T → T) ... ")
    att_T_mv_results = run_nn_model_pipeline(
        model_kind="ATT_MV",
        config=CONFIG,
        loaders_by_res=processed_data["mv_loaders"], # {"1h": {"train":DL,"val":DL,"test":DL}, ...}
        scalers=processed_data["scalers"],
        input_type="Multivariate",
        extra_test_loaders=processed_data["mv_test_loaders"]
    )
    all_results.extend(att_T_mv_results)

    # 8. Run Transformer-Encoder for feature and target "T"
    print("\n>>> Training and Evaluating Transformer-Encoder (univariate: T) ... ")
    trans_T_results = run_nn_model_pipeline(
        model_kind="TRANS",
        config=CONFIG,
        loaders_by_res=processed_data["uni_loaders"]["T"], # {"1h":{"train":DL,"val":DL,"test":DL}, ...}
        scalers=processed_data["scalers"],
        input_type="Univariate",
        extra_test_loaders=processed_data["uni_test_loaders"]
    )
    all_results.extend(trans_T_results)

    # 9. Run Transformer-Encoder for features ["SWDR", "rh", "T"] and target "T"
    print("\n>>> Training and Evaluating Transformer-Encoder (multivariate: SWDR, rh, T → T) ... ")
    trans_T_mv_results = run_nn_model_pipeline(
        model_kind="TRANS_MV",
        config=CONFIG,
        loaders_by_res=processed_data["mv_loaders"], # {"1h": {"train":DL,"val":DL,"test":DL}, ...}
        scalers=processed_data["scalers"],
        input_type="Multivariate",
        extra_test_loaders=processed_data["mv_test_loaders"]
    )
    all_results.extend(trans_T_mv_results)

    # all results into a df and log
    columns = ["Model", "Resolution", "Horizon", "Target Variable", "Input Type", "MAE", "RMSE", "MAPE (%)", "Fit Time", "# Params"]
    df = pd.DataFrame(all_results).reindex(columns=columns)

    table_str = df.to_markdown(index=False, tablefmt="github", floatfmt=".4f")
    write_log("==== Evaluation Results ====\n", filename="results/evaluation_results.txt")
    write_log(table_str + "\n", filename="results/evaluation_results.txt")

    # groupby results per resolution
    write_log("\n==== Evaluation Results by Resolution ====\n", filename="results/evaluation_results.txt")
    resolutions = CONFIG["resolutions"]
    for res in resolutions:
        sub = df[df["Resolution"] == res]
        if not sub.empty:
            write_log(f"\n-- {res} --\n", filename="results/evaluation_results.txt")
            table_str = sub.to_markdown(index=False, tablefmt="github", floatfmt=".4f")
            write_log(table_str + "\n", filename="results/evaluation_results.txt")

    # results in the terminal
    print("\nAll evaluation results have been saved to: ./results/evaluation_results.txt\n")
    # print(tabulate(all_results, headers="keys", tablefmt="github", floatfmt=".4f"))
    # print(" ")


if __name__ == "__main__":
    main()
