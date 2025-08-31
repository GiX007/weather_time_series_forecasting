# src.ar_models.py
#
# Forecasting models: AR/VAR models for the weather time-series forecasting.
#
import os, time
import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from statsmodels.tsa.api import VAR
from src.preprocessing import denormalize
from src.utils import evaluate_model, write_log, plot_residuals_and_predictions


def run_ar_one(splits_norm, scalers, config, level_splits):
    """
    Minimal AR(1) baseline; reuses ar_forecast_fixed_lags(fixed_lags=[1]).

    Args:
        splits_norm (dict): {feat: {res: (train_df, val_df, test_df)}} normalized target series per resolution
        scalers (dict): {(res, feat): {"mean": float, "std": float}} normalization stats for denorm
        config (dict): contains "target" (str), "resolutions" (list[str]), "horizons" (list[int])
        level_splits (dict): {feat: {res: (level_train_df, level_val_df, level_test_df)}} raw LEVEL series for DIFF inversion

    Returns:
        list[dict]: one row per (resolution, horizon) with metrics and metadata
    """
    results = []
    feat = config["target"]
    resolutions = config["resolutions"]
    horizon, h1, h2 = config["horizons"]
    diff_res = {"6h", "24h"}

    for res in resolutions:
        train_n, val_n, test_n = splits_norm[feat][res]
        mu  = scalers[(res, feat)]["mean"]
        std = scalers[(res, feat)]["std"]

        # fit on (train+val)
        y_fit = pd.concat([train_n.squeeze(), val_n.squeeze()])
        y_fit = y_fit.reset_index(drop=True) # reset datetimeIndex
        t0 = time.time()
        m = AutoReg(y_fit, lags=1, trend='c').fit()
        fit_time = time.time() - t0
        n_params = m.params.size
        m_summary = m.summary()

        # log training info
        write_log(
            f"[{feat} | {res}] - AR(1) model\n"
            f"- Train size: {len(y_fit)}\n"
            f"- Trainable params: {n_params}\n"
            f"- Training time (s): {fit_time}\n"
            f"- Model's Summary:\n{m_summary}\n\n",
            filename="results/training_results.txt")

        if res == "1h":
            horizons_for_res = [horizon, h2]
        elif res == "6h":
            horizons_for_res = [horizon, h1]
        elif res == "24h":
            horizons_for_res = [horizon]
        else:
            horizons_for_res = [horizon]

        # for DIFF inversion
        _, _, level_test_df   = level_splits[feat][res]
        level_test_series_raw = level_test_df.squeeze()
        is_diff = (res in diff_res)

        for h in horizons_for_res:
            if (not is_diff) or (h == 1):
                y_pred_level, y_true_level = ar_forecast_fixed_lags(
                    train_n=train_n, val_n=val_n, test_n=test_n,
                    res=res, feat=feat, horizon=h,
                    fixed_lags=[1],
                    mu=mu, std=std,
                    is_diff=is_diff,
                    level_series_raw=level_test_series_raw,
                    return_full_path=False
                )
            else:
                y_pred_level, y_true_level, _ = ar_forecast_fixed_lags(
                    train_n=train_n, val_n=val_n, test_n=test_n,
                    res=res, feat=feat, horizon=h,
                    fixed_lags=[1],
                    mu=mu, std=std,
                    is_diff=True,
                    level_series_raw=level_test_series_raw,
                    return_full_path=True
                )


            # save & eval (LEVEL space)
            os.makedirs("results/predictions", exist_ok=True)
            save_path = f"results/predictions/{res}_{feat}_AR(1)_(t+{h})_prediction_results.csv"
            pd.DataFrame({"true": y_true_level, "pred": y_pred_level}).to_csv(save_path, index=False)

            # plot and evaluate (LEVEL space)
            plot_residuals_and_predictions(y_true_level, y_pred_level, res, feat, h, model_type=f"AR(1)")
            mae, rmse, mape = evaluate_model(y_true_level.values, y_pred_level.values)

            results.append({
                "Model": "AR(1)",
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


def select_significant_lags(init_history_n, max_lags = 24, ic = "bic", signif = 0.05):
    """
    Selects significant AR lags once on (train+val) and reuses them for walk-forward forecasting. This avoids re-selecting at each step (faster, more stable)

    Args:
        init_history_n (pd.Series or pd.DataFrame): Normalized training history in modeling space (LEVEL or DIFF)
        max_lags (int, default=24): Maximum lag size considered for selection
        ic ({"aic", "bic"}, default="bic"): Information criterion used for lag order selection
        signif (float, default=0.05): P-value threshold for keeping lags; falls back to IC lags if none pass

    Returns:
        dict with:
            - "fixed_lags": list of significant lags (to reuse later)
            - "selected_lags_ic": lag set chosen by IC
            - "model": fitted AutoReg model with fixed_lags
            - "total_training_time": time (s) for final fit
    """
    y0 = init_history_n.squeeze().reset_index(drop=True)    # we use a RangeIndex (integer)

    sel = ar_select_order(y0, maxlag=max_lags, ic=ic, old_names=False)
    selected = sorted(list(sel.ar_lags)) if sel.ar_lags is not None else [1]

    init_fit = AutoReg(y0, lags=selected, old_names=False).fit()

    signif_lags = []
    for name, pval in init_fit.pvalues.items():
        if "L" in name and pval < signif:
            signif_lags.append(int(name.split("L")[-1]))
    signif_lags = sorted(set(signif_lags)) or selected  # fallback if none pass

    start = time.time()
    final_fit = AutoReg(y0, lags=signif_lags, old_names=False).fit()
    end = time.time()

    return {
        "fixed_lags": signif_lags,
        "selected_lags_ic": selected,
        "model": final_fit,
        "total_training_time": round((end-start), 4)
    }


def ar_forecast_fixed_lags(train_n, val_n, test_n, res, feat, horizon, fixed_lags, mu, std, is_diff, level_series_raw = None, return_full_path = False):
    """
    Rolling Forecast Origin (walk-forward, expanding the window) with a fixed lag set.

    There are TWO distinct “rollings”:

        1) Within-origin recursion (single origin t):
           - Calling forecast(h) returns [ŷ_{t+1}, …, ŷ_{t+h}]
           - Step 1 uses the last observed y_t
           - Step k (k>1) uses the PREVIOUS forecast ŷ_{t+k-1} because y_{t+k-1} is unknown at origin t
           - Example (h=2): ŷ_{t+1} = f(y_t, …), ŷ_{t+2} = f(ŷ_{t+1}, …)

        2) Across-origins update (t → t+1):
           - After time t occurs, the TRUE y_t is known
           - We append y_t to the history and refit/update the model for the next origin (expanding window)
           - Using the previous prediction here (“self-feeding”) is a different scenario and usually degrades realism

        Target space handling:
        - LEVEL: denormalize modeling-space predictions to LEVEL units.
        - DIFF: invert deltas to LEVEL using the base level at the origin:
            ŷ^{LEVEL}_{t+k} = y^{LEVEL}_t + cumulative_sum(Δŷ_{t+1…t+k})

        Alignment:
        - Each origin contributes the last-step ŷ_{t+h}. We align with the truth y_{t+h} by shifting the truth series by −h

    Args:
        train_n (pd.Series or pd.DataFrame): Normalized training history in modeling space (LEVEL or DIFF)
        val_n (pd.Series or pd.DataFrame): Normalized validation history in modeling space (LEVEL or DIFF)
        test_n (pd.Series or pd.DataFrame): Normalized test history in modeling space (LEVEL or DIFF)
        res (str): resolution label (e.g., "1h")
        feat (str): feature label (e.g., "T")
        horizon (int) : forecast horizon (1 or >1)
        fixed_lags (list[int]) : the AR lags to reuse (locked once)
        mu (float): target train mean (for denormalization)
        std (float): target train std (for denormalization)
        is_diff (bool) : True if modeling space is DIFF; else LEVEL
        level_series_raw (pd.Series) : raw LEVEL series (required if is_diff=True)
        return_full_path (bool, default=False) : if True, also return full denormalized paths

    Returns:
      y_pred_level : pd.Series (denormalized LEVEL preds, last-step-only, aligned to t+h)
      y_true_level : pd.Series (denormalized LEVEL truth, aligned to t+h)
      full_path_level : DataFrame with full LEVEL paths (step1 ... steph)
    """
    # step 1: ensure inputs are series (all sets are dfs)
    history_n = pd.concat([train_n.squeeze(), val_n.squeeze()]).sort_index()
    test_n = test_n.squeeze()

    preds_last_n = [] # last step in modeling space (Δ or LEVEL)
    need_paths = return_full_path or (is_diff and horizon > 1)
    paths_n = [] if need_paths else None # (for h>1, to return all hs)
    bases_level = [] # LEVEL y_t at each origin (only used when is_diff)

    # step 2: walk forward over test origins
    for t, y_t in test_n.items():
        y_hist = history_n.sort_index().reset_index(drop=True) # statsmodels sees integer steps (reset datetimeIndex)
        model = AutoReg(y_hist, lags=fixed_lags, old_names=False).fit() # fit AR with the fixed lags on all info up to t-1

        # recursive h-step forecast of shape (h,), statsmodels uses the LAST step (its own predictions to generate t+2, …, t+h) for t+h
        fc_n = np.asarray(model.forecast(steps=horizon), dtype=float)
        preds_last_n.append(fc_n[-1])
        if need_paths:
            paths_n.append(fc_n.copy())

        # store LEVEL base at origin (only for DIFF inversion)
        if is_diff:
            bases_level.append(level_series_raw.loc[t])

        # walk forward with TRUE normalized observation (across-origin update)
        history_n.loc[t] = y_t

        # walk forward with the current forecast (self-feeding alternative)
        # history_n.loc[t] = fc_n[0]

    # step 3: align to t+h by shifting ground truth (modeling-space:LEVEL or DIFF)
    preds_last_n = pd.Series(preds_last_n, index=test_n.index, name=f"{res} {feat}_pred_n") # preds_last_n holds ŷ_{t+h} per origin t, give it the test origins as index
    y_true_n = test_n.shift(-horizon).reindex(preds_last_n.index) # bring the truths to the same len as preds by shifting the truth series backward by h

    # after shifting truth by -h, the last h timestamps are NaN, keep only aligned (pred, truth) pairs to avoid NaNs in metrics
    mask = y_true_n.notna()
    preds_last_n = preds_last_n[mask]
    y_true_n = y_true_n[mask]

    # keep only valid origins for any stored arrays
    valid_len = len(preds_last_n)
    if need_paths:
        paths_n = paths_n[:valid_len]
    if is_diff:
        bases_level = bases_level[:valid_len]

    # step 4: LEVEL vs DIFF inversion
    if not is_diff:
        # LEVEL: just denormalize preds and truth
        y_pred_level = denormalize(preds_last_n, mu, std)
        y_true_level = denormalize(y_true_n, mu, std)

        # handle return_full_path when horizon==1
        if return_full_path and horizon == 1:
            full_path_level = y_pred_level.to_frame(name="step1")
            return y_pred_level, y_true_level, full_path_level

        if return_full_path:
            # full LEVEL paths by denorm of each modeling-space path
            cols = [f"step{k}" for k in range(1, horizon + 1)]
            full_path_level = pd.DataFrame(
                [denormalize(pd.Series(p, index=cols), mu, std) for p in paths_n] if need_paths else [],
                index=y_pred_level.index,
                columns=cols,
            )
            return y_pred_level, y_true_level, full_path_level

        return y_pred_level, y_true_level

    # DIFF: convert predicted Δy to LEVEL using base LEVEL at origin
    if horizon == 1:
        # last-step: ŷ_{t+1} = base_level + Δŷ_{t+1}
        bases_aligned = pd.Series(bases_level, index=preds_last_n.index)
        delta_level = denormalize(preds_last_n, mu, std)
        y_pred_level = bases_aligned + delta_level
        y_true_level = level_series_raw.shift(-1).reindex(preds_last_n.index) # align LEVEL truth at t+1 to the origin row t: we forecast ŷ_{t+1} at origin t, so shift(-1) moves y_{t+1} into index t

        # handle return_full_path when horizon==1
        if return_full_path:
            full_path_level = y_pred_level.to_frame(name="step1")
            return y_pred_level, y_true_level, full_path_level

        return y_pred_level, y_true_level

    # horizon > 1: accumulate Δŷ and add base level
    cols = [f"step{k}" for k in range(1, horizon + 1)]
    bases_series = pd.Series(bases_level, index=preds_last_n.index)

    last_steps = []
    full_rows = [] if return_full_path else None

    for b, p in zip(bases_series.values, paths_n):
        dn = denormalize(pd.Series(p, index=cols), mu, std)  # Δy in LEVEL units
        lv = b + dn.cumsum()  # level path
        last_steps.append(lv.iloc[-1])
        if return_full_path:
            full_rows.append(lv.values)

    y_pred_level = pd.Series(last_steps, index=preds_last_n.index, name=f"{res} {feat}_pred_level")
    y_true_level = level_series_raw.shift(-horizon).reindex(y_pred_level.index)

    if return_full_path:
        full_path_level = pd.DataFrame(full_rows, index=y_pred_level.index, columns=cols)
        return y_pred_level, y_true_level, full_path_level

    return y_pred_level, y_true_level


def select_var_order_bic(init_history_mv, max_lags=24):
    """
    Select lag order (p) for a VAR model using BIC on (train+val), then fit VAR(p) once.

    Difference vs. AR case:
      - AR: we start from max_lags, use IC (e.g., BIC), then keep only the most significant lags by p-value (sparse subset like [1,4,7])
      - VAR: statsmodels only supports dense lag order (1 ... p for all vars), so here we just pick a single p by BIC (bounded by sample size), no per-lag p-value filtering

    Args:
        init_history_mv (pd.DataFrame): normalized multivariate history (train+val)
        max_lags (int, default=24): maximum lag size considered for selection

    Returns:
        dict with:
            - "p": lag order used (dense lags 1 ... p)
            - "sel_lags": [1 ... p] chosen by BIC
            - "sig_lags": [] (kept empty for compatibility with AR case)
            - "model": fitted VAR(p) model
            - "n_params": number of estimated parameters
            - "fit_time": fitting time in seconds
    """
    # remove DatetimeIndex and force RangeIndex (kills freq warning)
    y = init_history_mv.reset_index(drop=True)

    # safe max lag (avoid asking for too many lags for small samples like in 24h res case)
    safe_max = max(1, min(len(y) // 10, max_lags))

    # select order by BIC (default to 1 if None)
    sel = VAR(y).select_order(maxlags=safe_max)
    p = sel.selected_orders.get("bic") or 1
    p = int(max(1, min(p, safe_max))) # make sure 1 <= p <= safe_max

    # fit VAR(p) once at p (dense 1 ... p)
    t0 = time.time()
    fit = VAR(y).fit(p)
    t1 = time.time()

    return {
        "p": p,
        "sel_lags": list(range(1, p + 1)),
        "sig_lags": [],  # empty: VAR cannot keep significant lags
        "model": fit,
        "n_params": int(fit.params.size),
        "fit_time": round(t1 - t0, 4),
    }


def var_forecast(train_mv, val_mv, test_mv, res, target, horizon, p, mu_t, std_t, is_diff_target, level_series_raw=None, return_full_path=False):
    """
    Rolling Forecast Origin (walk-forward, expanding the window) for VAR(p) with fixed order p.

        - Within-origin: statsmodels VAR forecast(h) rolls recursively on its own preds
        - Across-origins: append TRUE row to history and refit/update (every k steps)
        - Alignment: keep preds at origin index t; align LEVEL truth by shift(-horizon)

    Args:
        train_mv (pd.DataFrame): normalized train set (all features incl. target)
        val_mv (pd.DataFrame): normalized validation set
        test_mv (pd.DataFrame): normalized test set
        res (str): resolution label (e.g., "1h", "6h", "24h")
        target (str): target column name (e.g., "T")
        horizon (int): forecast horizon (1 or >1)
        p (int): fixed VAR order selected once by BIC
        mu_t (float): mean of target (for denormalization)
        std_t (float): std  of target (for denormalization)
        is_diff_target (bool): True if target modeled in DIFF space
        level_series_raw (pd.Series|None): raw LEVEL target (needed if DIFF)
        return_full_path (bool): if True, also return full denormalized paths

    Returns:
        y_pred_level (pd.Series): denormalized LEVEL predictions (LEVEL, last-step t+h)
        y_true_level (pd.Series): denormalized LEVEL ground truth (LEVEL, aligned to t+h)
        full_path_level (pd.DataFrame, opt): denormalized full forecast paths (step1 ... steph in LEVEL)
    """
    # skip 24h resolution to avoid SVD/MKL errors (too few daily samples)
    if res == "24h":
        raise RuntimeError("VAR is disabled for 24h resolution (insufficient data).")

    # expand history (normalized modeling space)
    history = pd.concat([train_mv, val_mv]).sort_index()
    test_mv = test_mv.sort_index()

    # containers
    preds_last_n = []  # last-step (normalized) for target
    keep_paths = return_full_path or (is_diff_target and horizon > 1)
    paths_n = [] if keep_paths else None  # per-origin normalized target paths
    bases_level = [] if is_diff_target else None  # base LEVEL y_t per origin (only if DIFF)

    # small refit to speed up
    refit_every = 6
    last_model = None
    last_p_eff = None
    since_refit = refit_every
    tgt_idx = history.columns.get_loc(target)

    # walk forward across test origins
    for t, row in test_mv.iterrows():
        if is_diff_target:
            bases_level.append(level_series_raw.loc[t]) # LEVEL base at origin

        # fit VAR on expanding history (RangeIndex avoids date-frequency issues)
        y_hist = history.reset_index(drop=True)
        # minimal cleanup
        y_hist = y_hist.replace([np.inf, -np.inf], np.nan).dropna().astype("float64")

        # safe effective lag order
        p_eff = max(1, min(int(p), len(y_hist) // 10, 8))

        # refit every k steps
        need_refit = (since_refit >= refit_every) or (last_model is None) or (p_eff != last_p_eff)
        if need_refit:
            model = VAR(y_hist).fit(p_eff)
            last_model = model
            last_p_eff = p_eff
            since_refit = 0
        else:
            model = last_model

        # step 2: forecast 'horizon' steps ahead for all variables, shape (h, n_features)
        # - input to forecast = last p rows (the most recent p observations of all features)
        # - output shape = (horizon, num_features), each row = one step ahead forecast for all variables
        fc = model.forecast(y_hist.values[-p_eff:], steps=horizon)  # normalized
        fc_t = fc[:, tgt_idx]  # normalized target path
        preds_last_n.append(fc_t[-1])  # last-step only
        if keep_paths:
            paths_n.append(fc_t.copy())  # keep the full path if needed

        # across-origins update: append TRUE row and move on
        history.loc[t] = row.values
        since_refit += 1

        # walk forward with the current forecast
        # history_mv.loc[t] = fc_t[0]

    # align predictions to test index (normalized space)
    preds_last_n = pd.Series(preds_last_n, index=test_mv.index, name=f"{res}_{target}_pred_n")

    # build preds series at origin index
    preds_last_n = pd.Series(preds_last_n, index=test_mv.index, name=f"{res}_{target}_pred_n")

    # build LEVEL truth aligned to t+h (shift LEVEL by -h)
    if not is_diff_target:
        y_true_level = denormalize(test_mv[target].shift(-horizon).reindex(preds_last_n.index), mu_t, std_t)
    else:
        y_true_level = level_series_raw.shift(-horizon).reindex(preds_last_n.index)

    # drop rows with no y_{t+h} (last h origins and any gaps)
    mask = y_true_level.notna()
    preds_last_n = preds_last_n[mask]
    y_true_level = y_true_level[mask]

    # keep lists aligned to valid origins
    valid_len = len(preds_last_n)
    if keep_paths:
        paths_n = paths_n[:valid_len]
    if is_diff_target:
        bases_level = bases_level[:valid_len]

    # produce LEVEL predictions
    if not is_diff_target:
        # LEVEL model: denorm last-step preds directly
        y_pred_level = denormalize(preds_last_n, mu_t, std_t)

        if return_full_path:
            cols = [f"step{k}" for k in range(1, horizon + 1)]
            # if horizon==1 → one column 'step1', else if h>1 → step1 ... steph
            if horizon == 1:
                full_path_level = y_pred_level.to_frame(name="step1")
            else:
                full_path_level = pd.DataFrame(
                    [denormalize(pd.Series(p, index=cols), mu_t, std_t) for p in paths_n],
                    index=y_pred_level.index, columns=cols
                )
            return y_pred_level, y_true_level, full_path_level

        return y_pred_level, y_true_level

    # DIFF model: invert Δy to LEVEL using base level
    if horizon == 1:
        # ŷ_{t+1} = y_t + denorm(Δŷ_{t+1})
        bases_aligned = pd.Series(bases_level, index=preds_last_n.index)
        delta_level   = denormalize(preds_last_n, mu_t, std_t)
        y_pred_level  = bases_aligned + delta_level

        if return_full_path:
            full_path_level = y_pred_level.to_frame(name="step1")
            return y_pred_level, y_true_level, full_path_level

        return y_pred_level, y_true_level

    # h>1: need Δ-path per origin → denorm, cumsum, add base level; keep only last step
    cols = [f"step{k}" for k in range(1, horizon + 1)]
    last_vals, full_rows = [], ([] if return_full_path else None)

    for base, path_n in zip(bases_level, paths_n):
        dn = denormalize(pd.Series(path_n, index=cols), mu_t, std_t)  # Δy in LEVEL units
        lv = base + dn.cumsum()                                       # LEVEL path
        last_vals.append(lv.iloc[-1])
        if return_full_path:
            full_rows.append(lv.values)

    y_pred_level = pd.Series(last_vals, index=preds_last_n.index, name=f"{res}_{target}_pred_level")

    if return_full_path:
        full_path_level = pd.DataFrame(full_rows, index=y_pred_level.index, columns=cols)
        return y_pred_level, y_true_level, full_path_level

    return y_pred_level, y_true_level
