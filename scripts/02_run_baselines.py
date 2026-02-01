from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from _common import bootstrap_src, get_targets, load_experiment_bundle


def _stack_slices(y: np.ndarray, *, horizon: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    n_origins = len(y) - horizon + 1
    return np.stack([y[i : i + horizon] for i in range(n_origins)], axis=0)


def _inverse_y(y_scaler, arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    return y_scaler.inverse_transform(a.reshape(-1, 1)).reshape(a.shape)


def main() -> None:
    bootstrap_src()

    from macro_bilstm.baselines.arima import rolling_forecast_sarimax
    from macro_bilstm.baselines.elasticnet import ElasticNetConfig, fit_predict_multistep, fit_predict_onestep
    from macro_bilstm.baselines.var import rolling_forecast_var
    from macro_bilstm.data.build import build_experiment_data
    from macro_bilstm.evaluation.evaluate import evaluate_direct_multistep, evaluate_onestep, metrics_to_dict
    from macro_bilstm.utils.artifacts import make_run_paths, save_json, save_predictions_npz
    from macro_bilstm.utils.experiment import experiment_fingerprint

    p = argparse.ArgumentParser(description="Run baseline models (ARIMA/ARIMAX/VAR/ElasticNet/XGBoost) for an experiment config.")
    p.add_argument("--config", default="configs/experiment/univariate_multistep.yaml")
    args = p.parse_args()

    exp_cfg_all, dataset_cfg, cov_cfg, baselines_cfg = load_experiment_bundle(args.config)
    exp: dict[str, Any] = exp_cfg_all["experiment"]
    horizon = int(exp["horizon"])
    lookback = int(exp["lookback"])
    feature_mode = str(exp["feature_mode"])
    results_dir = Path(exp.get("results_dir", "results"))

    exp_id, fingerprint = experiment_fingerprint(exp_cfg_all, dataset_cfg)
    experiment_name = exp["name"]

    targets = get_targets(exp_cfg_all)
    for target in targets:
        ed = build_experiment_data(
            raw_path=dataset_cfg["dataset"]["raw_path"],
            sheet_name=dataset_cfg["dataset"].get("sheet_name", 0),
            missing_strategy=dataset_cfg["dataset"].get("fill_missing", {}).get("strategy", "backfill_then_forwardfill"),
            train_size=int(dataset_cfg["train_test_split"]["train_size"]),
            target=target,
            feature_mode=feature_mode,
            covariates=cov_cfg,
            lookback=lookback,
            horizon=horizon,
        )

        y_train_full = ed.df[target].iloc[: ed.train_size].to_numpy(dtype=np.float64)
        y_test_full = ed.df[target].iloc[ed.train_size :].to_numpy(dtype=np.float64)
        y_true = _stack_slices(y_test_full, horizon=horizon)

        base_spec = baselines_cfg[target]
        arima_order = tuple(
            base_spec["arima"]["onestep_order"] if horizon == 1 else base_spec["arima"]["multistep_order"]
        )
        arimax_order = tuple(base_spec["arimax"]["order"])
        var_lags = int(base_spec["var"]["lags"])

        # ---------- ARIMA ----------
        try:
            y_pred = rolling_forecast_sarimax(
                y_train=y_train_full,
                y_test=y_test_full,
                exog_train=None,
                exog_test=None,
                order=arima_order,
                horizon=horizon,
            )
        except ImportError as e:
            print(f"[skip] ARIMA for {target}: {e}")
        else:
            res = evaluate_onestep(y_true[:, 0], y_pred[:, 0]) if horizon == 1 else evaluate_direct_multistep(y_true, y_pred)
            rp = make_run_paths(
                results_dir=results_dir,
                exp_id=exp_id,
                experiment_name=experiment_name,
                model_name="arima",
                target=target,
                seed=None,
            )
            rp.run_dir.mkdir(parents=True, exist_ok=True)
            save_json(
                rp.config_json,
                {"fingerprint": fingerprint, "model": {"name": "arima", "order": arima_order}, "target": target},
            )
            save_json(rp.metrics_json, metrics_to_dict(res))
            save_predictions_npz(rp.predictions_npz, y_true=y_true, y_pred=y_pred, meta={"horizon": horizon})

        # ---------- ARIMAX / VAR (multivariate only) ----------
        if feature_mode == "multivariate":
            covs = list(cov_cfg[target])
            exog_train = ed.df[covs].iloc[: ed.train_size].to_numpy(dtype=np.float64)
            exog_test = ed.df[covs].iloc[ed.train_size :].to_numpy(dtype=np.float64)

            try:
                y_pred = rolling_forecast_sarimax(
                    y_train=y_train_full,
                    y_test=y_test_full,
                    exog_train=exog_train,
                    exog_test=exog_test,
                    order=arimax_order,
                    horizon=horizon,
                )
            except ImportError as e:
                print(f"[skip] ARIMAX for {target}: {e}")
            else:
                res = (
                    evaluate_onestep(y_true[:, 0], y_pred[:, 0])
                    if horizon == 1
                    else evaluate_direct_multistep(y_true, y_pred)
                )
                rp = make_run_paths(
                    results_dir=results_dir,
                    exp_id=exp_id,
                    experiment_name=experiment_name,
                    model_name="arimax",
                    target=target,
                    seed=None,
                )
                rp.run_dir.mkdir(parents=True, exist_ok=True)
                save_json(
                    rp.config_json,
                    {"fingerprint": fingerprint, "model": {"name": "arimax", "order": arimax_order, "covariates": covs}, "target": target},
                )
                save_json(rp.metrics_json, metrics_to_dict(res))
                save_predictions_npz(rp.predictions_npz, y_true=y_true, y_pred=y_pred, meta={"horizon": horizon})

            # VAR: system is [target] + covariates
            Y_train = ed.df[[target] + covs].iloc[: ed.train_size].to_numpy(dtype=np.float64)
            Y_test = ed.df[[target] + covs].iloc[ed.train_size :].to_numpy(dtype=np.float64)
            try:
                y_pred = rolling_forecast_var(Y_train=Y_train, Y_test=Y_test, lags=var_lags, horizon=horizon, target_col=0)
            except ImportError as e:
                print(f"[skip] VAR for {target}: {e}")
            else:
                res = (
                    evaluate_onestep(y_true[:, 0], y_pred[:, 0])
                    if horizon == 1
                    else evaluate_direct_multistep(y_true, y_pred)
                )
                rp = make_run_paths(
                    results_dir=results_dir,
                    exp_id=exp_id,
                    experiment_name=experiment_name,
                    model_name="var",
                    target=target,
                    seed=None,
                )
                rp.run_dir.mkdir(parents=True, exist_ok=True)
                save_json(
                    rp.config_json,
                    {"fingerprint": fingerprint, "model": {"name": "var", "lags": var_lags, "covariates": covs}, "target": target},
                )
                save_json(rp.metrics_json, metrics_to_dict(res))
                save_predictions_npz(rp.predictions_npz, y_true=y_true, y_pred=y_pred, meta={"horizon": horizon})

        # ---------- Elastic Net ----------
        Xtr = ed.train_windows.X.reshape(len(ed.train_windows.X), -1)
        Xte = ed.test_windows.X.reshape(len(ed.test_windows.X), -1)
        if horizon == 1:
            ytr = ed.train_windows.y.reshape(-1)
            y_pred_scaled = fit_predict_onestep(Xtr, ytr, Xte, cfg=ElasticNetConfig())
        else:
            ytr = ed.train_windows.y
            y_pred_scaled = fit_predict_multistep(Xtr, ytr, Xte, cfg=ElasticNetConfig())

        y_pred = _inverse_y(ed.scalers.y_scaler, y_pred_scaled)
        if horizon == 1 and y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        res = (
            evaluate_onestep(y_true[:, 0], y_pred[:, 0])
            if horizon == 1
            else evaluate_direct_multistep(y_true, y_pred)
        )
        rp = make_run_paths(
            results_dir=results_dir,
            exp_id=exp_id,
            experiment_name=experiment_name,
            model_name="elasticnet",
            target=target,
            seed=None,
        )
        rp.run_dir.mkdir(parents=True, exist_ok=True)
        save_json(rp.config_json, {"fingerprint": fingerprint, "model": {"name": "elasticnet"}, "target": target})
        save_json(rp.metrics_json, metrics_to_dict(res))
        save_predictions_npz(rp.predictions_npz, y_true=y_true, y_pred=y_pred, meta={"horizon": horizon})

        # ---------- XGBoost (optional dependency) ----------
        try:
            from macro_bilstm.baselines.xgboost import XGBoostConfig, fit_predict_multistep as xgb_ms, fit_predict_onestep as xgb_os
        except Exception as e:
            print(f"[skip] XGBoost for {target}: {e}")
        else:
            if horizon == 1:
                y_pred_scaled = xgb_os(Xtr, ytr, Xte, cfg=XGBoostConfig())
            else:
                y_pred_scaled = xgb_ms(Xtr, ytr, Xte, cfg=XGBoostConfig())
            y_pred = _inverse_y(ed.scalers.y_scaler, y_pred_scaled)
            if horizon == 1 and y_pred.ndim == 1:
                y_pred = y_pred.reshape(-1, 1)
            res = (
                evaluate_onestep(y_true[:, 0], y_pred[:, 0])
                if horizon == 1
                else evaluate_direct_multistep(y_true, y_pred)
            )
            rp = make_run_paths(
                results_dir=results_dir,
                exp_id=exp_id,
                experiment_name=experiment_name,
                model_name="xgboost",
                target=target,
                seed=None,
            )
            rp.run_dir.mkdir(parents=True, exist_ok=True)
            save_json(rp.config_json, {"fingerprint": fingerprint, "model": {"name": "xgboost"}, "target": target})
            save_json(rp.metrics_json, metrics_to_dict(res))
            save_predictions_npz(rp.predictions_npz, y_true=y_true, y_pred=y_pred, meta={"horizon": horizon})

        print(f"[done] {target}")


if __name__ == "__main__":
    main()
