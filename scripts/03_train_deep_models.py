from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from _common import bootstrap_src, get_targets, load_experiment_bundle


def _stack_slices(y: np.ndarray, *, horizon: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    n_origins = len(y) - horizon + 1
    return np.stack([y[i : i + horizon] for i in range(n_origins)], axis=0)


def _inverse_y(y_scaler, arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    return y_scaler.inverse_transform(a.reshape(-1, 1)).reshape(a.shape)


def _train_val_split(X: np.ndarray, y: np.ndarray, *, val_fraction: float) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    if val_fraction <= 0:
        return X, y, None, None
    n = len(X)
    n_val = max(1, int(round(n * val_fraction)))
    if n_val >= n:
        n_val = max(1, n - 1)
    return X[:-n_val], y[:-n_val], X[-n_val:], y[-n_val:]


def main() -> None:
    bootstrap_src()

    from macro_bilstm.data.build import build_experiment_data
    from macro_bilstm.evaluation.evaluate import evaluate_direct_multistep, evaluate_onestep, metrics_to_dict
    from macro_bilstm.models.bilstm_torch import BiLSTMRegressor
    from macro_bilstm.models.cnn_bilstm_torch import CNNBiLSTM
    from macro_bilstm.training.train_torch import predict, train_model
    from macro_bilstm.training.tuning import deep_merge, random_search_tune, save_tuning_artifacts
    from macro_bilstm.utils.artifacts import make_run_paths, save_json, save_predictions_npz
    from macro_bilstm.utils.experiment import experiment_fingerprint
    from macro_bilstm.utils.seed import set_global_seed

    p = argparse.ArgumentParser(description="Train deep models (BiLSTM baseline + CNN–BiLSTM) with repeated seeds.")
    p.add_argument("--config", default="configs/experiment/univariate_multistep.yaml")
    p.add_argument("--seeds", nargs="*", type=int, default=None, help="Override seeds list (space-separated).")
    p.add_argument("--retune", action="store_true", help="Force re-running hyperparameter tuning even if cached.")
    args = p.parse_args()

    exp_cfg_all, dataset_cfg, cov_cfg, _ = load_experiment_bundle(args.config)
    exp: dict[str, Any] = exp_cfg_all["experiment"]
    training: dict[str, Any] = exp_cfg_all.get("training", {})
    model_cfg: dict[str, Any] = exp_cfg_all.get("model", {})

    horizon = int(exp["horizon"])
    lookback = int(exp["lookback"])
    feature_mode = str(exp["feature_mode"])
    results_dir = Path(exp.get("results_dir", "results"))

    exp_id, fingerprint = experiment_fingerprint(exp_cfg_all, dataset_cfg)
    experiment_name = exp["name"]

    seeds = args.seeds
    if seeds is None:
        seeds = list(exp_cfg_all.get("repeats", {}).get("seeds", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))

    epochs = int(training.get("epochs", 200))
    batch_size = int(training.get("batch_size", 32))
    learning_rate = float(training.get("learning_rate", 1e-3))
    max_norm = float(training.get("max_norm", 3.0))
    early_stopping = bool(training.get("early_stopping", True))
    patience = int(training.get("patience", 20))
    val_fraction = float(training.get("val_fraction", 0.1))
    weight_decay = float(training.get("weight_decay", 0.0))
    loss_name = str(training.get("loss", "mse"))
    grad_clip_norm = training.get("grad_clip_norm")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tuning_cfg = exp_cfg_all.get("tuning", {})
    tuning_enabled = bool(tuning_cfg.get("enabled", False))
    tuning_metric = str(tuning_cfg.get("metric", "mae"))
    tuning_trials = int(tuning_cfg.get("n_trials", 15))
    tuning_repeats = int(tuning_cfg.get("repeats", 1))
    tuning_seed = int(tuning_cfg.get("seed", 123))
    tuning_max_epochs = int(tuning_cfg.get("max_epochs", min(50, epochs)))
    tuning_patience = int(tuning_cfg.get("patience", min(10, patience)))
    tuning_search_space = tuning_cfg.get("search_space", {})
    tuning_models = tuning_cfg.get("models", ["cnn_bilstm"])  # default: tune only the main model

    tuning_root = results_dir / "tuning" / f"{experiment_name}__{exp_id}"

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

        y_test_full = ed.df[target].iloc[ed.train_size :].to_numpy(dtype=np.float64)
        y_true = _stack_slices(y_test_full, horizon=horizon)

        X_train_full = ed.train_windows.X
        y_train_full = ed.train_windows.y
        X_test = ed.test_windows.X

        X_train, y_train, X_val, y_val = _train_val_split(X_train_full, y_train_full, val_fraction=val_fraction)

        n_features = X_train.shape[-1]

        tuned: dict[str, dict[str, Any]] = {}
        if tuning_enabled:
            if X_val is None or y_val is None:
                raise SystemExit("tuning.enabled=true requires training.val_fraction > 0 (needs a validation split)")
            for model_name in tuning_models:
                if model_name not in {"cnn_bilstm", "bilstm"}:
                    raise SystemExit(f"Unknown tunable model: {model_name!r} (expected: cnn_bilstm|bilstm)")

                # Choose per-model space if provided, otherwise use top-level.
                space = tuning_search_space.get(model_name, tuning_search_space)
                if not isinstance(space, dict) or not space:
                    raise SystemExit("tuning.search_space must be a non-empty mapping")

                out_dir = tuning_root / model_name / target
                best_path = out_dir / "best.json"
                if best_path.exists() and not args.retune:
                    best_payload = json.loads(best_path.read_text(encoding="utf-8"))
                    tuned[model_name] = best_payload["best"]
                    print(f"[tuning cached] {target} {model_name} score={best_payload['best_score']:.4f}")
                    continue

                base_model_cfg = model_cfg.get(model_name, {})
                if not base_model_cfg:
                    raise SystemExit(f"Missing model config for '{model_name}'")

                base_training_cfg = {
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "loss": loss_name,
                    "grad_clip_norm": grad_clip_norm,
                }

                def build(mcfg: dict[str, Any]):
                    if model_name == "cnn_bilstm":
                        return CNNBiLSTM(
                            lookback=lookback,
                            n_features=n_features,
                            horizon=horizon,
                            conv_filters=int(mcfg.get("conv_filters", 64)),
                            conv_kernel=int(mcfg.get("conv_kernel", 3)),
                            pool_size=int(mcfg.get("pool_size", 2)),
                            lstm_units=tuple(mcfg.get("lstm_units", [200, 100, 50])),
                            fc_units=int(mcfg.get("fc_units", 25)),
                            dropout_cnn=float(mcfg.get("dropout_cnn", 0.1)),
                            dropout_lstm=float(mcfg.get("dropout_lstm", 0.5)),
                            dropout_fc=float(mcfg.get("dropout_fc", 0.1)),
                        )
                    return BiLSTMRegressor(
                        lookback=lookback,
                        n_features=n_features,
                        horizon=horizon,
                        lstm_units=tuple(mcfg.get("lstm_units", [200, 100, 50])),
                        fc_units=int(mcfg.get("fc_units", 25)),
                        dropout_lstm=float(mcfg.get("dropout_lstm", 0.5)),
                        dropout_fc=float(mcfg.get("dropout_fc", 0.1)),
                    )

                print(f"[tuning] {target} {model_name}: trials={tuning_trials}, metric={tuning_metric}")
                result = random_search_tune(
                    build_model=build,
                    base_model_cfg=base_model_cfg,
                    base_training_cfg=base_training_cfg,
                    search_space=space,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    y_scaler=ed.scalers.y_scaler,
                    metric=tuning_metric,
                    n_trials=tuning_trials,
                    repeats=tuning_repeats,
                    seed=tuning_seed,
                    device=device,
                    max_epochs=tuning_max_epochs,
                    batch_size=batch_size,
                    max_norm=max_norm,
                    early_stopping=early_stopping,
                    patience=tuning_patience,
                )
                save_tuning_artifacts(
                    out_dir=out_dir,
                    result=result,
                    extra={"experiment": experiment_name, "exp_id": exp_id, "target": target, "model": model_name},
                )
                tuned[model_name] = result.best
                print(f"[tuning done] {target} {model_name}: best_{tuning_metric}={result.best_score:.4f} (repeats={tuning_repeats})")

        # --------- BiLSTM baseline ---------
        bilstm_cfg = tuned.get("bilstm", {}).get("model") if "bilstm" in tuned else model_cfg.get("bilstm", {})
        bilstm_training = tuned.get("bilstm", {}).get("training", {}) if "bilstm" in tuned else {}
        eff_training_bilstm = deep_merge(
            {
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "loss": loss_name,
                "grad_clip_norm": grad_clip_norm,
            },
            bilstm_training,
        )
        for seed in seeds:
            set_global_seed(seed)
            model = BiLSTMRegressor(
                lookback=lookback,
                n_features=n_features,
                horizon=horizon,
                lstm_units=tuple(bilstm_cfg.get("lstm_units", [200, 100, 50])),
                fc_units=int(bilstm_cfg.get("fc_units", 25)),
                dropout_lstm=float(bilstm_cfg.get("dropout_lstm", 0.5)),
                dropout_fc=float(bilstm_cfg.get("dropout_fc", 0.1)),
            )

            tr = train_model(
                model,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=float(eff_training_bilstm.get("learning_rate", learning_rate)),
                weight_decay=float(eff_training_bilstm.get("weight_decay", 0.0)),
                loss=str(eff_training_bilstm.get("loss", "mse")),
                grad_clip_norm=eff_training_bilstm.get("grad_clip_norm"),
                max_norm=max_norm,
                early_stopping=early_stopping,
                patience=patience,
                device=device,
            )

            y_pred_scaled = predict(model, X_test, device=device)
            y_pred = _inverse_y(ed.scalers.y_scaler, y_pred_scaled)

            res = (
                evaluate_onestep(y_true[:, 0], y_pred[:, 0])
                if horizon == 1
                else evaluate_direct_multistep(y_true, y_pred)
            )

            rp = make_run_paths(
                results_dir=results_dir,
                exp_id=exp_id,
                experiment_name=experiment_name,
                model_name="bilstm",
                target=target,
                seed=seed,
            )
            rp.run_dir.mkdir(parents=True, exist_ok=True)
            save_json(
                rp.config_json,
                {
                    "fingerprint": fingerprint,
                    "seed": seed,
                    "target": target,
                    "features": ed.features,
                    "model": {"name": "bilstm", **bilstm_cfg},
                    "training": {**training, **eff_training_bilstm, "device": str(device)},
                },
            )
            save_json(rp.metrics_json, {**metrics_to_dict(res), "best_epoch": tr.best_epoch})
            save_predictions_npz(rp.predictions_npz, y_true=y_true, y_pred=y_pred, meta={"horizon": horizon})
            torch.save(model.state_dict(), rp.model_pt)

        # --------- CNN–BiLSTM ---------
        cnn_cfg = tuned.get("cnn_bilstm", {}).get("model") if "cnn_bilstm" in tuned else model_cfg.get("cnn_bilstm", {})
        cnn_training = tuned.get("cnn_bilstm", {}).get("training", {}) if "cnn_bilstm" in tuned else {}
        eff_training_cnn = deep_merge(
            {
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "loss": loss_name,
                "grad_clip_norm": grad_clip_norm,
            },
            cnn_training,
        )
        for seed in seeds:
            set_global_seed(seed)
            model = CNNBiLSTM(
                lookback=lookback,
                n_features=n_features,
                horizon=horizon,
                conv_filters=int(cnn_cfg.get("conv_filters", 64)),
                conv_kernel=int(cnn_cfg.get("conv_kernel", 3)),
                pool_size=int(cnn_cfg.get("pool_size", 2)),
                lstm_units=tuple(cnn_cfg.get("lstm_units", [200, 100, 50])),
                fc_units=int(cnn_cfg.get("fc_units", 25)),
                dropout_cnn=float(cnn_cfg.get("dropout_cnn", 0.1)),
                dropout_lstm=float(cnn_cfg.get("dropout_lstm", 0.5)),
                dropout_fc=float(cnn_cfg.get("dropout_fc", 0.1)),
            )

            tr = train_model(
                model,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=float(eff_training_cnn.get("learning_rate", learning_rate)),
                weight_decay=float(eff_training_cnn.get("weight_decay", 0.0)),
                loss=str(eff_training_cnn.get("loss", "mse")),
                grad_clip_norm=eff_training_cnn.get("grad_clip_norm"),
                max_norm=max_norm,
                early_stopping=early_stopping,
                patience=patience,
                device=device,
            )

            y_pred_scaled = predict(model, X_test, device=device)
            y_pred = _inverse_y(ed.scalers.y_scaler, y_pred_scaled)
            res = (
                evaluate_onestep(y_true[:, 0], y_pred[:, 0])
                if horizon == 1
                else evaluate_direct_multistep(y_true, y_pred)
            )

            rp = make_run_paths(
                results_dir=results_dir,
                exp_id=exp_id,
                experiment_name=experiment_name,
                model_name="cnn_bilstm",
                target=target,
                seed=seed,
            )
            rp.run_dir.mkdir(parents=True, exist_ok=True)
            save_json(
                rp.config_json,
                {
                    "fingerprint": fingerprint,
                    "seed": seed,
                    "target": target,
                    "features": ed.features,
                    "model": {"name": "cnn_bilstm", **cnn_cfg},
                    "training": {**training, **eff_training_cnn, "device": str(device)},
                },
            )
            save_json(rp.metrics_json, {**metrics_to_dict(res), "best_epoch": tr.best_epoch})
            save_predictions_npz(rp.predictions_npz, y_true=y_true, y_pred=y_pred, meta={"horizon": horizon})
            torch.save(model.state_dict(), rp.model_pt)

        print(f"[done] {target}")


if __name__ == "__main__":
    main()
