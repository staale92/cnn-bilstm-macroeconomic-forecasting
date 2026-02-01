# A CNN–BiLSTM architecture for macroeconomic time series forecasting

This repo contains a **reproducible experiment pipeline** for the paper *“A CNN–BiLSTM architecture for macroeconomic time series forecasting”*:

- deterministic preprocessing (time-respecting)
- scaling + windowing
- deep learning models (BiLSTM baseline + CNN–BiLSTM) with **10 repeated seeds**
- baselines (ARIMA/ARIMAX/VAR + Elastic Net + optional XGBoost)
- consistent evaluation (MAE/RMSE) + paired t-tests
- tables + quick sanity figures

## Data
The provided dataset is `data/data_all_variables.xls` (871 monthly observations, 18 variables + `date`).

Note: the XLS contains **7 leading NaNs** for `employment_rate`, `unemployment_rate`, `unempl_over_27w`. The pipeline fills these using `backfill_then_forwardfill` (configurable in `configs/dataset.yaml`).

## Install (optional)
Scripts are runnable without installing the package, but for development/testing you can do:

```bash
python -m pip install -e ".[dev,baselines]"
```

## Reproduce
Each experiment is configured via YAML under `configs/experiment/`:

- `univariate_multistep.yaml` (12-step direct)
- `univariate_onestep.yaml` (1-step repeated)
- `multivariate_multistep.yaml` (12-step direct, covariates from Table A1)
- `multivariate_onestep.yaml` (1-step repeated, covariates from Table A1)

Run an experiment end-to-end (example: univariate multi-step):

```bash
python scripts/00_validate_raw_data.py --dataset-config configs/dataset.yaml
python scripts/01_build_processed_dataset.py --dataset-config configs/dataset.yaml --experiment-config configs/experiment/univariate_multistep.yaml
python scripts/02_run_baselines.py --config configs/experiment/univariate_multistep.yaml
python scripts/03_train_deep_models.py --config configs/experiment/univariate_multistep.yaml
python scripts/04_evaluate_all.py --config configs/experiment/univariate_multistep.yaml
python scripts/05_make_tables_and_figures.py --config configs/experiment/univariate_multistep.yaml
```

## Hyperparameter tuning (optional)
`scripts/03_train_deep_models.py` supports **random-search tuning on a train/validation split only** (no test leakage). Enable it by adding a `tuning:` section to an experiment config (see `configs/experiment/smoke_tuned_univariate_multistep.yaml`).

- Tuning artifacts are written to `results/tuning/<experiment>__<hash>/<model>/<target>/`
- `tuning.repeats` (default 1) can be increased to evaluate each trial across multiple random initializations (more robust, slower).
- Cached `best.json` is reused automatically; force a re-tune with:

```bash
python scripts/03_train_deep_models.py --config <your_config.yaml> --retune
```

Outputs:
- `results/experiments/<experiment>__<hash>/...` (per-run artifacts: config, metrics, predictions, weights)
- `results/summary/<experiment>__<hash>/metrics_*.csv`
- `results/tables/*.csv`
- `results/figures/*.png`

## Configs sourced from the paper
- `configs/covariates_table_a1.yaml`: Table A1 (Granger-selected covariates per target)
- `configs/baselines.yaml`: Appendix A.2 (ARIMA/ARIMAX/VAR specifications)

## Citation
If you use this repository, please cite:

Staffini, A. (2023). A CNN–BiLSTM Architecture for Macroeconomic Time Series Forecasting. Engineering Proceedings, 39(1), 33. https://doi.org/10.3390/engproc2023039033

## Tests
```bash
python -m pytest
```
