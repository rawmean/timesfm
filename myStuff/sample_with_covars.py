import timesfm
import matplotlib.pyplot as plt
import numpy as np

model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")

model.compile(
    timesfm.ForecastConfig(
        max_context=7,
        max_horizon=7,
        normalize_inputs=True,
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        infer_is_positive=True,
        fix_quantile_crossing=True,
        return_backcast=True,
    )
)
# Example data for two products (ice cream and sunscreen)
inputs = [[30, 30, 4, 5, 7, 8, 10],    # observed sales (7 days)
          [5, 7, 12, 13, 5, 6, 10]]
# dynamic covariates must include both past 7 days and the next 7 days (total 14)
dynamic_categorical_covariates = {
    "Weekday": [[0,1,2,3,4,5,6, 0,1,2,3,4,5,6],
                [0,1,2,3,4,5,6, 0,1,2,3,4,5,6]],
    "Has_promotion": [[1,1,0,0,0,1,1, 1,1,1,0,0,1,0],
                      [0,0,1,1,0,0,0, 1,1,1,1,1,1,1]]
}
dynamic_numerical_covariates = {
    "Daily_temperature": [[31.0,24.3,19.4,26.2,24.6,30.0,31.1,
                           32.4,30.9,26.0,25.0,27.8,29.5,31.2],
                          [31.0,24.3,19.4,26.2,24.6,30.0,31.1,
                           32.4,30.9,26.0,25.0,27.8,29.5,31.2]]
}
static_categorical_covariates = {"Category": ["food", "skin product"]}
static_numerical_covariates = {"Base_price": [1.99, 29.99]}

# Forecast with covariates.
# For short histories, use "xreg + timesfm" to avoid zero-length residual fitting.
cov_forecast, linear_forecast = model.forecast_with_covariates(
    inputs=inputs,
    dynamic_numerical_covariates=dynamic_numerical_covariates,
    dynamic_categorical_covariates=dynamic_categorical_covariates,
    static_numerical_covariates=static_numerical_covariates,
    static_categorical_covariates=static_categorical_covariates,
    # freq=[0, 0],       # frequency indicator (0 for arbitrary)
    xreg_mode="xreg + timesfm",  # use "timesfm + xreg" only with longer history
    ridge=0.0,
    normalize_xreg_target_per_input=True
)

print("TSFM forecast with exogenous variables:", cov_forecast)

# Forecast without dynamic covariates (static only) for comparison.
cov_forecast_no_dynamic, linear_forecast_no_dynamic = model.forecast_with_covariates(
    inputs=inputs,
    static_numerical_covariates=static_numerical_covariates,
    static_categorical_covariates=static_categorical_covariates,
    xreg_mode="xreg + timesfm",
    ridge=0.0,
    normalize_xreg_target_per_input=True
)

print("TSFM forecast without dynamic covariates:", cov_forecast_no_dynamic)

# Force no-dynamic predictions to the same horizon as the dynamic-covariate run.
comparison_horizons = [len(np.asarray(pred)) for pred in cov_forecast]
cov_forecast_no_dynamic = [
    np.asarray(pred_no_dyn, dtype=float)[:h]
    for pred_no_dyn, h in zip(cov_forecast_no_dynamic, comparison_horizons)
]

# Visualize history + forecast comparison and dynamic covariates for each series.
series_names = ["Ice cream", "Sunscreen"]
num_series = len(inputs)
fig, axes = plt.subplots(
    num_series, 2, figsize=(14, 4 * num_series), squeeze=False, sharex=False
)

all_dynamic_covariates = {}
all_dynamic_covariates.update(dynamic_numerical_covariates)
all_dynamic_covariates.update(dynamic_categorical_covariates)

for i in range(num_series):
    ax_target = axes[i, 0]
    ax_cov = axes[i, 1]

    hist = np.asarray(inputs[i], dtype=float)
    pred_with_dyn = np.asarray(cov_forecast[i], dtype=float)
    pred_no_dyn = np.asarray(cov_forecast_no_dynamic[i], dtype=float)
    forecast_start = len(hist)

    x_hist = np.arange(len(hist))
    x_pred_with_dyn = np.arange(len(hist), len(hist) + len(pred_with_dyn))
    x_pred_no_dyn = np.arange(len(hist), len(hist) + len(pred_no_dyn))

    ax_target.plot(x_hist, hist, marker="o", linewidth=2, label="History")
    ax_target.plot(
        x_pred_with_dyn,
        pred_with_dyn,
        marker="o",
        linestyle="--",
        linewidth=2,
        label="Forecast (with dynamic covariates)",
    )
    ax_target.plot(
        x_pred_no_dyn,
        pred_no_dyn,
        marker="o",
        linestyle="-.",
        linewidth=2,
        label="Forecast (without dynamic covariates)",
    )
    ax_target.axvline(forecast_start - 0.5, color="gray", linestyle=":", linewidth=1)
    ax_target.set_title(
        f"{series_names[i] if i < len(series_names) else f'Series {i + 1}'}: Forecast Comparison"
    )
    ax_target.set_xlabel("Time step")
    ax_target.set_ylabel("Target")
    ax_target.grid(alpha=0.3)
    ax_target.legend(loc="best")

    cov_total_len = None
    for cov_values in all_dynamic_covariates.values():
        if i < len(cov_values):
            cov_total_len = len(cov_values[i])
            break
    if cov_total_len is None:
        cov_total_len = forecast_start + len(pred_with_dyn)

    x_cov = np.arange(cov_total_len)
    for cov_name, cov_values in dynamic_numerical_covariates.items():
        ax_cov.plot(x_cov, cov_values[i], linewidth=2, label=cov_name)
    for cov_name, cov_values in dynamic_categorical_covariates.items():
        ax_cov.step(x_cov, cov_values[i], where="mid", linewidth=1.5, label=cov_name)

    ax_cov.axvline(forecast_start - 0.5, color="gray", linestyle=":", linewidth=1)
    ax_cov.set_title(
        f"{series_names[i] if i < len(series_names) else f'Series {i + 1}'}: Dynamic Covariates"
    )
    ax_cov.set_xlabel("Time step")
    ax_cov.set_ylabel("Covariate value")
    ax_cov.grid(alpha=0.3)
    ax_cov.legend(loc="best")

    common_horizon = len(pred_with_dyn)
    delta = pred_with_dyn - pred_no_dyn
    print(
        f"{series_names[i] if i < len(series_names) else f'Series {i + 1}'} "
        f"delta on first {common_horizon} steps "
        f"(with_dynamic - without_dynamic): {np.round(delta, 4).tolist()}"
    )

plt.tight_layout()
plt.show()
