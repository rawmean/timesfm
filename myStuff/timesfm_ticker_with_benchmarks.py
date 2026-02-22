from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timesfm

from equity_benchmark_compare import (
    BENCHMARK_TICKERS,
    build_symbol_map,
    fetch_daily_close,
    load_config,
    normalize_from_start,
)


@dataclass(frozen=True)
class ForecastWindow:
    context_len: int = 30
    horizon_len: int = 7


def prepare_normalized_data(config_path: Path) -> tuple[str, pd.DataFrame]:
    config = load_config(config_path)
    symbols = build_symbol_map(config.ticker)
    close_prices = fetch_daily_close(symbols, config.lookback_days)
    normalized = normalize_from_start(close_prices).dropna()
    return config.ticker, normalized


def build_forecast_inputs(
    normalized: pd.DataFrame,
    target_ticker: str,
    window: ForecastWindow,
) -> tuple[list[float], np.ndarray, dict[str, list[list[float]]]]:
    required = window.context_len + window.horizon_len
    if len(normalized) < required:
        raise ValueError(
            f"Need at least {required} rows of normalized data, got {len(normalized)}."
        )

    frame = normalized.iloc[-required:].copy()
    target_series = frame[target_ticker].to_numpy(dtype=float)
    input_target = target_series[: window.context_len].tolist()
    actual_future = target_series[window.context_len :]

    dynamic_numerical_covariates: dict[str, list[list[float]]] = {}
    for name in BENCHMARK_TICKERS:
        dynamic_numerical_covariates[name] = [frame[name].to_numpy(dtype=float).tolist()]

    return input_target, actual_future, dynamic_numerical_covariates


def forecast_with_timesfm(
    input_target: list[float],
    dynamic_numerical_covariates: dict[str, list[list[float]]],
    window: ForecastWindow,
) -> tuple[np.ndarray, np.ndarray]:
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch"
    )
    model.compile(
        timesfm.ForecastConfig(
            max_context=window.context_len,
            max_horizon=window.horizon_len,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
            return_backcast=True,
        )
    )

    point_forecast_with_covars, _ = model.forecast_with_covariates(
        inputs=[input_target],
        dynamic_numerical_covariates=dynamic_numerical_covariates,
        xreg_mode="xreg + timesfm",
        ridge=0.0,
        normalize_xreg_target_per_input=True,
    )

    point_forecast_no_covars, _ = model.forecast(
        inputs=[input_target],
        horizon=window.horizon_len,
    )

    return (
        np.asarray(point_forecast_with_covars[0], dtype=float),
        np.asarray(point_forecast_no_covars[0], dtype=float),
    )


def plot_context_and_prediction(
    target_ticker: str,
    input_target: list[float],
    predicted_with_covars: np.ndarray,
    predicted_no_covars: np.ndarray,
    actual_future: np.ndarray,
) -> None:
    context_len = len(input_target)
    x_context = np.arange(context_len)
    x_future_covars = np.arange(
        context_len, context_len + len(predicted_with_covars)
    )
    x_future_no_covars = np.arange(
        context_len, context_len + len(predicted_no_covars)
    )

    plt.figure(figsize=(12, 5))
    plt.plot(x_context, input_target, marker="o", linewidth=2, label="Context (normalized)")
    plt.plot(
        x_future_covars,
        predicted_with_covars,
        marker="o",
        linestyle="--",
        linewidth=2,
        label="TimesFM prediction (with covariates)",
    )
    plt.plot(
        x_future_no_covars,
        predicted_no_covars,
        marker="o",
        linestyle="-.",
        linewidth=2,
        label="TimesFM prediction (without covariates)",
    )

    if len(actual_future) == len(predicted_with_covars):
        plt.plot(
            x_future_covars,
            actual_future,
            marker="o",
            linestyle=":",
            linewidth=2,
            label="Actual future",
        )

    plt.axvline(context_len - 0.5, color="gray", linestyle=":", linewidth=1)
    plt.axhline(1.0, color="black", linestyle=":", linewidth=1)
    plt.title(f"{target_ticker}: 30-day Context + 7-day Forecast (normalized)")
    plt.xlabel("Time step")
    plt.ylabel("Normalized value (start = 1.0)")
    plt.grid(alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()


def run(config_path: Path) -> np.ndarray:
    window = ForecastWindow(context_len=30, horizon_len=7)
    target_ticker, normalized = prepare_normalized_data(config_path)
    input_target, actual_future, dynamic_covariates = build_forecast_inputs(
        normalized=normalized,
        target_ticker=target_ticker,
        window=window,
    )
    predicted_with_covars, predicted_no_covars = forecast_with_timesfm(
        input_target=input_target,
        dynamic_numerical_covariates=dynamic_covariates,
        window=window,
    )
    plot_context_and_prediction(
        target_ticker=target_ticker,
        input_target=input_target,
        predicted_with_covars=predicted_with_covars,
        predicted_no_covars=predicted_no_covars,
        actual_future=actual_future,
    )
    return predicted_with_covars


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Forecast a ticker with TimesFM using benchmark indices as covariates."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("myStuff/equity_compare_config.toml"),
        help="Path to TOML config file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.config)
