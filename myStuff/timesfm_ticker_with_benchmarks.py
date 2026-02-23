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
    context_len: int = 600
    horizon_len: int = 7


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.bfill().fillna(50.0)


def prepare_normalized_data(config_path: Path) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    config = load_config(config_path)
    symbols = build_symbol_map(config.ticker)
    close_prices = fetch_daily_close(symbols, config.lookback_days)
    normalized = normalize_from_start(close_prices).dropna()
    raw_aligned = close_prices.loc[normalized.index].copy()
    return config.ticker, normalized, raw_aligned


def build_forecast_inputs(
    normalized: pd.DataFrame,
    raw_prices: pd.DataFrame,
    target_ticker: str,
    window: ForecastWindow,
) -> tuple[
    list[float],
    np.ndarray,
    dict[str, list[list[float]]],
    dict[str, list[list[int]]],
    np.ndarray,
    np.ndarray,
    float,
]:
    required = window.context_len + window.horizon_len
    if len(normalized) < required:
        raise ValueError(
            f"Need at least {required} rows of normalized data, got {len(normalized)}."
        )

    frame = normalized.iloc[-required:].copy()
    raw_frame = raw_prices.iloc[-required:].copy()
    base_row = frame.iloc[0].replace(0.0, np.nan)
    frame = frame.divide(base_row).ffill().bfill()

    target_series = frame[target_ticker].to_numpy(dtype=float)
    input_target = target_series[: window.context_len].tolist()
    actual_future = target_series[window.context_len :]
    raw_target_series = raw_frame[target_ticker].to_numpy(dtype=float)
    raw_context = raw_target_series[: window.context_len]
    raw_actual_future = raw_target_series[window.context_len :]
    raw_base = float(raw_context[0])
    full_target_rsi = compute_rsi(frame[target_ticker], period=14)
    rsi_frame = full_target_rsi.to_numpy(dtype=float)

    dynamic_numerical_covariates: dict[str, list[list[float]]] = {}
    for name in BENCHMARK_TICKERS:
        benchmark_series = frame[name].to_numpy(dtype=float)
        benchmark_context = benchmark_series[: window.context_len]
        last_known = benchmark_context[-1]
        future_proxy = np.full(window.horizon_len, last_known, dtype=float)
        strict_realtime_covariate = np.concatenate([benchmark_context, future_proxy])
        dynamic_numerical_covariates[name] = [strict_realtime_covariate.tolist()]

        benchmark_rsi = compute_rsi(frame[name], period=14).to_numpy(dtype=float)
        benchmark_rsi_context = benchmark_rsi[: window.context_len]
        benchmark_rsi_last_known = benchmark_rsi_context[-1]
        benchmark_rsi_future_proxy = np.full(
            window.horizon_len, benchmark_rsi_last_known, dtype=float
        )
        strict_realtime_benchmark_rsi = np.concatenate(
            [benchmark_rsi_context, benchmark_rsi_future_proxy]
        )
        dynamic_numerical_covariates[f"{name}_RSI_14"] = [
            strict_realtime_benchmark_rsi.tolist()
        ]

    rsi_context = rsi_frame[: window.context_len]
    rsi_last_known = rsi_context[-1]
    rsi_future_proxy = np.full(window.horizon_len, rsi_last_known, dtype=float)
    strict_realtime_rsi = np.concatenate([rsi_context, rsi_future_proxy])
    dynamic_numerical_covariates["Target_RSI_14"] = [strict_realtime_rsi.tolist()]

    day_of_week = frame.index.dayofweek.to_numpy(dtype=int)
    dynamic_categorical_covariates: dict[str, list[list[int]]] = {
        "DayOfWeek": [day_of_week.tolist()]
    }

    return (
        input_target,
        actual_future,
        dynamic_numerical_covariates,
        dynamic_categorical_covariates,
        raw_context,
        raw_actual_future,
        raw_base,
    )


def forecast_with_timesfm(
    input_target: list[float],
    dynamic_numerical_covariates: dict[str, list[list[float]]],
    dynamic_categorical_covariates: dict[str, list[list[int]]],
    window: ForecastWindow,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

    point_forecast_with_covars, quantile_forecast_with_covars = model.forecast_with_covariates(
        inputs=[input_target],
        dynamic_numerical_covariates=dynamic_numerical_covariates,
        dynamic_categorical_covariates=dynamic_categorical_covariates,
        xreg_mode="xreg + timesfm",
        ridge=0.0,
        normalize_xreg_target_per_input=True,
    )

    point_forecast_no_covars, quantile_forecast_no_covars = model.forecast(
        inputs=[input_target],
        horizon=window.horizon_len,
    )

    point_with_covars = np.asarray(point_forecast_with_covars[0], dtype=float)[
        -window.horizon_len :
    ]
    point_no_covars = np.asarray(point_forecast_no_covars[0], dtype=float)[
        -window.horizon_len :
    ]
    quantiles_with_covars = np.asarray(quantile_forecast_with_covars[0], dtype=float)[
        -window.horizon_len :
    ]
    quantiles_no_covars = np.asarray(quantile_forecast_no_covars[0], dtype=float)[
        -window.horizon_len :
    ]

    return (
        point_with_covars,
        point_no_covars,
        quantiles_with_covars[:, 1],
        quantiles_with_covars[:, 9],
        quantiles_no_covars[:, 1],
        quantiles_no_covars[:, 9],
    )


def plot_context_and_prediction(
    target_ticker: str,
    context_values: np.ndarray,
    predicted_with_covars: np.ndarray,
    predicted_no_covars: np.ndarray,
    p10_with_covars: np.ndarray,
    p90_with_covars: np.ndarray,
    p10_no_covars: np.ndarray,
    p90_no_covars: np.ndarray,
    actual_future: np.ndarray,
) -> None:
    context_len = len(context_values)
    plot_context_len = min(30, context_len)
    context_tail = np.asarray(context_values[-plot_context_len:], dtype=float)
    x_context = np.arange(plot_context_len)
    x_future_covars = np.arange(
        plot_context_len, plot_context_len + len(predicted_with_covars)
    )
    x_future_no_covars = np.arange(
        plot_context_len, plot_context_len + len(predicted_no_covars)
    )

    plt.figure(figsize=(12, 5))
    plt.plot(x_context, context_tail, marker="o", linewidth=2, label="Context (last 30)")
    plt.plot(
        x_future_covars,
        predicted_with_covars,
        marker="o",
        linestyle="--",
        linewidth=2,
        label="TimesFM prediction (with covariates)",
    )
    plt.fill_between(
        x_future_covars,
        p10_with_covars,
        p90_with_covars,
        alpha=0.2,
        color="tab:blue",
        label="With covariates p10-p90",
    )
    plt.plot(
        x_future_no_covars,
        predicted_no_covars,
        marker="o",
        linestyle="-.",
        linewidth=2,
        label="TimesFM prediction (without covariates)",
    )
    plt.fill_between(
        x_future_no_covars,
        p10_no_covars,
        p90_no_covars,
        alpha=0.15,
        color="tab:orange",
        label="Without covariates p10-p90",
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

    plt.axvline(plot_context_len - 0.5, color="gray", linestyle=":", linewidth=1)
    plt.title(
        f"{target_ticker}: Last {plot_context_len} Context Points (of {context_len}) + {len(predicted_with_covars)}-day Forecast"
    )
    plt.xlabel("Time step")
    plt.ylabel("Price")
    plt.grid(alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()


def run(config_path: Path) -> np.ndarray:
    window = ForecastWindow()
    target_ticker, normalized, raw_prices = prepare_normalized_data(config_path)
    (
        input_target,
        actual_future,
        dynamic_covariates,
        dynamic_categorical_covariates,
        raw_context,
        raw_actual_future,
        raw_base,
    ) = build_forecast_inputs(
        normalized=normalized,
        raw_prices=raw_prices,
        target_ticker=target_ticker,
        window=window,
    )
    (
        predicted_with_covars,
        predicted_no_covars,
        p10_with_covars,
        p90_with_covars,
        p10_no_covars,
        p90_no_covars,
    ) = forecast_with_timesfm(
        input_target=input_target,
        dynamic_numerical_covariates=dynamic_covariates,
        dynamic_categorical_covariates=dynamic_categorical_covariates,
        window=window,
    )
    predicted_with_covars_raw = predicted_with_covars * raw_base
    predicted_no_covars_raw = predicted_no_covars * raw_base
    p10_with_covars_raw = p10_with_covars * raw_base
    p90_with_covars_raw = p90_with_covars * raw_base
    p10_no_covars_raw = p10_no_covars * raw_base
    p90_no_covars_raw = p90_no_covars * raw_base
    plot_context_and_prediction(
        target_ticker=target_ticker,
        context_values=raw_context,
        predicted_with_covars=predicted_with_covars_raw,
        predicted_no_covars=predicted_no_covars_raw,
        p10_with_covars=p10_with_covars_raw,
        p90_with_covars=p90_with_covars_raw,
        p10_no_covars=p10_no_covars_raw,
        p90_no_covars=p90_no_covars_raw,
        actual_future=raw_actual_future,
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
