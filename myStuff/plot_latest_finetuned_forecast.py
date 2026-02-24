from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timesfm
import torch
import yfinance as yf


VALUE_METRIC_POSITIVE_RETURN = 0.05
VALUE_METRIC_SIGMOID_TARGET = 0.8


@dataclass(frozen=True)
class PlotConfig:
    checkpoint: Path | None = None
    checkpoint_glob: str = "myStuff/*.pt"
    ticker: str | None = None
    years_of_history: int = 5
    price_column: str = "Close"
    moving_average_window: int = 5
    context_length: int | None = None
    horizon: int | None = None
    model_id: str = "google/timesfm-2.5-200m-pytorch"
    device: str = "auto"


def parse_args() -> PlotConfig:
    parser = argparse.ArgumentParser(
        description="Plot latest context window and forecast using newest fine-tuned checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional explicit checkpoint path.")
    parser.add_argument(
        "--checkpoint-glob",
        type=str,
        default=PlotConfig.checkpoint_glob,
        help="Glob used to find the latest checkpoint when --checkpoint is not set.",
    )
    parser.add_argument("--ticker", type=str, default=None, help="Override ticker from checkpoint.")
    parser.add_argument("--years", type=int, default=PlotConfig.years_of_history)
    parser.add_argument("--moving-average-window", type=int, default=PlotConfig.moving_average_window)
    parser.add_argument("--context", type=int, default=None, help="Override context length from checkpoint.")
    parser.add_argument("--horizon", type=int, default=None, help="Override horizon from checkpoint.")
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "mps", "cpu"],
        default=PlotConfig.device,
        help="Inference device.",
    )
    args = parser.parse_args()

    return PlotConfig(
        checkpoint=args.checkpoint,
        checkpoint_glob=args.checkpoint_glob,
        ticker=args.ticker.upper() if args.ticker else None,
        years_of_history=args.years,
        moving_average_window=args.moving_average_window,
        context_length=args.context,
        horizon=args.horizon,
        device=args.device,
    )


def find_latest_checkpoint(config: PlotConfig) -> Path:
    if config.checkpoint is not None:
        if not config.checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {config.checkpoint}")
        return config.checkpoint

    candidates = sorted(Path().glob(config.checkpoint_glob), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoints found with pattern '{config.checkpoint_glob}'."
        )
    return candidates[-1]


def resolve_device(device_flag: str) -> torch.device:
    if device_flag == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        raise ValueError("Requested device 'mps' but Apple MPS is not available.")
    if device_flag == "cpu":
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def extract_torch_model(model):
    if hasattr(model, "model"):
        return model.model
    if hasattr(model, "_model"):
        return model._model
    return model


def configure_timesfm_device(model, device: torch.device) -> None:
    internal = extract_torch_model(model)
    if hasattr(internal, "device"):
        internal.device = device
    if hasattr(internal, "device_count"):
        internal.device_count = 1
    if hasattr(internal, "to"):
        internal.to(device)


def resolve_runtime_settings(config: PlotConfig, checkpoint_data: dict) -> tuple[str, int, int]:
    ck_cfg = checkpoint_data.get("config", {}) if isinstance(checkpoint_data, dict) else {}
    ticker = config.ticker or ck_cfg.get("ticker") or "VTI"

    context_len = config.context_length
    if context_len is None:
        context_len = ck_cfg.get("context_length")
    if context_len is None:
        raise ValueError("Context length not found in checkpoint config; pass --context.")

    horizon = config.horizon
    if horizon is None:
        horizon = ck_cfg.get("horizon")
    if horizon is None:
        raise ValueError("Horizon not found in checkpoint config; pass --horizon.")

    return ticker, int(context_len), int(horizon)


def fetch_ma_series(
    ticker: str,
    years: int,
    price_column: str,
    moving_average_window: int,
) -> pd.Series:
    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=365 * years)

    frame = yf.download(
        ticker,
        start=start_date.isoformat(),
        end=end_date.isoformat(),
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    if frame.empty:
        raise ValueError(f"No data returned from yfinance for '{ticker}'.")

    if isinstance(frame.columns, pd.MultiIndex):
        level0 = frame.columns.get_level_values(0)
        if price_column not in set(level0):
            raise ValueError(f"Column '{price_column}' not found for {ticker}.")
        price_frame = frame.xs(price_column, axis=1, level=0)
        if isinstance(price_frame, pd.DataFrame):
            series = price_frame.iloc[:, 0].dropna().astype(float)
        else:
            series = price_frame.dropna().astype(float)
    else:
        if price_column not in frame.columns:
            raise ValueError(f"Column '{price_column}' not found for {ticker}.")
        series = frame[price_column].dropna().astype(float)

    if moving_average_window < 1:
        raise ValueError("moving_average_window must be >= 1")

    ma = series.rolling(window=moving_average_window, min_periods=moving_average_window).mean().dropna()
    ma.name = f"{ticker}_MA{moving_average_window}"
    return ma


def load_model_with_checkpoint(
    model_id: str,
    context_len: int,
    horizon: int,
    checkpoint_data: dict,
    device: torch.device,
):
    torch.set_float32_matmul_precision("high")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(model_id)
    model.compile(
        timesfm.ForecastConfig(
            max_context=context_len,
            max_horizon=horizon,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
        )
    )
    configure_timesfm_device(model, device)

    state = checkpoint_data.get("model_state_dict") if isinstance(checkpoint_data, dict) else None
    if state:
        internal = extract_torch_model(model)
        missing, unexpected = internal.load_state_dict(state, strict=False)
        if missing:
            print(f"Warning: missing keys while loading checkpoint: {len(missing)}")
        if unexpected:
            print(f"Warning: unexpected keys while loading checkpoint: {len(unexpected)}")
    else:
        print("Warning: checkpoint has no model_state_dict; using base pretrained weights.")

    return model


def forecast_latest_window(
    model,
    series: pd.Series,
    context_len: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, pd.Series]:
    if len(series) < context_len:
        raise ValueError(f"Need at least {context_len} points, got {len(series)}.")

    context_series = series.iloc[-context_len:]
    context_values = context_series.to_numpy(dtype=np.float32)

    point_forecast, quantile_forecast = model.forecast(
        horizon=horizon,
        inputs=[context_values],
    )

    pred = np.asarray(point_forecast[0], dtype=float)[-horizon:]
    quantiles = np.asarray(quantile_forecast[0], dtype=float)[-horizon:, :]

    future_dates = pd.bdate_range(
        start=pd.Timestamp(context_series.index[-1]) + pd.Timedelta(days=1),
        periods=horizon,
    )
    return pred, quantiles, future_dates, context_series


def compute_value_metric(current_value: float, last_predicted_value: float) -> tuple[float, float]:
    if current_value == 0.0:
        raise ValueError("Current value is zero; cannot compute return-based value metric.")

    implied_return = (last_predicted_value - current_value) / current_value
    # Calibrate sigmoid so that +5% -> 0.8 and -5% -> 0.2.
    sigmoid_scale = np.log(VALUE_METRIC_SIGMOID_TARGET / (1.0 - VALUE_METRIC_SIGMOID_TARGET))
    sigmoid_scale = float(sigmoid_scale / VALUE_METRIC_POSITIVE_RETURN)
    sigmoid_value = 1.0 / (1.0 + np.exp(-sigmoid_scale * implied_return))
    value_metric = 100.0 * float(sigmoid_value)
    return value_metric, float(implied_return)


def plot_context_and_forecast(
    ticker: str,
    ma_window: int,
    context_series: pd.Series,
    pred: np.ndarray,
    quantiles: np.ndarray,
    future_dates: pd.DatetimeIndex,
    value_metric: float,
    implied_return: float,
) -> None:
    plt.figure(figsize=(13, 6))
    plt.plot(
        context_series.index,
        context_series.values,
        label=f"Context (MA{ma_window})",
        color="black",
        linewidth=2,
        marker="o",
        markersize=3,
    )
    plt.plot(
        future_dates,
        pred,
        label="Forecast",
        color="tab:red",
        linewidth=2,
        marker="o",
    )
    plt.fill_between(
        future_dates,
        quantiles[:, 1],
        quantiles[:, 9],
        color="tab:red",
        alpha=0.2,
        label="p10-p90",
    )
    plt.axvline(context_series.index[-1], color="gray", linestyle=":", linewidth=1)
    plt.title(f"{ticker} MA{ma_window}: Last {len(context_series)} Context Days + {len(pred)}-Day Forecast")
    plt.xlabel("Date")
    plt.ylabel("Moving Average")
    plt.text(
        0.02,
        0.98,
        f"Value metric: {value_metric:.1f}/100\nLast-step return: {implied_return * 100:.2f}%",
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )
    plt.grid(alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()


def main() -> None:
    config = parse_args()
    checkpoint_path = find_latest_checkpoint(config)
    checkpoint_data = torch.load(checkpoint_path, map_location="cpu")

    ticker, context_len, horizon = resolve_runtime_settings(config, checkpoint_data)
    device = resolve_device(config.device)

    print(f"Using checkpoint: {checkpoint_path}")
    print(f"Ticker: {ticker} | Context: {context_len} | Horizon: {horizon} | MA: {config.moving_average_window}")
    print(f"Device: {device}")

    series = fetch_ma_series(
        ticker=ticker,
        years=config.years_of_history,
        price_column=config.price_column,
        moving_average_window=config.moving_average_window,
    )

    model = load_model_with_checkpoint(
        model_id=config.model_id,
        context_len=context_len,
        horizon=horizon,
        checkpoint_data=checkpoint_data,
        device=device,
    )

    pred, quantiles, future_dates, context_series = forecast_latest_window(
        model=model,
        series=series,
        context_len=context_len,
        horizon=horizon,
    )
    current_value = float(context_series.iloc[-1])
    last_predicted_value = float(pred[-1])
    value_metric, implied_return = compute_value_metric(
        current_value=current_value,
        last_predicted_value=last_predicted_value,
    )

    print("\nForecast values:")
    for i in range(horizon):
        print(
            f"{future_dates[i].strftime('%Y-%m-%d')}: "
            f"pred={pred[i]:.2f}, p10={quantiles[i, 1]:.2f}, p90={quantiles[i, 9]:.2f}"
        )
    print(
        f"\nValue metric: {value_metric:.2f}/100 "
        f"(current={current_value:.2f}, last_pred={last_predicted_value:.2f}, "
        f"last-step return={implied_return * 100:.2f}%)"
    )

    plot_context_and_forecast(
        ticker=ticker,
        ma_window=config.moving_average_window,
        context_series=context_series,
        pred=pred,
        quantiles=quantiles,
        future_dates=future_dates,
        value_metric=value_metric,
        implied_return=implied_return,
    )


if __name__ == "__main__":
    main()
