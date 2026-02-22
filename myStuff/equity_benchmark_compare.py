from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
import tomllib

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

BENCHMARK_TICKERS = {
    "Dow Jones": "^DJI",
    "NASDAQ": "^IXIC",
    "S&P 500": "^GSPC",
    "Russell 2000": "^RUT",
}


@dataclass(frozen=True)
class AppConfig:
    ticker: str = "VTI"
    lookback_days: int = 365


def load_config(config_path: Path) -> AppConfig:
    with config_path.open("rb") as f:
        raw = tomllib.load(f)

    ticker = str(raw.get("ticker", "VTI")).strip().upper()
    lookback_days = int(raw.get("lookback_days", 365))
    if lookback_days <= 0:
        raise ValueError("lookback_days must be a positive integer")

    return AppConfig(ticker=ticker, lookback_days=lookback_days)


def build_symbol_map(primary_ticker: str) -> dict[str, str]:
    symbols = {primary_ticker: primary_ticker}
    symbols.update(BENCHMARK_TICKERS)
    return symbols


def fetch_daily_close(symbols: dict[str, str], lookback_days: int) -> pd.DataFrame:
    end_date = date.today() + timedelta(days=1)
    start_date = date.today() - timedelta(days=lookback_days)
    ticker_list = list(symbols.values())

    data = yf.download(
        tickers=ticker_list,
        start=start_date.isoformat(),
        end=end_date.isoformat(),
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=True,
    )

    if data.empty:
        raise RuntimeError("No data returned from yfinance. Check ticker symbols or network.")

    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"].copy()
    else:
        close = data[["Close"]].copy()
        close.columns = [ticker_list[0]]

    close = close.dropna(how="all")
    close = close[ticker_list]

    name_by_symbol = {symbol: name for name, symbol in symbols.items()}
    close = close.rename(columns=name_by_symbol)
    return close


def normalize_from_start(close_prices: pd.DataFrame) -> pd.DataFrame:
    baseline = close_prices.iloc[0]
    return close_prices.divide(baseline, axis=1)


def plot_normalized(normalized: pd.DataFrame, config: AppConfig) -> None:
    ax = normalized.plot(figsize=(12, 6), linewidth=2)
    ax.axhline(1.0, color="black", linewidth=1, linestyle=":")
    ax.set_title(
        f"Normalized Daily Close Performance (start at 1.0)\n"
        f"Ticker: {config.ticker}, Lookback: {config.lookback_days} days"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Price Ratio vs Start")
    ax.grid(alpha=0.3)
    ax.legend(title="Series")
    plt.tight_layout()
    plt.show()


def run(config_path: Path) -> pd.DataFrame:
    config = load_config(config_path)
    symbols = build_symbol_map(config.ticker)
    close_prices = fetch_daily_close(symbols, config.lookback_days)
    normalized = normalize_from_start(close_prices)
    plot_normalized(normalized, config)
    return normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a ticker to major US indices using normalized daily close data."
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
