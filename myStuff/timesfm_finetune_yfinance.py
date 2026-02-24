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
import torch.nn as nn
import torch.optim as optim
import yfinance as yf
from torch.utils.data import DataLoader, Dataset

from timesfm.torch.util import revin, update_running_stats

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - fallback when tqdm is unavailable
    tqdm = None

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - fallback when tensorboard is unavailable
    SummaryWriter = None


BENCHMARK_TICKERS = {
    "Dow Jones": "^DJI",
    "NASDAQ": "^IXIC",
    "S&P 500": "^GSPC",
    "Russell 2000": "^RUT",
}
VIX_COVARIATE_NAME = "VIX"
VIX_SYMBOL = "^VIX"


@dataclass(frozen=True)
class FineTuneConfig:
    ticker: str = "VTI"
    years_of_history: int = 5
    price_column: str = "Close"
    moving_average_window: int = 7
    use_calendar_covariates: bool = False
    covariate_boundary_correction: bool = True
    context_length: int = 32
    horizon: int = 7
    test_size: int = 120
    val_fraction: float = 0.15
    batch_size: int = 32
    num_epochs: int = 15
    learning_rate: float = 1e-4
    finetune_mode: str = "peft"
    peft_last_n_layers: int = 2
    device: str = "auto"
    show_progress: bool = True
    tensorboard: bool = True
    log_dir: Path = Path("myStuff/runs")
    model_id: str = "google/timesfm-2.5-200m-pytorch"
    save_path: Path = Path("myStuff/finetuned_timesfm_yfinance.pt")


class TimeSeriesDataset(Dataset):
    def __init__(self, data: np.ndarray, context_length: int, horizon: int):
        self.data = np.array(data, dtype=np.float32)
        self.context_length = context_length
        self.horizon = horizon
        self.samples: list[tuple[np.ndarray, np.ndarray]] = []

        for i in range(len(self.data) - context_length - horizon + 1):
            context = self.data[i : i + context_length]
            target = self.data[i + context_length : i + context_length + horizon]
            self.samples.append((context, target))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        context, target = self.samples[idx]
        return torch.tensor(context), torch.tensor(target)


def _to_1d_float_array(values: np.ndarray | pd.Series | pd.DataFrame) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    return arr.reshape(-1)


def parse_args() -> FineTuneConfig:
    parser = argparse.ArgumentParser(
        description="Fine-tune TimesFM on yfinance ticker data (5-year window)."
    )
    parser.add_argument("--ticker", type=str, default=FineTuneConfig.ticker)
    parser.add_argument("--epochs", type=int, default=FineTuneConfig.num_epochs)
    parser.add_argument("--batch-size", type=int, default=FineTuneConfig.batch_size)
    parser.add_argument("--learning-rate", type=float, default=FineTuneConfig.learning_rate)
    parser.add_argument(
        "--moving-average-window",
        type=int,
        default=FineTuneConfig.moving_average_window,
        help="Window size for moving-average target (set 5 for MA5).",
    )
    parser.add_argument(
        "--use-calendar-covariates",
        action="store_true",
        help="Include DayOfWeek and MonthOfYear as dynamic categorical covariates.",
    )
    parser.add_argument(
        "--no-covariate-boundary-correction",
        action="store_true",
        help="Disable boundary offset correction against no-covariate baseline.",
    )
    parser.add_argument(
        "--finetune-mode",
        type=str,
        choices=["peft", "full"],
        default=FineTuneConfig.finetune_mode,
        help="Use 'peft' (parameter-efficient) or 'full' fine-tuning.",
    )
    parser.add_argument(
        "--peft-last-n-layers",
        type=int,
        default=FineTuneConfig.peft_last_n_layers,
        help="For PEFT mode: additionally unfreeze last N transformer layers.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=FineTuneConfig.val_fraction,
        help="Fraction of pre-test data reserved for validation (temporal split).",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=FineTuneConfig.test_size,
        help="Number of latest points held out for final testing (must be >= context_length + horizon).",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "mps", "cpu"],
        default=FineTuneConfig.device,
        help="Training device. Use 'mps' for Apple GPU, or 'auto' to pick mps when available.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable TensorBoard logging.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=FineTuneConfig.log_dir,
        help="TensorBoard log directory.",
    )
    parser.add_argument("--save-path", type=Path, default=FineTuneConfig.save_path)
    args = parser.parse_args()

    return FineTuneConfig(
        ticker=args.ticker.upper(),
        moving_average_window=args.moving_average_window,
        use_calendar_covariates=args.use_calendar_covariates,
        covariate_boundary_correction=not args.no_covariate_boundary_correction,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        finetune_mode=args.finetune_mode,
        peft_last_n_layers=args.peft_last_n_layers,
        test_size=args.test_size,
        val_fraction=args.val_fraction,
        device=args.device,
        show_progress=not args.no_progress,
        tensorboard=not args.no_tensorboard,
        log_dir=args.log_dir,
        save_path=args.save_path,
    )


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


def configure_timesfm_device(model, device: torch.device) -> None:
    internal_model = _extract_torch_model(model)
    if hasattr(internal_model, "device"):
        internal_model.device = device
    if hasattr(internal_model, "device_count"):
        internal_model.device_count = 1
    if hasattr(internal_model, "to"):
        internal_model.to(device)


def maybe_tqdm(iterable, enabled: bool, **kwargs):
    if enabled and tqdm is not None:
        return tqdm(iterable, **kwargs)
    return iterable


def maybe_create_writer(config: FineTuneConfig):
    if not config.tensorboard:
        return None
    if SummaryWriter is None:
        print("TensorBoard disabled: torch.utils.tensorboard is not available.")
        return None
    run_name_base = (
        f"{config.ticker}_ctx{config.context_length}_hor{config.horizon}_"
        f"ep{config.num_epochs}_bs{config.batch_size}_lr{config.learning_rate}"
    )
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{run_name_base}_{run_ts}"
    writer = SummaryWriter(log_dir=str(config.log_dir / run_name))
    writer.add_text(
        "run/config",
        (
            f"ticker={config.ticker}, years={config.years_of_history}, "
            f"context={config.context_length}, horizon={config.horizon}, "
            f"test_size={config.test_size}, val_fraction={config.val_fraction}, "
            f"moving_average_window={config.moving_average_window}, "
            f"use_calendar_covariates={config.use_calendar_covariates}, "
            f"covariate_boundary_correction={config.covariate_boundary_correction}, "
            f"finetune_mode={config.finetune_mode}, peft_last_n_layers={config.peft_last_n_layers}, "
            f"epochs={config.num_epochs}, batch_size={config.batch_size}, "
            f"learning_rate={config.learning_rate}, device={config.device}"
        ),
    )
    return writer


def gradient_diagnostics(module: torch.nn.Module) -> tuple[int, int, float, float]:
    total_params = 0
    params_with_grad = 0
    total_grad_norm = 0.0
    max_abs_grad = 0.0
    for param in module.parameters():
        if not param.requires_grad:
            continue
        total_params += 1
        if param.grad is None:
            continue
        grad = param.grad.detach()
        if grad.numel() == 0:
            continue
        params_with_grad += 1
        total_grad_norm += float(torch.norm(grad).item())
        max_abs_grad = max(max_abs_grad, float(torch.max(torch.abs(grad)).item()))
    return total_params, params_with_grad, total_grad_norm, max_abs_grad


def configure_trainable_parameters(
    pytorch_model: torch.nn.Module, config: FineTuneConfig
) -> tuple[int, int]:
    total_params = sum(p.numel() for p in pytorch_model.parameters())
    for param in pytorch_model.parameters():
        param.requires_grad = False

    if config.finetune_mode == "full":
        for param in pytorch_model.parameters():
            param.requires_grad = True
    else:
        # PEFT mode: train small output/tokenizer blocks and optionally last N transformer layers.
        if hasattr(pytorch_model, "tokenizer"):
            for p in pytorch_model.tokenizer.parameters():
                p.requires_grad = True
        if hasattr(pytorch_model, "output_projection_point"):
            for p in pytorch_model.output_projection_point.parameters():
                p.requires_grad = True
        if hasattr(pytorch_model, "output_projection_quantiles"):
            for p in pytorch_model.output_projection_quantiles.parameters():
                p.requires_grad = True
        if hasattr(pytorch_model, "stacked_xf") and config.peft_last_n_layers > 0:
            n = min(config.peft_last_n_layers, len(pytorch_model.stacked_xf))
            for layer in list(pytorch_model.stacked_xf)[-n:]:
                for p in layer.parameters():
                    p.requires_grad = True

    trainable_params = sum(p.numel() for p in pytorch_model.parameters() if p.requires_grad)
    return trainable_params, total_params


def differentiable_point_forecast(
    module: torch.nn.Module, context: torch.Tensor, horizon: int
) -> torch.Tensor:
    """Differentiable point forecast for short horizons using module forward pass.

    This bypasses the public `forecast()/decode()` path, which is inference-only.
    """
    if horizon > module.o:
        raise ValueError(
            f"Differentiable helper supports horizon <= output patch ({module.o}), got {horizon}."
        )
    if context.ndim != 2:
        raise ValueError(f"Expected context shape [B, T], got {tuple(context.shape)}")

    batch_size, context_len = context.shape
    patch_len = module.p
    padded_len = int(np.ceil(context_len / patch_len) * patch_len)
    pad_len = padded_len - context_len

    if pad_len > 0:
        pad_values = torch.zeros(batch_size, pad_len, device=context.device, dtype=context.dtype)
        values = torch.cat([pad_values, context], dim=1)
        pad_mask = torch.ones(batch_size, pad_len, device=context.device, dtype=torch.bool)
        value_mask = torch.zeros(batch_size, context_len, device=context.device, dtype=torch.bool)
        masks = torch.cat([pad_mask, value_mask], dim=1)
    else:
        values = context
        masks = torch.zeros_like(values, dtype=torch.bool)

    patched_inputs = values.reshape(batch_size, -1, patch_len)
    patched_masks = masks.reshape(batch_size, -1, patch_len)

    n = torch.zeros(batch_size, device=context.device)
    mu = torch.zeros(batch_size, device=context.device)
    sigma = torch.zeros(batch_size, device=context.device)
    patch_mu: list[torch.Tensor] = []
    patch_sigma: list[torch.Tensor] = []
    for i in range(patched_inputs.shape[1]):
        (n, mu, sigma), _ = update_running_stats(
            n, mu, sigma, patched_inputs[:, i], patched_masks[:, i]
        )
        patch_mu.append(mu)
        patch_sigma.append(sigma)

    context_mu = torch.stack(patch_mu, dim=1)
    context_sigma = torch.stack(patch_sigma, dim=1)

    normed_inputs = revin(patched_inputs, context_mu, context_sigma, reverse=False)
    normed_inputs = torch.where(patched_masks, 0.0, normed_inputs)

    (_, _, normed_outputs, _), _ = module(normed_inputs, patched_masks, decode_caches=None)
    renormed_outputs = torch.reshape(
        revin(normed_outputs, context_mu, context_sigma, reverse=True),
        (batch_size, -1, module.o, module.q),
    )

    return renormed_outputs[:, -1, :horizon, module.aridx]


def fetch_ticker_history(config: FineTuneConfig) -> pd.Series:
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=365 * config.years_of_history)

    print(f"Downloading {config.ticker} from {start} to {end}...")
    frame = yf.download(
        config.ticker,
        start=start.isoformat(),
        end=end.isoformat(),
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    if frame.empty:
        raise ValueError(f"No data returned from yfinance for ticker '{config.ticker}'.")

    if isinstance(frame.columns, pd.MultiIndex):
        level0 = frame.columns.get_level_values(0)
        if config.price_column not in set(level0):
            available = ", ".join(sorted(set(level0.astype(str).tolist())))
            raise ValueError(
                f"Column '{config.price_column}' not found for {config.ticker}. "
                f"Available columns: {available}"
            )
        price_frame = frame.xs(config.price_column, axis=1, level=0)
        if isinstance(price_frame, pd.DataFrame):
            series = price_frame.iloc[:, 0].dropna().astype(float)
        else:
            series = price_frame.dropna().astype(float)
    else:
        if config.price_column not in frame.columns:
            available = ", ".join(frame.columns.astype(str).tolist())
            raise ValueError(
                f"Column '{config.price_column}' not found for {config.ticker}. "
                f"Available columns: {available}"
            )
        series = frame[config.price_column].dropna().astype(float)

    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]

    series = series.astype(float).copy()
    ma_window = config.moving_average_window
    if ma_window < 1:
        raise ValueError(f"moving_average_window must be >= 1, got {ma_window}.")
    series = series.rolling(window=ma_window, min_periods=ma_window).mean().dropna()
    series.name = f"{config.ticker}_MA{ma_window}"
    if len(series) < config.context_length + config.horizon + 1:
        raise ValueError(
            f"Not enough data points ({len(series)}). Need at least "
            f"{config.context_length + config.horizon + 1}."
        )

    print(
        f"Downloaded and transformed to {config.price_column} MA{ma_window} "
        f"with {len(series)} daily points."
    )
    return series


def fetch_adjusted_close_for_symbols(
    symbols: dict[str, str], lookback_days: int
) -> pd.DataFrame:
    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=lookback_days)
    ticker_list = list(symbols.values())

    data = yf.download(
        tickers=ticker_list,
        start=start_date.isoformat(),
        end=end_date.isoformat(),
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    if data.empty:
        raise RuntimeError(
            "No adjusted data returned from yfinance for covariates. Check ticker symbols or network."
        )

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


def load_model(config: FineTuneConfig):
    print("Loading TimesFM model...")
    torch.set_float32_matmul_precision("high")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(config.model_id)
    model.compile(
        timesfm.ForecastConfig(
            max_context=config.context_length,
            max_horizon=config.horizon,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
            return_backcast=True,
        )
    )
    device = resolve_device(config.device)
    configure_timesfm_device(model, device)
    print(f"TimesFM ready on device: {device}.\n")
    return model, device


def prepare_loader(data: np.ndarray, config: FineTuneConfig, shuffle: bool) -> DataLoader:
    dataset = TimeSeriesDataset(
        data=data,
        context_length=config.context_length,
        horizon=config.horizon,
    )
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle, num_workers=0)


def _extract_torch_model(model):
    if hasattr(model, "model"):
        return model.model
    if hasattr(model, "_model"):
        return model._model
    return model


def evaluate_loader_loss(
    pytorch_model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    horizon: int,
) -> float:
    losses: list[float] = []
    pytorch_model.eval()
    with torch.no_grad():
        for context, target in loader:
            context = context.to(device=device, dtype=torch.float32)
            target = target.to(device=device, dtype=torch.float32)
            pred_tensor = differentiable_point_forecast(
                pytorch_model, context=context, horizon=horizon
            )
            loss = criterion(pred_tensor, target)
            losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("nan")


def finetune_model(
    model,
    train_data: np.ndarray,
    val_data: np.ndarray,
    config: FineTuneConfig,
    device: torch.device,
):
    print("=" * 70)
    print("STARTING FINE-TUNING")
    print("=" * 70)
    print(f"Ticker: {config.ticker}")
    print(f"Training points: {len(train_data)}")
    print(f"Validation points: {len(val_data)}")
    print(f"Context length: {config.context_length}")
    print(f"Horizon: {config.horizon}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Finetune mode: {config.finetune_mode}")
    if config.finetune_mode == "peft":
        print(f"PEFT last transformer layers: {config.peft_last_n_layers}")
    print(f"Epochs: {config.num_epochs}\n")
    print(f"Device: {device}\n")

    train_loader = prepare_loader(train_data, config, shuffle=True)
    if len(train_loader) == 0:
        raise ValueError(
            f"Not enough data for training. Need at least {config.context_length + config.horizon} points."
        )
    val_loader = prepare_loader(val_data, config, shuffle=False)
    if len(val_loader) == 0:
        raise ValueError(
            f"Not enough data for validation. Need at least {config.context_length + config.horizon} points."
        )

    pytorch_model = _extract_torch_model(model)
    pytorch_model.train()
    trainable_numel, total_numel = configure_trainable_parameters(pytorch_model, config)
    trainable_params = [p for p in pytorch_model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters selected by fine-tuning configuration.")
    print(
        f"Trainable params: {trainable_numel:,} / {total_numel:,} "
        f"({100.0 * trainable_numel / max(1, total_numel):.2f}%)"
    )

    optimizer = optim.Adam(trainable_params, lr=config.learning_rate)
    criterion = nn.MSELoss()
    training_losses: list[float] = []
    validation_losses: list[float] = []
    writer = maybe_create_writer(config)
    global_step = 0

    epoch_iter = maybe_tqdm(
        range(config.num_epochs),
        enabled=config.show_progress,
        desc="Epochs",
        leave=True,
    )
    for epoch in epoch_iter:
        epoch_losses = []
        epoch_grad_params = 0
        epoch_total_params = 0
        epoch_grad_norms: list[float] = []
        epoch_grad_max: list[float] = []
        batch_iter = maybe_tqdm(
            train_loader,
            enabled=config.show_progress,
            desc=f"Epoch {epoch + 1}/{config.num_epochs}",
            leave=False,
            total=len(train_loader),
        )
        for batch_idx, (context, target) in enumerate(batch_iter):
            try:
                context = context.to(device=device, dtype=torch.float32)
                target = target.to(device=device, dtype=torch.float32)
                pred_tensor = differentiable_point_forecast(
                    pytorch_model, context=context, horizon=config.horizon
                )

                loss = criterion(pred_tensor, target)
                optimizer.zero_grad()
                loss.backward()
                total_params, params_with_grad, grad_norm, grad_max_abs = gradient_diagnostics(
                    pytorch_model
                )
                epoch_total_params = max(epoch_total_params, total_params)
                epoch_grad_params = max(epoch_grad_params, params_with_grad)
                epoch_grad_norms.append(grad_norm)
                epoch_grad_max.append(grad_max_abs)
                if writer is not None:
                    writer.add_scalar("debug/params_with_grad", params_with_grad, global_step)
                    writer.add_scalar("debug/grad_norm_sum", grad_norm, global_step)
                    writer.add_scalar("debug/grad_max_abs", grad_max_abs, global_step)
                optimizer.step()
                epoch_losses.append(float(loss.item()))
                if writer is not None:
                    writer.add_scalar("train/batch_loss", float(loss.item()), global_step)
                    global_step += 1
                if config.show_progress and tqdm is not None:
                    batch_iter.set_postfix(loss=f"{loss.item():.5f}")
            except Exception as exc:
                print(f"Training stopped at epoch {epoch + 1}, batch {batch_idx}: {exc}")
                print("Using pretrained model for inference only.")
                pytorch_model.eval()
                if writer is not None:
                    writer.close()
                return model, [], []

        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        val_loss = evaluate_loader_loss(
            pytorch_model, val_loader, criterion, device, config.horizon
        )
        avg_grad_norm = float(np.mean(epoch_grad_norms)) if epoch_grad_norms else 0.0
        max_grad_abs = float(np.max(epoch_grad_max)) if epoch_grad_max else 0.0
        training_losses.append(avg_loss)
        validation_losses.append(val_loss)
        if writer is not None:
            writer.add_scalar("train/epoch_loss", avg_loss, epoch + 1)
            writer.add_scalar("val/epoch_loss", val_loss, epoch + 1)
            writer.add_scalar("train/learning_rate", optimizer.param_groups[0]["lr"], epoch + 1)
            writer.add_scalar("debug/epoch_avg_grad_norm_sum", avg_grad_norm, epoch + 1)
            writer.add_scalar("debug/epoch_max_abs_grad", max_grad_abs, epoch + 1)
        print(
            f"Epoch {epoch + 1}/{config.num_epochs}, "
            f"train loss: {avg_loss:.6f}, val loss: {val_loss:.6f}"
        )
        print(
            f"Gradient diagnostic: params_with_grad={epoch_grad_params}/{epoch_total_params}, "
            f"avg_grad_norm_sum={avg_grad_norm:.6e}, max_abs_grad={max_grad_abs:.6e}"
        )

    if writer is not None:
        writer.flush()
        writer.close()
        print(f"TensorBoard logs written to: {config.log_dir}")

    pytorch_model.eval()
    _save_checkpoint(model, optimizer, training_losses, validation_losses, config)
    return model, training_losses, validation_losses


def load_checkpoint_if_available(model, checkpoint_path: Path) -> bool:
    if not checkpoint_path.exists():
        return False
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state = checkpoint.get("model_state_dict", {})
        if not state:
            print(f"Checkpoint found at {checkpoint_path}, but model_state_dict is empty.")
            return False
        internal_model = _extract_torch_model(model)
        missing, unexpected = internal_model.load_state_dict(state, strict=False)
        if missing:
            print(f"Warning: {len(missing)} missing keys while loading checkpoint.")
        if unexpected:
            print(f"Warning: {len(unexpected)} unexpected keys while loading checkpoint.")
        print(f"Loaded fine-tuned checkpoint: {checkpoint_path}")
        return True
    except Exception as exc:
        print(f"Failed to load checkpoint {checkpoint_path}: {exc}")
        return False


def prompt_yes_no(question: str, default_no: bool = True) -> bool:
    suffix = " [y/N]: " if default_no else " [Y/n]: "
    try:
        answer = input(question + suffix).strip().lower()
    except EOFError:
        return not default_no
    if not answer:
        return not default_no
    return answer in {"y", "yes"}


def _save_checkpoint(
    model,
    optimizer,
    training_losses: list[float],
    validation_losses: list[float],
    config: FineTuneConfig,
) -> None:
    config.save_path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "model"):
        model_state = model.model.state_dict()
    elif hasattr(model, "_model"):
        model_state = model._model.state_dict()
    else:
        model_state = {}

    torch.save(
        {
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "training_losses": training_losses,
            "validation_losses": validation_losses,
            "config": {
                "ticker": config.ticker,
                "context_length": config.context_length,
                "horizon": config.horizon,
                "num_epochs": config.num_epochs,
                "learning_rate": config.learning_rate,
                "val_fraction": config.val_fraction,
            },
        },
        config.save_path,
    )
    print(f"Saved checkpoint to {config.save_path}\n")


def forecast_last_window(
    model, context_series: pd.Series, future_index: pd.Index, config: FineTuneConfig
) -> tuple[np.ndarray, np.ndarray, list[pd.Timestamp]]:
    if len(context_series) < config.context_length:
        raise ValueError(
            f"Need at least {config.context_length} context points, got {len(context_series)}."
        )
    context_values = _to_1d_float_array(context_series.iloc[-config.context_length :])
    point_forecast, quantile_forecast = model.forecast(
        horizon=config.horizon,
        inputs=[context_values.astype(np.float32)],
    )

    pred = np.asarray(point_forecast[0], dtype=float)[-config.horizon :]
    quantiles = np.asarray(quantile_forecast[0], dtype=float)[-config.horizon :, :]
    future_dates = pd.DatetimeIndex(future_index).to_list()
    return pred, quantiles, future_dates


def forecast_with_benchmark_covariates(
    model,
    config: FineTuneConfig,
) -> tuple[np.ndarray, np.ndarray, list[pd.Timestamp], np.ndarray]:
    required = config.context_length + config.horizon
    lookback_days = max(
        config.years_of_history * 365, config.test_size + required + config.context_length
    )

    symbols = {config.ticker: config.ticker}
    symbols.update(BENCHMARK_TICKERS)
    symbols[VIX_COVARIATE_NAME] = VIX_SYMBOL
    close_prices = fetch_adjusted_close_for_symbols(symbols, lookback_days=lookback_days)
    target_ma = (
        close_prices[config.ticker]
        .astype(float)
        .rolling(
            window=config.moving_average_window,
            min_periods=config.moving_average_window,
        )
        .mean()
        .dropna()
    )
    if len(target_ma) < required:
        raise ValueError(
            f"Need at least {required} MA points for covariate forecast, got {len(target_ma)}."
        )
    covariate_columns = list(BENCHMARK_TICKERS.keys()) + [VIX_COVARIATE_NAME]
    missing_covars = [c for c in covariate_columns if c not in close_prices.columns]
    if missing_covars:
        raise ValueError(
            f"Missing covariate columns in adjusted close frame: {missing_covars}. "
            f"Available: {close_prices.columns.tolist()}"
        )

    aligned = close_prices[[config.ticker] + covariate_columns].dropna(how="any").copy()
    target_ma = (
        aligned[config.ticker]
        .astype(float)
        .rolling(
            window=config.moving_average_window,
            min_periods=config.moving_average_window,
        )
        .mean()
        .dropna()
    )
    if len(target_ma) < required:
        raise ValueError(
            f"Need at least {required} aligned MA points, got {len(target_ma)}."
        )

    frame_index = target_ma.index[-required:]
    target_ma = target_ma.loc[frame_index]
    cov_raw = aligned.loc[frame_index, covariate_columns].astype(float).copy()
    xreg_mode = "timesfm + xreg"
    internal_model = _extract_torch_model(model)
    patch_len = int(getattr(internal_model, "p", 32))
    xreg_train_len = max(1, config.context_length - patch_len)
    train_start = config.context_length - xreg_train_len
    cov_frame = pd.DataFrame(index=cov_raw.index)
    for cov_name in covariate_columns:
        cov_arr = cov_raw[cov_name].to_numpy(dtype=float)
        train_slice = cov_arr[train_start : config.context_length]
        mu = float(np.mean(train_slice))
        sigma = float(np.std(train_slice))
        if sigma < 1e-8:
            sigma = 1.0
        cov_frame[cov_name] = (cov_arr - mu) / sigma
    if cov_frame.isna().any().any():
        raise ValueError("Covariate standardization produced NaNs; check zero/missing covariates.")

    target_series = target_ma.to_numpy(dtype=np.float32)
    input_target = target_series[: config.context_length]
    raw_target = target_ma.to_numpy(dtype=float)
    raw_context = raw_target[: config.context_length]
    raw_actual_future = raw_target[config.context_length :]

    day_of_week = frame_index.dayofweek.to_numpy(dtype=int)
    month_of_year = frame_index.month.to_numpy(dtype=int)
    dynamic_numerical_covariates: dict[str, list[list[float]]] = {
        cov_name: [cov_frame[cov_name].to_numpy(dtype=float).tolist()]
        for cov_name in covariate_columns
    }
    dynamic_categorical_covariates: dict[str, list[list[int]]] | None = None
    if config.use_calendar_covariates:
        dynamic_categorical_covariates = {
            "DayOfWeek": [day_of_week.tolist()],
            "MonthOfYear": [month_of_year.tolist()],
        }

    point_forecast_with_covars, _ = model.forecast_with_covariates(
        inputs=[input_target.astype(np.float32)],
        dynamic_numerical_covariates=dynamic_numerical_covariates,
        dynamic_categorical_covariates=dynamic_categorical_covariates,
        xreg_mode=xreg_mode,
        ridge=0.1,
        normalize_xreg_target_per_input=True,
    )

    pred_raw = np.asarray(point_forecast_with_covars[0], dtype=float)[-config.horizon :]
    point_forecast_no_covars, _ = model.forecast(
        horizon=config.horizon,
        inputs=[input_target.astype(np.float32)],
    )
    pred_no_covars = np.asarray(point_forecast_no_covars[0], dtype=float)[-config.horizon :]
    if config.covariate_boundary_correction:
        boundary_offset = pred_raw[0] - pred_no_covars[0]
        pred_raw = pred_raw - boundary_offset
    forecast_dates = frame_index[-config.horizon :].to_list()
    return pred_raw, raw_actual_future, forecast_dates, raw_context


def plot_results(
    series: pd.Series,
    pred_finetuned: np.ndarray,
    quantiles_finetuned: np.ndarray,
    pred_base: np.ndarray,
    quantiles_base: np.ndarray,
    future_dates: list[pd.Timestamp],
    config: FineTuneConfig,
) -> None:
    recent = series.iloc[-120:]
    plt.figure(figsize=(13, 6))
    plt.plot(
        recent.index,
        recent.values,
        label=f"Historical {config.price_column} MA{config.moving_average_window}",
        color="black",
        linewidth=2,
    )
    plt.plot(
        future_dates,
        pred_finetuned,
        label="Fine-tuned forecast",
        color="tab:red",
        linewidth=2,
        marker="o",
    )
    plt.fill_between(
        future_dates,
        quantiles_finetuned[:, 1],
        quantiles_finetuned[:, 9],
        color="tab:red",
        alpha=0.2,
        label="Fine-tuned p10-p90",
    )
    plt.plot(
        future_dates,
        pred_base,
        label="Base model forecast",
        color="tab:blue",
        linewidth=2,
        marker="s",
        linestyle="--",
    )
    plt.fill_between(
        future_dates,
        quantiles_base[:, 1],
        quantiles_base[:, 9],
        color="tab:blue",
        alpha=0.15,
        label="Base p10-p90",
    )
    plt.title(
        f"{config.ticker} TimesFM forecast after fine-tuning "
        f"({config.price_column} MA{config.moving_average_window})"
    )
    plt.xlabel("Date")
    plt.ylabel(config.price_column)
    plt.grid(alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()


def plot_covariate_results(
    context_values: np.ndarray,
    pred_covars: np.ndarray,
    actual_future: np.ndarray,
    forecast_dates: list[pd.Timestamp],
    config: FineTuneConfig,
) -> None:
    plot_context_len = min(config.context_length, len(context_values))
    context_tail = np.asarray(context_values[-plot_context_len:], dtype=float)
    x_context = np.arange(plot_context_len)
    x_forecast = np.arange(plot_context_len, plot_context_len + len(pred_covars))
    context_label_len = min(config.context_length, len(context_values))

    plt.figure(figsize=(12, 5))
    plt.plot(
        x_context,
        context_tail,
        marker="o",
        linewidth=2,
        label=f"Context (last {context_label_len})",
    )
    plt.plot(
        x_forecast,
        pred_covars,
        marker="o",
        linestyle="--",
        linewidth=2,
        label="Forecast with covariates",
    )
    if len(actual_future) == len(pred_covars):
        plt.plot(
            x_forecast,
            actual_future,
            marker="o",
            linestyle=":",
            linewidth=2,
            label="Actual future",
        )

    plt.axvline(plot_context_len - 0.5, color="gray", linestyle=":", linewidth=1)
    date_range = f"{forecast_dates[0].strftime('%Y-%m-%d')} to {forecast_dates[-1].strftime('%Y-%m-%d')}"
    plt.title(f"{config.ticker} covariate forecast ({date_range})")
    plt.xlabel("Time step")
    plt.ylabel(config.price_column)
    plt.grid(alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()


def main() -> None:
    config = parse_args()
    series = fetch_ticker_history(config)
    if not (0.0 < config.val_fraction < 1.0):
        raise ValueError(f"val_fraction must be in (0, 1), got {config.val_fraction}.")
    min_test_size = config.context_length + config.horizon
    if config.test_size < min_test_size:
        raise ValueError(
            f"test_size ({config.test_size}) must be >= context_length + horizon "
            f"({config.context_length} + {config.horizon} = {min_test_size})."
        )
    if len(series) < config.context_length + config.test_size + 1:
        raise ValueError(
            f"Not enough data for split. Need at least {config.context_length + config.test_size + 1}, got {len(series)}."
        )

    # Hold out an explicit test set, then carve out validation from pre-test data.
    trainval_series = series.iloc[: -config.test_size]
    test_series = series.iloc[-config.test_size :]
    val_size = int(np.ceil(len(trainval_series) * config.val_fraction))
    min_window = config.context_length + config.horizon
    if val_size < min_window:
        val_size = min_window
    if len(trainval_series) - val_size < min_window:
        raise ValueError(
            "Not enough pre-test data for both train and validation windows. "
            f"Need at least {2 * min_window} pre-test points, got {len(trainval_series)}."
        )
    train_series = trainval_series.iloc[: -val_size]
    val_series = trainval_series.iloc[-val_size:]
    holdout_series = series.iloc[-config.horizon :]
    eval_context = series.iloc[-(config.context_length + config.horizon) : -config.horizon]
    split_date = pd.Timestamp(trainval_series.index[-1])
    print(
        f"Train/val/test split at {split_date.strftime('%Y-%m-%d')} | "
        f"train={len(train_series)} points, val={len(val_series)} points, test={len(test_series)} points"
    )
    print(
        f"Final evaluation window (tail): "
        f"{holdout_series.index[0].strftime('%Y-%m-%d')} to "
        f"{holdout_series.index[-1].strftime('%Y-%m-%d')}"
    )

    model, device = load_model(config)
    losses: list[float] = []
    val_losses: list[float] = []
    if config.save_path.exists():
        retrain = prompt_yes_no(
            f"Found existing fine-tuned checkpoint at {config.save_path}. Fine-tune again?",
            default_no=True,
        )
        if retrain:
            model, losses, val_losses = finetune_model(
                model,
                train_series.to_numpy(dtype=np.float32),
                val_series.to_numpy(dtype=np.float32),
                config,
                device,
            )
        else:
            loaded = load_checkpoint_if_available(model, config.save_path)
            if not loaded:
                print("Checkpoint load failed; proceeding with fresh fine-tuning.")
                model, losses, val_losses = finetune_model(
                    model,
                    train_series.to_numpy(dtype=np.float32),
                    val_series.to_numpy(dtype=np.float32),
                    config,
                    device,
                )
    else:
        model, losses, val_losses = finetune_model(
            model,
            train_series.to_numpy(dtype=np.float32),
            val_series.to_numpy(dtype=np.float32),
            config,
            device,
        )

    print(f"\nGenerating forecast from the latest {config.context_length}-day context...")
    pred_finetuned, quantiles_finetuned, future_dates = forecast_last_window(
        model,
        context_series=eval_context,
        future_index=holdout_series.index,
        config=config,
    )
    base_model, _ = load_model(config)
    pred_base, quantiles_base, _ = forecast_last_window(
        base_model,
        context_series=eval_context,
        future_index=holdout_series.index,
        config=config,
    )

    print("\nForecast summary (fine-tuned vs base)")
    print("=" * 70)
    for i in range(config.horizon):
        date_str = future_dates[i].strftime("%Y-%m-%d")
        p10_ft = quantiles_finetuned[i, 1]
        p90_ft = quantiles_finetuned[i, 9]
        p10_base = quantiles_base[i, 1]
        p90_base = quantiles_base[i, 9]
        actual = holdout_series.iloc[i] if i < len(holdout_series) else float("nan")
        print(
            f"{date_str}: finetuned={pred_finetuned[i]:.2f} "
            f"(p10={p10_ft:.2f}, p90={p90_ft:.2f}), "
            f"base={pred_base[i]:.2f} (p10={p10_base:.2f}, p90={p90_base:.2f}), "
            f"actual={float(actual):.2f}"
        )
    actual_vals = holdout_series.to_numpy(dtype=float)
    mae_finetuned = float(np.mean(np.abs(pred_finetuned - actual_vals)))
    mae_base = float(np.mean(np.abs(pred_base - actual_vals)))
    print(f"\nMAE over horizon | fine-tuned: {mae_finetuned:.4f} | base: {mae_base:.4f}")

    if losses:
        plt.figure(figsize=(9, 4))
        plt.plot(range(1, len(losses) + 1), losses, marker="o", label="Train")
        if val_losses:
            plt.plot(range(1, len(val_losses) + 1), val_losses, marker="s", label="Validation")
        plt.title("Fine-tuning loss")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.grid(alpha=0.3)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()

    print("\nGenerating forecast with benchmark covariates...")
    try:
        cov_pred, cov_actual, cov_dates, cov_context = forecast_with_benchmark_covariates(
            model, config
        )
        print("\nCovariate Forecast summary (benchmark-style setup)")
        print("=" * 70)
        for i in range(config.horizon):
            date_str = cov_dates[i].strftime("%Y-%m-%d")
            actual = cov_actual[i] if i < len(cov_actual) else float("nan")
            print(f"{date_str}: pred_covars={cov_pred[i]:.2f}, actual={float(actual):.2f}")
        plot_covariate_results(cov_context, cov_pred, cov_actual, cov_dates, config)
    except Exception as exc:
        print(f"Covariate forecast failed: {exc}")

    plot_results(
        series=series,
        pred_finetuned=pred_finetuned,
        quantiles_finetuned=quantiles_finetuned,
        pred_base=pred_base,
        quantiles_base=quantiles_base,
        future_dates=future_dates,
        config=config,
    )


if __name__ == "__main__":
    main()
