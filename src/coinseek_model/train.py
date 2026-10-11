"""Train a LightGBM return model and export a checked ONNX artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import onnxruntime as ort
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType

from coinseek_model import FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from coinseek_model.data import fetch_candles, fetch_funding_rates, parse_utc
from coinseek_model.features import build_dataset, rank_portfolio_returns


def train(args: argparse.Namespace) -> Path:
    start = parse_utc(args.start)
    end = parse_utc(args.end)
    if not end > start:
        raise ValueError("end must be later than start")
    interval_ms = {"1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}[args.interval]
    lookback_start = start - timedelta(milliseconds=(args.lookback + args.horizon) * interval_ms)
    candle_sets = {
        symbol: fetch_candles(args.market, symbol, args.interval, lookback_start, end)
        for symbol in args.symbols
    }
    funding_sets = (
        {
            symbol: fetch_funding_rates(symbol, start, end)
            for symbol in args.symbols
        }
        if args.market == "usdm"
        else {}
    )
    x, y, times, symbols = build_dataset(
        candle_sets,
        args.interval,
        args.lookback,
        args.horizon,
        int(start.timestamp() * 1000),
        int(end.timestamp() * 1000),
        args.fee_bps,
        args.slippage_bps,
        funding_sets,
    )
    unique_times = np.unique(times)
    split_index = max(1, int(len(unique_times) * 0.8))
    if split_index >= len(unique_times):
        raise ValueError("need at least two timestamps for a time-ordered validation split")
    split_time = unique_times[split_index]
    embargo_time = split_time - args.horizon * interval_ms
    train_mask = times < embargo_time
    validation_times = unique_times[split_index::args.horizon]
    valid_mask = np.isin(times, validation_times)
    if not np.any(train_mask) or not np.any(valid_mask):
        raise ValueError("insufficient rows after applying the horizon embargo")
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=400,
        learning_rate=0.03,
        max_depth=5,
        num_leaves=15,
        min_child_samples=30,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=2,
        verbosity=-1,
    )
    model.fit(x[train_mask], y[train_mask], feature_name=list(FEATURE_NAMES))
    predictions = model.predict(x[valid_mask]).astype(np.float32)
    validation_returns = rank_portfolio_returns(
        predictions,
        y[valid_mask],
        times[valid_mask],
        args.top_n,
        2 * (args.fee_bps + args.slippage_bps) / 10_000,
    )
    if len(validation_returns) == 0:
        raise ValueError("validation window has too few symbols for long-short ranking")
    mean_return = float(np.mean(validation_returns))
    std_return = float(np.std(validation_returns))
    periods_per_year = {"1h": 24 * 365, "4h": 6 * 365, "1d": 365}[args.interval]
    periods_per_year /= args.horizon
    sharpe = mean_return / std_return * np.sqrt(periods_per_year) if std_return > 0 else 0.0
    equity = np.cumprod(1 + validation_returns)
    peaks = np.maximum.accumulate(np.concatenate(([1.0], equity)))[1:]
    drawdown = float(np.max((peaks - equity) / peaks))

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    onnx_model = onnxmltools.convert_lightgbm(
        model,
        initial_types=[("features", FloatTensorType([None, len(FEATURE_NAMES)]))],
        target_opset=17,
    )
    model_path = output / "model.onnx"
    onnxmltools.utils.save_model(onnx_model, str(model_path))
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    onnx_predictions = session.run(None, {"features": x[valid_mask][:64]})[0]
    onnx_predictions = np.asarray(onnx_predictions).reshape(-1)
    native_predictions = model.predict(x[valid_mask][:64]).reshape(-1)
    max_abs_error = float(np.max(np.abs(onnx_predictions - native_predictions)))
    if not np.allclose(onnx_predictions, native_predictions, rtol=1e-4, atol=1e-5):
        raise ValueError(f"ONNX parity check failed; max absolute error={max_abs_error}")

    model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = "unknown"
    manifest = {
        "model_id": args.job_id,
        "model_family": "lightgbm_regressor",
        "model_format": "onnx",
        "onnx_opset": 17,
        "input_name": "features",
        "output_name": session.get_outputs()[0].name,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "market": args.market,
        "interval": args.interval,
        "lookback_bars": args.lookback,
        "prediction_horizon_bars": args.horizon,
        "symbols": args.symbols,
        "training_start": start.isoformat(),
        "training_end": end.isoformat(),
        "funding_rates_in_labels": args.market == "usdm",
        "validation_start": datetime.fromtimestamp(int(split_time) / 1000, timezone.utc).isoformat(),
        "validation_metrics": {
            "mean_long_short_return": mean_return,
            "annualized_sharpe": float(sharpe),
            "max_drawdown": drawdown,
            "validation_rebalances": int(len(validation_returns)),
            "onnx_max_abs_error": max_abs_error,
        },
        "training_rows": int(np.sum(train_mask)),
        "validation_rows": int(np.sum(valid_mask)),
        "code_commit": git_commit,
        "python_version": platform.python_version(),
        "lightgbm_version": lgb.__version__,
        "onnxruntime_version": ort.__version__,
        "model_sha256": model_sha256,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", default=os.getenv("GITHUB_RUN_ID", "local"))
    parser.add_argument("--market", choices=("spot", "usdm"), required=True)
    parser.add_argument("--interval", choices=("1h", "4h", "1d"), required=True)
    parser.add_argument("--symbols", required=True, help="comma-separated Binance symbols")
    parser.add_argument("--start", required=True, help="ISO-8601 UTC date/time")
    parser.add_argument("--end", required=True, help="ISO-8601 UTC date/time")
    parser.add_argument("--lookback", type=int, default=48)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--top-n", type=int, default=2)
    parser.add_argument("--fee-bps", type=float, default=4.0)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--output", default="model-output")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", args.job_id):
        parser.error("job_id must be 1-64 safe identifier characters")
    args.symbols = sorted({symbol.strip().upper() for symbol in args.symbols.split(",") if symbol.strip()})
    if len(args.symbols) < 4 or len(args.symbols) > 30:
        parser.error("symbols must contain 4 to 30 distinct symbols")
    if any(not re.fullmatch(r"[A-Z0-9]{3,24}USDT", symbol) for symbol in args.symbols):
        parser.error("symbols must be USDT-quoted Binance symbols")
    if (
        args.lookback < 8
        or args.lookback > 720
        or args.horizon < 1
        or args.horizon > 720
        or args.top_n < 1
        or args.top_n > len(args.symbols) // 2
        or not math.isfinite(args.fee_bps)
        or args.fee_bps < 0
        or args.fee_bps > 100
        or not math.isfinite(args.slippage_bps)
        or args.slippage_bps < 0
        or args.slippage_bps > 100
    ):
        parser.error("invalid lookback, horizon, top_n, fee_bps, or slippage_bps")
    args.start = parse_utc(args.start).isoformat()
    args.end = parse_utc(args.end).isoformat()
    if parse_utc(args.end) - parse_utc(args.start) > timedelta(days=365 * 10):
        parser.error("training date range cannot exceed 10 years")
    return args


if __name__ == "__main__":
    train(parse_args())
