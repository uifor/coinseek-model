"""Versioned features and forward-return labels shared with the Go contract."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from coinseek_model import FEATURE_NAMES
from coinseek_model.data import Candle, FundingRate, INTERVAL_MILLISECONDS


def build_dataset(
    candles_by_symbol: Mapping[str, Sequence[Candle]],
    interval: str,
    lookback: int,
    horizon: int,
    start_ms: int,
    end_ms: int,
    fee_bps: float,
    slippage_bps: float,
    funding_by_symbol: Mapping[str, Sequence[FundingRate]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if interval not in INTERVAL_MILLISECONDS or lookback < 2 or horizon < 1:
        raise ValueError("invalid feature window")
    interval_ms = INTERVAL_MILLISECONDS[interval]
    indexed = {
        symbol: {candle.open_time: candle for candle in rows}
        for symbol, rows in candles_by_symbol.items()
    }
    timeline = sorted(set.intersection(*(set(rows) for rows in indexed.values())))
    if len(timeline) < lookback + horizon:
        raise ValueError("insufficient common candles for lookback and horizon")

    features: list[tuple[float, float, float]] = []
    labels: list[float] = []
    times: list[int] = []
    symbols: list[str] = []
    round_trip_cost = 2 * (fee_bps + slippage_bps) / 10_000
    for at in range(lookback - 1, len(timeline) - horizon):
        if timeline[at] < start_ms or timeline[at] >= end_ms:
            continue
        window_times = timeline[at - lookback + 1 : at + horizon + 1]
        if any(b - a != interval_ms for a, b in zip(window_times, window_times[1:])):
            continue
        for symbol, rows in indexed.items():
            window = [rows[timestamp] for timestamp in timeline[at - lookback + 1 : at + 1]]
            closes = [candle.close for candle in window]
            if any(price <= 0 or not math.isfinite(price) for price in closes):
                continue
            width = max(1, lookback // 4)
            recent_volume = sum(candle.quote_volume for candle in window[-width:])
            prior_volume = sum(candle.quote_volume for candle in window[-2 * width : -width])
            returns = np.diff(np.asarray(closes, dtype=np.float64)) / np.asarray(
                closes[:-1], dtype=np.float64
            )
            volume_trend = (
                math.log(recent_volume / prior_volume)
                if recent_volume > 0 and prior_volume > 0
                else 0.0
            )
            feature = (
                closes[-1] / closes[0] - 1,
                -float(np.std(returns)),
                volume_trend,
            )
            first_future = rows[timeline[at + 1]]
            last_future = rows[timeline[at + horizon]]
            funding_cost = 0.0
            if funding_by_symbol is not None:
                funding_cost = sum(
                    rate.rate
                    for rate in funding_by_symbol.get(symbol, ())
                    if first_future.open_time <= rate.funding_time < last_future.open_time + interval_ms
                )
            target = last_future.close / first_future.open - 1 - round_trip_cost - funding_cost
            if all(math.isfinite(value) for value in (*feature, target)):
                features.append(feature)
                labels.append(target)
                times.append(timeline[at])
                symbols.append(symbol)

    if not features:
        raise ValueError("no valid feature rows in requested date range")
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=np.float32),
        np.asarray(times, dtype=np.int64),
        np.asarray(symbols, dtype=str),
    )


def rank_portfolio_returns(
    predictions: np.ndarray,
    labels: np.ndarray,
    times: np.ndarray,
    top_n: int,
    round_trip_cost: float,
) -> np.ndarray:
    returns: list[float] = []
    for timestamp in np.unique(times):
        indices = np.flatnonzero(times == timestamp)
        count = min(top_n, len(indices) // 2)
        if count == 0:
            continue
        order = indices[np.argsort(predictions[indices])]
        shorts = order[:count]
        longs = order[-count:]
        returns.append(
            float(np.mean(labels[longs]) - np.mean(labels[shorts]) - 2 * round_trip_cost)
        )
    return np.asarray(returns, dtype=np.float64)
