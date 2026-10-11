"""Public Binance market-data collection used by offline training."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone


INTERVAL_MILLISECONDS = {"1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}


@dataclass(frozen=True)
class Candle:
    open_time: int
    open: float
    high: float
    low: float
    close: float
    quote_volume: float


@dataclass(frozen=True)
class FundingRate:
    funding_time: int
    rate: float


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def fetch_candles(
    market: str, symbol: str, interval: str, start: datetime, end: datetime
) -> list[Candle]:
    if market not in ("spot", "usdm") or interval not in INTERVAL_MILLISECONDS:
        raise ValueError("unsupported market or interval")
    host = "https://api.binance.com/api/v3/klines"
    if market == "usdm":
        host = "https://fapi.binance.com/fapi/v1/klines"
    cursor = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    candles: list[Candle] = []
    while cursor < end_ms:
        query = urllib.parse.urlencode(
            {
                "symbol": symbol,
                "interval": interval,
                "startTime": cursor,
                "endTime": end_ms,
                "limit": 1000,
            }
        )
        request = urllib.request.Request(
            f"{host}?{query}", headers={"User-Agent": "coinseek-model/0.1"}
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                rows = json.loads(response.read())
        except (urllib.error.URLError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Binance candle request failed for {symbol}") from error
        if not rows:
            break
        for row in rows:
            candles.append(
                Candle(
                    open_time=int(row[0]),
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    quote_volume=float(row[7]),
                )
            )
        next_cursor = int(rows[-1][0]) + INTERVAL_MILLISECONDS[interval]
        if next_cursor <= cursor:
            raise RuntimeError(f"Binance candle pagination stalled for {symbol}")
        cursor = next_cursor
        time.sleep(0.15)
    return [candle for candle in candles if start.timestamp() * 1000 <= candle.open_time < end_ms]


def fetch_funding_rates(symbol: str, start: datetime, end: datetime) -> list[FundingRate]:
    cursor = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    rates: list[FundingRate] = []
    while cursor < end_ms:
        query = urllib.parse.urlencode(
            {"symbol": symbol, "startTime": cursor, "endTime": end_ms, "limit": 1000}
        )
        request = urllib.request.Request(
            f"https://fapi.binance.com/fapi/v1/fundingRate?{query}",
            headers={"User-Agent": "coinseek-model/0.1"},
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                rows = json.loads(response.read())
        except (urllib.error.URLError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Binance funding-rate request failed for {symbol}") from error
        if not rows:
            break
        rates.extend(
            FundingRate(funding_time=int(row["fundingTime"]), rate=float(row["fundingRate"]))
            for row in rows
        )
        next_cursor = int(rows[-1]["fundingTime"]) + 1
        if next_cursor <= cursor:
            raise RuntimeError(f"Binance funding-rate pagination stalled for {symbol}")
        cursor = next_cursor
        time.sleep(0.15)
    return [rate for rate in rates if int(start.timestamp() * 1000) <= rate.funding_time <= end_ms]
