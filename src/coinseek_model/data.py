"""Public Binance market-data collection used by offline training."""

from __future__ import annotations

import csv
import io
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone


INTERVAL_MILLISECONDS = {"1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}
BINANCE_DATA_BASE_URL = "https://data.binance.vision/data"


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
    archive_market = "spot" if market == "spot" else "futures/um"
    rows = _fetch_archive_rows(
        archive_market,
        "klines",
        symbol,
        interval,
        start,
        end,
    )
    start_ms, end_ms = int(start.timestamp() * 1000), int(end.timestamp() * 1000)
    candles: list[Candle] = []
    for row in rows:
        if row[0].lower() == "open_time":
            continue
        try:
            open_time = int(row[0])
            if start_ms <= open_time < end_ms:
                candles.append(
                    Candle(
                        open_time=open_time,
                        open=float(row[1]),
                        high=float(row[2]),
                        low=float(row[3]),
                        close=float(row[4]),
                        quote_volume=float(row[7]),
                    )
                )
        except (IndexError, ValueError) as error:
            raise RuntimeError(f"invalid Binance kline archive row for {symbol}") from error
    return candles


def fetch_funding_rates(symbol: str, start: datetime, end: datetime) -> list[FundingRate]:
    rows = _fetch_archive_rows(
        "futures/um",
        "fundingRate",
        symbol,
        None,
        start,
        end,
    )
    start_ms, end_ms = int(start.timestamp() * 1000), int(end.timestamp() * 1000)
    rates: list[FundingRate] = []
    for row in rows:
        if row[0].lower() == "calc_time":
            continue
        try:
            funding_time = int(row[0])
            if start_ms <= funding_time <= end_ms:
                rates.append(FundingRate(funding_time=funding_time, rate=float(row[2])))
        except (IndexError, ValueError) as error:
            raise RuntimeError(f"invalid Binance funding archive row for {symbol}") from error
    return rates


def _fetch_archive_rows(
    market_path: str,
    data_type: str,
    symbol: str,
    interval: str | None,
    start: datetime,
    end: datetime,
) -> list[list[str]]:
    rows: list[list[str]] = []
    last_included_date = (end - timedelta(milliseconds=1)).date()
    month = date(start.year, start.month, 1)
    while month <= last_included_date:
        next_month = (
            date(month.year + 1, 1, 1)
            if month.month == 12
            else date(month.year, month.month + 1, 1)
        )
        month_start = datetime.combine(month, datetime.min.time(), timezone.utc)
        next_month_start = datetime.combine(next_month, datetime.min.time(), timezone.utc)
        if interval is None:
            filename = f"{symbol}-fundingRate-{month:%Y-%m}.zip"
            directory = f"{market_path}/monthly/{data_type}/{symbol}"
        else:
            filename = f"{symbol}-{interval}-{month:%Y-%m}.zip"
            directory = f"{market_path}/monthly/{data_type}/{symbol}/{interval}"
        archive_rows = _download_csv_archive(f"{directory}/{filename}")
        if archive_rows is not None:
            rows.extend(archive_rows)
        elif start > month_start or end < next_month_start:
            if interval is None:
                raise RuntimeError(
                    f"Binance funding archive for {symbol} {month:%Y-%m} is not available yet; "
                    f"set the USD-M training end to {month_start:%Y-%m-%d} UTC or retry after month-end"
                )
            day = max(start.date(), month)
            last_day = min(last_included_date, next_month - timedelta(days=1))
            while day <= last_day:
                filename = f"{symbol}-{interval}-{day:%Y-%m-%d}.zip"
                directory = f"{market_path}/daily/{data_type}/{symbol}/{interval}"
                daily_rows = _download_csv_archive(f"{directory}/{filename}")
                if daily_rows is not None:
                    rows.extend(daily_rows)
                day += timedelta(days=1)
        month = next_month
    return rows


def _download_csv_archive(path: str) -> list[list[str]] | None:
    url = f"{BINANCE_DATA_BASE_URL}/{path}"
    request = urllib.request.Request(url, headers={"User-Agent": "coinseek-model/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            content = response.read()
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise RuntimeError(
            f"Binance public data download failed: HTTP {error.code} for {url}"
        ) from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Binance public data download failed for {url}: {error}") from error
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            csv_name = next(name for name in archive.namelist() if name.endswith(".csv"))
            with archive.open(csv_name) as source:
                reader = csv.reader(io.TextIOWrapper(source, encoding="utf-8"))
                return [row for row in reader if row]
    except (StopIteration, OSError, UnicodeDecodeError, zipfile.BadZipFile) as error:
        raise RuntimeError(f"invalid Binance public data archive: {url}") from error
