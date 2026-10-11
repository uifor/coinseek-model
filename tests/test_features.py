import numpy as np

from coinseek_model.data import Candle, FundingRate
from coinseek_model.features import build_dataset, rank_portfolio_returns


def test_build_dataset_uses_only_contiguous_common_timestamps():
    rows = [
        Candle(i * 3_600_000, 100 + i, 102 + i, 99 + i, 101 + i, 1000 + i)
        for i in range(12)
    ]
    sparse = [row for row in rows if row.open_time != 5 * 3_600_000]
    features, labels, times, symbols = build_dataset(
        {"AAAUSDT": rows, "BBBUSDT": sparse},
        "1h",
        4,
        2,
        3 * 3_600_000,
        12 * 3_600_000,
        4,
        2,
    )
    assert features.shape[1] == 3
    assert len(features) == len(labels) == len(times) == len(symbols)
    assert 5 * 3_600_000 not in times


def test_rank_portfolio_returns_uses_top_and_bottom_predictions():
    returns = rank_portfolio_returns(
        predictions=np.array([-2.0, -1.0, 1.0, 2.0]),
        labels=np.array([-0.04, -0.02, 0.03, 0.05]),
        times=np.array([1, 1, 1, 1]),
        top_n=1,
        round_trip_cost=0.001,
    )
    assert returns.tolist() == [0.088]


def test_build_dataset_deducts_funding_rates_inside_holding_period():
    rows = [
        Candle(i * 3_600_000, 100, 102, 99, 101 + i, 1000)
        for i in range(6)
    ]
    _, labels, times, symbols = build_dataset(
        {"AAAUSDT": rows, "BBBUSDT": rows},
        "1h",
        2,
        2,
        3_600_000,
        4 * 3_600_000,
        0,
        0,
        {"AAAUSDT": [FundingRate(2 * 3_600_000, 0.01)]},
    )
    at_time = np.flatnonzero(times == 3_600_000)
    by_symbol = {symbols[index]: labels[index] for index in at_time}
    assert np.isclose(by_symbol["AAAUSDT"], by_symbol["BBBUSDT"] - 0.01)
