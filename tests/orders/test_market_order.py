"""市价单：覆盖已启用的沪深市价类型，并检查剩余量保留或撤销。"""

import pytest
from conftest import OPEN_DIRECTIONS, START, SYMBOLS, OrderBacktest

from backtest import Direction, MarketOrder, OrderStatus, PriceType


@pytest.mark.parametrize(
    "symbol, price_type",
    [
        ("000001.XSHE", PriceType.BEST_FIVE_IOC),
        ("600000.XSHG", PriceType.BEST_FIVE_IOC),
    ],
)
@pytest.mark.parametrize("direction", OPEN_DIRECTIONS)
def test_ioc_sweeps_depth_and_cancels_remainder(
    engine: OrderBacktest,
    symbol: str,
    price_type: PriceType,
    direction: Direction,
) -> None:
    engine.quote(symbol=symbol)
    order_id = engine.submit_order(
        MarketOrder(
            symbol=symbol,
            time=START,
            direction=direction,
            quantity=600,
            price_type=price_type,
        )
    )
    fills = engine.fills(order_id)
    assert sum(row.tradeQty for row in fills) == 500
    assert sum(row.tradePrice * row.tradeQty for row in fills) == pytest.approx(
        5150 if direction == Direction.BUY_OPEN else 4850
    )
    report = engine.get_orders()[-1]
    assert report.status == OrderStatus.CANCELED
    assert report.tradeQty == 100
    engine.quote(3, symbol=symbol)
    assert sum(row.tradeQty for row in engine.fills(order_id)) == 500


@pytest.mark.parametrize("direction", OPEN_DIRECTIONS)
def test_sh_remainder_becomes_limit(engine: OrderBacktest, direction: Direction) -> None:
    engine.quote(symbol="600000.XSHG")
    order_id = engine.submit_order(
        MarketOrder(
            symbol="600000.XSHG",
            time=START,
            direction=direction,
            quantity=600,
            price_type=PriceType.SH_BEST_FIVE_TO_LIMIT,
        )
    )
    assert sum(row.tradeQty for row in engine.fills(order_id)) == 500
    assert not any(row.status == OrderStatus.CANCELED for row in engine.get_orders())
    # 转换后的限价应是最后成交档位：买单 10.5，卖单 9.5。
    engine.quote(3, symbol="600000.XSHG", ask=10.6, bid=9.4)
    assert sum(row.tradeQty for row in engine.fills(order_id)) == 500
    engine.quote(6, symbol="600000.XSHG", ask=10.5, bid=9.5)
    fills = engine.fills(order_id)
    assert sum(row.tradeQty for row in fills) == 600
    assert fills[-1].tradePrice == (10.5 if direction == Direction.BUY_OPEN else 9.5)


@pytest.mark.parametrize("symbol", SYMBOLS)
@pytest.mark.parametrize("direction", OPEN_DIRECTIONS)
def test_best_own_waits_at_original_price(
    engine: OrderBacktest,
    symbol: str,
    direction: Direction,
) -> None:
    engine.quote(symbol=symbol)
    order_id = engine.submit_order(
        MarketOrder(
            symbol=symbol,
            time=START,
            direction=direction,
            quantity=100,
            price_type=PriceType.BEST_OWN,
        )
    )
    assert not engine.fills(order_id)
    engine.quote(3, symbol=symbol)
    assert not engine.fills(order_id)
    engine.quote(
        6,
        symbol=symbol,
        ask=9.9 if direction == Direction.BUY_OPEN else 10.2,
        bid=10.1 if direction == Direction.SELL_OPEN else 9.8,
    )
    assert [(row.tradeQty, row.tradePrice) for row in engine.fills(order_id)] == [
        (100, 9.9 if direction == Direction.BUY_OPEN else 10.1)
    ]


@pytest.mark.parametrize("symbol", SYMBOLS)
@pytest.mark.parametrize("direction", OPEN_DIRECTIONS)
def test_best_counterparty_only_takes_first_level(
    engine: OrderBacktest,
    symbol: str,
    direction: Direction,
) -> None:
    engine.quote(symbol=symbol)
    order_id = engine.submit_order(
        MarketOrder(
            symbol=symbol,
            time=START,
            direction=direction,
            quantity=200,
            price_type=PriceType.BEST_COUNTERPARTY,
        )
    )
    assert sum(row.tradeQty for row in engine.fills(order_id)) == 100
    engine.quote(3, symbol=symbol)
    fills = engine.fills(order_id)
    assert sum(row.tradeQty for row in fills) == 200
    assert all(
        row.tradePrice == (10.1 if direction == Direction.BUY_OPEN else 9.9) for row in fills
    )
