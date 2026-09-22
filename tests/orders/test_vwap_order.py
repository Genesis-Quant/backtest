"""VWAP：按给定成交量分布拆单，验证数量、时间、价格及母子单关联。"""

from datetime import time, timedelta

import pytest
from conftest import OPEN_DIRECTIONS, START, SYMBOLS, OrderBacktest

from backtest import AlgoPriceType, Direction, DosVar, OrderStatus, VWAPOrder


@pytest.mark.parametrize("symbol", SYMBOLS)
@pytest.mark.parametrize("direction", OPEN_DIRECTIONS)
@pytest.mark.parametrize(
    "weights, expected",
    [
        ((1.0, 2.0, 3.0), (100, 200, 300)),
        ((3.0, 2.0, 1.0), (300, 200, 100)),
    ],
)
def test_slices_follow_volume_distribution(
    engine: OrderBacktest,
    symbol: str,
    direction: Direction,
    weights: tuple[float, float, float],
    expected: tuple[int, int, int],
) -> None:
    engine.quote(symbol=symbol, quantities=(1000, 1000, 1000, 1000, 1000))
    engine.session.upload({"volumeWeights": list(weights)})
    engine.session.run("""
        volumeDist=table(take(quoteSymbol,3) as symbol,
            [09:30:10,09:30:30,09:30:50] as time,volumeWeights as value)
    """)
    engine.set_trading_volume_dist(DosVar("volumeDist"))
    order_id = engine.submit_order(
        VWAPOrder(
            symbol=symbol,
            time=START,
            direction=direction,
            quantity=600,
            price=20.0 if direction == Direction.BUY_OPEN else 1.0,
            start_time=time(9, 30, 10),
            end_time=time(9, 30, 50),
            interval=20,
            max_single_order_volume=600,
            order_price_type=AlgoPriceType.BEST_COUNTERPARTY,
        )
    )
    engine.quote(9, symbol=symbol)
    assert not engine.children(order_id)
    for index, seconds in enumerate((10, 30, 50), start=1):
        engine.quote(seconds, symbol=symbol, quantities=(1000, 1000, 1000, 1000, 1000))
        assert len(engine.children(order_id)) == index - 1
        engine.quote(seconds + 1, symbol=symbol, quantities=(1000, 1000, 1000, 1000, 1000))
        children = engine.children(order_id)
        assert [row.orderQty for row in children] == list(expected[:index])
        assert [row.sendTime for row in children] == [
            START + timedelta(seconds=10 + 20 * i) for i in range(index)
        ]
        child_ids = {row.orderId for row in children}
        fills = [row for row in engine.fills() if row.orderId in child_ids]
        assert sum(row.tradeQty for row in fills) == sum(expected[:index])
        assert all(
            row.tradePrice == (10.1 if direction == Direction.BUY_OPEN else 9.9) for row in fills
        )
    engine.quote(60, symbol=symbol)
    assert len(engine.children(order_id)) == 3


def test_maximum_child_quantity(engine: OrderBacktest) -> None:
    engine.quote()
    engine.session.run("""
        volumeDist=table(take(quoteSymbol,3) as symbol,
            [09:30:10,09:30:30,09:30:50] as time,[1.,2.,3.] as value)
    """)
    engine.set_trading_volume_dist(DosVar("volumeDist"))
    order_id = engine.submit_order(
        VWAPOrder(
            symbol="000001.XSHE",
            time=START,
            direction=Direction.BUY_OPEN,
            quantity=600,
            price=20.0,
            start_time=time(9, 30, 10),
            end_time=time(9, 30, 50),
            max_single_order_volume=100,
            order_price_type=AlgoPriceType.BEST_COUNTERPARTY,
        )
    )
    for seconds in (10, 11, 30, 31, 50, 51):
        engine.quote(seconds, quantities=(1000, 1000, 1000, 1000, 1000))
    children = engine.children(order_id)
    assert [row.orderQty for row in children] == [100] * 6
    assert [row.sendTime for row in children] == [
        START + timedelta(seconds=seconds) for seconds in (10, 30, 30, 50, 50, 50)
    ]
    assert sum(row.tradeQty for row in engine.fills()) == 600


def test_cancel_parent_stops_future_slices(engine: OrderBacktest) -> None:
    engine.quote()
    engine.session.run("""
        volumeDist=table(take(quoteSymbol,3) as symbol,
            [09:30:10,09:30:30,09:30:50] as time,[1.,2.,3.] as value)
    """)
    engine.set_trading_volume_dist(DosVar("volumeDist"))
    order_id = engine.submit_order(
        VWAPOrder(
            symbol="000001.XSHE",
            time=START,
            direction=Direction.BUY_OPEN,
            quantity=600,
            price=20.0,
            start_time=time(9, 30, 10),
            end_time=time(9, 30, 50),
            order_price_type=AlgoPriceType.BEST_COUNTERPARTY,
        )
    )
    for seconds in (10, 11, 12):
        engine.quote(seconds)
    assert sum(row.tradeQty for row in engine.fills()) == 100
    engine.cancel_order(orders=[order_id])
    canceled = [
        row
        for row in engine.get_trade_details()
        if row.orderId == order_id and row.orderStatus == OrderStatus.CANCELED
    ]
    assert len(canceled) == 1 and canceled[0].tradeQty == 500
    for seconds in (30, 31, 50, 51, 60):
        engine.quote(seconds)
    assert len(engine.children(order_id)) == 1
    assert sum(row.tradeQty for row in engine.fills()) == 100


def test_child_partial_fill_continues_on_next_quote(engine: OrderBacktest) -> None:
    engine.quote()
    engine.session.run("""
        volumeDist=table(take(quoteSymbol,1) as symbol,[09:30:10] as time,[1.] as value)
    """)
    engine.set_trading_volume_dist(DosVar("volumeDist"))
    order_id = engine.submit_order(
        VWAPOrder(
            symbol="000001.XSHE",
            time=START,
            direction=Direction.BUY_OPEN,
            quantity=200,
            price=20.0,
            start_time=time(9, 30, 10),
            end_time=time(9, 30, 29),
            order_price_type=AlgoPriceType.BEST_COUNTERPARTY,
        )
    )
    engine.quote(10, quantities=(100, 0, 0, 0, 0))
    engine.quote(11, quantities=(0, 0, 0, 0, 0))
    child_id = engine.children(order_id)[0].orderId
    assert sum(row.tradeQty for row in engine.fills(child_id)) == 100
    engine.quote(12, quantities=(100, 0, 0, 0, 0))
    assert sum(row.tradeQty for row in engine.fills(child_id)) == 200
    engine.quote(30)
    assert len(engine.children(order_id)) == 1


@pytest.mark.parametrize(
    "price_type, expected",
    [
        (AlgoPriceType.LAST, 10.0),
        (AlgoPriceType.BEST_OWN, 9.9),
        (AlgoPriceType.BEST_COUNTERPARTY, 10.1),
    ],
)
def test_default_start_and_child_price_source(
    engine: OrderBacktest,
    price_type: AlgoPriceType,
    expected: float,
) -> None:
    engine.quote()
    engine.session.run("""
        volumeDist=table(take(quoteSymbol,1) as symbol,[09:30:00] as time,[1.] as value)
    """)
    engine.set_trading_volume_dist(DosVar("volumeDist"))
    order_id = engine.submit_order(
        VWAPOrder(
            symbol="000001.XSHE",
            time=START,
            direction=Direction.BUY_OPEN,
            quantity=100,
            price=20.0,
            end_time=time(9, 30, 19),
            order_price_type=price_type,
        )
    )
    engine.quote(1)
    assert not engine.children(order_id)
    engine.quote(2)
    children = engine.children(order_id)
    assert len(children) == 1 and children[0].sendTime == START + timedelta(seconds=1)
    assert children[0].orderPrice == expected
    if price_type != AlgoPriceType.BEST_COUNTERPARTY:
        assert not engine.fills(children[0].orderId)
        engine.quote(3, bid=9.7, ask=expected)
    assert sum(row.tradeQty for row in engine.fills(children[0].orderId)) == 100
