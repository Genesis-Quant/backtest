"""限价单：价格边界、逐档成交、剩余量延续、撤单与多空开平。"""

import pytest
from conftest import OPEN_DIRECTIONS, START, SYMBOLS, OrderBacktest

from backtest import Direction, LimitOrder, OrderStatus


@pytest.mark.parametrize("symbol", SYMBOLS)
@pytest.mark.parametrize("direction", OPEN_DIRECTIONS)
def test_sweep_only_marketable_levels(
    engine: OrderBacktest, symbol: str, direction: Direction
) -> None:
    engine.quote(symbol=symbol)
    order_id = engine.submit_order(
        LimitOrder(
            symbol=symbol,
            time=START,
            direction=direction,
            quantity=300,
            price=10.2 if direction == Direction.BUY_OPEN else 9.8,
        )
    )
    fills = engine.fills(order_id)
    assert sum(row.tradeQty for row in fills) == 200
    assert sum(row.tradePrice * row.tradeQty for row in fills) == pytest.approx(
        2030 if direction == Direction.BUY_OPEN else 1970
    )
    assert engine.get_orders()[-1].status == OrderStatus.PARTIALLY_FILLED

    # 下一条同价盘口补充可成交量，只成交剩余 100 股。
    engine.quote(3, symbol=symbol)
    assert sum(row.tradeQty for row in engine.fills(order_id)) == 300
    assert engine.get_orders()[-1].status == OrderStatus.FILLED
    position = engine.get_position()[symbol]
    assert (position.longPosition, position.shortPosition) == (
        (300, 0) if direction == Direction.BUY_OPEN else (0, 300)
    )


@pytest.mark.parametrize("direction", OPEN_DIRECTIONS)
def test_wait_until_limit_is_reached(engine: OrderBacktest, direction: Direction) -> None:
    engine.quote()
    order_id = engine.submit_order(
        LimitOrder(
            symbol="000001.XSHE",
            time=START,
            direction=direction,
            quantity=100,
            price=10.0,
        )
    )
    assert not engine.fills(order_id)
    engine.quote(3)
    assert not engine.fills(order_id)
    engine.quote(
        6,
        bid=10.0 if direction == Direction.SELL_OPEN else 9.8,
        ask=10.0 if direction == Direction.BUY_OPEN else 10.2,
    )
    assert [(row.tradeQty, row.tradePrice) for row in engine.fills(order_id)] == [(100, 10.0)]


def test_cancel_remaining_quantity(engine: OrderBacktest) -> None:
    engine.quote()
    order_id = engine.submit_order(
        LimitOrder(
            symbol="000001.XSHE",
            time=START,
            direction=Direction.BUY_OPEN,
            quantity=300,
            price=10.1,
        )
    )
    assert sum(row.tradeQty for row in engine.fills(order_id)) == 100
    engine.quote(3, ask=10.5)
    engine.cancel_order(orders=[order_id])
    report = engine.get_orders()[-1]
    assert report.status == OrderStatus.CANCELED
    assert report.tradeQty == 200
    engine.quote(6, ask=10.0)
    assert sum(row.tradeQty for row in engine.fills(order_id)) == 100


@pytest.mark.parametrize("direction", OPEN_DIRECTIONS)
def test_close_position(engine: OrderBacktest, direction: Direction) -> None:
    engine.quote()
    engine.submit_order(
        LimitOrder(
            symbol="000001.XSHE",
            time=START,
            direction=direction,
            quantity=100,
            price=10.1 if direction == Direction.BUY_OPEN else 9.9,
        )
    )
    close_direction = (
        Direction.SELL_CLOSE if direction == Direction.BUY_OPEN else Direction.BUY_CLOSE
    )
    order_id = engine.submit_order(
        LimitOrder(
            symbol="000001.XSHE",
            time=START,
            direction=close_direction,
            quantity=100,
            price=9.9 if close_direction == Direction.SELL_CLOSE else 10.1,
        )
    )
    assert sum(row.tradeQty for row in engine.fills(order_id)) == 100
    assert not engine.get_position()


def test_reject_close_without_position(engine: OrderBacktest) -> None:
    engine.quote()
    order_id = engine.submit_order(
        LimitOrder(
            symbol="000001.XSHE",
            time=START,
            direction=Direction.SELL_CLOSE,
            quantity=100,
            price=9.9,
        )
    )
    assert not engine.fills(order_id)
    assert engine.get_orders()[-1].status == OrderStatus.REJECTED
