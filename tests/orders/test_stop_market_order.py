"""市价止盈止损单：阈值触发、跳空按盘口成交和深度不足。"""

from datetime import timedelta

import pytest
from conftest import OPEN_DIRECTIONS, START, SYMBOLS, OrderBacktest

from backtest import Direction, OrderStatus, StopMarketOrder


@pytest.mark.parametrize("symbol", SYMBOLS)
@pytest.mark.parametrize("direction", OPEN_DIRECTIONS)
@pytest.mark.parametrize("trigger", ["stop_loss", "take_profit"])
def test_trigger_exits_at_market(
    engine: OrderBacktest,
    symbol: str,
    direction: Direction,
    trigger: str,
) -> None:
    engine.quote(symbol=symbol)
    long = direction == Direction.BUY_OPEN
    order_id = engine.submit_order(
        StopMarketOrder(
            symbol=symbol,
            exchange=symbol.split(".")[1],
            time=START,
            direction=direction,
            quantity=100,
            price=20.0 if long else 1.0,
            stop_loss=9.5 if long else 10.5,
            take_profit=10.5 if long else 9.5,
            expires_at=START + timedelta(minutes=1),
        )
    )
    assert sum(row.tradeQty for row in engine.fills(order_id)) == 100
    engine.quote(3, symbol=symbol)
    close_direction = Direction.SELL_CLOSE if long else Direction.BUY_CLOSE
    assert not any(row.direction == close_direction for row in engine.get_trade_details())
    trigger_price = 9.4 if (long == (trigger == "stop_loss")) else 10.6
    engine.quote(
        6, symbol=symbol, last=trigger_price, bid=trigger_price - 0.1, ask=trigger_price + 0.1
    )
    exit_fills = [row for row in engine.fills() if row.direction == close_direction]
    assert sum(row.tradeQty for row in exit_fills) == 100
    assert all(
        row.tradePrice == pytest.approx(trigger_price + (-0.1 if long else 0.1))
        for row in exit_fills
    )
    assert not engine.get_position()
    engine.quote(9, symbol=symbol, last=9.4 if trigger_price == 10.6 else 10.6)
    assert sum(row.tradeQty for row in engine.fills() if row.direction == close_direction) == 100


def test_entry_sweeps_multiple_levels(engine: OrderBacktest) -> None:
    engine.quote()
    order_id = engine.submit_order(
        StopMarketOrder(
            symbol="000001.XSHE",
            exchange="XSHE",
            time=START,
            direction=Direction.BUY_OPEN,
            quantity=300,
            price=20.0,
            stop_loss=9.0,
            take_profit=11.0,
            expires_at=START + timedelta(minutes=1),
        )
    )
    fills = engine.fills(order_id)
    assert sum(row.tradeQty for row in fills) == 300
    assert sum(row.tradePrice * row.tradeQty for row in fills) == pytest.approx(3060)


def test_partial_entry_cancels_unfilled_remainder(engine: OrderBacktest) -> None:
    engine.quote()
    order_id = engine.submit_order(
        StopMarketOrder(
            symbol="000001.XSHE",
            exchange="XSHE",
            time=START,
            direction=Direction.BUY_OPEN,
            quantity=600,
            price=20.0,
            stop_loss=9.0,
            take_profit=11.0,
            expires_at=START + timedelta(minutes=1),
        )
    )
    assert sum(row.tradeQty for row in engine.fills(order_id)) == 500
    report = engine.get_orders()[-1]
    assert report.status == OrderStatus.CANCELED and report.tradeQty == 100
    engine.quote(3, quantities=(1000, 1000, 1000, 1000, 1000))
    assert sum(row.tradeQty for row in engine.fills(order_id)) == 500
