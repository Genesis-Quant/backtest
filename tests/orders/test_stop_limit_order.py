"""限价止盈止损单：开仓、未触发、止损/止盈触发后的限价约束。"""

from datetime import timedelta

import pytest
from conftest import OPEN_DIRECTIONS, START, SYMBOLS, OrderBacktest

from backtest import Direction, OrderStatus, StopLimitOrder


@pytest.mark.parametrize("symbol", SYMBOLS)
@pytest.mark.parametrize("direction", OPEN_DIRECTIONS)
@pytest.mark.parametrize("trigger", ["stop_loss", "take_profit"])
def test_trigger_creates_limit_exit(
    engine: OrderBacktest,
    symbol: str,
    direction: Direction,
    trigger: str,
) -> None:
    engine.quote(symbol=symbol)
    long = direction == Direction.BUY_OPEN
    order_id = engine.submit_order(
        StopLimitOrder(
            symbol=symbol,
            exchange=symbol.split(".")[1],
            time=START,
            direction=direction,
            quantity=100,
            price=10.1 if long else 9.9,
            stop_loss=9.5 if long else 10.5,
            take_profit=10.5 if long else 9.5,
            expires_at=START + timedelta(minutes=1),
        )
    )
    assert sum(row.tradeQty for row in engine.fills(order_id)) == 100
    engine.quote(3, symbol=symbol)
    close_direction = Direction.SELL_CLOSE if long else Direction.BUY_CLOSE
    assert not any(row.direction == close_direction for row in engine.get_trade_details())

    trigger_price = 9.5 if (long == (trigger == "stop_loss")) else 10.5
    # 最新成交价越过阈值；对手盘口仍劣于触发价，限价平仓不应穿价成交。
    engine.quote(
        6, symbol=symbol, last=trigger_price, bid=trigger_price - 0.1, ask=trigger_price + 0.1
    )
    exits = [row for row in engine.get_trade_details() if row.direction == close_direction]
    assert exits
    assert all(row.orderPrice == trigger_price for row in exits)
    assert not any(row.direction == close_direction for row in engine.fills())

    engine.quote(
        9,
        symbol=symbol,
        last=trigger_price,
        bid=trigger_price if long else trigger_price - 0.2,
        ask=trigger_price + 0.2 if long else trigger_price,
    )
    exit_fills = [row for row in engine.fills() if row.direction == close_direction]
    assert [(row.tradeQty, row.tradePrice) for row in exit_fills] == [(100, trigger_price)]
    assert not engine.get_position()
    # 另一侧阈值再触发时，已平仓的算法单不能再次开出平仓单。
    engine.quote(12, symbol=symbol, last=10.5 if trigger_price == 9.5 else 9.5)
    assert sum(row.tradeQty for row in engine.fills() if row.direction == close_direction) == 100


def test_no_exit_before_entry_fills(engine: OrderBacktest) -> None:
    engine.quote()
    order_id = engine.submit_order(
        StopLimitOrder(
            symbol="000001.XSHE",
            exchange="XSHE",
            time=START,
            direction=Direction.BUY_OPEN,
            quantity=100,
            price=9.0,
            stop_loss=8.5,
            take_profit=10.5,
            expires_at=START + timedelta(minutes=1),
        )
    )
    engine.quote(3, last=10.6, bid=10.5, ask=10.7)
    assert not engine.fills(order_id)
    assert not any(row.direction == Direction.SELL_CLOSE for row in engine.get_trade_details())


def test_cancel_unfilled_entry_disables_exit(engine: OrderBacktest) -> None:
    engine.quote()
    order_id = engine.submit_order(
        StopLimitOrder(
            symbol="000001.XSHE",
            exchange="XSHE",
            time=START,
            direction=Direction.BUY_OPEN,
            quantity=100,
            price=9.0,
            stop_loss=8.5,
            take_profit=10.5,
            expires_at=START + timedelta(minutes=1),
        )
    )
    engine.quote(3)
    engine.cancel_order(orders=[order_id])
    assert engine.get_orders()[-1].status == OrderStatus.CANCELED
    engine.quote(6, last=8.4, bid=8.3, ask=8.5)
    assert not engine.fills()
    assert not any(row.direction == Direction.SELL_CLOSE for row in engine.get_trade_details())
