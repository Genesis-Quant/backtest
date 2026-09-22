"""沪深股票 T+1：当日禁售、次日可售、新旧仓位隔离与挂单占用可卖量。"""

import numpy as np
import pytest

from backtest import Direction, DosVar, OrderStatus

from .conftest import A, B, Quote, StrategyAlgo, StrategyFactory, assert_days, assert_fills


@pytest.mark.parametrize("symbol", [A, B])
@pytest.mark.parametrize("batch_hours", [12, 24])
@pytest.mark.parametrize("t_plus_one", [True, False], ids=["t-plus-one", "same-day-allowed"])
def test_same_day_sale_and_next_day_unlock(
    make_strategy: StrategyFactory,
    symbol: str,
    batch_hours: int,
    t_plus_one: bool,
) -> None:
    class SellAttempt(StrategyAlgo):
        step: int = 0
        same_day_order: int = 0

        def on_snapshot(self, msg: DosVar) -> None:
            if self.step == 0:
                self.submit(symbol, Direction.BUY_OPEN, 100, 10.1)
            elif self.step == 1:
                self.same_day_order = self.submit(symbol, Direction.SELL_CLOSE, 100, 10.9)
            else:
                position = self.backtest.get_position().get(symbol)
                if position and position.longPosition:
                    assert position.longPosition == position.lastDayLongPosition == 100
                    self.submit(symbol, Direction.SELL_CLOSE, 100, 11.9)
            self.step += 1

    algo = SellAttempt()
    engine = make_strategy(
        [
            Quote("2026-06-10T09:30:00", symbol, 10, 9.9, 10.1),
            Quote("2026-06-10T14:30:00", symbol, 11, 10.9, 11.1),
            Quote("2026-06-11T09:30:00", symbol, 12, 11.9, 12.1, previous=11),
        ],
        algo,
        t_plus_one=t_plus_one,
    )
    engine.run(np.timedelta64(batch_hours, "h"))

    rejected = [algo.same_day_order] if t_plus_one else []
    assert_fills(
        engine,
        [
            ("2026-06-10T09:30:00", symbol, Direction.BUY_OPEN, 100, 10.1),
            (
                "2026-06-11T09:30:00" if t_plus_one else "2026-06-10T14:30:00",
                symbol,
                Direction.SELL_CLOSE,
                100,
                11.9 if t_plus_one else 10.9,
            ),
        ],
        rejected_orders=rejected,
    )
    assert [
        row.orderId for row in engine.ctx.orders if row.status == OrderStatus.REJECTED
    ] == rejected
    # 拒单不得产生额外成交或扣款；关闭限制的对照组当日赚 80，开启后次日卖出赚 180。
    assert_days(
        engine,
        [
            ("2026-06-10", 98990.0, 100090.0, {symbol: (100, 0)}),
            ("2026-06-11", 100180.0, 100180.0, {}),
        ]
        if t_plus_one
        else [
            ("2026-06-10", 100080.0, 100080.0, {}),
            ("2026-06-11", 100080.0, 100080.0, {}),
        ],
    )


@pytest.mark.parametrize("symbol", [A, B])
@pytest.mark.parametrize("batch_hours", [12, 24])
def test_only_yesterdays_shares_can_be_sold_after_adding_today(
    make_strategy: StrategyFactory,
    symbol: str,
    batch_hours: int,
) -> None:
    class MixedPosition(StrategyAlgo):
        step: int = 0

        def __init__(self) -> None:
            super().__init__()
            self.rejected: list[int] = []

        def on_snapshot(self, msg: DosVar) -> None:
            if self.step == 0:
                self.submit(symbol, Direction.BUY_OPEN, 200, 10.1)
            elif self.step == 1:
                self.submit(symbol, Direction.BUY_OPEN, 100, 11.1)
            elif self.step == 2:
                position = self.backtest.get_position()[symbol]
                assert position.longPosition == 300
                assert position.lastDayLongPosition == 200
                assert position.todayBuyVolume == 100
                self.rejected.append(self.submit(symbol, Direction.SELL_CLOSE, 300, 10.9))
            elif self.step == 3:
                # 超量请求应整单拒绝，不能自动裁成 200 股；昨日仓位仍完整可卖。
                assert self.backtest.get_position()[symbol].longPosition == 300
                self.submit(symbol, Direction.SELL_CLOSE, 200, 10.9)
            elif self.step == 4:
                assert self.backtest.get_position()[symbol].longPosition == 100
                self.rejected.append(self.submit(symbol, Direction.SELL_CLOSE, 100, 10.9))
            else:
                self.submit(symbol, Direction.SELL_CLOSE, 100, 11.9)
            self.step += 1

    algo = MixedPosition()
    engine = make_strategy(
        [
            Quote("2026-06-10T09:30:00", symbol, 10, 9.9, 10.1),
            *[
                Quote(f"2026-06-11T09:30:{second:02}", symbol, 11, 10.9, 11.1)
                for second in (0, 3, 6, 9)
            ],
            Quote("2026-06-12T09:30:00", symbol, 12, 11.9, 12.1, previous=11),
        ],
        algo,
    )
    engine.run(np.timedelta64(batch_hours, "h"))
    assert_fills(
        engine,
        [
            ("2026-06-10T09:30:00", symbol, Direction.BUY_OPEN, 200, 10.1),
            ("2026-06-11T09:30:00", symbol, Direction.BUY_OPEN, 100, 11.1),
            ("2026-06-11T09:30:06", symbol, Direction.SELL_CLOSE, 200, 10.9),
            ("2026-06-12T09:30:00", symbol, Direction.SELL_CLOSE, 100, 11.9),
        ],
        rejected_orders=algo.rejected,
    )
    assert len(algo.rejected) == 2
    assert [
        row.orderId for row in engine.ctx.orders if row.status == OrderStatus.REJECTED
    ] == algo.rejected
    # D1 -2020；D2 -1110+2180；D3 +1190，总盈利 240。
    assert_days(
        engine,
        [
            ("2026-06-10", 97980.0, 99980.0, {symbol: (200, 0)}),
            ("2026-06-11", 99050.0, 100150.0, {symbol: (100, 0)}),
            ("2026-06-12", 100240.0, 100240.0, {}),
        ],
    )


@pytest.mark.parametrize("symbol", [A, B])
@pytest.mark.parametrize("batch_hours", [12, 24])
def test_pending_sell_reserves_sellable_shares_and_cancel_releases_them(
    make_strategy: StrategyFactory,
    symbol: str,
    batch_hours: int,
) -> None:
    class ReservedPosition(StrategyAlgo):
        step: int = 0
        pending_order: int = 0
        rejected_order: int = 0

        def on_snapshot(self, msg: DosVar) -> None:
            if self.step == 0:
                self.submit(symbol, Direction.BUY_OPEN, 300, 10.1)
            elif self.step == 1:
                self.pending_order = self.submit(symbol, Direction.SELL_CLOSE, 200, 12.0)
            elif self.step == 2:
                self.rejected_order = self.submit(symbol, Direction.SELL_CLOSE, 200, 10.9)
            elif self.step == 3:
                self.submit(symbol, Direction.SELL_CLOSE, 100, 10.9)
            else:
                self.backtest.cancel_order(orders=[self.pending_order])
                self.submit(symbol, Direction.SELL_CLOSE, 200, 10.9)
            self.step += 1

    algo = ReservedPosition()
    engine = make_strategy(
        [
            Quote("2026-06-10T09:30:00", symbol, 10, 9.9, 10.1),
            *[
                Quote(f"2026-06-11T09:30:{second:02}", symbol, 11, 10.9, 11.1)
                for second in (0, 3, 6, 9)
            ],
        ],
        algo,
    )
    engine.run(np.timedelta64(batch_hours, "h"))
    assert_fills(
        engine,
        [
            ("2026-06-10T09:30:00", symbol, Direction.BUY_OPEN, 300, 10.1),
            ("2026-06-11T09:30:06", symbol, Direction.SELL_CLOSE, 100, 10.9),
            ("2026-06-11T09:30:09", symbol, Direction.SELL_CLOSE, 200, 10.9),
        ],
        rejected_orders=[algo.rejected_order],
    )
    canceled = [row for row in engine.ctx.orders if row.status == OrderStatus.CANCELED]
    assert [(row.orderId, row.tradeQty) for row in canceled] == [(algo.pending_order, 200)]
    assert_days(
        engine,
        [
            ("2026-06-10", 96970.0, 99970.0, {symbol: (300, 0)}),
            ("2026-06-11", 100240.0, 100240.0, {}),
        ],
    )
