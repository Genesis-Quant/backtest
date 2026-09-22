"""价格比回归：卖空高价股、买入低价股，第三日收敛后双边平仓。

按当前 stock 模式实测账务口径：空头开仓金额从可用现金扣除，
持有时空头权益为开仓金额加浮盈，平仓时释放开仓金额并结算盈亏。
这是插件普通股票账户的核算，未模拟真实融券业务的保证金及借券费。
"""

import numpy as np
import pytest

from backtest import Direction, DosVar

from .conftest import A, B, Quote, StrategyAlgo, StrategyFactory, assert_days, assert_fills


@pytest.mark.parametrize("batch_hours", [12, 24])
def test_pair_convergence_long_and_short(make_strategy: StrategyFactory, batch_hours: int) -> None:
    class Pairs(StrategyAlgo):
        def on_snapshot(self, msg: DosVar) -> None:
            prices = self.backtest.session.run(f"dict({msg}.symbol, {msg}.lastPrice)")
            ratio = prices[A] / prices[B]
            if ratio >= 2 and not self.ctx.started:
                self.ctx.started = True
                self.submit(A, Direction.SELL_OPEN, 500, prices[A] - 0.1)
                self.submit(B, Direction.BUY_OPEN, 1000, prices[B] + 0.1)
            elif ratio <= 1.7 and self.ctx.started and not self.ctx.exited:
                self.ctx.exited = True
                positions = self.backtest.get_position()
                self.submit(A, Direction.BUY_CLOSE, positions[A].shortPosition, prices[A] + 0.1)
                self.submit(B, Direction.SELL_CLOSE, positions[B].longPosition, prices[B] - 0.1)

    quotes = [
        Quote("2026-06-10T09:30:00", A, 20, 19.9, 20.1, previous=20),
        Quote("2026-06-10T09:30:00", B, 10, 9.9, 10.1),
        Quote("2026-06-11T09:30:00", A, 19, 18.9, 19.1, previous=20),
        Quote("2026-06-11T09:30:00", B, 10.5, 10.4, 10.6),
        Quote("2026-06-12T09:30:00", A, 18, 17.9, 18.1, previous=19),
        Quote("2026-06-12T09:30:00", B, 11, 10.9, 11.1, previous=10.5),
    ]
    engine = make_strategy(quotes, Pairs())
    engine.run(np.timedelta64(batch_hours, "h"))
    assert_fills(
        engine,
        [
            ("2026-06-10T09:30:00", A, Direction.SELL_OPEN, 500, 19.9),
            ("2026-06-10T09:30:00", B, Direction.BUY_OPEN, 1000, 10.1),
            ("2026-06-12T09:30:00", A, Direction.BUY_CLOSE, 500, 18.1),
            ("2026-06-12T09:30:00", B, Direction.SELL_CLOSE, 1000, 10.9),
        ],
    )
    # 开仓现金 100000-19.9*500-10.1*1000=79950；首日浮亏 150。
    # 次日空头浮盈 (19.9-19)*500=450，多头浮盈 (10.5-10.1)*1000=400。
    # 最终空头赚 900，多头赚 800，总盈利 1700。
    assert_days(
        engine,
        [
            ("2026-06-10", 79950.0, 99850.0, {A: (0, 500), B: (1000, 0)}),
            ("2026-06-11", 79950.0, 100850.0, {A: (0, 500), B: (1000, 0)}),
            ("2026-06-12", 101700.0, 101700.0, {}),
        ],
    )
