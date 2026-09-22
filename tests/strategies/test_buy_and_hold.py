"""双股票持有三日：资金扣除佣金，逐日权益只随持仓市值变化。"""

import numpy as np
import pytest

from backtest import Direction, DosVar

from .conftest import A, B, Quote, StrategyAlgo, StrategyFactory, assert_days, assert_fills


@pytest.mark.parametrize("batch_hours", [12, 24])
def test_buy_and_hold_with_commission(make_strategy: StrategyFactory, batch_hours: int) -> None:
    class BuyAndHold(StrategyAlgo):
        def on_snapshot(self, msg: DosVar) -> None:
            if not self.ctx.started:
                self.ctx.started = True
                self.submit(A, Direction.BUY_OPEN, 1000, 10.1)
                self.submit(B, Direction.BUY_OPEN, 500, 20.2)

    quotes = [
        Quote("2026-06-10T09:30:00", A, 10, 9.9, 10.1),
        Quote("2026-06-10T09:30:00", B, 20, 19.8, 20.2, previous=20),
        Quote("2026-06-11T14:59:00", A, 11, 10.9, 11.1),
        Quote("2026-06-11T14:59:00", B, 19, 18.8, 19.2, previous=20),
        Quote("2026-06-12T14:59:00", A, 12, 11.9, 12.1, previous=11),
        Quote("2026-06-12T14:59:00", B, 21, 20.8, 21.2, previous=19),
    ]
    engine = make_strategy(quotes, BuyAndHold(), commission=0.001)
    engine.run(np.timedelta64(batch_hours, "h"))
    assert_fills(
        engine,
        [
            ("2026-06-10T09:30:00", A, Direction.BUY_OPEN, 1000, 10.1),
            ("2026-06-10T09:30:00", B, Direction.BUY_OPEN, 500, 20.2),
        ],
        fees=[10.1, 10.1],
    )
    # 买入金额 20200，佣金 20.2，现金 79779.8；市值依次 20000、20500、22500。
    assert_days(
        engine,
        [
            ("2026-06-10", 79779.8, 99779.8, {A: (1000, 0), B: (500, 0)}),
            ("2026-06-11", 79779.8, 100279.8, {A: (1000, 0), B: (500, 0)}),
            ("2026-06-12", 79779.8, 102279.8, {A: (1000, 0), B: (500, 0)}),
        ],
    )
