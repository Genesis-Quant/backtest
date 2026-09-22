"""两点/三点均线交叉，明确关闭 T+1 限制以验证日内往返。"""

import numpy as np
import pytest

from backtest import Direction, DosVar

from .conftest import A, Quote, StrategyAlgo, StrategyFactory, assert_days, assert_fills


@pytest.mark.parametrize("batch_hours", [12, 24])
def test_moving_average_cross_round_trip(make_strategy: StrategyFactory, batch_hours: int) -> None:
    class MovingAverage(StrategyAlgo):
        def on_snapshot(self, msg: DosVar) -> None:
            price = float(self.backtest.session.run(f"{msg}.lastPrice[0]"))
            self.ctx.prices.append(price)
            if len(self.ctx.prices) < 3:
                return
            fast, slow = sum(self.ctx.prices[-2:]) / 2, sum(self.ctx.prices[-3:]) / 3
            position = self.backtest.get_position().get(A)
            held = position.longPosition if position else 0
            if fast > slow and not held:
                self.submit(A, Direction.BUY_OPEN, 1000, price + 0.1)
            elif fast < slow and held:
                self.submit(A, Direction.SELL_CLOSE, held, price - 0.1)

    prices = [10, 9, 8, 11, 12, 9, 8]
    quotes = [
        Quote(f"2026-06-10T09:30:{i * 3:02}", A, price, price - 0.1, price + 0.1)
        for i, price in enumerate(prices)
    ]
    engine = make_strategy(quotes, MovingAverage(), t_plus_one=False)
    engine.run(np.timedelta64(batch_hours, "h"))
    # 第四点 9.5 > 9⅓ 买入；第六点 10.5 < 10⅔ 平仓。盈亏 (8.9-11.1)*1000=-2200。
    assert_fills(
        engine,
        [
            ("2026-06-10T09:30:09", A, Direction.BUY_OPEN, 1000, 11.1),
            ("2026-06-10T09:30:15", A, Direction.SELL_CLOSE, 1000, 8.9),
        ],
    )
    assert_days(engine, [("2026-06-10", 97800.0, 97800.0, {})])
    assert engine.ctx.prices == prices
