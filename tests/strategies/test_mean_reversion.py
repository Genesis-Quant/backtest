"""低吸订单跨快照部分成交，反弹后撤销余量并只平实际持仓。

matchingMode=1、matchingRatio=0 时，后续挂单按原委托价成交：
https://docs.dolphindb.cn/zh/tutorials/matching_engine_simulator.html
"""

import numpy as np
import pytest

from backtest import Direction, DosVar, OrderStatus

from .conftest import A, Quote, StrategyAlgo, StrategyFactory, assert_days, assert_fills


@pytest.mark.parametrize("batch_hours", [12, 24])
def test_partial_entry_cancel_and_take_profit(
    make_strategy: StrategyFactory, batch_hours: int
) -> None:
    class MeanReversion(StrategyAlgo):
        def on_snapshot(self, msg: DosVar) -> None:
            price = float(self.backtest.session.run(f"{msg}.lastPrice[0]"))
            if price <= 9 and not self.ctx.started:
                self.ctx.started = True
                self.ctx.entry_id = self.submit(A, Direction.BUY_OPEN, 300, 9.0)
            elif price >= 9.5 and self.ctx.started and not self.ctx.exited:
                self.ctx.exited = True
                assert self.ctx.entry_id is not None
                self.backtest.cancel_order(orders=[self.ctx.entry_id])
                held = self.backtest.get_position()[A].longPosition
                self.submit(A, Direction.SELL_CLOSE, held, price - 0.1)

    quotes = [
        Quote("2026-06-10T09:30:00", A, 10, 9.9, 10.1),
        Quote("2026-06-10T09:30:03", A, 9, 8.9, 9.0, size=100),
        Quote("2026-06-10T09:30:06", A, 8.8, 8.7, 8.9, size=100),
        Quote("2026-06-10T09:30:09", A, 9.6, 9.5, 9.7),
        Quote("2026-06-10T09:30:12", A, 10, 9.9, 10.1),
    ]
    engine = make_strategy(quotes, MeanReversion(), t_plus_one=False)
    engine.run(np.timedelta64(batch_hours, "h"))
    # 首次吃单 100*9，后续挂单仍以委托价 9 成交 100 股；买入共 1800。
    # 撤销余下 100 股；卖出 200*9.5=1900，盈利 100。
    assert_fills(
        engine,
        [
            ("2026-06-10T09:30:03", A, Direction.BUY_OPEN, 100, 9.0),
            ("2026-06-10T09:30:06", A, Direction.BUY_OPEN, 100, 9.0),
            ("2026-06-10T09:30:09", A, Direction.SELL_CLOSE, 200, 9.5),
        ],
    )
    assert_days(engine, [("2026-06-10", 100100.0, 100100.0, {})])
    canceled = [row for row in engine.ctx.orders if row.status == OrderStatus.CANCELED]
    assert [(row.orderId, row.tradeQty) for row in canceled] == [(engine.ctx.entry_id, 100)]
