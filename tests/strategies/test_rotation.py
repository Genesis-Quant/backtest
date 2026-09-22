"""按相对昨收涨幅轮动，跨日先卖后买，含双边佣金与卖出税。"""

import numpy as np
import pytest

from backtest import Direction, DosVar

from .conftest import A, B, Quote, StrategyAlgo, StrategyFactory, assert_days, assert_fills


@pytest.mark.parametrize("batch_hours", [12, 24])
def test_relative_strength_rotation(make_strategy: StrategyFactory, batch_hours: int) -> None:
    class Rotation(StrategyAlgo):
        def on_snapshot(self, msg: DosVar) -> None:
            selected = str(
                self.backtest.session.run(
                    f"(exec symbol from {msg} order by lastPrice / prevClosePrice desc)[0]"
                )
            )
            if selected == self.ctx.selected:
                return
            for symbol, position in self.backtest.get_position().items():
                if position.longPosition and symbol != selected:
                    self.backtest.session.upload({"selectedSymbol": symbol})
                    bid = float(
                        self.backtest.session.run(
                            f"(exec bidPrice from {msg} where symbol=selectedSymbol)[0][0]"
                        )
                    )
                    self.submit(symbol, Direction.SELL_CLOSE, position.longPosition, bid)
            self.backtest.session.upload({"selectedSymbol": selected})
            ask = float(
                self.backtest.session.run(
                    f"(exec offerPrice from {msg} where symbol=selectedSymbol)[0][0]"
                )
            )
            self.submit(selected, Direction.BUY_OPEN, 1000, ask)
            self.ctx.selected = selected

    quotes = [
        Quote("2026-06-10T09:30:00", A, 11, 10.9, 11.1, previous=10),
        Quote("2026-06-10T09:30:00", B, 20, 19.9, 20.1, previous=20),
        Quote("2026-06-11T09:30:00", A, 10, 9.9, 10.1, previous=11),
        Quote("2026-06-11T09:30:00", B, 22, 21.9, 22.1, previous=20),
        Quote("2026-06-12T09:30:00", A, 12, 11.9, 12.1, previous=10),
        Quote("2026-06-12T09:30:00", B, 21, 20.9, 21.1, previous=22),
    ]
    engine = make_strategy(quotes, Rotation(), commission=0.001, tax=0.001)
    engine.run(np.timedelta64(batch_hours, "h"))
    assert_fills(
        engine,
        [
            ("2026-06-10T09:30:00", A, Direction.BUY_OPEN, 1000, 11.1),
            ("2026-06-11T09:30:00", A, Direction.SELL_CLOSE, 1000, 9.9),
            ("2026-06-11T09:30:00", B, Direction.BUY_OPEN, 1000, 22.1),
            ("2026-06-12T09:30:00", B, Direction.SELL_CLOSE, 1000, 20.9),
            ("2026-06-12T09:30:00", A, Direction.BUY_OPEN, 1000, 12.1),
        ],
        fees=[11.1, 19.8, 22.1, 41.8, 12.1],
    )
    # D1: 100000-11100*1.001；D2: +9900*0.998-22100*1.001；D3: +20900*0.998-12100*1.001。
    assert_days(
        engine,
        [
            ("2026-06-10", 88888.9, 99888.9, {A: (1000, 0)}),
            ("2026-06-11", 76647.0, 98647.0, {B: (1000, 0)}),
            ("2026-06-12", 85393.1, 97393.1, {A: (1000, 0)}),
        ],
    )
    assert engine.ctx.selected == A
