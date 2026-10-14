"""人工快照与独立引擎；沿用实际 Backtest 构造、回放和事件分发。"""

from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack
from datetime import datetime, timedelta
from typing import TypeAlias
from uuid import uuid4

import numpy as np
import pytest
from pydantic import BaseModel, Field

from backtest import Algo, Backtest, Direction, DosVar, LimitOrder, OrderReport, TradeReport

START = datetime(2026, 6, 10, 9, 30)


class SharedContext(BaseModel):
    value: int = 0
    events: list[tuple[str, str, int]] = Field(default_factory=list)
    counts: dict[str, int] = Field(default_factory=dict)


class RecordingAlgo(Algo[str, SharedContext]):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.orders: list[list[OrderReport]] = []
        self.trades: list[list[TradeReport]] = []
        self.snapshots: list[tuple[np.datetime64, int]] = []

    def record(self, event: str) -> None:
        self.ctx.value += 1
        self.ctx.counts[event] = self.ctx.counts.get(event, 0) + 1
        self.ctx.events.append((event, self.params, self.ctx.value))

    def initialize(self) -> None:
        self.record("initialize")

    def on_snapshot(self, msg: DosVar) -> None:
        self.record("snapshot")
        self.snapshots.append((self.backtest.time, int(self.backtest.session.run(f"size({msg})"))))

    def on_order(self, orders: Sequence[OrderReport]) -> None:
        self.record("order")
        self.orders.append(list(orders))

    def on_trade(self, trades: Sequence[TradeReport]) -> None:
        self.record("trade")
        self.trades.append(list(trades))


class AlgoBacktest(Backtest[SharedContext]):
    def load_messages(self, start: np.datetime64, end: np.datetime64) -> DosVar:
        self.session.upload({"batchStart": start, "batchEnd": end})
        self.session.run(
            "messages = select * from source where timestamp >= batchStart, timestamp < batchEnd"
        )
        return DosVar("messages")

    def quote(self, seconds: int = 0) -> None:
        self.time = np.datetime64(START + timedelta(seconds=seconds), "ms")
        self.session.upload({"quoteTime": self.time})
        self.session.run("msg = select top 2 * from source; update msg set timestamp=quoteTime")
        self.advance(DosVar("msg"))


EngineFactory: TypeAlias = Callable[
    [Sequence[Algo[str, SharedContext]], SharedContext | None], AlgoBacktest
]


@pytest.fixture
def make_engine(connection: tuple[str, int, str, str]) -> Iterator[EngineFactory]:
    with ExitStack() as stack:

        def create(
            algos: Sequence[Algo[str, SharedContext]], ctx: SharedContext | None = None
        ) -> AlgoBacktest:
            engine = stack.enter_context(
                AlgoBacktest(
                    *connection,
                    engine_name="solo_algo_test_" + uuid4().hex,
                    config={
                        "startDate": "2026-06-10",
                        "endDate": "2026-06-11",
                        "strategyGroup": "stock",
                        "cash": 1_000_000.0,
                        "dataType": np.int32(1),
                        "frequency": np.int32(0),
                        "msgAsTable": True,
                        "msgAsPiecesOnSnapshot": True,
                        "callbackForSnapshot": np.int32(0),
                        "matchingMode": np.int32(1),
                        "matchingRatio": 0.0,
                        "orderBookMatchingRatio": 1.0,
                        "latency": np.int32(0),
                        "commission": 0.0,
                        "tax": 0.0,
                        "enableSellCloseRestrict": False,
                    },
                    ctx=ctx if ctx is not None else SharedContext(),
                    algos=algos,
                )
            )
            engine.session.run("""
                source = table(
                    take(["000001.XSHE","600000.XSHG"],6) as symbol,
                    take(["XSHE","XSHG"],6) as symbolSource,
                    [2026.06.10T09:30:00.000,2026.06.10T09:30:00.000,
                     2026.06.10T09:30:03.000,2026.06.10T09:30:03.000,
                     2026.06.11T09:30:00.000,2026.06.11T09:30:00.000] as timestamp,
                    take(10.,6) as lastPrice, take(10.,6) as prevClosePrice,
                    take(20.,6) as upLimitPrice, take(1.,6) as downLimitPrice,
                    take(0l,6) as totalBidQty, take(0l,6) as totalOfferQty)
                source["bidPrice"] = arrayVector([5,10,15,20,25,30],take([9.9,9.8,9.7,9.6,9.5],30))
                source["offerPrice"] = arrayVector([5,10,15,20,25,30],take([10.1,10.2,10.3,10.4,10.5],30))
                source["bidQty"] = arrayVector([5,10,15,20,25,30],take(100l,30))
                source["offerQty"] = arrayVector([5,10,15,20,25,30],take(100l,30))
            """)
            return engine

        yield create


def buy(engine: Backtest[SharedContext], quantity: int = 100, price: float = 20.0) -> int:
    return engine.submit_order(
        LimitOrder(
            symbol="000001.XSHE",
            time=engine.time.astype("datetime64[us]").item(),
            direction=Direction.BUY_OPEN,
            quantity=quantity,
            price=price,
        )
    )
