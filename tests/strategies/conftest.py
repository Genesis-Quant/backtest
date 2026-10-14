"""真实引擎、确定性盘口与手工账本断言；不模拟撮合或账户记账。"""

from collections.abc import Iterator, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Protocol, TypeAlias
from uuid import uuid4

import numpy as np
import pytest
from pydantic import BaseModel, Field

from backtest import (
    Algo,
    Backtest,
    Direction,
    DosVar,
    LimitOrder,
    OrderReport,
    OrderStatus,
    TradeReport,
)

A = "000001.XSHE"
B = "600000.XSHG"
CASH = 100_000.0


@dataclass(frozen=True)
class Quote:
    at: str
    symbol: str
    last: float
    bid: float
    ask: float
    size: int = 10_000
    previous: float = 10.0


class StrategyState(BaseModel):
    started: bool = False
    exited: bool = False
    selected: str | None = None
    entry_id: int | None = None
    prices: list[float] = Field(default_factory=list)
    orders: list[OrderReport] = Field(default_factory=list)
    trades: list[TradeReport] = Field(default_factory=list)


class StrategyAlgo(Algo[None, StrategyState]):
    def __init__(self) -> None:
        super().__init__(None)

    def on_order(self, orders: Sequence[OrderReport]) -> None:
        self.ctx.orders.extend(orders)

    def on_trade(self, trades: Sequence[TradeReport]) -> None:
        self.ctx.trades.extend(trades)

    def submit(self, symbol: str, direction: Direction, quantity: int, price: float) -> int:
        return self.backtest.submit_order(
            LimitOrder(
                symbol=symbol,
                time=self.backtest.time.astype("datetime64[us]").item(),
                direction=direction,
                quantity=quantity,
                price=price,
            )
        )


class StrategyBacktest(Backtest[StrategyState]):
    def load_messages(self, start: np.datetime64, end: np.datetime64) -> DosVar:
        self.session.upload({"batchStart": start, "batchEnd": end})
        self.session.run(
            "messages = select * from source where timestamp >= batchStart, timestamp < batchEnd"
        )
        return DosVar("messages")


class StrategyFactory(Protocol):
    def __call__(
        self,
        quotes: Sequence[Quote],
        algo: StrategyAlgo,
        *,
        commission: float = 0.0,
        tax: float = 0.0,
        t_plus_one: bool = True,
    ) -> StrategyBacktest: ...


@pytest.fixture
def make_strategy(connection: tuple[str, int, str, str]) -> Iterator[StrategyFactory]:
    with ExitStack() as stack:

        def create(
            quotes: Sequence[Quote],
            algo: StrategyAlgo,
            *,
            commission: float = 0.0,
            tax: float = 0.0,
            t_plus_one: bool = True,
        ) -> StrategyBacktest:
            times = np.array([row.at for row in quotes], dtype="datetime64[ms]")
            engine = stack.enter_context(
                StrategyBacktest(
                    *connection,
                    engine_name="solo_strategy_" + uuid4().hex,
                    config={
                        "startDate": times.min().astype("datetime64[D]"),
                        "endDate": times.max().astype("datetime64[D]"),
                        "strategyGroup": "stock",
                        "cash": CASH,
                        "dataType": np.int32(1),
                        "frequency": np.int32(0),
                        "msgAsTable": True,
                        "msgAsPiecesOnSnapshot": True,
                        "callbackForSnapshot": np.int32(0),
                        "matchingMode": np.int32(1),
                        "matchingRatio": 0.0,
                        "orderBookMatchingRatio": 1.0,
                        "latency": np.int32(0),
                        "commission": commission,
                        "tax": tax,
                        "enableSellCloseRestrict": t_plus_one,
                    },
                    ctx=StrategyState(),
                    algos=[algo],
                )
            )
            engine.session.upload(
                {
                    "qTimes": times,
                    "qSymbols": [row.symbol for row in quotes],
                    "qSources": [row.symbol.split(".")[1] for row in quotes],
                    "qLast": np.array([row.last for row in quotes], dtype=float),
                    "qPrevious": np.array([row.previous for row in quotes], dtype=float),
                    "qBid": np.array([row.bid for row in quotes], dtype=float),
                    "qAsk": np.array([row.ask for row in quotes], dtype=float),
                    "qSize": np.array([row.size for row in quotes], dtype=np.int64),
                }
            )
            engine.session.run("""
                source = table(qSymbols as symbol, qSources as symbolSource,
                    qTimes as timestamp, qLast as lastPrice, qPrevious as prevClosePrice,
                    take(100.,size(qTimes)) as upLimitPrice, take(1.,size(qTimes)) as downLimitPrice,
                    take(0l,size(qTimes)) as totalBidQty, take(0l,size(qTimes)) as totalOfferQty)
                source["bidPrice"] = fixedLengthArrayVector(qBid,qBid-0.1,qBid-0.2,qBid-0.3,qBid-0.4)
                source["offerPrice"] = fixedLengthArrayVector(qAsk,qAsk+0.1,qAsk+0.2,qAsk+0.3,qAsk+0.4)
                zeroQty = take(0l,size(qTimes))
                source["bidQty"] = fixedLengthArrayVector(qSize,zeroQty,zeroQty,zeroQty,zeroQty)
                source["offerQty"] = fixedLengthArrayVector(qSize,zeroQty,zeroQty,zeroQty,zeroQty)
            """)
            return engine

        yield create


ExpectedFill: TypeAlias = tuple[str, str, Direction, int, float]
ExpectedDay: TypeAlias = tuple[str, float, float, dict[str, tuple[int, int]]]


def assert_fills(
    engine: StrategyBacktest,
    expected: list[ExpectedFill],
    fees: list[float] | None = None,
    *,
    rejected_orders: Sequence[int] = (),
) -> None:
    details = engine.get_trade_details()
    fills = [
        row
        for row in details
        if row.orderStatus in (OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED)
    ]
    actual = [
        (str(np.datetime64(row.tradeTime, "s")), row.symbol, row.direction, row.tradeQty)
        for row in fills
    ]
    assert actual == [row[:4] for row in expected]
    assert [row.tradePrice for row in fills] == pytest.approx([row[4] for row in expected])
    callbacks = [
        (str(np.datetime64(row.tradeTime, "s")), row.symbol, row.direction, row.tradeQty)
        for row in engine.ctx.trades
    ]
    assert callbacks == actual
    assert [row.tradePrice for row in engine.ctx.trades] == pytest.approx(
        [row[4] for row in expected]
    )
    if fees is not None:
        assert [row.totalFee for row in engine.ctx.trades] == pytest.approx(fees)
    assert [row.orderId for row in details if row.orderStatus == OrderStatus.REJECTED] == list(
        rejected_orders
    )
    assert engine.get_orders() == engine.get_trades() == []


def assert_days(engine: StrategyBacktest, expected: list[ExpectedDay]) -> None:
    portfolios = engine.get_daily_total_portfolios()
    positions = engine.get_daily_position()
    assert [row.tradeDate.isoformat() for row in portfolios] == [row[0] for row in expected]
    for portfolio, (day, cash, equity, holdings) in zip(portfolios, expected, strict=True):
        assert portfolio.cash == pytest.approx(cash), day
        assert portfolio.totalEquity == pytest.approx(equity), day
        assert portfolio.totalMarketValue == pytest.approx(equity - cash), day
        assert portfolio.totalPnl == pytest.approx(equity - CASH), day
        assert portfolio.netValue == pytest.approx(equity / CASH), day
        actual = {
            row.symbol: (row.longPosition, row.shortPosition)
            for row in positions
            if row.tradeDate.isoformat() == day and (row.longPosition or row.shortPosition)
        }
        assert actual == holdings, day
    final = engine.get_position()
    assert {
        symbol: (row.longPosition, row.shortPosition)
        for symbol, row in final.items()
        if row.longPosition or row.shortPosition
    } == expected[-1][3]
