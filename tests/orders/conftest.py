"""真实插件撮合测试：仅上传人工五档盘口，不读取行情库。

运行：uv run pytest tests/orders --ddb-config ../../arena/.env
配置文件使用 DOLPHIN_HOST、DOLPHIN_PORT、DOLPHIN_RUNTIME_USERNAME、
DOLPHIN_RUNTIME_PASSWORD；每个用例独立创建并释放引擎。
"""

from collections.abc import Iterator
from datetime import datetime, timedelta
from uuid import uuid4

import numpy as np
import pytest

from backtest import Backtest, Direction, DosRecord, DosVar, OrderStatus, TradeDetail

START = datetime(2026, 6, 10, 9, 30)
SYMBOLS = ("000001.XSHE", "600000.XSHG")
OPEN_DIRECTIONS = (Direction.BUY_OPEN, Direction.SELL_OPEN)


class OrderBacktest(Backtest[DosRecord]):
    def load_messages(self, start: np.datetime64, end: np.datetime64) -> DosVar:
        raise NotImplementedError("这些测试逐步输入人工盘口，不使用历史数据加载。")

    def quote(
        self,
        seconds: int = 0,
        *,
        symbol: str = "000001.XSHE",
        last: float = 10.0,
        bid: float = 9.9,
        ask: float = 10.1,
        quantities: tuple[int, int, int, int, int] = (100, 100, 100, 100, 100),
    ) -> None:
        """一条快照：买价逐档减 0.1，卖价逐档加 0.1，双边使用指定档量。"""
        self.time = np.datetime64(START + timedelta(seconds=seconds), "ms")
        self.session.upload(
            {
                "quoteSymbol": symbol,
                "quoteExchange": symbol.split(".")[1],
                "quoteTime": self.time,
                "quoteLast": float(last),
                "quoteBid": np.array([round(bid - i * 0.1, 2) for i in range(5)]),
                "quoteAsk": np.array([round(ask + i * 0.1, 2) for i in range(5)]),
                "quoteQty": np.array(quantities, dtype=np.int64),
            }
        )
        self.session.run("""
            msg = table(symbol(take(quoteSymbol,1)) as symbol,
                symbol(take(quoteExchange,1)) as symbolSource,
                take(quoteTime,1) as timestamp,take(quoteLast,1) as lastPrice,
                [20.] as upLimitPrice,[1.] as downLimitPrice,
                [0l] as totalBidQty,[0l] as totalOfferQty,[10.] as prevClosePrice)
            msg["bidPrice"] = arrayVector([5],quoteBid)
            msg["offerPrice"] = arrayVector([5],quoteAsk)
            msg["bidQty"] = arrayVector([5],quoteQty)
            msg["offerQty"] = arrayVector([5],quoteQty)
        """)
        self.advance(DosVar("msg"))

    def fills(self, order_id: int | None = None) -> list[TradeDetail]:
        return [
            row
            for row in self.get_trade_details()
            if row.orderStatus in (OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED)
            and (order_id is None or row.orderId == order_id)
        ]

    def children(self, order_id: int) -> list[TradeDetail]:
        return [
            row
            for row in self.get_trade_details()
            if row.motherOrderId == order_id and row.orderStatus == OrderStatus.REPORTED
        ]


@pytest.fixture
def engine(connection: tuple[str, int, str, str]) -> Iterator[OrderBacktest]:
    with OrderBacktest(
        *connection,
        engine_name="solo_order_test_" + uuid4().hex,
        config={
            "startDate": "2026-06-10",
            "endDate": "2026-06-10",
            "cash": 1_000_000.0,
            "strategyGroup": "stock",
            "msgAsTable": True,
            "msgAsPiecesOnSnapshot": True,
            "dataType": np.int32(1),
            "frequency": np.int32(0),
            "callbackForSnapshot": np.int32(0),
            "matchingMode": np.int32(1),
            "matchingRatio": 0.0,
            "orderBookMatchingRatio": 1.0,
            "latency": np.int32(0),
            "commission": 0.0,
            "tax": 0.0,
            "enableSellCloseRestrict": False,
        },
        ctx=DosRecord(),
    ) as backtest:
        yield backtest
