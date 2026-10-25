"""DolphinDB 回测插件的 Python 时间线驱动器。"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from types import TracebackType
from typing import Any, Generic, Self, Sequence, TypeVar, final

import dolphindb as ddb
import numpy as np
from pydantic import BaseModel

from .algo import Algo
from .models import (
    DailyPortfolio,
    DailyPosition,
    DailyTradingStatistics,
    EngineContext,
    OrderReport,
    Position,
    TradeDetail,
    TradeReport,
)
from .orders import Order
from .types import DosRecord, DosVar

__all__ = ["Backtest"]

C = TypeVar("C", bound=BaseModel, infer_variance=True)
T = TypeVar("T", bound=DosRecord, infer_variance=True)


class Moment:
    """一个行情时点的回放情况；该时点的成交要等时间线推进后才完整，因此推进后再输出。"""

    def __init__(self, time: np.datetime64, quotes: int) -> None:
        self.time = time
        self.quotes = quotes
        self.orders = 0
        self.trades = 0

    def __str__(self) -> str:
        return f"行情时点 {self.time}：报价 {self.quotes}，委托 {self.orders}，成交 {self.trades}"


class Backtest(ABC, Generic[C]):
    """持有回测引擎及其会话，负责回放、回调和订单交互。"""

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        username: str = "",
        password: str = "",
        *,
        session: Any | None = None,
        engine_name: str,
        config: dict[str, Any],
        ctx: C,
        algos: Sequence[Algo[Any, C]] = (),
    ) -> None:
        """传入已连接 session 时将其生命周期交给引擎，不再建立连接。"""
        if session is None and (not host or port is None):
            raise ValueError("必须提供已连接会话或主机与端口")
        self.session = session if session is not None else ddb.session()
        self.closed = False
        created = False
        try:
            self.config = {
                "enableAlgoOrder": True,
                **config,
                "startDate": np.datetime64(config["startDate"], "D"),
                "endDate": np.datetime64(config["endDate"], "D"),
            }
            if (
                np.isnat(self.config["startDate"])
                or np.isnat(self.config["endDate"])
                or self.config["startDate"] > self.config["endDate"]
            ):
                raise ValueError("回测日期必须有效且开始日不晚于结束日")
            self.time = np.datetime64("NaT", "ms")
            self.moment = Moment(self.time, 0)
            self.ctx = ctx
            self.algos = tuple(algos)
            for algo in self.algos:
                algo.backtest = self
                algo.ctx = self.ctx
            if session is None:
                if not self.session.connect(host, port, username, password):
                    raise ConnectionError(f"无法连接 DolphinDB {host}:{port}")
            self.session.run("""
                def initialize(mutable ctx) {
                    ctx["orders"] = array(ANY, 0)
                    ctx["trades"] = array(ANY, 0)
                }
                def beforeTrading(mutable ctx) {}
                def afterTrading(mutable ctx) {}
                def finalize(mutable ctx) {}
                def onSnapshot(mutable ctx, msg, indicator) {}
                def onOrder(mutable ctx, orders) {
                    for (order in orders) ctx["orders"].append!(dict(order.keys(), each(copy, order.values())))
                }
                def onTrade(mutable ctx, trades) {
                    for (trade in trades) ctx["trades"].append!(dict(trade.keys(), each(copy, trade.values())))
                }
                def getOrders(engine, clear=false) {
                    ctx = Backtest::getContextDict(engine)
                    orders = ctx["orders"]
                    if (clear) ctx["orders"] = array(ANY, 0)
                    return orders
                }
                def getTrades(engine, clear=false) {
                    ctx = Backtest::getContextDict(engine)
                    trades = ctx["trades"]
                    if (clear) ctx["trades"] = array(ANY, 0)
                    return trades
                }
                NULL
            """)
            self.session.upload({"engineName": engine_name, "config": self.config})
            self.session.run("""
                engine = Backtest::createBacktestEngine(engineName, config,, initialize, beforeTrading,, onSnapshot, onOrder, onTrade, afterTrading, finalize)
                NULL
            """)
            created = True
            self.release(["engineName", "config"])
        except BaseException:
            try:
                if created:
                    self.close()
                else:
                    self.session.close()
                    self.closed = True
            except Exception:
                pass
            raise

    def release(self, variables: Sequence[str]) -> None:
        if not variables:
            return
        statements = [
            f"if ({json.dumps(name)} in objs(true).name) undef({json.dumps(name)}, VAR)"
            for name in variables
        ]
        try:
            self.session.run("\n".join([*statements, "NULL"]))
        except Exception:
            pass

    @abstractmethod
    def load_messages(self, start: np.datetime64, end: np.datetime64) -> DosVar:
        """加载 [start,end) 内的行情，返回当前会话中的表名。"""
        ...

    def run(self, batch_size: np.timedelta64 = np.timedelta64(1, "D")) -> None:
        if np.isnat(batch_size) or batch_size <= np.timedelta64(0, "ms"):
            raise ValueError("行情加载跨度必须大于零")
        start = np.datetime64(self.config["startDate"], "ms")
        end = np.datetime64(self.config["endDate"], "ms") + np.timedelta64(1, "D")
        initialized = False
        try:
            while start < end:
                batch_end = min(start + batch_size, end)
                message = self.load_messages(start, batch_end)
                timeline = self.session.run(f"sort(distinct({message}.timestamp))")
                if len(timeline) and (timeline[0] < start or timeline[-1] >= batch_end):
                    raise ValueError("行情超出本批日期范围")
                for self.time in timeline:
                    if not initialized:
                        self.initialize()
                        initialized = True
                    self.session.upload({"inputTime": self.time})
                    self.session.run(
                        f"msg = select * from {message} where timestamp = inputTime; NULL"
                    )
                    self.advance(DosVar("msg"))
                    self.process_events(DosVar("msg"))
                start = batch_end
            if not initialized:
                raise ValueError("行情为空")
            self.finish(DosVar("msg"))
        finally:
            self.release(["inputTime", "msg", "endMsg"])

    @final
    def initialize(self) -> None:
        for algo in self.algos:
            algo.initialize()

    @final
    def on_snapshot(self, msg: DosVar) -> None:
        self.on_event("on_snapshot", msg)

    @final
    def on_order(self, orders: Sequence[OrderReport]) -> None:
        self.on_event("on_order", orders)

    @final
    def on_trade(self, trades: Sequence[TradeReport]) -> None:
        self.on_event("on_trade", trades)

    def on_event(self, callback: str, payload: Any) -> None:
        for algo in reversed(self.algos):
            getattr(algo, callback)(payload)

    def process_events(self, msg: DosVar | None = None) -> None:
        while True:
            orders = self.session.run("getOrders(engine, true)")
            if len(orders):
                self.on_order([OrderReport.model_validate(record) for record in orders])
                continue
            trades = self.session.run("getTrades(engine, true)")
            if len(trades):
                self.moment.trades += len(trades)
                self.on_trade([TradeReport.model_validate(record) for record in trades])
                continue
            if msg is None:
                return
            current = msg
            msg = None
            self.on_snapshot(current)

    def advance(self, msg: DosVar) -> None:
        quotes = self.session.run(f"""
            (def(engine, message) {{
                stat = Backtest::getBacktestEngineStat(engine)
                if (stat.status[0] == "FATAL") throw "回测引擎发生错误：" + stat.lastErrMsg[0]
                Backtest::appendQuotationMsg(engine, message)
                return message.rows()
            }})(engine, {msg})
        """)
        if not np.isnat(self.moment.time):
            print(self.moment, flush=True)
        self.moment = Moment(self.time, int(quotes))

    def submit_order(self, order: Order) -> int:
        try:
            self.session.upload(
                {
                    "orderMsg": order.to_dos(),
                    "orderKind": np.int32(order.kind),
                    "orderLabel": order.label,
                }
            )
            parameters = order.algo_params()
            if parameters is not None:
                self.session.upload({"algoOrderParam": parameters})
                expression = """
                    (def(engine, message, label, kind, settings) {
                        params = copy(settings)
                        params["endTime"] = second(params["endTime"])
                        if ("startTime" in params.keys()) params["startTime"] = second(params["startTime"])
                        return Backtest::submitOrder(engine, message, label, kind,, params)[0]
                    })(engine, orderMsg, orderLabel, orderKind, algoOrderParam)
                """
            else:
                expression = "Backtest::submitOrder(engine, orderMsg, orderLabel, orderKind)[0]"
            identifier = int(self.session.run(expression))
        finally:
            self.release(["orderMsg", "orderKind", "orderLabel", "algoOrderParam"])
        self.moment.orders += 1
        return identifier

    def set_trading_volume_dist(self, volume: DosVar) -> None:
        self.session.run(f"""
            Backtest::setTradingVolumeDist(engine, {volume})
            NULL
        """)

    def cancel_order(
        self,
        symbol: str = "",
        orders: Sequence[int] | None = None,
        label: str = "",
    ) -> None:
        try:
            self.session.upload(
                {
                    "cancelSymbol": symbol,
                    "cancelLabel": label,
                    "cancelIds": np.asarray(orders if orders is not None else [], dtype=np.int64),
                }
            )
            self.session.run("""
                Backtest::cancelOrder(engine, cancelSymbol, cancelIds, cancelLabel)
                NULL
            """)
        finally:
            self.release(["cancelSymbol", "cancelLabel", "cancelIds"])

    def get_records(self, expression: str, model: type[T]) -> list[T]:
        columns = self.session.run(
            f"(def(records) {{ return dict(records.columnNames(), records.values()) }})({expression})"
        )
        return [
            model.model_validate(dict(zip(columns, row, strict=True)))
            for row in zip(*columns.values(), strict=True)
        ]

    def get_position(self) -> dict[str, Position]:
        return {
            position.symbol: position
            for position in self.get_records("Backtest::getPosition(engine)", Position)
        }

    def get_orders(self) -> list[OrderReport]:
        return [
            OrderReport.model_validate(record) for record in self.session.run("getOrders(engine)")
        ]

    def get_trades(self) -> list[TradeReport]:
        return [
            TradeReport.model_validate(record) for record in self.session.run("getTrades(engine)")
        ]

    def get_context(self) -> EngineContext:
        return EngineContext.model_validate(self.session.run("Backtest::getContextDict(engine)"))

    def get_trade_details(self) -> list[TradeDetail]:
        return self.get_records("Backtest::getTradeDetails(engine)", TradeDetail)

    def get_daily_position(self) -> list[DailyPosition]:
        return self.get_records("Backtest::getDailyPosition(engine)", DailyPosition)

    def get_daily_total_portfolios(self) -> list[DailyPortfolio]:
        return self.get_records("Backtest::getDailyTotalPortfolios(engine)", DailyPortfolio)

    def get_daily_trading_statistics(self) -> list[DailyTradingStatistics]:
        return self.get_records(
            "Backtest::getDailyTradingStatistics(engine)", DailyTradingStatistics
        )

    def finish(self, msg: DosVar) -> None:
        try:
            self.session.run(f"""
                endMsg = select top 1 * from {msg}
                update endMsg set symbol = "END", timestamp = timestamp + 1
                NULL
            """)
            self.advance(DosVar("endMsg"))
            self.process_events()
        finally:
            self.release(["endMsg"])

    def close(self) -> None:
        if self.closed:
            return
        try:
            self.session.run("""
                Backtest::dropBacktestEngine(engine)
                NULL
            """)
        finally:
            try:
                self.session.close()
            finally:
                self.closed = True

    def __enter__(self) -> Self:
        if self.closed:
            raise RuntimeError("回测引擎已关闭")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            self.close()
        except Exception:
            if exc_type is None:
                raise
