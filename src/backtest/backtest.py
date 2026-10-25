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
from .enums import OrderStatus
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
        self.rejected = 0

    def __str__(self) -> str:
        return (
            f"行情时点 {self.time}：报价 {self.quotes}，委托 {self.orders}，"
            f"成交 {self.trades}，拒单 {self.rejected}"
        )


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
            self.trusted_event_cache = False
            self.deferred_trade_ack = False
            self._events_empty: bool | None = None
            self._prefetched_trades: tuple[int, list[Any]] | None = None
            self._pending_trade_ack: int | None = None
            self._event_rpc_active = False
            self._message_source: DosVar | None = None
            self._finishing = False
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
                    ctx["tradeReadVersion"] = 0l
                    ctx["tradePeekVersion"] = 0l
                    ctx["tradePeek"] = array(ANY, 0)
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
                    if (clear) {
                        ctx["trades"] = array(ANY, 0)
                        ctx["tradeReadVersion"] += 1
                    }
                    return trades
                }
                def soloAckTrades(engine, version) {
                    if (version < 0) return
                    ctx = Backtest::getContextDict(engine)
                    expected = ctx["tradePeek"]
                    trades = ctx["trades"]
                    count = size(expected)
                    if (version != ctx["tradePeekVersion"] ||
                        ctx["tradeReadVersion"] != ctx["tradePeekReadVersion"] ||
                        count == 0 || size(trades) < count)
                        throw "递延成交确认已失效，回报未被清除"
                    for (i in 0:count) {
                        if (!eqObj(expected[i].keys(), trades[i].keys()))
                            throw "递延成交确认内容不匹配，回报未被清除"
                        for (key in expected[i].keys()) {
                            if (!eqObj(expected[i][key], trades[i][key]))
                                throw "递延成交确认内容不匹配，回报未被清除"
                        }
                    }
                    ctx["trades"] = trades[count:size(trades)]
                    ctx["tradeReadVersion"] += 1
                    ctx["tradePeek"] = array(ANY, 0)
                }
                def soloFlushTrades(engineName, version) {
                    soloAckTrades(objByName(engineName), version)
                    NULL
                }
                def soloEventsEmpty(engine) {
                    ctx = Backtest::getContextDict(engine)
                    return size(ctx["orders"]) == 0 && size(ctx["trades"]) == 0
                }
                def soloNextEvents(engineName, ackVersion, prefetch) {
                    engine = objByName(engineName)
                    soloAckTrades(engine, ackVersion)
                    ctx = Backtest::getContextDict(engine)
                    kind = ""
                    reports = array(ANY, 0)
                    if (size(ctx["orders"])) {
                        kind = "order"
                        reports = getOrders(engine, true)
                    } else if (size(ctx["trades"])) {
                        kind = "trade"
                        reports = getTrades(engine, true)
                    }
                    peek = array(ANY, 0)
                    version = -1l
                    if (prefetch && kind == "order" && size(ctx["trades"])) {
                        peek = getTrades(engine)
                        ctx["tradePeek"] = each(copy, peek)
                        ctx["tradePeekVersion"] += 1
                        ctx["tradePeekReadVersion"] = ctx["tradeReadVersion"]
                        version = ctx["tradePeekVersion"]
                    }
                    return dict(`kind`reports`eventsEmpty`peekTrades`peekVersion,
                        [kind, reports, soloEventsEmpty(engine), peek, version])
                }
                def soloSubmitOrders(engineName, messages, labels, kinds, settings, hasSettings, ackVersion) {
                    engine = objByName(engineName)
                    soloAckTrades(engine, ackVersion)
                    identifiers = array(LONG, 0)
                    for (i in 0:size(messages)) {
                        if (!hasSettings[i]) {
                            identifier = Backtest::submitOrder(engine, messages[i], labels[i], kinds[i])[0]
                        } else {
                            params = copy(settings[i])
                            params["endTime"] = second(params["endTime"])
                            if ("startTime" in params.keys()) params["startTime"] = second(params["startTime"])
                            identifier = Backtest::submitOrder(engine, messages[i], labels[i], kinds[i],, params)[0]
                        }
                        identifiers.append!(long(identifier))
                    }
                    return dict(`ids`eventsEmpty, [identifiers, soloEventsEmpty(engine)])
                }
                def soloCancelOrder(engineName, symbol, identifiers, label, ackVersion) {
                    engine = objByName(engineName)
                    soloAckTrades(engine, ackVersion)
                    Backtest::cancelOrder(engine, symbol, identifiers, label)
                    return soloEventsEmpty(engine)
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

    def load_timeline(
        self, message: DosVar, start: np.datetime64, end: np.datetime64
    ) -> Sequence[np.datetime64]:
        """返回本批 [start,end) 的有序行情时点，可覆盖为预载的本地时间轴。"""
        return self.session.run(f"sort(distinct({message}.timestamp))")

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
                timeline = self.load_timeline(message, start, batch_end)
                if len(timeline) and (timeline[0] < start or timeline[-1] >= batch_end):
                    raise ValueError("行情超出本批日期范围")
                if any(left >= right for left, right in zip(timeline, timeline[1:])):
                    raise ValueError("行情时间轴必须严格递增")
                for self.time in timeline:
                    if not initialized:
                        self.initialize()
                        initialized = True
                    self._message_source = message
                    try:
                        self.advance(DosVar("msg"))
                    finally:
                        self._message_source = None
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

    def invalidate_events(self) -> None:
        """原始 session 调用可能改变插件回报时，使空提示和未消费的预取失效。"""
        self._events_empty = None
        self._prefetched_trades = None

    def flush_pending_events(self, session: Any | None = None) -> None:
        """原始 session 读写前确认已交付成交；代理应传入未包装的 session。"""
        if self._event_rpc_active or self._pending_trade_ack is None:
            return
        runner = self.session if session is None else session
        self._run_event_rpc("soloFlushTrades", "engine", self._ack_version(), session=runner)

    def _ack_version(self) -> np.int64:
        return np.int64(-1 if self._pending_trade_ack is None else self._pending_trade_ack)

    def _run_event_rpc(self, script: str, *args: Any, session: Any | None = None) -> Any:
        """已携带待确认版本的 RPC；防止 session 代理重复确认或递归。"""
        self.invalidate_events()
        self._event_rpc_active = True
        try:
            result = (self.session if session is None else session).run(script, *args)
        except BaseException:
            # ACK 可能已执行；不重试具有交易副作用的 RPC。
            raise
        else:
            self._pending_trade_ack = None
            return result
        finally:
            self._event_rpc_active = False

    def next_events(self) -> tuple[str, list[OrderReport] | list[TradeReport]]:
        """仅消费一个非空回报队列，始终先委托后成交；回调前已逻辑清空。"""
        if self.trusted_event_cache:
            if self._prefetched_trades is not None:
                version, records = self._prefetched_trades
                reports = [TradeReport.model_validate(record) for record in records]
                self._prefetched_trades = None
                self._pending_trade_ack = version
                self._events_empty = True
                return "trade", reports
            if self._events_empty is True:
                return "", []
        result = self._run_event_rpc(
            "soloNextEvents",
            "engine",
            self._ack_version(),
            self.deferred_trade_ack and self.trusted_event_cache,
        )
        self._events_empty = bool(result["eventsEmpty"])
        if len(result["peekTrades"]):
            self._prefetched_trades = (int(result["peekVersion"]), list(result["peekTrades"]))
        kind = result["kind"] or ""
        model = OrderReport if kind == "order" else TradeReport
        return kind, [model.model_validate(record) for record in result["reports"]]

    def process_events(self, msg: DosVar | None = None) -> None:
        # 默认不信任其他调用者是否通过原始 session 修改了插件。
        if not self.trusted_event_cache:
            self.invalidate_events()
        while True:
            kind, reports = self.next_events()
            if kind:
                try:
                    if kind == "order":
                        self.moment.rejected += sum(
                            report.status == OrderStatus.REJECTED for report in reports
                        )
                        self.on_order(reports)
                    else:
                        self.moment.trades += len(reports)
                        self.on_trade(reports)
                finally:
                    if not self.trusted_event_cache:
                        self.invalidate_events()
                continue
            if msg is None:
                return
            current = msg
            msg = None
            try:
                self.on_snapshot(current)
            finally:
                if not self.trusted_event_cache:
                    self.invalidate_events()

    def snapshot_fields(self, msg: DosVar) -> dict[str, str]:
        """扩展一步回放的返回值：字段名映射为 append 之后执行的 DOS 表达式。"""
        return {}

    def on_advance(self, snapshot: dict[str, Any]) -> None:
        """append 成功后调用；可缓存 snapshot_fields 返回的价格或账户快照。"""
        if not np.isnat(self.moment.time):
            self.emit_moment(self.moment)
        self.moment = Moment(
            np.datetime64("NaT", "ms") if self._finishing else self.time,
            int(snapshot["quotes"]),
        )

    def emit_moment(self, moment: Moment) -> None:
        """仅在后继行情成功推进后输出一次；上层可覆盖为统一封存输出。"""
        print(moment, flush=True)

    def advance(self, msg: DosVar) -> None:
        selection = ""
        if self._message_source is not None:
            millis = int(np.datetime64(self.time, "ms").astype(np.int64))
            selection = (
                f"{msg} = select * from {self._message_source} "
                f"where timestamp = timestamp({millis}l);"
            )
        extra = self.snapshot_fields(msg)
        if {"quotes", "eventsEmpty"} & extra.keys():
            raise ValueError("回放快照扩展字段不能覆盖 quotes 或 eventsEmpty")
        names = ["quotes", "eventsEmpty", *extra]
        expressions = [f"{msg}.rows()", "soloEventsEmpty(engine)", *extra.values()]
        snapshot = self._run_event_rpc(f"""
            soloAckTrades(engine, {int(self._ack_version())}l)
            {selection}
            (def(engine, message) {{
                stat = Backtest::getBacktestEngineStat(engine)
                if (stat.status[0] == "FATAL") throw "回测引擎发生错误：" + stat.lastErrMsg[0]
                Backtest::appendQuotationMsg(engine, message)
            }})(engine, {msg})
            dict({json.dumps(names)}, [{", ".join(f"({expression})" for expression in expressions)}])
        """)
        self._events_empty = bool(snapshot["eventsEmpty"])
        self.on_advance(snapshot)

    def submit_order(self, order: Order) -> int:
        """同步提交单笔委托并返回真实插件编号，不延迟下单。"""
        return self.submit_orders([order])[0]

    def submit_orders(self, orders: Sequence[Order]) -> list[int]:
        """一次往返按输入顺序逐笔调用插件，不重排、不交错分发回报。"""
        if not orders:
            return []
        messages = [order.to_dos() for order in orders]
        labels = np.asarray([order.label for order in orders], dtype=str)
        kinds = np.asarray([order.kind for order in orders], dtype=np.int32)
        settings = [order.algo_params() for order in orders]
        result = self._run_event_rpc(
            "soloSubmitOrders",
            "engine",
            messages,
            labels,
            kinds,
            settings,
            np.asarray([item is not None for item in settings], dtype=bool),
            self._ack_version(),
        )
        identifiers = [int(identifier) for identifier in result["ids"]]
        self._events_empty = bool(result["eventsEmpty"])
        self.moment.orders += len(identifiers)
        return identifiers

    def set_trading_volume_dist(self, volume: DosVar) -> None:
        self._run_event_rpc(f"""
            soloAckTrades(engine, {int(self._ack_version())}l)
            Backtest::setTradingVolumeDist(engine, {volume})
            NULL
        """)

    def cancel_order(
        self,
        symbol: str = "",
        orders: Sequence[int] | None = None,
        label: str = "",
    ) -> None:
        self._events_empty = bool(
            self._run_event_rpc(
                "soloCancelOrder",
                "engine",
                symbol,
                np.asarray(orders if orders is not None else [], dtype=np.int64),
                label,
                self._ack_version(),
            )
        )

    def get_records(self, expression: str, model: type[T]) -> list[T]:
        columns = self._run_event_rpc(
            f"soloAckTrades(engine, {int(self._ack_version())}l); "
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

    def _read_events(self, expression: str) -> Any:
        return self._run_event_rpc(
            f"soloAckTrades(engine, {int(self._ack_version())}l); {expression}"
        )

    def get_orders(self) -> list[OrderReport]:
        return [
            OrderReport.model_validate(record) for record in self._read_events("getOrders(engine)")
        ]

    def get_trades(self) -> list[TradeReport]:
        return [
            TradeReport.model_validate(record) for record in self._read_events("getTrades(engine)")
        ]

    def get_context(self) -> EngineContext:
        return EngineContext.model_validate(self._read_events("Backtest::getContextDict(engine)"))

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
        self._finishing = True
        try:
            self._run_event_rpc(f"""
                soloAckTrades(engine, {int(self._ack_version())}l)
                endMsg = select top 1 * from {msg}
                update endMsg set symbol = "END", timestamp = timestamp + 1
                NULL
            """)
            self.advance(DosVar("endMsg"))
            self.process_events()
        finally:
            self._finishing = False
            self.release(["endMsg"])

    def close(self) -> None:
        if self.closed:
            return
        try:
            self._run_event_rpc(f"""
                soloAckTrades(engine, {int(self._ack_version())}l)
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
