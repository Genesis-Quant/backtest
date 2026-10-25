"""DolphinDB 回测插件的 Python 时间线驱动器。"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
from types import TracebackType
from typing import Any, Generic, Literal, Self, Sequence, TypeVar, final

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

__all__ = ["Backtest", "EventBatch"]

C = TypeVar("C", bound=BaseModel, infer_variance=True)
T = TypeVar("T", bound=DosRecord, infer_variance=True)


@dataclass(frozen=True, slots=True)
class EventBatch:
    """已同步消费的单个回报队列；空状态仅代表本次 RPC 返回瞬间，不是缓存。"""

    kind: Literal["", "order", "trade"]
    reports: list[OrderReport] | list[TradeReport]
    events_empty: bool


class Moment:
    """当前行情时点的累计统计；日志按有操作或回报的事件批次输出。"""

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


@dataclass(slots=True)
class BatchLog:
    time: np.datetime64
    event: str
    sequence: int = 0
    orders: int = 0
    cancellations: int = 0
    order_reports: int = 0
    trades: int = 0
    rejected: int = 0
    failed: bool = False

    @property
    def changed(self) -> bool:
        return bool(self.orders or self.cancellations or self.order_reports or self.trades)

    def __str__(self) -> str:
        labels = {
            "initialize": "初始化",
            "snapshot": "行情回调",
            "order": "委托回调",
            "trade": "成交回调",
            "submit": "批量委托",
            "cancel": "撤单",
        }
        return (
            f"{self.time} 批次 {self.sequence} [{labels[self.event]}]："
            f"提交 {self.orders}，撤单请求 {self.cancellations}，"
            f"委托回报 {self.order_reports}，成交 {self.trades}，拒单 {self.rejected}"
            + ("；处理失败" if self.failed else "")
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
            self._finishing = False
            self._batch_log: BatchLog | None = None
            self._batch_day = np.datetime64("NaT", "D")
            self._batch_number = 0
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
                def soloEventsEmpty(engine) {
                    ctx = Backtest::getContextDict(engine)
                    return size(ctx["orders"]) == 0 && size(ctx["trades"]) == 0
                }
                def soloNextEvents(engineName) {
                    engine = objByName(engineName)
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
                    return dict(`kind`reports`eventsEmpty,
                        [kind, reports, soloEventsEmpty(engine)])
                }
                def soloSubmitOrders(engineName, messages, labels, kinds, settings, hasSettings) {
                    engine = objByName(engineName)
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
                def soloCancelOrder(engineName, symbol, identifiers, label) {
                    engine = objByName(engineName)
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
                    self.step(DosVar("msg"), source=message)
                start = batch_end
            if not initialized:
                raise ValueError("行情为空")
            self.finish(DosVar("msg"))
        finally:
            self.release(["inputTime", "msg", "endMsg"])

    def _flush_batch_log(self, batch: BatchLog) -> None:
        if not batch.changed:
            return
        day = np.datetime64(batch.time, "D")
        if np.isnat(self._batch_day) or day != self._batch_day:
            self._batch_day = day
            self._batch_number = 0
        self._batch_number += 1
        self.emit_batch(replace(batch, sequence=self._batch_number))
        batch.orders = batch.cancellations = batch.order_reports = batch.trades = batch.rejected = 0

    @contextmanager
    def _event_log(self, event: str):
        previous = self._batch_log
        if previous is not None:
            self._flush_batch_log(previous)
        batch = BatchLog(self.time, event)
        self._batch_log = batch
        try:
            yield batch
        except BaseException:
            batch.failed = True
            raise
        finally:
            self._batch_log = previous
            self._flush_batch_log(batch)

    def emit_batch(self, batch: BatchLog) -> None:
        """一批 Python 回调及其新增操作完成后打印；没有变动的批次不输出。"""
        print(batch, flush=True)

    def _record_operation(self, event: str, count: int) -> None:
        batch = self._batch_log
        if batch is None:
            with self._event_log(event) as batch:
                if event == "submit":
                    batch.orders += count
                else:
                    batch.cancellations += count
        elif event == "submit":
            batch.orders += count
        else:
            batch.cancellations += count

    @final
    def initialize(self) -> None:
        with self._event_log("initialize"):
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

    def next_event_batch(self) -> EventBatch:
        """实时读取并同步清空一个队列，orders 优先；不预取、不延迟确认。"""
        result = self.session.run("soloNextEvents", "engine")
        kind = result["kind"] or ""
        model = OrderReport if kind == "order" else TradeReport
        return EventBatch(
            kind,
            [model.model_validate(record) for record in result["reports"]],
            bool(result["eventsEmpty"]),
        )

    def next_events(self) -> tuple[str, list[OrderReport] | list[TradeReport]]:
        """兼容二元组接口；每次都查询服务端，不依赖前一次返回的空状态。"""
        batch = self.next_event_batch()
        return batch.kind, batch.reports

    def process_events(self, msg: DosVar | None = None) -> None:
        """排空回报后分发行情，再排空回调产生的回报；允许回调内重入。"""
        while True:
            kind, reports = self.next_events()
            if kind:
                with self._event_log(kind) as batch:
                    if kind == "order":
                        batch.order_reports = len(reports)
                        batch.rejected = sum(
                            report.status == OrderStatus.REJECTED for report in reports
                        )
                        self.moment.rejected += batch.rejected
                        self.on_order(reports)
                    else:
                        batch.trades = len(reports)
                        self.moment.trades += batch.trades
                        self.on_trade(reports)
                # 任意 Python 回调都可能直接调用 session；必须重新查询队列。
                continue
            if msg is None:
                return
            current = msg
            msg = None
            with self._event_log("snapshot"):
                self.on_snapshot(current)

    def snapshot_fields(self, msg: DosVar) -> dict[str, str]:
        """扩展 append 后的瞬时返回值，不由内核缓存或自动用于账户查询。

        字段名映射为 DOS 表达式；任何后续 session 操作、下单、回调或推进均
        可能使值过时。扩展自行持有的副本不具备跨这些边界的实时性保证。
        """
        return {}

    def on_advance(self, snapshot: dict[str, Any]) -> None:
        """append 成功后调用；snapshot 只描述本次 RPC 返回瞬间的状态。"""
        self.moment = Moment(
            np.datetime64("NaT", "ms") if self._finishing else self.time,
            int(snapshot["quotes"]),
        )

    def step(self, msg: DosVar, *, source: DosVar | None = None) -> None:
        """推进并分发一个时点；source 非空时按 self.time 选择本批行情。

        默认实现融合选择和 append；旧 advance(msg) 覆写仍按原签名调用，
        该兼容路径先单独选取行情，不要求子类接收新增关键字参数。
        """
        if source is None:
            self.advance(msg)
        else:
            millis = int(np.datetime64(self.time, "ms").astype(np.int64))
            selection = f"{msg} = select * from {source} where timestamp = timestamp({millis}l);"
            if getattr(self.advance, "__func__", None) is Backtest.advance:
                self._advance(msg, selection)
            else:
                self.session.run(f"{selection}\nNULL")
                self.advance(msg)
        self.process_events(msg)

    def advance(self, msg: DosVar) -> None:
        """推进已备好的 msg，不消费回报；保留子类的 advance(msg) hook。"""
        self._advance(msg)

    def _advance(self, msg: DosVar, selection: str = "") -> None:
        """一次 RPC 完成可选行情选择、状态检查、append 和瞬时扩展快照。"""
        extra = self.snapshot_fields(msg)
        if {"quotes", "eventsEmpty"} & extra.keys():
            raise ValueError("回放快照扩展字段不能覆盖 quotes 或 eventsEmpty")
        names = ["quotes", "eventsEmpty", *extra]
        expressions = [f"{msg}.rows()", "soloEventsEmpty(engine)", *extra.values()]
        snapshot = self.session.run(f"""
            {selection}
            (def(engine, message) {{
                stat = Backtest::getBacktestEngineStat(engine)
                if (stat.status[0] == "FATAL") throw "回测引擎发生错误：" + stat.lastErrMsg[0]
                Backtest::appendQuotationMsg(engine, message)
            }})(engine, {msg})
            dict({json.dumps(names)}, [{", ".join(f"({expression})" for expression in expressions)}])
        """)
        self.on_advance(snapshot)

    def submit_order(self, order: Order) -> int:
        """同步提交单笔委托；直接进入运输层，不反向调用公开批量 hook。"""
        return self._submit_orders([order])[0]

    def submit_orders(self, orders: Sequence[Order]) -> list[int]:
        """按输入顺序同步提交，默认一次 RPC，不交错分发回报。

        子类若覆写 submit_order，则逐笔调用该 hook，保留其价格或风控逻辑；
        hook 内可安全调用 super().submit_order，不会递归进入本批量接口。
        """
        submit = self.submit_order
        if getattr(submit, "__func__", None) is not Backtest.submit_order:
            scope = self._event_log("submit") if self._batch_log is None else nullcontext()
            with scope:
                return [submit(order) for order in orders]
        return self._submit_orders(orders)

    def _submit_orders(self, orders: Sequence[Order]) -> list[int]:
        """一次 RPC 按序运输委托并返回真实编号；不调用公开下单 hook。"""
        if not orders:
            return []
        messages = [order.to_dos() for order in orders]
        labels = np.asarray([order.label for order in orders], dtype=str)
        kinds = np.asarray([order.kind for order in orders], dtype=np.int32)
        settings = [order.algo_params() for order in orders]
        result = self.session.run(
            "soloSubmitOrders",
            "engine",
            messages,
            labels,
            kinds,
            settings,
            np.asarray([item is not None for item in settings], dtype=bool),
        )
        identifiers = [int(identifier) for identifier in result["ids"]]
        self.moment.orders += len(identifiers)
        self._record_operation("submit", len(identifiers))
        return identifiers

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
        self.session.run(
            "soloCancelOrder",
            "engine",
            symbol,
            np.asarray(orders if orders is not None else [], dtype=np.int64),
            label,
        )
        self._record_operation("cancel", 1)

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
        self._finishing = True
        try:
            self.session.run(f"""
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
