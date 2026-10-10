"""DolphinDB 回测插件的 Python 时间线驱动器。

插件接口：https://docs.dolphindb.cn/zh/plugins/backtest/interface_description.html
配置与行情结构：https://docs.dolphindb.cn/zh/plugins/backtest/stock.html
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from importlib.resources import files
from time import perf_counter
from types import TracebackType
from typing import Any, Self, Sequence, final

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

# 库只产生标准库日志，由使用方配置输出与级别。
logger = logging.getLogger("backtest")


def _moment(value: np.datetime64) -> str:
    """日级时点只显示日期，盘中时点精确到秒。"""
    text = np.datetime_as_string(value, unit="s").replace("T", " ")
    return text.removesuffix(" 00:00:00")


def _span(value: np.timedelta64) -> str:
    days, remainder = divmod(value, np.timedelta64(1, "D"))
    return f"{int(days)} 天" if remainder == np.timedelta64(0, "D") else str(value)


def _window(start: np.datetime64, end: np.datetime64) -> str:
    """把 [start, end) 显示为日期或含结束日的日期区间，盘中边界保留半开写法。"""
    day = np.timedelta64(1, "D")
    if (start - start.astype("datetime64[D]")) or (end - end.astype("datetime64[D]")):
        return f"[{_moment(start)}, {_moment(end)})"
    if end - start == day:
        return _moment(start)
    return f"{_moment(start)} 至 {_moment(end - day)}"


class Backtest[C: BaseModel](ABC):
    """持有一个 DolphinDB 会话和引擎，组织股票快照回放与 Algo 回调。

    下单和账户记账由插件完成；Python 负责加载行情、推进时间和分发回报。
    当前订单元组与结果模型适配股票，不因 config 更改就自动适配其他资产。
    """

    def __init__(
        self,
        host: str,
        port: int,
        username: str,
        password: str,
        *,
        engine_name: str,
        config: dict[str, Any],
        ctx: C,
        algos: Sequence[Algo[Any, C]] = (),
    ) -> None:
        """连接服务并通过 createBacktestEngine 创建引擎。

        Args:
            host: DolphinDB 服务地址。
            port: DolphinDB 服务端口。
            username: 登录用户。
            password: 登录密码。
            engine_name: 本次引擎名称，应避免与已有引擎重名。
            config: 插件原生配置；必须含 startDate、endDate，股票回测还需
                strategyGroup、cash、dataType 等。日期转换为 DATE；其他字段
                按 SDK 类型上传，例如要求 INT 的配置使用 np.int32。
                enableAlgoOrder 默认开启，可由传入配置覆盖。
            ctx: 所有 Algo 共用的 Python 模型实例，与 DOS 引擎 context 分离。
            algos: 按上游到下游排列；初始化正序，每次事件回调逆序。

        服务端须已加载 MatchingEngineSimulator 和 Backtest 插件。
        """
        self.config = {
            "enableAlgoOrder": True,
            **config,
            "startDate": np.datetime64(config["startDate"], "D"),
            "endDate": np.datetime64(config["endDate"], "D"),
        }
        self.time: np.datetime64 = np.datetime64("NaT", "ms")  # run 当前输入的行情时间。
        self.ctx: C = ctx
        self.algos = tuple(algos)
        for algo in self.algos:
            algo.backtest = self
            algo.ctx = self.ctx
        self.session = ddb.session()
        try:
            if not self.session.connect(host, port, username, password):
                raise ConnectionError("DolphinDB connection failed")
            self.session.run(
                files(__package__).joinpath("backtest.dos").read_text(encoding="utf-8")
            )
            self.session.upload({"engineName": engine_name, "config": self.config})
            self.session.run("""
                engine = Backtest::createBacktestEngine(engineName,config,,initialize,beforeTrading,,onSnapshot,onOrder,onTrade,afterTrading,finalize)
            """)
        except Exception:
            self.session.close()
            raise
        logger.debug(
            "回测引擎 %s 已创建：%s 至 %s，Algo %s",
            engine_name,
            self.config["startDate"],
            self.config["endDate"],
            " → ".join(type(algo).__name__ for algo in self.algos) or "无",
        )

    @abstractmethod
    def load_messages(self, start: np.datetime64, end: np.datetime64) -> DosVar:
        """加载 [start, end) 的行情，返回当前会话内的表变量名。

        子类负责查询及字段转换。表必须有 TIMESTAMP 类型的 timestamp 列，
        其他字段满足 dataType 对应的插件行情结构；只返回引用，不下载整张表。
        """
        ...

    def run(self, batch_size: np.timedelta64 = np.timedelta64(1, "D")) -> None:
        """回放 startDate 至 endDate（含结束日），随后执行 finish。

        batch_size 是数据加载跨度，不是策略触发频率。每个时间戳的全部行
        一次传给 advance；回报处理完毕后才推进下一时点。首次非空行情前
        初始化 Algo。空批次跳过；整个区间为空、批次跨度无效或行情越界时报错。
        """
        if np.isnat(batch_size) or batch_size <= np.timedelta64(0, "ms"):
            raise ValueError("batch_size 必须大于零")
        start = np.datetime64(self.config["startDate"], "ms")
        end = np.datetime64(self.config["endDate"], "ms") + np.timedelta64(1, "D")
        last_time: np.datetime64 | None = None
        started = perf_counter()
        batches = moments = 0
        logger.debug("开始回放 %s，每批 %s", _window(start, end), _span(batch_size))
        while start < end:
            batch_end = min(start + batch_size, end)
            loading = perf_counter()
            message = self.load_messages(start, batch_end)
            timeline: np.ndarray = self.session.run(f"sort(distinct({message}.timestamp))")
            if len(timeline) and (timeline[0] < start or timeline[-1] >= batch_end):
                raise ValueError("load_messages 返回了 [start, end) 之外的行情")
            batches += 1
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "批次 %s：%d 行行情、%d 个时点，加载 %.2f 秒",
                    _window(start, batch_end), self.session.run(f"size({message})"),
                    len(timeline), perf_counter() - loading,
                )
            start = batch_end
            if len(timeline) == 0:
                continue
            if last_time is None:
                self.time = timeline[0]
                logger.info("首个行情时点 %s，初始化 %d 个 Algo", _moment(timeline[0]), len(self.algos))
                self.initialize()
            for self.time in timeline:
                self.session.upload({"inputTime": self.time})
                self.session.run(f"msg = select * from {message} where timestamp=inputTime")
                self.advance(DosVar("msg"))
                self.process_events(DosVar("msg"))
            moments += len(timeline)
            last_time = self.time
        if last_time is None:
            raise ValueError("行情为空")
        self.finish(DosVar("msg"))
        logger.debug(
            "回放完成：%d 个批次、%d 个行情时点，用时 %.1f 秒", batches, moments, perf_counter() - started
        )

    @final
    def initialize(self) -> None:
        """按传入顺序初始化 Algo；run 在第一份行情进入插件前调用。"""
        for algo in self.algos:
            started = perf_counter()
            algo.initialize()
            seconds = perf_counter() - started
            # 训练、预加载等耗时初始化值得在默认级别显示，瞬时完成的只在调试时显示。
            logger.log(
                logging.INFO if seconds >= 1 else logging.DEBUG,
                "%s 初始化完成，用时 %.2f 秒", type(algo).__name__, seconds,
            )

    @final
    def on_snapshot(self, msg: DosVar) -> None:
        """将当前行情表的引用交给 Algo；这是 Python 驱动层的通知。"""
        self.on_event("on_snapshot", msg)

    @final
    def on_order(self, orders: Sequence[OrderReport]) -> None:
        """分发一批订单状态回报，同一订单可能在该批中出现多次。"""
        self.on_event("on_order", orders)

    @final
    def on_trade(self, trades: Sequence[TradeReport]) -> None:
        """分发一批实际成交回报；单笔委托可产生多次部分成交。"""
        self.on_event("on_trade", trades)

    def on_event(self, callback: str, payload: Any) -> None:
        """逆序调用指定 Algo 回调，Algo 通过共享 ctx 交换状态。"""
        for algo in reversed(self.algos):
            getattr(algo, callback)(payload)

    def process_events(self, msg: DosVar | None = None) -> None:
        """按订单回报、成交回报、当前行情的优先级循环分发。

        这是本封装的调度约定。每批回报从 DOS 缓冲区取出时即清空对应缓冲区；
        回调中新产生的回报留待下一轮处理。msg 只通知一次，None 表示仅处理
        回报。此方法不会推进 self.time，也不会再次向插件追加行情。
        """
        while True:
            if len(orders := self.session.run("getOrders(engine, true)")):
                logger.debug("分发 %d 条订单回报", len(orders))
                self.on_order([OrderReport.model_validate(order) for order in orders])
            elif len(trades := self.session.run("getTrades(engine, true)")):
                logger.debug("分发 %d 条成交回报", len(trades))
                self.on_trade([TradeReport.model_validate(trade) for trade in trades])
            elif msg is not None:
                current_msg, msg = msg, None
                self.on_snapshot(current_msg)
            else:
                return

    def advance(self, msg: DosVar) -> None:
        """将 msg 引用的表传给 appendQuotationMsg，推进插件撮合。

        DOS 包装在追加前检查 FATAL 状态并抛错，异常经 SDK 传回 Python。
        本方法不更新 self.time、不分发 Python 回调；手动驱动时由调用者
        设置时间并随后调用 process_events。
        """
        self.session.run(f"advance(engine, {msg})")

    def submit_order(self, order: Order) -> int:
        """提交单个订单对象，返回插件分配的订单号。

        order.to_dos() 生成 msg 元组，order.kind 作为外层 orderType；
        TWAP/VWAP 另传 algoOrderParam。当前单笔接口取返回数组的首个编号，
        算法拆单的子单关联可从 TradeDetail.motherOrderId 读取。
        编号不表示成交；状态变化由后续 process_events 分发。
        """
        self.session.upload(
            {
                "orderMsg": order.to_dos(),
                "orderKind": np.int32(order.kind),
                "orderLabel": order.label,
            }
        )
        if (params := order.algo_params()) is not None:
            self.session.upload({"algoOrderParam": params})
            order_id = int(
                self.session.run(
                    "submitOrder(engine, orderMsg, orderLabel, orderKind, algoOrderParam)[0]"
                )
            )
        else:
            order_id = int(self.session.run("submitOrder(engine, orderMsg, orderLabel, orderKind)[0]"))
        logger.debug("提交订单 #%d：%r", order_id, order)
        return order_id

    def set_trading_volume_dist(self, volume: DosVar) -> None:
        """设置 VWAP 拆量依据，volume 是当前会话内的成交量分布表名。

        表列为 symbol（STRING/SYMBOL）、time（SECOND）、value（INT/DOUBLE）；
        value 表示成交量或相对权重。应在提交 VWAPOrder 前设置。
        """
        self.session.run(f"setTradingVolumeDist(engine, {volume})")

    def cancel_order(
        self,
        symbol: str = "",
        orders: Sequence[int] | None = None,
        label: str = "",
    ) -> None:
        """通过 cancelOrder 提交撤单请求。

        使用 symbol 指定标的，或 orders 指定订单编号，或 label 指定标签；
        选择方式按插件接口约定使用。本封装将 None/空列表编码为空 LONG 向量。
        撤单是否成功通过 OrderStatus.CANCELED / CANCEL_REJECTED 回报确认。
        """
        self.session.upload(
            {
                "cancelSymbol": symbol,
                "cancelLabel": label,
                "cancelIds": np.asarray(orders or [], dtype=np.int64),
            }
        )
        self.session.run("cancelOrder(engine, cancelSymbol, cancelIds, cancelLabel)")
        logger.debug("提交撤单：symbol=%r orders=%s label=%r", symbol, list(orders or []), label)

    def get_records[T: DosRecord](self, query: str, model: type[T]) -> list[T]:
        """将 DOS 结果表按列取回、逐行转换为模型，避免生成 DataFrame。"""
        columns = self.session.run(f"result = {query}; dict(result.columnNames(), result.values())")
        return [
            model.model_validate(dict(zip(columns, row, strict=True)))
            for row in zip(*columns.values(), strict=True)
        ]

    def get_position(self) -> dict[str, Position]:
        """查询插件当前持仓，按 symbol 索引；无记录时返回空字典。"""
        return {
            position.symbol: position
            for position in self.get_records("getPosition(engine)", Position)
        }

    def get_context(self) -> EngineContext:
        """读取 DOS 引擎 context 的本次快照；修改返回模型不会写回服务端。"""
        return EngineContext.model_validate(self.session.run("getContext(engine)"))

    def get_orders(self) -> list[OrderReport]:
        """查看尚未被 process_events 消费的订单回报，不清空缓冲区。

        这是 backtest.dos 提供的缓冲读取接口，不是活动委托查询 getOpenOrders。
        已分发的回报不再出现在此列表中。
        """
        return [
            OrderReport.model_validate(order) for order in self.session.run("getOrders(engine)")
        ]

    def get_trades(self) -> list[TradeReport]:
        """查看尚未分发的成交回报，不清空缓冲区；完整历史另查交易明细。"""
        return [
            TradeReport.model_validate(trade) for trade in self.session.run("getTrades(engine)")
        ]

    def get_trade_details(self) -> list[TradeDetail]:
        """读取 getTradeDetails 的订单事件明细，保留同一订单的多条记录。

        统计成交时按 PARTIALLY_FILLED / FILLED 筛选；撤单记录的数量不能直接
        累加为成交量。完整结果应在 finish 后、close 前获取。
        """
        return self.get_records("getTradeDetails(engine)", TradeDetail)

    def get_daily_position(self) -> list[DailyPosition]:
        """读取已生成的日终持仓记录；盘中实时持仓使用 get_position。"""
        return self.get_records("getDailyPosition(engine)", DailyPosition)

    def get_daily_total_portfolios(self) -> list[DailyPortfolio]:
        """读取已生成的每日账户权益记录；最后一日需结束处理后才完整。"""
        return self.get_records("getDailyTotalPortfolios(engine)", DailyPortfolio)

    def get_daily_trading_statistics(self) -> list[DailyTradingStatistics]:
        """读取按标的、交易日统计的四种开平方向的成交量、金额和均价。"""
        return self.get_records("getDailyTradingStatistics(engine)", DailyTradingStatistics)

    def finish(self, msg: DosVar) -> None:
        """基于最后一份非空行情构造 END 消息，完成日终与结束处理。

        END 使用源表首行时间加 1 毫秒；仅处理产生的回报，不通知行情回调。
        此操作保留引擎供结果查询，资源释放由 close 完成。
        """
        self.session.run(f"""
            endMsg = select top 1 * from {msg}
            update endMsg set symbol="END", timestamp=timestamp+1
        """)
        logger.debug("发送 END 消息，完成日终与结束处理")
        self.advance(DosVar("endMsg"))
        self.process_events()

    def close(self) -> None:
        """删除当前回测引擎，并在删除失败时仍关闭 SDK 会话。"""
        logger.debug("释放回测引擎与会话")
        try:
            self.session.run("dropEngine(engine)")
        finally:
            self.session.close()

    def __enter__(self) -> Self:
        """进入 with 块，返回当前实例。"""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """离开 with 块时释放引擎和会话，不自动执行 finish。"""
        self.close()
