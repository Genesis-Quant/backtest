"""股票回测的账户、回报与历史结果模型，字段名沿用插件协议。

接口字段：https://docs.dolphindb.cn/zh/plugins/backtest/interface_description.html
回报语义：https://docs.dolphindb.cn/zh/plugins/backtest.html
包含当前适配版本额外返回的 totalVolume、motherOrderId 等字段。
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import numpy as np
from pydantic import field_validator

from .enums import Direction, OrderKind, OrderStatus, TimeInForce
from .types import DosRecord

__all__ = [
    "DailyPortfolio",
    "DailyPosition",
    "DailyTradingStatistics",
    "EngineContext",
    "OrderReport",
    "Position",
    "TradeDetail",
    "TradeReport",
]


class Position(DosRecord):
    """当前持仓的一条记录，数量为股票股数；由 get_position 按 symbol 索引。"""

    symbol: str  # 标的代码，与行情一致。
    lastDayLongPosition: int  # 上一交易日的多仓数量。
    lastDayShortPosition: int  # 上一交易日的空仓数量。
    longPosition: int  # 多仓数量。
    longPositionAvgPrice: float  # 买持仓的成交均价。
    shortPosition: int  # 空仓数量。
    shortPositionAvgPrice: float  # 卖持仓的成交均价。
    todayBuyVolume: int  # 当日买入成交数量。
    todayBuyValue: float  # 当日买入成交金额。
    todaySellVolume: int  # 当日卖出成交数量。
    todaySellValue: float  # 当日卖出成交金额。


class DailyPosition(Position):
    """日终持仓，继承当前持仓字段并增加交易日和收盘价。"""

    tradeDate: date  # 交易日期，转换为 Python date。
    closePrice: float  # 该交易日收盘价。


class OrderReport(DosRecord):
    """一条委托状态回报；tradeQty 通常为累计成交量，撤单成功时含义改变。"""

    orderId: int  # 插件分配的订单编号。
    symbol: str  # 标的代码，与行情一致。
    direction: Direction  # 委托开平方向。
    timestamp: datetime  # 原委托时间。
    updateTime: datetime  # 本次状态更新时间。
    price: float  # 原委托价格。
    qty: int  # 原委托数量。
    tradeQty: int  # 累计成交数量；CANCELED 时表示本次撤销的数量。
    tradeValue: float  # 累计成交金额；CANCELED 时插件返回 0。
    status: OrderStatus  # 本次回报的委托状态。
    label: str | None  # 提交委托时的标签；SDK 返回空值时可为 None。


class TradeReport(DosRecord):
    """一条实际成交回报；当次成交字段与累计字段分开保留。"""

    orderId: int  # 插件分配的订单编号。
    symbol: str  # 标的代码，与行情一致。
    direction: Direction  # 委托开平方向。
    tradeTime: datetime  # 本次成交时间。
    orderPrice: float  # 原委托价格。
    tradePrice: float  # 本次成交价格。
    tradeQty: int  # 本次成交数量。
    tradeValue: float  # 本次成交金额。
    totalVolume: int  # 当前适配版本返回的订单累计成交数量。
    totalValue: float  # 当前适配版本返回的订单累计成交金额。
    totalFee: float  # 插件返回的总费用。
    label: str | None  # 提交委托时的标签；SDK 返回空值时可为 None。


class TradeDetail(DosRecord):
    """getTradeDetails 的一行订单事件；同一 orderId 可对应多行。"""

    orderId: int  # 插件分配的订单编号。
    symbol: str  # 标的代码，与行情一致。
    direction: Direction  # 委托开平方向。
    sendTime: datetime  # 委托提交时间。
    orderPrice: float  # 原委托价格。
    orderQty: int  # 原委托数量。
    tradeTime: datetime | None  # 该事件记录的成交/处理时间；无有效时间时为 None。
    tradePrice: float  # 事件记录中的成交价格，非成交状态不作为成交价使用。
    tradeQty: int  # 事件记录数量；仅成交状态用于统计成交，撤单等记录需按状态解释。
    orderStatus: OrderStatus  # 此条明细对应的订单状态。
    label: str | None  # 提交委托时的标签；SDK 返回空值时可为 None。
    algoOrderType: OrderKind | None = None  # 外层算法类型；未返回该列时为 None。
    motherOrderId: int | None = None  # 关联母单编号；普通订单或空值为 None。
    timeInForce: TimeInForce | None = None  # 委托有效性；未返回该列时为 None。
    expireTime: datetime | None = None  # 委托到期时间；空值转为 None。
    orderInfo: str | None = None  # 当前插件版本的附加订单信息，如拒单原因；受输出配置影响。

    @field_validator("motherOrderId", mode="before")
    @classmethod
    def convert_mother_order_id(cls, value: Any) -> Any:
        """将 SDK 用 NaN 表示的空母单编号转为 None。"""
        return None if isinstance(value, (float, np.floating)) and np.isnan(value) else value


class DailyPortfolio(DosRecord):
    """股票账户每日权益；收益率为比例数值，不是乘以 100 后的百分数。"""

    tradeDate: date  # 交易日期，转换为 Python date。
    floatingPnl: float  # 浮动盈亏。
    realizedPnl: float  # 已实现盈亏。
    totalPnl: float  # 账户总盈亏。
    cash: float  # 账户可用资金。
    totalMarketValue: float  # 账户总市值。
    totalEquity: float  # 账户总权益。
    netValue: float  # 账户单位净值。
    totalReturn: float  # 截至该日的累计收益率。
    ratio: float  # 当日收益率。
    pnl: float  # 当日盈亏金额。
    totalFee: float  # 插件返回的总费用。
    frozenFunds: float  # 订单冻结资金。


class DailyTradingStatistics(DosRecord):
    """单个标的单个交易日的成交统计，分开记录买开、卖开、卖平、买平。"""

    symbol: str  # 标的代码，与行情一致。
    tradeDate: date  # 交易日期，转换为 Python date。
    todayBuyOpenTradeVolume: int  # 当日买开成交数量。
    todayBuyOpenTradeValue: float  # 当日买开成交金额。
    todayBuyOpenAvgPrice: float  # 当日买开成交均价。
    todaySellOpenTradeVolume: int  # 当日卖开成交数量。
    todaySellOpenTradeValue: float  # 当日卖开成交金额。
    todaySellOpenAvgPrice: float  # 当日卖开成交均价。
    todaySellCloseTradeVolume: int  # 当日卖平成交数量。
    todaySellCloseTradeValue: float  # 当日卖平成交金额。
    todaySellCloseAvgPrice: float  # 当日卖平成交均价。
    todayBuyCloseTradeVolume: int  # 当日买平成交数量。
    todayBuyCloseTradeValue: float  # 当日买平成交金额。
    todayBuyCloseAvgPrice: float  # 当日买平成交均价。


class EngineContext(DosRecord):
    """DOS 引擎 context 的读取结果；与 Python 算法共享的 ctx 是不同对象。"""

    engine: int  # 服务端引擎句柄，仅在对应会话和引擎生命周期内有效。
    tradeDate: date | None  # 交易日期，转换为 Python date。
    tradeTime: datetime | None  # 插件记录的最新行情时间，可能与 Python 通知时序不同。
    barTime: datetime | None  # 插件合成的当前 Bar 时间，未生成有效值时可能为空或初始时间。
    orders: list[OrderReport]  # backtest.dos 累积、尚未被 Python 消费的订单回报。
    trades: list[TradeReport]  # backtest.dos 累积、尚未被 Python 消费的成交回报。
