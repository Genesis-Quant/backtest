"""股票下单模型；to_dos 的字段顺序对应插件 submitOrder 协议。

示例中的 engine 为已创建并推进至示例委托时间的 Backtest 实例，
且已接收对应股票的行情；算法订单须开启 enableAlgoOrder。

https://docs.dolphindb.cn/zh/plugins/backtest/interface_description.html#submitorder
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, time
from typing import Any, ClassVar

import numpy as np
from pydantic import ConfigDict, Field, field_validator

from .enums import AlgoPriceType, Direction, OrderKind, PriceType, TimeInForce
from .types import DosRecord

__all__ = [
    "LimitOrder",
    "MarketOrder",
    "Order",
    "StopLimitOrder",
    "StopMarketOrder",
    "TWAPOrder",
    "VWAPOrder",
]


class Order(DosRecord, ABC):
    """单笔委托的共同字段；quantity 在股票回测中以股为单位。

    time 是传给插件的委托时间，按交易数据时区提供无时区时间，编码为
    TIMESTAMP。kind 控制外层订单类别，与元组内的委托价格类型分开。

    本类不可直接实例化，可用于统一接收各种具体订单::

        from backtest import Order

        def place_order(order: Order) -> int:
            return engine.submit_order(order)
    """

    model_config = ConfigDict(extra="forbid")
    kind: ClassVar[OrderKind] = OrderKind.NORMAL
    symbol: str  # 与行情中的标的代码一致，如 000333.SZ。
    time: datetime  # 委托时间，转换为毫秒精度。
    quantity: int = Field(gt=0)  # 股数，上传为 LONG。
    direction: Direction  # 买开、卖开、卖平或买平。
    label: str = ""  # 随回报返回的标签，也可用于按标签撤单。

    @abstractmethod
    def to_dos(self) -> tuple[Any, ...]:
        """按当前订单类型生成插件 msg 元组，并明确整数和时间的 DOS 类型。"""
        ...

    def algo_params(self) -> dict[str, Any] | None:
        """提供 TWAP/VWAP 专用参数；其他订单省略该参数。"""
        return None


class LimitOrder(Order):
    """普通限价委托；撮合、剩余量和交易限制由插件处理。

    以最高 10.1 元买入 100 股::

        from datetime import datetime
        from backtest import Direction, LimitOrder

        order_id = engine.submit_order(LimitOrder(
            symbol="000001.XSHE",
            time=datetime(2026, 6, 10, 9, 30),
            direction=Direction.BUY_OPEN,
            quantity=100,
            price=10.1,
        ))
    """

    price: float = Field(gt=0)

    def to_dos(self) -> tuple[Any, ...]:
        """编码为（标的、委托时间、价格类型、价格、数量、方向）。"""
        return (
            self.symbol,
            np.datetime64(self.time, "ms"),
            np.int32(PriceType.LIMIT),
            self.price,
            np.int64(self.quantity),
            np.int32(self.direction),
        )


class MarketOrder(Order):
    """普通市价委托；price_type 必须采用标的所在交易所的有效类型。

    编码中的价格填 0，实际成交价由对应市价规则和盘口决定。

    买入 600 股，扫最优五档，未成交部分撤销::

        from datetime import datetime
        from backtest import Direction, MarketOrder, PriceType

        order_id = engine.submit_order(MarketOrder(
            symbol="000001.XSHE",
            time=datetime(2026, 6, 10, 9, 30),
            direction=Direction.BUY_OPEN,
            quantity=600,
            price_type=PriceType.BEST_FIVE_IOC,
        ))
    """

    price_type: PriceType = PriceType.BEST_FIVE_IOC

    @field_validator("price_type")
    @classmethod
    def validate_market_price_type(cls, value: PriceType) -> PriceType:
        """禁止通过市价模型提交限价委托。"""
        if value == PriceType.LIMIT:
            raise ValueError("限价委托请使用 LimitOrder")
        return value

    def to_dos(self) -> tuple[Any, ...]:
        """编码为普通股票委托元组，保留指定的市价规则。"""
        return (
            self.symbol,
            np.datetime64(self.time, "ms"),
            np.int32(self.price_type),
            0.0,
            np.int64(self.quantity),
            np.int32(self.direction),
        )


class StopLimitOrder(Order):
    """带止盈止损条件的限价算法单；引擎须开启 enableAlgoOrder。

    元组同时携带进场委托、止损/止盈价格、滑点、有效性与到期时间。
    触发后的委托仍需撮合，不保证按触发价格成交。

    限价买入 100 股，设置 9.5 元止损、10.5 元止盈::

        from datetime import datetime
        from backtest import Direction, StopLimitOrder, TimeInForce

        order_id = engine.submit_order(StopLimitOrder(
            symbol="000001.XSHE",
            exchange="XSHE",
            time=datetime(2026, 6, 10, 9, 30),
            direction=Direction.BUY_OPEN,
            quantity=100,
            price=10.1,
            stop_loss=9.5,
            take_profit=10.5,
            expires_at=datetime(2026, 6, 10, 9, 31),
            validity=TimeInForce.DAY,
        ))
    """

    kind: ClassVar[OrderKind] = OrderKind.STOP_LIMIT
    price_type: ClassVar[PriceType] = PriceType.LIMIT
    exchange: str  # 交易所代码，与行情 symbolSource 一致。
    price: float = Field(gt=0)  # 委托价格。
    stop_loss: float = Field(ge=0)  # 止损价格，直接传入插件。
    take_profit: float = Field(ge=0)  # 止盈价格，直接传入插件。
    expires_at: datetime  # 委托到期时间，编码为 TIMESTAMP。
    slippage: float = Field(default=0.0, ge=0)  # 插件算法单的滑点参数。
    validity: TimeInForce = TimeInForce.DAY  # 当日有效、FOK 或 FAK。

    def to_dos(self) -> tuple[Any, ...]:
        """编码为止盈止损算法单的十二字段元组。"""
        return (
            self.symbol,
            self.exchange,
            np.datetime64(self.time, "ms"),
            np.int32(self.price_type),
            self.price,
            self.stop_loss,
            self.take_profit,
            np.int64(self.quantity),
            np.int32(self.direction),
            self.slippage,
            np.int32(self.validity),
            np.datetime64(self.expires_at, "ms"),
        )


class StopMarketOrder(StopLimitOrder):
    """市价止盈止损算法单，沿用十二字段协议。

    当前已验证插件版本仍校验委托 price，因此本模型保留该必填字段。

    市价买入 100 股，止盈或止损触发后按市价规则平仓::

        from datetime import datetime
        from backtest import Direction, StopMarketOrder, TimeInForce

        order_id = engine.submit_order(StopMarketOrder(
            symbol="000001.XSHE",
            exchange="XSHE",
            time=datetime(2026, 6, 10, 9, 30),
            direction=Direction.BUY_OPEN,
            quantity=100,
            price=10.1,
            stop_loss=9.5,
            take_profit=10.5,
            expires_at=datetime(2026, 6, 10, 9, 31),
            validity=TimeInForce.DAY,
        ))
    """

    kind: ClassVar[OrderKind] = OrderKind.STOP_MARKET
    price_type: ClassVar[PriceType] = PriceType.BEST_FIVE_IOC


class TWAPOrder(LimitOrder):
    """由插件按时间间隔拆单，适用于股票快照回测。

    start_time 缺省时采用引擎最新行情时间，指定时也不会早于该时间。
    计划切分次数为 floor((endTime - 实际开始时间) / interval) + 1；
    子单的实际成交仍取决于随后输入的行情和撮合设置。

    max_single_order_volume 对应文档的“每笔订单最大下单手数”；
    封装按 LONG 原值传递，不执行手与股的换算。

    从 09:30:10 到 09:30:50 每隔 20 秒拆单，买入共 300 股::

        from datetime import datetime, time
        from backtest import AlgoPriceType, Direction, TWAPOrder

        order_id = engine.submit_order(TWAPOrder(
            symbol="000001.XSHE",
            time=datetime(2026, 6, 10, 9, 30),
            direction=Direction.BUY_OPEN,
            quantity=300,
            price=10.5,
            start_time=time(9, 30, 10),
            end_time=time(9, 30, 50),
            interval=20,
            max_single_order_volume=100,
            order_price_type=AlgoPriceType.BEST_COUNTERPARTY,
        ))
    """

    kind: ClassVar[OrderKind] = OrderKind.TWAP
    end_time: time  # 当日拆单结束时间，提交时由会话内 DOS 转换为 SECOND。
    start_time: time | None = None  # 当日开始时间；None 时省略该键。
    interval: int = Field(default=20, gt=0)  # 切分间隔，单位秒，上传为 INT。
    max_single_order_volume: int | None = Field(
        default=None, gt=0
    )  # 子单数量上限，LONG；None 时由插件决定。
    order_price_type: AlgoPriceType = AlgoPriceType.LAST  # 子单取价依据。

    def algo_params(self) -> dict[str, Any]:
        """构造插件参数字典，提交时由会话内 DOS 将开始/结束时间转成 SECOND。"""
        params: dict[str, Any] = {
            "endTime": self.end_time.isoformat(),
            "interval": np.int32(self.interval),
            "orderPriceType": np.int32(self.order_price_type),
        }
        if self.start_time is not None:
            params["startTime"] = self.start_time.isoformat()
        if self.max_single_order_volume is not None:
            params["maxSingleOrderVolume"] = np.int64(self.max_single_order_volume)
        return params


class VWAPOrder(TWAPOrder):
    """按成交量分布拆单；先调用 set_trading_volume_dist 设置分布表。

    时间窗口和报价参数沿用 TWAPOrder，子单计划数量依据分布权重计算。

    设置三个时点的权重为 1:2:3，将 600 股计划分为 100、200、300 股::

        from datetime import datetime, time
        from backtest import AlgoPriceType, Direction, DosVar, VWAPOrder

        engine.session.run(
            'volumeDist = table(take("000001.XSHE", 3) as symbol, '
            '[09:30:10,09:30:30,09:30:50] as time, [1.,2.,3.] as value)'
        )
        engine.set_trading_volume_dist(DosVar("volumeDist"))
        order_id = engine.submit_order(VWAPOrder(
            symbol="000001.XSHE",
            time=datetime(2026, 6, 10, 9, 30),
            direction=Direction.BUY_OPEN,
            quantity=600,
            price=10.5,
            start_time=time(9, 30, 10),
            end_time=time(9, 30, 50),
            interval=20,
            max_single_order_volume=600,
            order_price_type=AlgoPriceType.BEST_COUNTERPARTY,
        ))
    """

    kind: ClassVar[OrderKind] = OrderKind.VWAP
