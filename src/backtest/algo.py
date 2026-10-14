from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Generic, Sequence, TypeVar

from pydantic import BaseModel

from .models import OrderReport, TradeReport
from .types import DosVar

if TYPE_CHECKING:
    from .backtest import Backtest

__all__ = ["Algo"]

P = TypeVar("P", infer_variance=True)
C = TypeVar("C", bound=BaseModel, infer_variance=True)


class Algo(ABC, Generic[P, C]):
    """Python 侧算法基类；P 是算法参数类型，C 是共享上下文模型类型。

    回调返回 None，算法在回调内读取账户、下单或修改 ctx。
    逆序分发和批量回报是本封装的约定，不是原生 DOS 回调签名。
    """

    backtest: Backtest[C]  # 由 Backtest 构造函数绑定，可访问时间、查询和下单接口。
    ctx: C  # 与 Backtest.ctx 为同一对象，不自动同步到 DOS context。

    def __init__(self, params: P) -> None:
        """保存算法参数；backtest 和 ctx 会在接入回测时绑定。"""
        self.params = params

    def initialize(self) -> None:
        """首次非空行情进入插件前调用；此时已绑定 backtest 和 ctx。"""
        pass

    def on_snapshot(self, msg: DosVar) -> None:
        """当前时间戳行情已送入插件、已有回报已分发后调用一次。

        msg 是会话中的行情表名，可按需通过 backtest.session 查询。
        表变量会被后续行情覆盖，需要留存的数据应由算法自行保存。
        """
        pass

    def on_order(self, orders: Sequence[OrderReport]) -> None:
        """处理本批委托状态变化；可包含同一订单的多次变化。

        回调开始前，本批对应的 DOS 缓冲区已清空；新回报进入下一批。
        """
        pass

    def on_trade(self, trades: Sequence[TradeReport]) -> None:
        """处理本批实际成交；tradeQty 是当次量，totalVolume 是订单累计量。"""
        pass
