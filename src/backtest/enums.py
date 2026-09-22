"""插件订单协议编码，股票市价类型按沪深交易所区分。

https://docs.dolphindb.cn/zh/plugins/backtest/interface_description.html#submitorder
https://docs.dolphindb.cn/zh/plugins/backtest.html
"""

from enum import IntEnum

__all__ = [
    "AlgoPriceType",
    "Direction",
    "OrderKind",
    "OrderStatus",
    "PriceType",
    "TimeInForce",
]


class Direction(IntEnum):
    """订单元组及回报中的开平方向。"""

    BUY_OPEN = 1  # 买入开多仓。
    SELL_OPEN = 2  # 卖出开空仓。
    SELL_CLOSE = 3  # 卖出平多仓。
    BUY_CLOSE = 4  # 买入平空仓。


class OrderKind(IntEnum):
    """submitOrder 外层 orderType，本封装支持的类别。"""

    NORMAL = 0  # 普通委托。
    STOP_LIMIT = 5  # 限价止盈止损算法单。
    STOP_MARKET = 6  # 市价止盈止损算法单。
    TWAP = 10  # 按时间间隔拆单。
    VWAP = 11  # 按成交量分布拆单。


class PriceType(IntEnum):
    """股票 msg 元组内的委托类型，与外层 OrderKind 编码互相独立。"""

    BEST_FIVE_IOC = 0  # 最优五档成交，剩余撤销。
    SH_BEST_FIVE_TO_LIMIT = 1  # 上海：最优五档成交，剩余转限价。
    # SZ_IOC = 1  # 暂停使用：当前 Backtest 股票快照路径将余量转为限价单。
    BEST_OWN = 2  # 本方最优价。
    BEST_COUNTERPARTY = 3  # 对手方最优价。
    # SZ_FOK = 4  # 暂停使用：当前 Backtest 股票快照路径报订单类型错误。 https://ask.dolphindb.cn/question/4822
    LIMIT = 5  # 限价委托。


class TimeInForce(IntEnum):
    """算法订单元组中的委托有效性。"""

    DAY = 0  # 当日有效。
    FOK = 1  # 立即全部成交，否则撤销。
    FAK = 2  # 立即成交可成交部分，其余撤销。


class AlgoPriceType(IntEnum):
    """TWAP/VWAP algoOrderParam.orderPriceType 的子单取价方式。"""

    LAST = 0  # 最新成交价。
    BEST_OWN = 1  # 本方第一档报价。
    BEST_COUNTERPARTY = 2  # 对手方第一档报价。


class OrderStatus(IntEnum):
    """订单回报和交易明细中的状态；撤单拒绝不表示原委托已结束。"""

    UNFILLED = -3  # 未成交记录，见 getTradeDetails。
    CANCEL_REJECTED = -2  # 撤单请求被拒绝。
    REJECTED = -1  # 委托审批拒绝。
    PARTIALLY_FILLED = 0  # 部分成交。
    FILLED = 1  # 全部成交。
    CANCELED = 2  # 撤单成功。
    REPORTED = 4  # 已报。
