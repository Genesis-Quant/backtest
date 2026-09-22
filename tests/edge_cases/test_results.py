"""空账户、回报读取副作用与结束回测后的持仓、净值、成交统计。"""

from datetime import date

import pytest
from algos.conftest import EngineFactory, RecordingAlgo, buy

from backtest import DosVar, OrderStatus


def test_empty_account_queries(make_engine: EngineFactory) -> None:
    engine = make_engine([], None)
    engine.quote()
    assert engine.get_position() == {}
    assert engine.get_orders() == engine.get_trades() == engine.get_trade_details() == []
    assert engine.get_daily_position() == []
    context = engine.get_context()
    assert context.tradeDate == date(2026, 6, 10)
    assert context.orders == context.trades == []


def test_reading_reports_and_context_does_not_consume_or_modify_server_buffer(
    make_engine: EngineFactory,
) -> None:
    engine = make_engine([], None)
    engine.quote()
    order_id = buy(engine)
    orders, trades = engine.get_orders(), engine.get_trades()
    context = engine.get_context()
    assert orders and trades
    assert context.orders == orders and context.trades == trades
    context.orders.clear()
    context.trades.clear()
    orders[-1].status = OrderStatus.REJECTED
    trades[0].tradeQty = 999
    assert engine.get_orders()[-1].status == OrderStatus.FILLED
    assert engine.get_trades()[0].tradeQty == 100
    assert engine.get_trades()[0].orderId == order_id
    engine.process_events()
    assert engine.get_orders() == engine.get_trades() == []
    assert engine.get_trade_details()  # 消费通知不会删除历史记录。


def test_daily_results_include_last_day_after_finish(make_engine: EngineFactory) -> None:
    class Buyer(RecordingAlgo):
        def on_snapshot(self, msg: DosVar) -> None:
            super().on_snapshot(msg)
            if len(self.snapshots) == 1:
                buy(self.backtest)

    engine = make_engine([Buyer("selection")], None)
    engine.run()
    positions = engine.get_daily_position()
    portfolios = engine.get_daily_total_portfolios()
    statistics = engine.get_daily_trading_statistics()
    days = [date(2026, 6, 10), date(2026, 6, 11)]
    assert [row.tradeDate for row in portfolios] == days
    held = [row for row in positions if row.symbol == "000001.XSHE"]
    assert [row.tradeDate for row in held] == days
    assert all(row.longPosition == 100 and row.shortPosition == 0 for row in held)
    assert portfolios[-1].cash == pytest.approx(998990.0)
    assert portfolios[-1].totalMarketValue == pytest.approx(1000.0)
    assert portfolios[-1].totalEquity == pytest.approx(999990.0)
    assert portfolios[-1].netValue == pytest.approx(0.99999)
    assert sum(row.todayBuyOpenTradeVolume for row in statistics) == 100
    assert sum(row.todayBuyOpenTradeValue for row in statistics) == pytest.approx(1010.0)
    assert sum(row.todaySellCloseTradeVolume for row in statistics) == 0
