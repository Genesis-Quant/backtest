"""会话内 DOS 定义及插件直调用契约；使用 SDK 替身，不连接 DolphinDB。"""

import ast
import logging
import re
from collections.abc import Iterator
from datetime import datetime, time
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from pydantic import BaseModel

import backtest.backtest as module
from backtest import (
    Backtest,
    Direction,
    DosVar,
    LimitOrder,
    MarketOrder,
    Order,
    OrderReport,
    OrderStatus,
    StopLimitOrder,
    StopMarketOrder,
    TradeReport,
    TWAPOrder,
    VWAPOrder,
)

MOMENT = datetime(2026, 6, 10, 9, 30)


class Context(BaseModel):
    pass


class SessionBacktest(Backtest[Context]):
    def load_messages(self, start: np.datetime64, end: np.datetime64) -> DosVar:
        return DosVar("messages")


@pytest.fixture
def engine(monkeypatch: pytest.MonkeyPatch) -> Iterator[SessionBacktest]:
    session = Mock()
    session.connect.return_value = True
    session.run.return_value = None
    monkeypatch.setattr(module.ddb, "session", lambda: session)
    instance = SessionBacktest(
        "unused", 0, "unused", "unused", engine_name="session_dos",
        config={"startDate": "2026-06-10", "endDate": "2026-06-10"}, ctx=Context(),
    )
    yield instance
    session.run.side_effect = None
    instance.close()


@pytest.mark.parametrize("failure", ["session", "connect_false", "connect_error", "callbacks", "upload", "engine"])
def test_constructor_failure_closes_created_session_without_logging_secrets(monkeypatch, caplog, failure):
    caplog.set_level(logging.DEBUG, logger="backtest")
    session = Mock()
    session.connect.return_value = failure != "connect_false"
    error = RuntimeError("password-sentinel config-sentinel")
    if failure == "connect_error":
        session.connect.side_effect = error
    elif failure == "callbacks":
        session.run.side_effect = error
    elif failure == "upload":
        session.upload.side_effect = error
    elif failure == "engine":
        session.run.side_effect = [None, error]
    factory = Mock(return_value=session, side_effect=error if failure == "session" else None)
    monkeypatch.setattr(module.ddb, "session", factory)
    expected = ConnectionError if failure == "connect_false" else RuntimeError
    with pytest.raises(expected):
        SessionBacktest("ddb.example", 8900, "user-sentinel", "password-sentinel", engine_name="failure",
                        config={"startDate": "2026-06-10", "endDate": "2026-06-10"}, ctx=Context())
    assert session.close.call_count == int(failure != "session")
    if failure == "session":
        session.connect.assert_not_called()
    else:
        session.connect.assert_called_once_with("ddb.example", 8900, "user-sentinel", "password-sentinel")
    assert "password-sentinel" not in caplog.text
    assert "config-sentinel" not in caplog.text
    assert "user-sentinel" not in caplog.text


@pytest.mark.parametrize("failure", ["drop", "close", "both"])
def test_close_always_closes_session_and_preserves_native_failure_order(engine, failure):
    session = engine.session
    drop_error = ValueError("drop failed")
    close_error = OSError("close failed")
    events = []
    def drop(script):
        assert script == "Backtest::dropBacktestEngine(engine)"
        events.append("drop")
        if failure in ("drop", "both"):
            raise drop_error
    def close():
        events.append("close")
        if failure in ("close", "both"):
            raise close_error
    session.run.side_effect = drop
    session.close.side_effect = close
    try:
        expected = drop_error if failure == "drop" else close_error
        with pytest.raises(type(expected)) as caught:
            engine.close()
        assert caught.value is expected
        assert events == ["drop", "close"]
        if failure == "both":
            assert caught.value.__context__ is drop_error
    finally:
        session.run.side_effect = None
        session.close.side_effect = None


def test_no_external_dos_resource() -> None:
    path = Path(module.__file__)
    source = path.read_text(encoding="utf-8")
    assert not list(path.parent.rglob("*.dos"))
    assert "importlib.resources" not in source
    assert "backtest.dos" not in source


def test_only_reused_dos_functions_are_defined_in_initialization(engine: SessionBacktest) -> None:
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    definitions = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
        and re.search(r"\bdef\s+\w+\(", node.value)
    ]
    assert len(definitions) == 1
    definition = definitions[0]
    initializer = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    assert any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "run" and node.args == [definition]
        for node in ast.walk(initializer)
    )
    script = engine.session.run.call_args_list[0].args[0]
    assert script == definition.value
    assert re.findall(r"\bdef\s+(\w+)\(", script) == [
        "initialize", "beforeTrading", "afterTrading", "finalize", "onSnapshot",
        "onOrder", "onTrade", "getOrders", "getTrades",
    ]
    for singular, plural in [("order", "orders"), ("trade", "trades")]:
        assert f'ctx["{plural}"] = array(ANY, 0)' in script
        assert (
            f'ctx["{plural}"].append!(dict({singular}.keys(), each(copy, {singular}.values())))'
        ) in script
        body = re.search(rf"def get{plural.title()}\(engine, clear=false\) {{(.*?)}}", script, re.S)
        assert body is not None
        assert " ".join(body[1].split()) == (
            f'ctx = Backtest::getContextDict(engine) {plural} = ctx["{plural}"] '
            f'if (clear) ctx["{plural}"] = array(ANY, 0) return {plural}'
        )
    assert " ".join(engine.session.run.call_args_list[1].args[0].split()) == (
        "engine = Backtest::createBacktestEngine(engineName,config,,initialize,"
        "beforeTrading,,onSnapshot,onOrder,onTrade,afterTrading,finalize)"
    )


@pytest.mark.parametrize(
    "method, plugin, expected",
    [
        ("get_position", "getPosition", {}),
        ("get_trade_details", "getTradeDetails", []),
        ("get_daily_position", "getDailyPosition", []),
        ("get_daily_total_portfolios", "getDailyTotalPortfolios", []),
        ("get_daily_trading_statistics", "getDailyTradingStatistics", []),
    ],
)
def test_result_queries_call_native_plugin(engine: SessionBacktest, method, plugin, expected):
    engine.session.run.reset_mock()
    engine.session.run.return_value = {}
    assert getattr(engine, method)() == expected
    engine.session.run.assert_called_once_with(
        f"result = Backtest::{plugin}(engine); dict(result.columnNames(), result.values())"
    )


def test_context_query_calls_native_plugin(engine: SessionBacktest) -> None:
    engine.session.run.return_value = {
        "engine": 1, "tradeDate": None, "tradeTime": None, "barTime": None,
        "orders": [], "trades": [],
    }
    assert engine.get_context().engine == 1
    engine.session.run.assert_called_with("Backtest::getContextDict(engine)")


def test_volume_distribution_calls_native_plugin(engine: SessionBacktest) -> None:
    engine.set_trading_volume_dist(DosVar("volumeDist"))
    engine.session.run.assert_called_with("Backtest::setTradingVolumeDist(engine, volumeDist)")


@pytest.mark.parametrize("orders", [None, [], [7, 8]])
def test_cancel_calls_native_plugin_with_long_ids(engine: SessionBacktest, orders) -> None:
    engine.cancel_order(symbol="000001.XSHE", orders=orders, label="rebalance")
    values = engine.session.upload.call_args.args[0]
    assert values["cancelSymbol"] == "000001.XSHE"
    assert values["cancelLabel"] == "rebalance"
    np.testing.assert_array_equal(values["cancelIds"], orders or [])
    assert values["cancelIds"].dtype == np.int64
    engine.session.run.assert_called_with(
        "Backtest::cancelOrder(engine, cancelSymbol, cancelIds, cancelLabel)"
    )


def test_advance_checks_fatal_before_appending_directly(engine: SessionBacktest) -> None:
    engine.advance(DosVar("quotes"))
    assert " ".join(engine.session.run.call_args.args[0].split()) == (
        "stat = Backtest::getBacktestEngineStat(engine) "
        'if (stat.status[0]=="FATAL") throw "Backtest engine is FATAL: " + stat.lastErrMsg[0] '
        "Backtest::appendQuotationMsg(engine, quotes)"
    )


@pytest.mark.parametrize("order_type", [LimitOrder, MarketOrder, StopLimitOrder, StopMarketOrder])
def test_non_sliced_orders_call_native_submit_order(
    engine: SessionBacktest, order_type: type[Order],
) -> None:
    values = dict(
        symbol="000001.XSHE", time=MOMENT, quantity=100,
        direction=Direction.BUY_OPEN, label="rebalance",
    )
    if issubclass(order_type, (LimitOrder, StopLimitOrder)):
        values["price"] = 10.0
    if issubclass(order_type, StopLimitOrder):
        values.update(exchange="XSHE", stop_loss=9.5, take_profit=10.5, expires_at=MOMENT)
    order = order_type(**values)
    engine.session.run.return_value = np.int64(7)
    engine.session.upload.reset_mock()
    order_id = engine.submit_order(order)
    assert type(order_id) is int and order_id == 7
    engine.session.run.assert_called_with(
        "Backtest::submitOrder(engine, orderMsg, orderLabel, orderKind)[0]"
    )
    engine.session.upload.assert_called_once_with({
        "orderMsg": order.to_dos(), "orderKind": np.int32(order.kind), "orderLabel": order.label,
    })
    assert isinstance(engine.session.upload.call_args.args[0]["orderKind"], np.int32)


@pytest.mark.parametrize("order_type", [TWAPOrder, VWAPOrder])
@pytest.mark.parametrize("start_time", [None, time(9, 30, 10)])
def test_sliced_orders_convert_copied_parameters_at_call_site(
    engine: SessionBacktest, order_type: type[TWAPOrder], start_time: time | None,
) -> None:
    order = order_type(
        symbol="000001.XSHE", time=MOMENT, quantity=300, price=10.0,
        direction=Direction.BUY_OPEN, label="sliced", start_time=start_time,
        end_time=time(9, 31), interval=20, max_single_order_volume=100,
    )
    engine.session.run.return_value = np.int64(8)
    engine.session.upload.reset_mock()
    assert engine.submit_order(order) == 8
    assert " ".join(engine.session.run.call_args.args[0].split()) == (
        'params = copy(algoOrderParam) params["endTime"] = second(params["endTime"]) '
        'if ("startTime" in params.keys()) params["startTime"] = second(params["startTime"]) '
        "Backtest::submitOrder(engine, orderMsg, orderLabel, orderKind,,params)[0]"
    )
    uploads = engine.session.upload.call_args_list
    assert len(uploads) == 2
    assert uploads[0].args[0]["orderMsg"] == order.to_dos()
    assert uploads[0].args[0]["orderLabel"] == "sliced"
    assert uploads[0].args[0]["orderKind"] == order.kind
    assert isinstance(uploads[0].args[0]["orderKind"], np.int32)
    params = uploads[1].args[0]["algoOrderParam"]
    assert params == order.algo_params()
    assert params["endTime"] == "09:31:00"
    assert ("startTime" in params) == (start_time is not None)
    if start_time is not None:
        assert params["startTime"] == "09:30:10"
    assert isinstance(params["interval"], np.int32)
    assert isinstance(params["orderPriceType"], np.int32)
    assert isinstance(params["maxSingleOrderVolume"], np.int64)


def test_event_buffers_are_cleared_before_notification_and_reused(
    engine: SessionBacktest, monkeypatch: pytest.MonkeyPatch,
) -> None:
    order = OrderReport(
        orderId=7, symbol="000001.XSHE", direction=Direction.BUY_OPEN,
        timestamp=MOMENT, updateTime=MOMENT, price=10.0, qty=100,
        tradeQty=100, tradeValue=1000.0, status=OrderStatus.FILLED, label="rebalance",
    )
    trade = TradeReport(
        orderId=7, symbol="000001.XSHE", direction=Direction.BUY_OPEN,
        tradeTime=MOMENT, orderPrice=10.0, tradePrice=10.0, tradeQty=100,
        tradeValue=1000.0, totalVolume=100, totalValue=1000.0, totalFee=0.0,
        label="rebalance",
    )
    buffers = {"orders": [order.model_dump()], "trades": [trade.model_dump()]}

    def run(script: str):
        name, clear = {
            "getOrders(engine)": ("orders", False),
            "getTrades(engine)": ("trades", False),
            "getOrders(engine, true)": ("orders", True),
            "getTrades(engine, true)": ("trades", True),
        }[script]
        batch = buffers[name]
        if clear:
            buffers[name] = []
        return batch

    engine.session.run.side_effect = run
    events = []

    def on_order(reports):
        assert engine.get_orders() == []
        events.append(("order", [report.orderId for report in reports]))
        if len(events) == 1:
            buffers["orders"].append(order.model_copy(update={"orderId": 8}).model_dump())

    def on_trade(reports):
        assert engine.get_trades() == []
        events.append(("trade", [report.orderId for report in reports]))

    monkeypatch.setattr(engine, "on_order", on_order)
    monkeypatch.setattr(engine, "on_trade", on_trade)
    monkeypatch.setattr(engine, "on_snapshot", lambda msg: events.append(("snapshot", msg)))
    assert engine.get_orders() == engine.get_orders() == [order]
    assert engine.get_trades() == engine.get_trades() == [trade]
    engine.process_events(DosVar("quotes"))
    assert events == [("order", [7]), ("order", [8]), ("trade", [7]), ("snapshot", "quotes")]
    assert engine.get_orders() == engine.get_trades() == []
    engine.process_events()
    assert len(events) == 4
