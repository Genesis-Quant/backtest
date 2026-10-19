"""回放日志分级，不连接 DolphinDB。"""

import logging
from datetime import datetime

import numpy as np
import pytest
from pydantic import BaseModel

import backtest.backtest as module
from backtest import Algo, Backtest, Direction, DosVar, LimitOrder, MarketOrder


class Context(BaseModel):
    pass


class Probe(Algo[None, Context]):
    def initialize(self) -> None:
        self.ready = True


class FakeSession:
    def __init__(self, batches: list[list[str]]) -> None:
        self.batches = [np.array(batch, dtype="datetime64[ms]") for batch in batches]
        self.scripts: list[str] = []

    def connect(self, *args: object) -> bool:
        return True

    def run(self, script: str) -> object:
        self.scripts.append(script)
        if script.startswith("sort(distinct("):
            return self.batches.pop(0)
        if script.startswith("Backtest::submitOrder("):
            return 7
        if "getOrders(engine, true)" in script or "getTrades(engine, true)" in script:
            return []
        return None

    def upload(self, values: dict) -> None:
        pass

    def close(self) -> None:
        pass


class Replay(Backtest[Context]):
    def load_messages(self, start: np.datetime64, end: np.datetime64) -> DosVar:
        return DosVar("messages")


@pytest.fixture
def session(monkeypatch):
    session = FakeSession([["2023-01-03T09:30", "2023-01-03T15:00"], []])
    monkeypatch.setattr(module.ddb, "session", lambda: session)
    return session


def replay(*algos: Algo) -> None:
    with Replay(
        "localhost", 8848, "user", "secret", engine_name="probe",
        config={"startDate": "2023-01-03", "endDate": "2023-01-04"}, ctx=Context(), algos=algos,
    ) as engine:
        engine.run()


def messages(caplog, level: int) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.levelno == level]


def test_info_reports_connection_engine_and_first_quote(session, caplog):
    caplog.set_level(logging.INFO, logger="backtest")
    probe = Probe(None)
    replay(probe)
    assert probe.ready
    assert messages(caplog, logging.INFO) == [
        "DolphinDB: localhost:8848",
        "回测引擎 probe：2023-01-03 至 2023-01-04，Algo=Probe",
        "首个行情时点 2023-01-03 09:30:00，初始化 1 个 Algo",
    ]
    assert "secret" not in caplog.text
    assert not any(script.startswith("size(") for script in session.scripts)


def test_debug_reports_callbacks_batches_and_resource_cleanup(session, caplog):
    caplog.set_level(logging.DEBUG, logger="backtest")
    replay(Probe(None))
    debug = messages(caplog, logging.DEBUG)
    assert "定义订单和成交回调：engine=probe" in debug
    assert "创建回测引擎：probe" in debug
    assert "批次 2023-01-03：2 个行情时点" in debug
    assert "批次 2023-01-04：0 个行情时点" in debug
    assert "初始化 Algo：Probe" in debug
    assert debug[-3:] == [
        "发送 END 消息，完成日终与结束处理",
        "回放完成：2 个批次、2 个行情时点",
        "释放回测引擎和连接",
    ]
    assert "secret" not in caplog.text


def test_order_submission_is_one_compact_debug_line(session, caplog):
    caplog.set_level(logging.DEBUG, logger="backtest")
    moment = datetime(2023, 1, 3, 9, 30)
    with Replay(
        "localhost", 8848, "user", "secret", engine_name="probe",
        config={"startDate": "2023-01-03", "endDate": "2023-01-04"}, ctx=Context(),
    ) as engine:
        engine.submit_order(LimitOrder(
            symbol="600000.XSHG", time=moment, quantity=200, direction=Direction.BUY_OPEN,
            price=10.5, label="rebalance",
        ))
        engine.submit_order(MarketOrder(
            symbol="000001.XSHE", time=moment, quantity=100, direction=Direction.SELL_CLOSE,
        ))
    debug = messages(caplog, logging.DEBUG)
    assert "提交订单 #7：LimitOrder BUY_OPEN 600000.XSHG 200 股 @ 10.5，标签 rebalance" in debug
    assert "提交订单 #7：MarketOrder SELL_CLOSE 000001.XSHE 100 股 @ 市价" in debug


def test_window_keeps_intraday_bounds_half_open():
    start = np.datetime64("2023-01-03T09:30", "ms")
    assert module._window(start, start + np.timedelta64(1, "h")) == (
        "[2023-01-03 09:30:00, 2023-01-03 10:30:00)"
    )
    day = np.datetime64("2023-01-03", "ms")
    assert module._window(day, day + np.timedelta64(30, "D")) == "2023-01-03 至 2023-02-01"
