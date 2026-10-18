"""回放日志：用假会话驱动时间线，不连接 DolphinDB。"""

import logging
from datetime import datetime

import numpy as np
import pytest
from pydantic import BaseModel

import backtest.backtest as module
from backtest import Algo, Backtest, Direction, DosVar, LimitOrder, MarketOrder


class Context(BaseModel):
    pass


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class Probe(Algo[None, Context]):
    def initialize(self) -> None:
        self.ready = True


class Slow(Algo[Clock, Context]):
    def initialize(self) -> None:
        self.params.now += 2.5


class FakeSession:
    def __init__(self, batches: list[list[str]]) -> None:
        self.batches = [np.array(batch, dtype="datetime64[ms]") for batch in batches]
        self.current = self.batches[0]
        self.scripts: list[str] = []

    def connect(self, *args: object) -> bool:
        return True

    def run(self, script: str) -> object:
        self.scripts.append(script)
        if script.startswith("sort(distinct("):
            self.current = self.batches.pop(0)
            return self.current
        if script.startswith("size("):
            return 3 * len(self.current)
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


def test_info_shows_first_moment_and_slow_initialization_only(session, monkeypatch, caplog):
    clock = Clock()
    monkeypatch.setattr(module, "perf_counter", clock)
    caplog.set_level(logging.INFO, logger="backtest")
    probe = Probe(None)
    replay(probe, Slow(clock))
    assert probe.ready
    assert messages(caplog, logging.INFO) == [
        "首个行情时点 2023-01-03 09:30:00，初始化 2 个 Algo",
        "Slow 初始化完成，用时 2.50 秒",
    ]
    # 批次行数只在 DEBUG 时查询，默认级别不增加会话往返。
    assert not any(script.startswith("size(") for script in session.scripts)


def test_debug_shows_engine_batches_and_replay_totals(session, caplog):
    caplog.set_level(logging.DEBUG, logger="backtest")
    replay(Probe(None))
    debug = messages(caplog, logging.DEBUG)
    assert debug[0] == "回测引擎 probe 已创建：2023-01-03 至 2023-01-04，Algo Probe"
    assert debug[1] == "开始回放 2023-01-03 至 2023-01-04，每批 1 天"
    assert debug[2].startswith("批次 2023-01-03：6 行行情、2 个时点，加载 ")
    assert any(message.startswith("Probe 初始化完成，用时 ") for message in debug)
    assert any(message.startswith("批次 2023-01-04：0 行行情、0 个时点") for message in debug)
    assert debug[-3] == "发送 END 消息，完成日终与结束处理"
    assert debug[-2].startswith("回放完成：2 个批次、2 个行情时点，用时 ")
    assert debug[-1] == "释放回测引擎与会话"
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
