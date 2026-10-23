"""回放日志和边界；不连接 DolphinDB。"""

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
        self.closed = 0

    def connect(self, *args: object) -> bool:
        return True

    def run(self, script: str) -> object:
        self.scripts.append(script)
        if script.startswith("sort(distinct("):
            return self.batches.pop(0)
        if "orderId = Backtest::submitOrder(" in script:
            return 7
        if "getOrders(engine, true)" in script or "getTrades(engine, true)" in script:
            return []
        return None

    def upload(self, values: dict) -> None:
        pass

    def close(self) -> None:
        self.closed += 1


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
        "localhost",
        8848,
        "user",
        "secret",
        engine_name="probe",
        config={"startDate": "2023-01-03", "endDate": "2023-01-04"},
        ctx=Context(),
        algos=algos,
    ) as engine:
        engine.run()


def messages(caplog, level: int) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.levelno == level]


def test_info_reports_engine_without_secrets(session, caplog):
    caplog.set_level(logging.INFO, logger="backtest")
    probe = Probe(None)
    replay(probe)
    assert probe.ready
    assert messages(caplog, logging.INFO) == [
        "DolphinDB: localhost:8848",
        "回测引擎 probe：2023-01-03 至 2023-01-04",
    ]
    assert "secret" not in caplog.text
    assert session.closed == 1


def test_data_exchange_hooks_and_end_cleanup(session, caplog):
    caplog.set_level(logging.DEBUG, logger="backtest")
    replay(Probe(None))
    debug = messages(caplog, logging.DEBUG)
    assert "上传：回测引擎配置" in debug
    assert "读取：本批行情时点" in debug
    assert "读取：订单回报" in debug
    assert "读取：成交回报" in debug
    ending = next(script for script in session.scripts if 'symbol = "END"' in script)
    assert (
        ending.index("[DOS 回测] 开始准备结束消息")
        < ending.index("endMsg = select")
        < ending.index("[DOS 回测] 结束消息已准备")
    )
    assert any('undef("endMsg", VAR)' in script for script in session.scripts)
    assert "secret" not in caplog.text


def test_order_submission_uses_exchange_hook_and_cleans_parameters(session, caplog):
    caplog.set_level(logging.DEBUG, logger="backtest")
    moment = datetime(2023, 1, 3, 9, 30)
    with Replay(
        "localhost",
        8848,
        "user",
        "secret",
        engine_name="probe",
        config={"startDate": "2023-01-03", "endDate": "2023-01-04"},
        ctx=Context(),
    ) as engine:
        assert (
            engine.submit_order(
                LimitOrder(
                    symbol="600000.XSHG",
                    time=moment,
                    quantity=200,
                    direction=Direction.BUY_OPEN,
                    price=10.5,
                    label="rebalance",
                )
            )
            == 7
        )
        assert (
            engine.submit_order(
                MarketOrder(
                    symbol="000001.XSHE",
                    time=moment,
                    quantity=100,
                    direction=Direction.SELL_CLOSE,
                )
            )
            == 7
        )
    debug = messages(caplog, logging.DEBUG)
    assert debug.count("上传：委托参数") == 2
    assert debug.count("读取：委托编号") == 2
    assert sum('undef("orderMsg", VAR)' in script for script in session.scripts) == 2


def test_replay_rejects_out_of_batch_quotes(monkeypatch):
    session = FakeSession([["2023-01-04T09:30"]])
    monkeypatch.setattr(module.ddb, "session", lambda: session)
    with Replay(
        "localhost",
        8848,
        "user",
        "secret",
        engine_name="bounds",
        config={"startDate": "2023-01-03", "endDate": "2023-01-03"},
        ctx=Context(),
    ) as engine:
        with pytest.raises(ValueError, match="行情超出本批日期范围"):
            engine.run()
    assert session.closed == 1
