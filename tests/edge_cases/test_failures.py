"""真实 FATAL 状态检查；连接与清理失败通过 SDK 故障注入验证。"""

from unittest.mock import Mock

import pytest
from algos.conftest import AlgoBacktest, EngineFactory, SharedContext

from backtest import DosVar


@pytest.mark.parametrize(
    "stage", ["connect_false", "connect_error", "load_script", "create_engine"]
)
def test_constructor_failure_closes_session(monkeypatch: pytest.MonkeyPatch, stage: str) -> None:
    session = Mock()
    session.connect.return_value = True
    if stage == "connect_false":
        session.connect.return_value = False
    elif stage == "connect_error":
        session.connect.side_effect = RuntimeError("connection failed")
    elif stage == "load_script":
        session.run.side_effect = RuntimeError("script failed")
    else:
        session.run.side_effect = [None, RuntimeError("create failed")]
    monkeypatch.setattr("backtest.backtest.ddb.session", lambda: session)
    message = "无法连接" if stage == "connect_false" else "failed"
    with pytest.raises((ConnectionError, RuntimeError), match=message):
        AlgoBacktest(
            "unused",
            0,
            "unused",
            "unused",
            engine_name="edge_failure",
            config={"startDate": "2026-06-10", "endDate": "2026-06-10"},
            ctx=SharedContext(),
        )
    session.close.assert_called_once_with()
    if stage.startswith("connect"):
        session.run.assert_not_called()


def test_close_still_disconnects_when_engine_drop_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    session = Mock()
    session.connect.return_value = True
    monkeypatch.setattr("backtest.backtest.ddb.session", lambda: session)
    engine = AlgoBacktest(
        "unused",
        0,
        "unused",
        "unused",
        engine_name="edge_close",
        config={"startDate": "2026-06-10", "endDate": "2026-06-10"},
        ctx=SharedContext(),
    )
    session.run.side_effect = RuntimeError("drop failed")
    with pytest.raises(RuntimeError, match="drop failed"):
        engine.close()
    script = session.run.call_args.args[0]
    assert (
        script.index("[DOS 回测] 开始回收引擎")
        < script.index("Backtest::dropBacktestEngine(engine)")
        < script.index("[DOS 回测] 引擎已回收")
    )
    session.close.assert_called_once_with()


def test_context_manager_preserves_body_exception_and_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = Mock()
    session.connect.return_value = True
    monkeypatch.setattr("backtest.backtest.ddb.session", lambda: session)
    with pytest.raises(ValueError, match="research failed"):
        with AlgoBacktest(
            "unused",
            0,
            "unused",
            "unused",
            engine_name="edge_context",
            config={"startDate": "2026-06-10", "endDate": "2026-06-10"},
            ctx=SharedContext(),
        ):
            raise ValueError("research failed")
    script = session.run.call_args.args[0]
    assert (
        script.index("[DOS 回测] 开始回收引擎")
        < script.index("Backtest::dropBacktestEngine(engine)")
        < script.index("[DOS 回测] 引擎已回收")
    )
    session.close.assert_called_once_with()


def test_fatal_engine_is_rejected_before_more_quotes_are_appended(
    make_engine: EngineFactory,
) -> None:
    engine = make_engine([], None)
    engine.session.upload({"engineName": "fatal_probe", "config": engine.config})
    engine.session.run("""
        def failSnapshot(mutable ctx, msg, indicator) { throw "edge snapshot failure" }
        Backtest::dropBacktestEngine(engine)
        engine = Backtest::createBacktester(engineName, config, {onSnapshot:failSnapshot}, false)
    """)
    engine.quote()
    with pytest.raises(Exception, match="edge snapshot failure"):
        engine.quote(3)
    before = engine.session.run("Backtest::getBacktestEngineStat(engine)")
    assert before.status[0] == "FATAL"
    with pytest.raises(Exception, match="回测引擎发生错误：.*edge snapshot failure"):
        engine.advance(DosVar("msg"))
    after = engine.session.run("Backtest::getBacktestEngineStat(engine)")
    assert after.snapshotTimestamp[0] == before.snapshotTimestamp[0]
    assert after.lastErrMsg[0] == before.lastErrMsg[0]
