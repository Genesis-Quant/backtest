"""空批次、非法跨度、半开区间边界和乱序行情。"""

from unittest.mock import Mock

import numpy as np
import pytest
from algos.conftest import EngineFactory, RecordingAlgo

from backtest import Algo, DosVar


@pytest.mark.parametrize(
    "span", [np.timedelta64(0, "D"), np.timedelta64(-1, "s"), np.timedelta64("NaT", "ms")]
)
def test_invalid_batch_size(make_engine: EngineFactory, span: np.timedelta64) -> None:
    algo = RecordingAlgo("selection")
    engine = make_engine([algo], None)
    with pytest.raises(ValueError, match="行情加载跨度必须大于零"):
        engine.run(span)
    assert engine.ctx.events == []
    assert np.isnat(engine.time)


def test_empty_entire_range(make_engine: EngineFactory) -> None:
    engine = make_engine([RecordingAlgo("selection")], None)
    engine.session.run("delete from source")
    with pytest.raises(ValueError, match="行情为空"):
        engine.run()
    assert engine.ctx.events == []
    assert engine.get_orders() == engine.get_trades() == []


@pytest.mark.parametrize("kept_day, snapshots", [("2026.06.10", 2), ("2026.06.11", 1)])
def test_empty_leading_and_trailing_batches(
    make_engine: EngineFactory,
    kept_day: str,
    snapshots: int,
) -> None:
    algo = RecordingAlgo("selection")
    engine = make_engine([algo], None)
    engine.session.run(f"delete from source where date(timestamp) != {kept_day}")
    engine.run(np.timedelta64(12, "h"))
    assert engine.ctx.counts == {"initialize": 1, "snapshot": snapshots}
    assert len(algo.snapshots) == snapshots
    assert engine.session.run("Backtest::getBacktestEngineStat(engine).status[0]") == "END"


@pytest.mark.parametrize("outside", ["2026.06.09T23:59:59.999", "2026.06.11T00:00:00.000"])
def test_loader_must_respect_half_open_batch(
    make_engine: EngineFactory,
    monkeypatch: pytest.MonkeyPatch,
    outside: str,
) -> None:
    engine = make_engine([RecordingAlgo("selection")], None)
    engine.session.run(f"update source set timestamp={outside}")
    # 故意返回查询范围外的数据，验证 Backtest 本身会阻止推进。
    monkeypatch.setattr(engine, "load_messages", lambda start, end: DosVar("source"))
    with pytest.raises(ValueError, match="行情超出本批日期范围"):
        engine.run()
    assert engine.ctx.events == []
    assert np.isnat(engine.time)


def test_exact_batch_boundary_is_delivered_once_and_times_are_sorted(
    make_engine: EngineFactory,
) -> None:
    algo = RecordingAlgo("selection")
    engine = make_engine([algo], None)
    engine.session.run("""
        delete from source where date(timestamp)=2026.06.11
        update source set timestamp=iif(timestamp==2026.06.10T09:30:00.000,
            2026.06.10T09:59:59.999,2026.06.10T10:00:00.000)
        source = select * from source order by timestamp desc
    """)
    engine.run(np.timedelta64(10, "h"))
    assert algo.snapshots == [
        (np.datetime64("2026-06-10T09:59:59.999", "ms"), 2),
        (np.datetime64("2026-06-10T10:00:00.000", "ms"), 2),
    ]
    assert engine.ctx.counts == {"initialize": 1, "snapshot": 2}


def test_default_algo_callbacks_are_optional(make_engine: EngineFactory) -> None:
    algo = Algo("silent")
    engine = make_engine([algo], None)
    engine.run()
    engine.on_order([])
    engine.on_trade([])
    assert algo.ctx is engine.ctx
    assert engine.ctx.events == []
    assert engine.session.run("Backtest::getBacktestEngineStat(engine).status[0]") == "END"


def test_preloaded_messages_use_local_timeline_and_one_rpc_per_step(
    make_engine: EngineFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = make_engine([], None)
    times = np.asarray(
        ["2026-06-10T09:30:00", "2026-06-10T09:30:03", "2026-06-11T09:30:00"],
        dtype="datetime64[ms]",
    )
    monkeypatch.setattr(engine, "load_messages", lambda start, end: DosVar("source"))
    monkeypatch.setattr(
        engine,
        "load_timeline",
        lambda message, start, end: times[(times >= start) & (times < end)],
    )
    engine.trusted_event_cache = True
    run = Mock(wraps=engine.session.run)
    upload = Mock(wraps=engine.session.upload)
    monkeypatch.setattr(engine.session, "run", run)
    monkeypatch.setattr(engine.session, "upload", upload)
    engine.run(np.timedelta64(12, "h"))
    scripts = [call.args[0] for call in run.call_args_list]
    assert len(scripts) == 7  # 三次 step、END 准备/推进及两次结束清理。
    assert sum("where timestamp = timestamp(" in script for script in scripts) == 3
    assert not any("soloNextEvents" in script for script in scripts)
    upload.assert_not_called()


@pytest.mark.parametrize(
    "times",
    [
        ["2026-06-10T09:30:03", "2026-06-10T09:30:00"],
        ["2026-06-10T09:30:00", "2026-06-10T09:30:00"],
    ],
)
def test_local_timeline_must_be_strictly_increasing(
    make_engine: EngineFactory, monkeypatch: pytest.MonkeyPatch, times: list[str]
) -> None:
    engine = make_engine([], None)
    monkeypatch.setattr(
        engine,
        "load_timeline",
        lambda message, start, end: np.asarray(times, dtype="datetime64[ms]"),
    )
    with pytest.raises(ValueError, match="行情时间轴必须严格递增"):
        engine.run()
    assert np.isnat(engine.time)


def test_moment_is_emitted_only_after_advance_and_end_flushes_once(
    make_engine: EngineFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = make_engine([], None)
    emitted = []

    def emit(moment) -> None:
        if engine._finishing:
            assert engine.session.run("Backtest::getBacktestEngineStat(engine).status[0]") == "END"
        else:
            server_time = engine.get_context().tradeTime
            assert server_time is not None and np.datetime64(server_time, "ms") > moment.time
        emitted.append(moment)

    monkeypatch.setattr(engine, "emit_moment", emit)
    engine.quote()
    assert emitted == []
    engine.quote(3)
    assert [moment.time for moment in emitted] == [np.datetime64("2026-06-10T09:30:00", "ms")]
    engine.finish(DosVar("msg"))
    assert [moment.time for moment in emitted] == [
        np.datetime64("2026-06-10T09:30:00", "ms"),
        np.datetime64("2026-06-10T09:30:03", "ms"),
    ]
    assert np.isnat(engine.moment.time)


def test_snapshot_extension_reads_prices_equity_and_position_in_same_advance(
    make_engine: EngineFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = make_engine([], None)
    snapshots = []
    original_hook = engine.on_advance

    def advanced(snapshot) -> None:
        original_hook(snapshot)
        snapshots.append(snapshot)

    monkeypatch.setattr(
        engine,
        "snapshot_fields",
        lambda msg: {
            "prices": f"exec dict(string(symbol), lastPrice) from {msg}",
            "equity": "exec first(totalEquity) from Backtest::getTotalPortfolios(engine)",
            "positions": "(def(p) { return dict(p.columnNames(), p.values()) })(Backtest::getPosition(engine))",
        },
    )
    monkeypatch.setattr(engine, "on_advance", advanced)
    engine.quote()
    assert len(snapshots) == 1
    assert snapshots[0]["prices"] == {"000001.XSHE": 10.0, "600000.XSHG": 10.0}
    assert snapshots[0]["equity"] == 1_000_000
    assert snapshots[0]["quotes"] == 2
    assert snapshots[0]["eventsEmpty"]
    assert all(len(column) == 0 for column in snapshots[0]["positions"].values())
