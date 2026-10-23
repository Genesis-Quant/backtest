"""空批次、非法跨度、半开区间边界和乱序行情。"""

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
