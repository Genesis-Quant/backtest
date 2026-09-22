"""验证 ctx 引用、嵌套状态、跨批次延续以及不同回测之间的隔离。"""

import numpy as np
import pytest

from backtest import DosVar

from .conftest import EngineFactory, RecordingAlgo, SharedContext, buy


@pytest.mark.parametrize("batch_hours", [12, 24])
def test_context_survives_batches_and_events_finish_before_time_advances(
    make_engine: EngineFactory,
    batch_hours: int,
) -> None:
    class Buyer(RecordingAlgo):
        def on_snapshot(self, msg: DosVar) -> None:
            super().on_snapshot(msg)
            if len(self.snapshots) == 1:
                buy(self.backtest)
            else:
                assert self.ctx.counts["order"] == self.ctx.counts["trade"] == 2
                assert sum(row.tradeQty for batch in self.trades for row in batch) == 100

    ctx = SharedContext(value=10, counts={"existing": 7})
    selection, execution = Buyer("selection"), RecordingAlgo("execution")
    engine = make_engine([selection, execution], ctx)
    engine.run(np.timedelta64(batch_hours, "h"))

    assert engine.ctx is selection.ctx is execution.ctx is ctx
    assert ctx.events == [
        ("initialize", "selection", 11),
        ("initialize", "execution", 12),
        ("snapshot", "execution", 13),
        ("snapshot", "selection", 14),
        ("order", "execution", 15),
        ("order", "selection", 16),
        ("trade", "execution", 17),
        ("trade", "selection", 18),
        ("snapshot", "execution", 19),
        ("snapshot", "selection", 20),
        ("snapshot", "execution", 21),
        ("snapshot", "selection", 22),
    ]
    assert ctx.counts == {"existing": 7, "initialize": 2, "snapshot": 6, "order": 2, "trade": 2}
    assert ctx.value == 22
    assert (
        selection.snapshots
        == execution.snapshots
        == [
            (np.datetime64("2026-06-10T09:30:00", "ms"), 2),
            (np.datetime64("2026-06-10T09:30:03", "ms"), 2),
            (np.datetime64("2026-06-11T09:30:00", "ms"), 2),
        ]
    )
    assert engine.get_orders() == engine.get_trades() == []


def test_contexts_are_isolated_between_engines(make_engine: EngineFactory) -> None:
    first_algo, second_algo = RecordingAlgo("first"), RecordingAlgo("second")
    first, second = make_engine([first_algo], None), make_engine([second_algo], None)
    first.quote()
    first.process_events(DosVar("msg"))
    first.ctx.counts["custom"] = 42
    assert first.ctx is not second.ctx
    assert first.ctx.events is not second.ctx.events
    assert first.ctx.counts is not second.ctx.counts
    assert second.ctx == SharedContext()

    second.quote()
    second.process_events(DosVar("msg"))
    assert second.ctx.events == [("snapshot", "second", 1)]
    assert first.ctx.events == [("snapshot", "first", 1)]
    assert "custom" not in second.ctx.counts
