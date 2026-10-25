from collections.abc import Sequence

import numpy as np
import pytest

from backtest import Direction, DosVar, LimitOrder, TradeReport

from .conftest import START, EngineFactory, RecordingAlgo


def orders():
    return [
        LimitOrder(
            symbol=symbol, time=START, quantity=100, price=10.1, direction=Direction.BUY_OPEN
        )
        for symbol in ("000001.XSHE", "600000.XSHG")
    ]


def test_order_batch_is_logged_once_not_per_order(make_engine: EngineFactory, capsys):
    engine = make_engine([], None)
    engine.quote()
    identifiers = engine.submit_orders(orders())
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1
    assert "[批量委托]" in lines[0] and "提交 2" in lines[0]
    engine.process_events()
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2
    assert "[委托回调]" in lines[0] and "委托回报 4" in lines[0]
    assert "[成交回调]" in lines[1] and "成交 2" in lines[1]
    assert len(identifiers) == 2
    engine.process_events()
    engine.quote(3)
    engine.process_events(DosVar("msg"))
    assert capsys.readouterr().out == ""


def test_callback_batches_drain_same_time_before_next_day(make_engine: EngineFactory, monkeypatch):
    class Chained(RecordingAlgo):
        submitted = False
        reacted = False

        def on_snapshot(self, msg: DosVar):
            super().on_snapshot(msg)
            if not self.submitted:
                self.submitted = True
                self.backtest.submit_orders(orders())

        def on_trade(self, trades: Sequence[TradeReport]):
            super().on_trade(trades)
            if not self.reacted:
                self.reacted = True
                self.backtest.submit_orders(
                    [
                        order.model_copy(update={"direction": Direction.SELL_CLOSE, "price": 9.9})
                        for order in orders()
                    ]
                )

    algo = Chained("chain")
    engine = make_engine([algo], None)
    batches = []
    monkeypatch.setattr(engine, "emit_batch", batches.append)
    engine.run()
    assert [batch.event for batch in batches] == ["snapshot", "order", "trade", "order", "trade"]
    assert [batch.sequence for batch in batches] == [1, 2, 3, 4, 5]
    assert all(batch.time == np.datetime64(START, "ms") for batch in batches)
    assert [batch.orders for batch in batches] == [2, 0, 2, 0, 0]
    assert [batch.trades for batch in batches] == [0, 0, 2, 0, 2]
    assert [event for event, _, _ in engine.ctx.events] == [
        "initialize",
        "snapshot",
        "order",
        "trade",
        "order",
        "trade",
        "snapshot",
        "snapshot",
    ]
    assert engine.get_orders() == engine.get_trades() == []


def test_raw_session_side_effects_log_as_returned_event_batches(
    make_engine: EngineFactory, monkeypatch
):
    class Raw(RecordingAlgo):
        submitted = False

        def on_snapshot(self, msg: DosVar):
            if not self.submitted:
                self.submitted = True
                self.backtest.session.run("""
                    Backtest::submitOrder(engine,
                        ("600000.XSHG",2026.06.10T09:30:00.000,5,10.1,100l,1))
                """)

    engine = make_engine([Raw("raw")], None)
    batches = []
    monkeypatch.setattr(engine, "emit_batch", batches.append)
    engine.run()
    assert [batch.event for batch in batches] == ["order", "trade"]
    assert sum(batch.trades for batch in batches) == 1
    assert all(batch.time == np.datetime64(START, "ms") for batch in batches)


def test_nested_drain_neither_reorders_logs_nor_double_counts(
    make_engine: EngineFactory, monkeypatch
):
    class Reentrant(RecordingAlgo):
        submitted = False

        def on_snapshot(self, msg: DosVar):
            if not self.submitted:
                self.submitted = True
                self.backtest.submit_orders(orders())
                self.backtest.process_events()

    engine = make_engine([Reentrant("nested")], None)
    batches = []
    monkeypatch.setattr(engine, "emit_batch", batches.append)
    engine.run()
    assert [batch.event for batch in batches] == ["snapshot", "order", "trade"]
    assert sum(batch.orders for batch in batches) == 2
    assert sum(batch.order_reports for batch in batches) == 4
    assert sum(batch.trades for batch in batches) == 2
    assert [batch.sequence for batch in batches] == [1, 2, 3]


def test_empty_submissions_and_callbacks_produce_no_logs(make_engine: EngineFactory, capsys):
    engine = make_engine([RecordingAlgo("observer")], None)
    assert engine.submit_orders([]) == []
    engine.run()
    assert capsys.readouterr().out == ""


def test_callback_failure_logs_actual_changes_without_advancing(
    make_engine: EngineFactory, monkeypatch
):
    class Fails(RecordingAlgo):
        def on_snapshot(self, msg: DosVar):
            self.backtest.submit_orders(orders())
            raise RuntimeError("failed after submit")

    engine = make_engine([Fails("failure")], None)
    batches = []
    monkeypatch.setattr(engine, "emit_batch", batches.append)
    with pytest.raises(RuntimeError, match="failed after submit"):
        engine.run()
    assert len(batches) == 1 and batches[0].failed
    assert batches[0].orders == 2
    assert engine.time == np.datetime64(START, "ms")


def test_log_sequence_restarts_for_each_day(make_engine: EngineFactory, monkeypatch):
    class Daily(RecordingAlgo):
        seen = None

        def on_snapshot(self, msg: DosVar):
            day = self.backtest.time.astype("datetime64[D]")
            if self.seen is None or self.seen != day:
                self.seen = day
                self.backtest.submit_orders(
                    [
                        orders()[0].model_copy(
                            update={"time": self.backtest.time.astype("datetime64[us]").item()}
                        )
                    ]
                )

    engine = make_engine([Daily("daily")], None)
    batches = []
    monkeypatch.setattr(engine, "emit_batch", batches.append)
    engine.run()
    days = sorted(set(batch.time.astype("datetime64[D]") for batch in batches))
    assert len(days) == 2
    for day in days:
        sequence = [
            batch.sequence for batch in batches if batch.time.astype("datetime64[D]") == day
        ]
        assert sequence == list(range(1, len(sequence) + 1))
