"""验证回调优先级、批量回报、回调内下单及异常传播。"""

from collections.abc import Sequence
from unittest.mock import Mock

import pytest

from backtest import Direction, DosVar, LimitOrder, OrderReport, OrderStatus, TradeReport

from .conftest import START, EngineFactory, RecordingAlgo, buy


@pytest.mark.parametrize("trusted", [False, True])
@pytest.mark.parametrize("count", [0, 1, 3, 5])
def test_callback_order_and_shared_context(
    make_engine: EngineFactory, count: int, trusted: bool
) -> None:
    algos = [RecordingAlgo(str(i)) for i in range(count)]
    engine = make_engine(algos, None)
    engine.trusted_event_cache = trusted
    assert all(algo.backtest is engine and algo.ctx is engine.ctx for algo in algos)
    engine.quote()
    engine.initialize()
    order_id = buy(engine)
    engine.process_events(DosVar("msg"))

    expected = [("initialize", str(i)) for i in range(count)] + [
        (event, str(i)) for event in ("order", "trade", "snapshot") for i in reversed(range(count))
    ]
    assert engine.ctx.events == [(event, name, i + 1) for i, (event, name) in enumerate(expected)]
    assert engine.ctx.value == 4 * count
    assert engine.get_orders() == engine.get_trades() == []
    for algo in algos:
        assert len(algo.orders) == len(algo.trades) == len(algo.snapshots) == 1
        assert all(
            isinstance(row, OrderReport) and row.orderId == order_id for row in algo.orders[0]
        )
        assert all(
            isinstance(row, TradeReport) and row.orderId == order_id for row in algo.trades[0]
        )
        assert sum(row.tradeQty for row in algo.trades[0]) == 100
        assert algo.snapshots == [(engine.time, 2)]

    # 缓冲已消费：重复排空既不重复通知，也不修改 ctx。
    before = engine.ctx.model_copy(deep=True)
    engine.process_events()
    assert engine.ctx == before


def test_multiple_orders_and_partial_fills_remain_batched(make_engine: EngineFactory) -> None:
    algo = RecordingAlgo("selection")
    engine = make_engine([algo], None)
    engine.quote()
    first = buy(engine, quantity=200, price=10.1)
    second = buy(engine, quantity=100, price=9.0)
    engine.process_events()
    assert len(algo.orders) == len(algo.trades) == 1
    assert {row.orderId for row in algo.orders[0]} == {first, second}
    assert [row.tradeQty for row in algo.trades[0]] == [100]
    first_batch = [row.model_copy(deep=True) for row in algo.orders[0]]

    engine.quote(3)
    engine.process_events(DosVar("msg"))
    assert len(algo.orders) == len(algo.trades) == 2
    assert algo.orders[0] == first_batch  # 后续 DOS 回调不能覆盖已交付的记录。
    assert algo.orders[1][-1].status == OrderStatus.FILLED
    assert algo.orders[1][-1].orderId == first
    assert [row.tradeQty for batch in algo.trades for row in batch] == [100, 100]
    assert algo.trades[1][-1].totalVolume == 200
    assert [event for event, _, _ in engine.ctx.events] == [
        "order",
        "trade",
        "order",
        "trade",
        "snapshot",
    ]


@pytest.mark.parametrize("deferred", [False, True])
@pytest.mark.parametrize("trusted", [False, True])
@pytest.mark.parametrize(
    "trigger, expected",
    [
        ("order", ["order", "order", "trade", "snapshot"]),
        ("trade", ["order", "trade", "order", "trade", "snapshot"]),
        ("snapshot", ["snapshot", "order", "trade"]),
    ],
)
def test_callback_submits_order_before_next_snapshot(
    make_engine: EngineFactory,
    trigger: str,
    expected: list[str],
    trusted: bool,
    deferred: bool,
) -> None:
    class SubmitAlgo(RecordingAlgo):
        fired = False

        def record(self, event: str) -> None:
            super().record(event)
            if event == trigger and not self.fired:
                self.fired = True
                # 本批回报在回调前就已清空；无需等 Python 处理完成。
                if event == "order":
                    assert self.backtest.get_orders() == []
                if event == "trade":
                    assert self.backtest.get_trades() == []
                buy(self.backtest)

    algos = [RecordingAlgo("selection"), SubmitAlgo("portfolio"), RecordingAlgo("execution")]
    engine = make_engine(algos, None)
    engine.trusted_event_cache = trusted
    engine.deferred_trade_ack = deferred
    engine.quote()
    if trigger != "snapshot":
        buy(engine)
    before_time = engine.time
    engine.process_events(DosVar("msg"))
    assert [(event, name) for event, name, _ in engine.ctx.events] == [
        (event, name) for event in expected for name in ("execution", "portfolio", "selection")
    ]
    assert engine.time == before_time
    assert engine.ctx.value == len(expected) * 3
    assert engine.get_orders() == engine.get_trades() == []
    assert sum(row.tradeQty for batch in algos[0].trades for row in batch) == (
        100 if trigger == "snapshot" else 200
    )
    assert all(len(algo.snapshots) == 1 for algo in algos)


def test_selection_can_react_to_rejection(make_engine: EngineFactory) -> None:
    class Selection(RecordingAlgo):
        def on_order(self, orders: Sequence[OrderReport]) -> None:
            super().on_order(orders)
            if any(row.status == OrderStatus.REJECTED for row in orders):
                self.ctx.counts["rejections"] = self.ctx.counts.get("rejections", 0) + 1
                buy(self.backtest)

    selection = Selection("selection")
    engine = make_engine([selection, RecordingAlgo("execution")], None)
    engine.quote()
    engine.submit_order(
        LimitOrder(
            symbol="000001.XSHE",
            time=START,
            quantity=100,
            price=9.9,
            direction=Direction.SELL_CLOSE,
        )
    )
    engine.process_events(DosVar("msg"))
    assert engine.ctx.counts["rejections"] == 1
    assert [row.status for row in selection.orders[0]] == [
        OrderStatus.REPORTED,
        OrderStatus.REJECTED,
    ]
    assert sum(row.tradeQty for batch in selection.trades for row in batch) == 100
    assert engine.get_position()["000001.XSHE"].longPosition == 100


def test_callback_exception_stops_dispatch(make_engine: EngineFactory) -> None:
    class FailingAlgo(RecordingAlgo):
        def on_snapshot(self, msg: DosVar) -> None:
            self.ctx.counts["before_error"] = 1
            raise RuntimeError("callback failed")

    engine = make_engine(
        [
            RecordingAlgo("selection"),
            FailingAlgo("portfolio"),
            RecordingAlgo("execution"),
        ],
        None,
    )
    engine.quote()
    with pytest.raises(RuntimeError, match="callback failed"):
        engine.process_events(DosVar("msg"))
    assert engine.ctx.events == [("snapshot", "execution", 1)]
    assert engine.ctx.counts["before_error"] == 1


def test_submit_orders_uses_one_rpc_and_preserves_submission_order(
    make_engine: EngineFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = make_engine([], None)
    engine.quote()
    orders = [
        LimitOrder(
            symbol="000001.XSHE",
            time=START,
            direction=direction,
            quantity=100,
            price=price,
            label=label,
        )
        for direction, price, label in [
            (Direction.BUY_OPEN, 10.1, "open"),
            (Direction.SELL_CLOSE, 9.9, "close"),
        ]
    ]
    run = Mock(wraps=engine.session.run)
    monkeypatch.setattr(engine.session, "run", run)
    identifiers = engine.submit_orders(orders)
    assert len(identifiers) == 2 and len(set(identifiers)) == 2
    assert run.call_count == 1
    assert run.call_args.args[0] == "soloSubmitOrders"
    assert engine.moment.orders == 2
    assert engine.get_position() == {}
    trades = engine.get_trades()
    assert [row.orderId for row in trades] == identifiers
    assert [row.label for row in trades] == ["open", "close"]
    run.reset_mock()
    assert engine.submit_orders([]) == []
    run.assert_not_called()


def test_next_events_only_takes_orders_then_trades(make_engine: EngineFactory) -> None:
    engine = make_engine([], None)
    engine.quote()
    identifier = buy(engine)
    original_trades = engine.get_trades()
    kind, orders = engine.next_events()
    assert kind == "order" and {row.orderId for row in orders} == {identifier}
    assert engine.get_orders() == []
    assert engine.get_trades() == original_trades
    kind, trades = engine.next_events()
    assert kind == "trade" and trades == original_trades
    assert engine.next_events() == ("", [])


def test_trusted_empty_events_skip_rpc_until_invalidated(
    make_engine: EngineFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = make_engine([], None)
    engine.trusted_event_cache = True
    engine.quote()
    run = Mock(wraps=engine.session.run)
    monkeypatch.setattr(engine.session, "run", run)
    engine.process_events(DosVar("msg"))
    run.assert_not_called()
    engine.invalidate_events()
    engine.process_events()
    assert run.call_count == 1
    assert run.call_args.args[0:2] == ("soloNextEvents", "engine")


@pytest.mark.parametrize("trusted", [False, True])
def test_raw_session_orders_are_dispatched_without_changing_callback_order(
    make_engine: EngineFactory, trusted: bool
) -> None:
    class RawBuyer(RecordingAlgo):
        def on_snapshot(self, msg: DosVar) -> None:
            super().on_snapshot(msg)
            self.backtest.session.run("""
                Backtest::submitOrder(engine,
                    ("000001.XSHE", 2026.06.10T09:30:00.000, 5, 10.1, 100l, 1))
            """)
            if trusted:
                self.backtest.invalidate_events()

    algo = RawBuyer("raw")
    engine = make_engine([algo], None)
    engine.trusted_event_cache = trusted
    engine.quote()
    engine.process_events(DosVar("msg"))
    assert [event for event, _, _ in engine.ctx.events] == ["snapshot", "order", "trade"]
    assert sum(row.tradeQty for batch in algo.trades for row in batch) == 100


def test_rejections_are_counted_in_current_moment(make_engine: EngineFactory) -> None:
    engine = make_engine([], None)
    engine.quote()
    engine.submit_order(
        LimitOrder(
            symbol="000001.XSHE",
            time=START,
            direction=Direction.SELL_CLOSE,
            quantity=100,
            price=9.9,
        )
    )
    engine.process_events()
    assert engine.moment.rejected == 1


def test_deferred_trade_is_delivered_locally_and_ack_joins_next_submit(
    make_engine: EngineFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = make_engine([], None)
    engine.trusted_event_cache = engine.deferred_trade_ack = True
    engine.quote()
    first = buy(engine)
    raw_run = engine.session.run
    run = Mock(wraps=raw_run)
    monkeypatch.setattr(engine.session, "run", run)
    assert engine.next_events()[0] == "order"
    assert run.call_count == 1
    kind, trades = engine.next_events()
    assert kind == "trade" and [row.orderId for row in trades] == [first]
    assert run.call_count == 1
    assert engine._pending_trade_ack is not None
    assert engine.next_events() == ("", [])
    second = buy(engine)
    assert run.call_count == 2
    assert engine._pending_trade_ack is None
    assert engine.get_trades()[0].orderId == second
    assert engine.get_context().trades[0].orderId == second


def test_deferred_trade_public_peeks_flush_only_consumed_reports(
    make_engine: EngineFactory,
) -> None:
    engine = make_engine([], None)
    engine.trusted_event_cache = engine.deferred_trade_ack = True
    engine.quote()
    buy(engine)
    assert engine.next_events()[0] == "order"
    assert engine.next_events()[0] == "trade"
    assert engine.get_context().trades == []
    assert engine.get_trades() == []
    assert engine._pending_trade_ack is None


def test_deferred_trade_ack_checks_entire_prefix_before_clearing(
    make_engine: EngineFactory,
) -> None:
    engine = make_engine([], None)
    engine.trusted_event_cache = engine.deferred_trade_ack = True
    engine.quote()
    buy(engine)
    assert engine.next_events()[0] == "order"
    assert engine.next_events()[0] == "trade"
    engine.session.run(
        'ctx = Backtest::getContextDict(engine); ctx["trades"][0]["tradeQty"] = 999l'
    )
    with pytest.raises(Exception, match="递延成交确认内容不匹配"):
        engine.flush_pending_events()
    assert engine.session.run('Backtest::getContextDict(engine)["trades"][0]["tradeQty"]') == 999
    engine.session.run(
        'ctx = Backtest::getContextDict(engine); ctx["trades"][0]["tradeQty"] = 100l'
    )
    engine.flush_pending_events()
    assert engine.get_trades() == []


def test_deferred_trade_ack_preserves_new_suffix_from_untracked_write(
    make_engine: EngineFactory,
) -> None:
    engine = make_engine([], None)
    engine.trusted_event_cache = engine.deferred_trade_ack = True
    engine.quote()
    first = buy(engine)
    assert engine.next_events()[0] == "order"
    assert engine.next_events()[0] == "trade"
    engine.session.run("""
        Backtest::submitOrder(engine,
            ("600000.XSHG", 2026.06.10T09:30:00.000, 5, 10.1, 100l, 1))
    """)
    engine.flush_pending_events()
    remaining = engine.get_trades()
    assert len(remaining) == 1 and remaining[0].orderId != first


def test_deferred_trade_proxy_flushes_raw_callback_reads_and_writes(
    make_engine: EngineFactory,
) -> None:
    class RawTradeBuyer(RecordingAlgo):
        fired = False

        def on_trade(self, trades) -> None:
            super().on_trade(trades)
            assert self.backtest.session.run("size(getTrades(engine))") == 0
            if not self.fired:
                self.fired = True
                self.backtest.session.run("""
                    Backtest::submitOrder(engine,
                        ("600000.XSHG", 2026.06.10T09:30:00.000, 5, 10.1, 100l, 1))
                """)

    algo = RawTradeBuyer("raw")
    engine = make_engine([algo], None)
    raw = engine.session

    class Proxy:
        def run(self, *args, **kwargs):
            engine.flush_pending_events(session=raw)
            engine.invalidate_events()
            return raw.run(*args, **kwargs)

        def upload(self, *args, **kwargs):
            engine.flush_pending_events(session=raw)
            engine.invalidate_events()
            return raw.upload(*args, **kwargs)

        def close(self):
            return raw.close()

    engine.session = Proxy()
    engine.trusted_event_cache = engine.deferred_trade_ack = True
    engine.quote()
    buy(engine)
    engine.process_events(DosVar("msg"))
    assert [event for event, _, _ in engine.ctx.events] == [
        "order",
        "trade",
        "order",
        "trade",
        "snapshot",
    ]
    assert sum(row.tradeQty for batch in algo.trades for row in batch) == 200
    assert engine.get_context().orders == engine.get_context().trades == []
