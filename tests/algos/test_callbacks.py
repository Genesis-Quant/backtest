"""验证同步消费、回调优先级、保序下单及不经过代理的 session 读写。"""

from collections.abc import Sequence
from unittest.mock import Mock

import pytest

from backtest import (
    Algo,
    Direction,
    DosVar,
    EventBatch,
    LimitOrder,
    Order,
    OrderReport,
    OrderStatus,
    TradeReport,
)

from .conftest import START, AlgoBacktest, EngineFactory, RecordingAlgo, SharedContext, buy


@pytest.mark.parametrize("count", [0, 1, 3, 5])
def test_callback_order_and_shared_context(make_engine: EngineFactory, count: int) -> None:
    algos = [RecordingAlgo(str(i)) for i in range(count)]
    engine = make_engine(algos, None)
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
    assert algo.orders[0] == first_batch
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


@pytest.mark.parametrize("raw_session", [False, True])
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
    raw_session: bool,
) -> None:
    class SubmitAlgo(RecordingAlgo):
        fired = False

        def record(self, event: str) -> None:
            super().record(event)
            if event == trigger and not self.fired:
                self.fired = True
                # 回调开始前，服务端对应队列已物理清空，无需 flush 或 session 代理。
                if event == "order":
                    assert self.backtest.session.run("size(getOrders(engine))") == 0
                    assert self.backtest.session.run("size(getTrades(engine))") > 0
                if event == "trade":
                    assert self.backtest.session.run("size(getTrades(engine))") == 0
                if raw_session:
                    self.backtest.session.run("""
                        Backtest::submitOrder(engine,
                            ("600000.XSHG", 2026.06.10T09:30:00.000, 5, 10.1, 100l, 1))
                    """)
                else:
                    buy(self.backtest)

    algos = [RecordingAlgo("selection"), SubmitAlgo("portfolio"), RecordingAlgo("execution")]
    engine = make_engine(algos, None)
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


@pytest.mark.parametrize(
    "trigger, expected",
    [
        ("order", ["order", "order", "trade", "snapshot"]),
        ("trade", ["order", "trade", "order", "trade", "snapshot"]),
        ("snapshot", ["snapshot", "order", "trade"]),
    ],
)
def test_process_events_can_reenter_from_any_callback(
    make_engine: EngineFactory, trigger: str, expected: list[str]
) -> None:
    class ReentrantAlgo(RecordingAlgo):
        fired = False

        def record(self, event: str) -> None:
            super().record(event)
            if event == trigger and not self.fired:
                self.fired = True
                buy(self.backtest)
                self.backtest.process_events()

    algo = ReentrantAlgo("reentrant")
    engine = make_engine([algo], None)
    engine.quote()
    if trigger != "snapshot":
        buy(engine)
    engine.process_events(DosVar("msg"))
    assert [event for event, _, _ in engine.ctx.events] == expected
    assert engine.get_context().orders == engine.get_context().trades == []
    assert engine.moment.trades == (1 if trigger == "snapshot" else 2)


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
        [RecordingAlgo("selection"), FailingAlgo("portfolio"), RecordingAlgo("execution")],
        None,
    )
    engine.quote()
    with pytest.raises(RuntimeError, match="callback failed"):
        engine.process_events(DosVar("msg"))
    assert engine.ctx.events == [("snapshot", "execution", 1)]
    assert engine.ctx.counts["before_error"] == 1


def test_trade_callback_exception_leaves_no_unacknowledged_reports(
    make_engine: EngineFactory,
) -> None:
    class FailingTrade(RecordingAlgo):
        def on_trade(self, trades: Sequence[TradeReport]) -> None:
            assert self.backtest.session.run("size(getTrades(engine))") == 0
            raise RuntimeError("trade callback failed")

    engine = make_engine([FailingTrade("failing")], None)
    engine.quote()
    buy(engine)
    with pytest.raises(RuntimeError, match="trade callback failed"):
        engine.process_events()
    assert engine.session.run("size(getTrades(engine))") == 0
    assert engine.next_events() == ("", [])
    assert engine.get_trade_details()


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


@pytest.mark.parametrize("inherited", [False, True])
def test_submit_orders_respects_single_override_and_super_without_recursion(
    make_engine: EngineFactory, monkeypatch: pytest.MonkeyPatch, inherited: bool
) -> None:
    hooked: list[Order] = []
    returned: list[int] = []

    class CustomBacktest(AlgoBacktest):
        def submit_order(self, order: Order) -> int:
            hooked.append(order)
            adjusted = order.model_copy(update={"price": 10.1, "label": f"hook:{order.label}"})
            identifier = super().submit_order(adjusted)
            returned.append(identifier)
            return identifier

    class InheritedBacktest(CustomBacktest):
        pass

    monkeypatch.setattr(
        f"{AlgoBacktest.__module__}.AlgoBacktest",
        InheritedBacktest if inherited else CustomBacktest,
    )
    algo = RecordingAlgo("observer")
    engine = make_engine([algo], None)
    engine.quote()
    orders = [
        LimitOrder(
            symbol=symbol,
            time=START,
            direction=Direction.BUY_OPEN,
            quantity=100,
            price=9.0,
            label=label,
        )
        for symbol, label in [("000001.XSHE", "first"), ("600000.XSHG", "second")]
    ]
    run = Mock(wraps=engine.session.run)
    monkeypatch.setattr(engine.session, "run", run)
    identifiers = engine.submit_orders(orders)
    assert hooked == orders
    assert identifiers == returned and len(set(identifiers)) == 2
    assert run.call_count == 2
    assert all(call.args[0] == "soloSubmitOrders" for call in run.call_args_list)
    assert all(len(call.args[2]) == 1 for call in run.call_args_list)
    assert engine.moment.orders == 2
    assert engine.ctx.events == []  # 运输不会交错分发 Python 回调。
    trades = engine.get_trades()
    assert [row.orderId for row in trades] == identifiers
    assert [row.label for row in trades] == ["hook:first", "hook:second"]
    assert [row.orderPrice for row in trades] == [10.1, 10.1]
    assert [order.price for order in orders] == [9.0, 9.0]
    engine.process_events()
    assert [event for event, _, _ in engine.ctx.events] == ["order", "trade"]
    run.reset_mock()
    assert engine.submit_orders([]) == []
    assert hooked == orders
    run.assert_not_called()


def test_submit_orders_keeps_single_override_risk_checks_and_stops_on_error(
    make_engine: EngineFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    hooked: list[str] = []

    class RiskCheckedBacktest(AlgoBacktest):
        def submit_order(self, order: Order) -> int:
            hooked.append(order.label)
            if self.get_position():
                raise ValueError("position limit reached")
            return super().submit_order(order)

    monkeypatch.setattr(f"{AlgoBacktest.__module__}.AlgoBacktest", RiskCheckedBacktest)
    engine = make_engine([], None)
    engine.quote()
    orders = [
        LimitOrder(
            symbol="000001.XSHE",
            time=START,
            direction=Direction.BUY_OPEN,
            quantity=100,
            price=10.1,
            label=label,
        )
        for label in ["accepted", "blocked", "not reached"]
    ]
    with pytest.raises(ValueError, match="position limit reached"):
        engine.submit_orders(orders)
    assert hooked == ["accepted", "blocked"]
    assert engine.moment.orders == 1
    trades = engine.get_trades()
    assert len(trades) == 1 and trades[0].label == "accepted"
    assert engine.get_position()["000001.XSHE"].longPosition == 100


def test_single_submit_does_not_reenter_custom_batch_hook(
    make_engine: EngineFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    batches: list[list[Order]] = []

    class CustomBatchBacktest(AlgoBacktest):
        def submit_orders(self, orders: Sequence[Order]) -> list[int]:
            batches.append(list(orders))
            return [self.submit_order(order) for order in orders]

    monkeypatch.setattr(f"{AlgoBacktest.__module__}.AlgoBacktest", CustomBatchBacktest)
    engine = make_engine([], None)
    engine.quote()
    order = LimitOrder(
        symbol="000001.XSHE",
        time=START,
        direction=Direction.BUY_OPEN,
        quantity=100,
        price=10.1,
        label="single",
    )
    run = Mock(wraps=engine.session.run)
    monkeypatch.setattr(engine.session, "run", run)
    first = engine.submit_order(order)
    assert run.call_count == 1 and batches == []
    others = [
        order.model_copy(update={"label": "second", "symbol": "600000.XSHG"}),
        order.model_copy(
            update={"label": "third", "direction": Direction.SELL_CLOSE, "price": 9.9}
        ),
    ]
    identifiers = engine.submit_orders(others)
    assert run.call_count == 3
    assert batches == [others]
    trades = engine.get_trades()
    assert [row.orderId for row in trades] == [first, *identifiers]
    assert [row.label for row in trades] == ["single", "second", "third"]
    assert engine.moment.orders == 3


def test_next_events_only_takes_orders_then_trades(make_engine: EngineFactory) -> None:
    engine = make_engine([], None)
    engine.quote()
    identifier = buy(engine)
    original_trades = engine.get_trades()
    kind, orders = engine.next_events()
    assert kind == "order" and {row.orderId for row in orders} == {identifier}
    assert engine.session.run("size(getOrders(engine))") == 0
    assert engine.get_trades() == original_trades
    kind, trades = engine.next_events()
    assert kind == "trade" and trades == original_trades
    assert engine.session.run("size(getTrades(engine))") == 0
    assert engine.next_events() == ("", [])


def test_empty_public_next_events_always_queries_server(
    make_engine: EngineFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = make_engine([], None)
    engine.quote()
    raw_run = engine.session.run
    run = Mock(wraps=raw_run)
    monkeypatch.setattr(engine.session, "run", run)
    assert engine.next_events() == engine.next_events() == ("", [])
    assert run.call_count == 2
    assert all(call.args == ("soloNextEvents", "engine") for call in run.call_args_list)
    raw_run("""
        Backtest::submitOrder(engine,
            ("600000.XSHG", 2026.06.10T09:30:00.000, 5, 10.1, 100l, 1))
    """)
    assert engine.next_events()[0] == "order"
    assert engine.next_events()[0] == "trade"
    assert engine.next_events() == ("", [])


def test_batch_state_does_not_cache_or_preconsume_trades(make_engine: EngineFactory) -> None:
    engine = make_engine([], None)
    engine.quote()
    first = buy(engine)
    batch = engine.next_event_batch()
    assert isinstance(batch, EventBatch)
    assert batch.kind == "order" and not batch.events_empty
    assert {row.orderId for row in batch.reports} == {first}
    assert engine.get_trades()[0].orderId == first

    engine.session.run("""
        Backtest::submitOrder(engine,
            ("600000.XSHG", 2026.06.10T09:30:00.000, 5, 10.1, 100l, 1))
    """)
    # 原来的成交尚未消费；新委托仍应优先，不能交付预取成交。
    second_batch = engine.next_event_batch()
    assert second_batch.kind == "order" and not second_batch.events_empty
    second = second_batch.reports[0].orderId
    assert second != first
    trades = engine.next_event_batch()
    assert trades.kind == "trade" and trades.events_empty
    assert [row.orderId for row in trades.reports] == [first, second]
    assert engine.session.run("size(getTrades(engine))") == 0
    assert engine.next_event_batch() == EventBatch("", [], True)


def test_raw_session_can_consume_or_replace_unread_trades(make_engine: EngineFactory) -> None:
    engine = make_engine([], None)
    engine.quote()
    first = buy(engine)
    assert engine.next_events()[0] == "order"
    consumed = engine.session.run("getTrades(engine, true)")
    assert [int(row["orderId"]) for row in consumed] == [first]
    assert engine.next_events() == ("", [])
    engine.session.run("""
        Backtest::submitOrder(engine,
            ("600000.XSHG", 2026.06.10T09:30:00.000, 5, 10.1, 100l, 1))
    """)
    assert engine.next_events()[0] == "order"
    kind, trades = engine.next_events()
    assert kind == "trade" and len(trades) == 1 and trades[0].orderId != first
    assert engine.get_context().trades == []


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


def test_external_python_signals_need_no_ddb_history_or_readable_params(
    make_engine: EngineFactory,
) -> None:
    class ExternalSignals(Algo[None, SharedContext]):
        def __init__(self) -> None:
            # 代表来自 CSV/API/内存的外部信号，而非任何 DOS 历史组件。
            self.signals = {"2026-06-10T09:30:00.000": 100}
            self.received: list[str] = []
            self.traded = 0

        @property
        def params(self) -> None:
            raise AssertionError("回测内核不应读取 Algo 参数")

        def initialize(self) -> None:
            self.received.append("initialize")

        def on_snapshot(self, msg: DosVar) -> None:
            self.received.append("snapshot")
            quantity = self.signals.get(str(self.backtest.time), 0)
            if quantity:
                buy(self.backtest, quantity=quantity, price=10.1)

        def on_order(self, orders: Sequence[OrderReport]) -> None:
            self.received.append("order")

        def on_trade(self, trades: Sequence[TradeReport]) -> None:
            self.received.append("trade")
            self.traded += sum(row.tradeQty for row in trades)

    algo = ExternalSignals()
    engine = make_engine([algo], None)
    engine.run()
    assert algo.received == ["initialize", "snapshot", "order", "trade", "snapshot", "snapshot"]
    assert algo.traded == 100
    assert engine.get_position()["000001.XSHE"].longPosition == 100
    assert engine.get_context().orders == engine.get_context().trades == []
