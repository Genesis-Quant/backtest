"""验证回调优先级、批量回报、回调内下单及异常传播。"""

from collections.abc import Sequence

import pytest

from backtest import Direction, DosVar, LimitOrder, OrderReport, OrderStatus, TradeReport

from .conftest import START, EngineFactory, RecordingAlgo, buy


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
