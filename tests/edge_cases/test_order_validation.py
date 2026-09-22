"""错误订单在模型构造时被拒绝，无需向撮合引擎提交。"""

from datetime import time
from typing import Any

import pytest
from algos.conftest import START
from pydantic import ValidationError

from backtest import Direction, LimitOrder, MarketOrder, PriceType, StopLimitOrder, TWAPOrder


@pytest.mark.parametrize(
    "field, value",
    [
        ("quantity", 0),
        ("quantity", -100),
        ("quantity", 1.5),
        ("price", 0.0),
        ("price", -1.0),
        ("direction", 99),
        ("time", "not-a-date"),
        ("unexpected", True),
    ],
)
def test_invalid_limit_order(field: str, value: Any) -> None:
    params = dict(
        symbol="000001.XSHE", time=START, direction=Direction.BUY_OPEN, quantity=100, price=10.0
    )
    params[field] = value
    with pytest.raises(ValidationError) as error:
        LimitOrder(**params)
    assert error.value.errors()[0]["loc"] == (field,)


@pytest.mark.parametrize("price_type", [PriceType.LIMIT, 4, 99])
def test_market_order_rejects_limit_disabled_fok_and_unknown_type(price_type: int) -> None:
    with pytest.raises(ValidationError) as error:
        MarketOrder(
            symbol="000001.XSHE",
            time=START,
            direction=Direction.BUY_OPEN,
            quantity=100,
            price_type=price_type,
        )
    assert error.value.errors()[0]["loc"] == ("price_type",)


@pytest.mark.parametrize("field", ["stop_loss", "take_profit", "slippage"])
def test_negative_stop_parameters(field: str) -> None:
    params = dict(
        symbol="000001.XSHE",
        exchange="XSHE",
        time=START,
        direction=Direction.BUY_OPEN,
        quantity=100,
        price=10.0,
        stop_loss=9.5,
        take_profit=10.5,
        expires_at=START,
    )
    params[field] = -0.1
    with pytest.raises(ValidationError) as error:
        StopLimitOrder(**params)
    assert error.value.errors()[0]["loc"] == (field,)


@pytest.mark.parametrize(
    "field, value",
    [
        ("interval", 0),
        ("interval", -1),
        ("max_single_order_volume", 0),
        ("max_single_order_volume", -100),
        ("end_time", "25:00:00"),
        ("order_price_type", 99),
    ],
)
def test_invalid_twap_parameters(field: str, value: Any) -> None:
    params = dict(
        symbol="000001.XSHE",
        time=START,
        direction=Direction.BUY_OPEN,
        quantity=100,
        price=10.0,
        end_time=time(9, 31),
    )
    params[field] = value
    with pytest.raises(ValidationError) as error:
        TWAPOrder(**params)
    assert error.value.errors()[0]["loc"] == (field,)
