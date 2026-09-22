"""官方股票快照限价单示例的回归测试。

来源：https://docs.dolphindb.cn/zh/plugins/backtest/quick_start.html
附件：股票快照.dos。直接调用原生插件，不经过 Python Backtest 封装。
"""

from uuid import uuid4

import dolphindb as ddb
import pytest

from backtest import OrderStatus, PriceType


@pytest.mark.parametrize(
    "price_type, quantity",
    [(PriceType.LIMIT, 100)],
    ids=["official-limit"],
)
def test_official_stock_example(
    connection: tuple[str, int, str, str],
    price_type: PriceType,
    quantity: int,
) -> None:
    session = ddb.session()
    name = "solo_official_" + uuid4().hex
    created = False
    try:
        session.connect(*connection)
        session.upload({"engineName": name})
        session.run(r"""
            @state
            def pctChg(lastPrice, prevClosePrice) { return lastPrice\prevClosePrice - 1 }
            def initialize(mutable context) {
                d = dict(STRING, ANY)
                d["pctChg"] = <pctChg(lastPrice, prevClosePrice)>
                Backtest::subscribeIndicator(context["engine"], "snapshot", d)
                context["maxPos"] = 500
            }
            def beforeTrading(mutable context) {
                Backtest::setUniverse(context["engine"], ["000001.XSHE"])
                context["open"] = dict(STRING, BOOL)
            }
        """)
        # 保留官方选股条件与限价单参数。
        session.run(f"""
            def onSnapshot(mutable context, msg, indicator) {{
                &pos = Backtest::getPosition(context["engine"], msg.symbol)
                if (indicator.pctChg > 0.01 and pos.longPosition <= context.maxPos and
                    context["open"][msg.symbol] != true) {{
                    Backtest::submitOrder(context["engine"], (msg.symbol,
                        context["tradeTime"], {int(price_type)}, msg.offerPrice[0],
                        {quantity}, 1), "buy")
                    context["open"][msg.symbol] = true
                }}
            }}
            config = {{startDate:2022.04.11, endDate:2022.04.11, strategyGroup:`stock,
                frequency:0, cash:100000000, commission:0.00015, tax:0.001,
                msgAsTable:false, dataType:1, "context":{{maxPos:500, open:{{A:true}}}}}}
            callbacks = {{initialize:initialize, beforeTrading:beforeTrading,
                onSnapshot:onSnapshot}}
            engine = Backtest::createBacktester(engineName, config, callbacks, false)
        """)
        created = True
        session.run("""
            names = ["symbol","symbolSource","timestamp","lastPrice","upLimitPrice",
                "downLimitPrice","totalBidQty","totalOfferQty","bidPrice","bidQty",
                "offerPrice","offerQty","signal","prevClosePrice"]
            types = [STRING,STRING,TIMESTAMP,DOUBLE,DOUBLE,DOUBLE,LONG,LONG,
                DOUBLE[],LONG[],DOUBLE[],LONG[],DOUBLE[],DOUBLE]
            messageTable = table(100:0, names, types)
            insert into messageTable values("000001.XSHE", "XSHE",
                2022.04.11T10:10:00.000, 7, 15, 5, 10000, 10000,
                arrayVector([10], [6.9,6.8,6.7,6.6,6.5,6.4,6.3,6.2,6.1,6.0]),
                arrayVector([10], [800,900,1000,1100,1200,1000,1000,1000,1000,1000]),
                arrayVector([10], [7.1,7.2,7.3,7.4,7.5,7.6,7.7,7.8,7.9,8.0]),
                arrayVector([10], [1000,1000,1000,1000,1000,1000,1000,1000,1000,1000]),
                , 6.5)
            insert into messageTable values("000001.XSHE", "XSHE",
                2022.04.11T10:10:03.000, 7.5, 15, 5, 10000, 10000,
                arrayVector([10], [7.4,6.8,6.7,6.6,6.5,6.4,6.3,6.2,6.1,6.0]),
                arrayVector([10], [800,900,1000,1100,1200,1000,1000,1000,1000,1000]),
                arrayVector([10], [7.6,7.7,7.8,7.9,8.0,8.1,8.2,8.3,8.4,8.5]),
                arrayVector([10], [1000,1000,1000,1000,1000,1000,1000,1000,1000,1000]),
                , 7)
            Backtest::appendQuotationMsg(engine, messageTable)
            messageTable = select top 1 * from messageTable where timestamp=max(timestamp)
            update messageTable set symbol="END", timestamp=2022.04.11T16:00:00.000
            Backtest::appendQuotationMsg(engine, messageTable)
        """)
        details = session.run("Backtest::getTradeDetails(engine)")
        fills = details[
            details.orderStatus.isin([OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED])
        ]
        assert list(zip(fills.tradeQty, fills.tradePrice, strict=True)) == [(100, 7.1)]
    finally:
        try:
            if created:
                session.run("Backtest::dropBacktestEngine(engineName)")
        finally:
            session.close()
