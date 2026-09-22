"""复用人工行情与独立引擎，每个用例结束后释放资源。"""

from algos.conftest import make_engine as make_engine
