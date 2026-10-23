"""真实 DolphinDB 测试共用连接配置，不在仓库保存凭据。"""

from pathlib import Path

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--ddb-config", type=Path, help="DolphinDB 连接配置文件路径")


@pytest.fixture(scope="session")
def connection(pytestconfig: pytest.Config) -> tuple[str, int, str, str]:
    path = pytestconfig.getoption("--ddb-config")
    if path is None:
        raise pytest.UsageError("请用 --ddb-config 指定 DolphinDB 连接配置文件。")
    values = {
        key.strip(): value.strip().strip('"').strip("'")
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if "=" in line and not line.lstrip().startswith("#")
        for key, value in [line.split("=", 1)]
    }
    host, port = values["DOLPHINDB_ENDPOINTS"].split(",")[0].strip().rsplit(":", 1)
    return (
        host.removeprefix("[").removesuffix("]"),
        int(port),
        values["DOLPHINDB_USERNAME"],
        values["DOLPHINDB_PASSWORD"],
    )
