from __future__ import annotations

from typing import Any, NewType

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator

__all__ = ["DosRecord", "DosVar"]

# 当前 DolphinDB 会话中的变量名，如 DosVar("msg")；不是查询语句或数据副本。
# NewType 只用于静态类型区分，运行时仍是字符串，不负责转义或验证变量名。
DosVar = NewType("DosVar", str)


class DosRecord(BaseModel):
    """DOS 数据转换基类，保留插件额外返回的列，避免丢弃版本扩展字段。"""

    model_config = ConfigDict(extra="allow")

    @field_validator("*", mode="before")
    @classmethod
    def convert_datetime(cls, value: Any) -> Any:
        """把 SDK 的 numpy 时间转为 Python datetime，NaT 转为 None。

        DATE 字段再由其 date 类型注解转换；不在此步骤推断或转换时区。
        """
        if isinstance(value, np.datetime64):
            return None if np.isnat(value) else value.astype("datetime64[us]").item()
        return value
