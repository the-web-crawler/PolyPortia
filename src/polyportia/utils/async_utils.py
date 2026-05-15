"""Async helpers."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import TypeVar

T = TypeVar("T")


async def bounded_gather(
    coros: list[Awaitable[T]], *, return_exceptions: bool = True
) -> list[T | BaseException]:
    return await asyncio.gather(*coros, return_exceptions=return_exceptions)
