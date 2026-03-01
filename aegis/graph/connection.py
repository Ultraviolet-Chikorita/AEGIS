from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from neo4j import AsyncGraphDatabase, READ_ACCESS, WRITE_ACCESS
from neo4j.exceptions import ServiceUnavailable, SessionExpired, WriteServiceUnavailable


logger = logging.getLogger(__name__)

_WRITE_QUERY_PATTERN = re.compile(
    r"\b(create|merge|set|delete|remove|detach|drop|load\s+csv|call\s+db\.|call\s+apoc\.)\b",
    re.IGNORECASE,
)


class Neo4jConnection:
    def __init__(self, uri: str, user: str, password: str) -> None:
        self._driver = AsyncGraphDatabase.driver(uri, auth=(user, password))

    async def connect(self) -> None:
        await self._driver.verify_connectivity()

    async def close(self) -> None:
        await self._driver.close()

    async def query(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        params = params or {}
        is_write = _WRITE_QUERY_PATTERN.search(cypher) is not None
        access_mode = WRITE_ACCESS if is_write else READ_ACCESS

        max_attempts = 5
        for attempt in range(1, max_attempts + 1):
            try:
                async with self._driver.session(default_access_mode=access_mode) as session:
                    result = await session.run(cypher, **params)
                    return [record.data() async for record in result]
            except (WriteServiceUnavailable, SessionExpired, ServiceUnavailable):
                if attempt >= max_attempts:
                    raise
                backoff_seconds = min(1.5, 0.12 * (2 ** (attempt - 1)))
                logger.warning(
                    "Neo4j query retrying after transient connectivity issue "
                    "(attempt %s/%s, mode=%s, sleep=%.2fs)",
                    attempt,
                    max_attempts,
                    "WRITE" if is_write else "READ",
                    backoff_seconds,
                )
                await asyncio.sleep(backoff_seconds)
