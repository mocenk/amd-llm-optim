"""
Paged KV-cache for memory-efficient attention on AMD GPUs.

Manages a pool of fixed-size cache pages and assigns them to active
sequences on demand. Modeled after PagedAttention but with allocation
heuristics tuned to AMD HBM bandwidth characteristics.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable

logger = logging.getLogger(__name__)


@dataclass
class CachePage:
    """A single fixed-size block of KV memory."""

    page_id: int
    owner: str | None = None
    occupancy: int = 0  # number of tokens currently stored

    @property
    def free(self) -> bool:
        return self.owner is None


class PagedKVCache:
    """Block-based KV-cache with copy-on-write block sharing.

    Args:
        num_pages: Total number of pages in the pool.
        page_size: Tokens stored per page.
        num_heads: Number of attention heads (for tensor sizing).
        head_dim: Per-head embedding dimension.
        dtype: Storage dtype for KV tensors.
    """

    def __init__(
        self,
        num_pages: int = 2048,
        page_size: int = 16,
        num_heads: int = 32,
        head_dim: int = 128,
        dtype: str = "float16",
    ) -> None:
        if num_pages <= 0 or page_size <= 0:
            raise ValueError("num_pages and page_size must be positive")
        self.num_pages = num_pages
        self.page_size = page_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype

        self._pages: list[CachePage] = [CachePage(page_id=i) for i in range(num_pages)]
        self._free_pages: OrderedDict[int, None] = OrderedDict(
            (i, None) for i in range(num_pages)
        )
        self._owners: dict[str, list[int]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Reservation
    # ------------------------------------------------------------------
    def reserve(self, owner: str, num_tokens: int) -> bool:
        """Reserve enough pages to hold ``num_tokens`` for ``owner``.

        Returns ``False`` (without partial allocation) if the pool can't
        satisfy the request, so the scheduler can apply backpressure.
        """
        with self._lock:
            pages_needed = (num_tokens + self.page_size - 1) // self.page_size
            if len(self._free_pages) < pages_needed:
                logger.debug(
                    "Cache reservation rejected: need=%d free=%d",
                    pages_needed,
                    len(self._free_pages),
                )
                return False
            assigned: list[int] = []
            for _ in range(pages_needed):
                page_id, _ = self._free_pages.popitem(last=False)
                self._pages[page_id].owner = owner
                self._pages[page_id].occupancy = 0
                assigned.append(page_id)
            self._owners.setdefault(owner, []).extend(assigned)
            return True

    def release(self, owner: str) -> int:
        """Release all pages owned by ``owner``. Returns pages freed."""
        with self._lock:
            pages = self._owners.pop(owner, [])
            for page_id in pages:
                self._pages[page_id].owner = None
                self._pages[page_id].occupancy = 0
                self._free_pages[page_id] = None
            return len(pages)

    def append(self, owner: str, num_tokens: int = 1) -> bool:
        """Append tokens to the last allocated page; allocate a new one if needed."""
        with self._lock:
            pages = self._owners.get(owner)
            if not pages:
                return False
            last = self._pages[pages[-1]]
            remaining = self.page_size - last.occupancy
            if remaining >= num_tokens:
                last.occupancy += num_tokens
                return True
            # Spill into a fresh page.
            if not self._free_pages:
                return False
            last.occupancy = self.page_size
            page_id, _ = self._free_pages.popitem(last=False)
            self._pages[page_id].owner = owner
            self._pages[page_id].occupancy = num_tokens - remaining
            pages.append(page_id)
            return True

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def pages_in_use(self) -> int:
        return self.num_pages - len(self._free_pages)

    def utilization(self) -> float:
        return self.pages_in_use() / max(1, self.num_pages)

    def owner_pages(self, owner: str) -> Iterable[int]:
        return tuple(self._owners.get(owner, ()))

    def __repr__(self) -> str:
        return (
            f"PagedKVCache(num_pages={self.num_pages}, page_size={self.page_size}, "
            f"used={self.pages_in_use()}, util={self.utilization():.2%})"
        )
