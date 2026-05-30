"""Unit tests for the dynamic batching engine."""

from __future__ import annotations

import pytest

from optimizer.batch_engine import ContinuousBatchEngine, Request, RequestQueue
from optimizer.kv_cache import PagedKVCache


@pytest.fixture
def kv_cache():
    return PagedKVCache(num_pages=64, page_size=16, num_heads=8, head_dim=64)


class TestRequestQueue:
    def test_fifo_ordering(self):
        q = RequestQueue()
        q.put(Request(request_id="a", prompt="p1", max_tokens=8))
        q.put(Request(request_id="b", prompt="p2", max_tokens=8))
        assert q.get().request_id == "a"
        assert q.get().request_id == "b"
        assert q.empty()

    def test_size_tracks_inserts_and_pops(self):
        q = RequestQueue()
        assert len(q) == 0
        q.put(Request(request_id="a", prompt="p", max_tokens=4))
        q.put(Request(request_id="b", prompt="p", max_tokens=4))
        assert len(q) == 2
        q.get()
        assert len(q) == 1


class TestContinuousBatchEngine:
    def test_initialization(self, kv_cache):
        engine = ContinuousBatchEngine(max_batch_size=32, kv_cache=kv_cache)
        assert engine.max_batch_size == 32
        assert engine.kv_cache is kv_cache

    def test_admit_respects_batch_limit(self, kv_cache):
        engine = ContinuousBatchEngine(max_batch_size=2, kv_cache=kv_cache)
        for i in range(5):
            engine.enqueue(Request(request_id=f"r{i}", prompt="p", max_tokens=8))
        active = engine._admit_new_requests()
        assert len(active) <= engine.max_batch_size

    def test_enqueue_grows_pending(self, kv_cache):
        engine = ContinuousBatchEngine(max_batch_size=4, kv_cache=kv_cache)
        engine.enqueue(Request(request_id="r0", prompt="hello", max_tokens=4))
        assert len(engine._pending) == 1
