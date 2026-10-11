"""Tests for the versioned evaluation cache."""

from unittest.mock import patch

from backend.app.services.persistent_cache import load_cached_json, save_cached_json


def test_persistent_cache_round_trip_and_expiry(tmp_path):
    cache_path = tmp_path / "cache.sqlite3"
    key = {"query": "HotPotQA", "limit": 40}
    with patch(
        "backend.app.services.persistent_cache._cache_path",
        return_value=cache_path,
    ):
        save_cached_json("academic:arxiv", key, {"papers": [{"id": "one"}]})
        assert load_cached_json(
            "academic:arxiv",
            key,
            ttl_seconds=60,
        ) == {"papers": [{"id": "one"}]}
        assert load_cached_json(
            "academic:arxiv",
            key,
            ttl_seconds=0,
        ) is None


def test_persistent_cache_keys_include_request_parameters(tmp_path):
    cache_path = tmp_path / "cache.sqlite3"
    with patch(
        "backend.app.services.persistent_cache._cache_path",
        return_value=cache_path,
    ):
        save_cached_json("tavily", {"query": "one"}, {"value": 1})
        assert load_cached_json(
            "tavily",
            {"query": "two"},
            ttl_seconds=60,
        ) is None


def test_frozen_snapshot_can_replay_an_expired_entry(tmp_path):
    cache_path = tmp_path / "cache.sqlite3"
    key = {"query": "stable benchmark"}
    with patch(
        "backend.app.services.persistent_cache._cache_path",
        return_value=cache_path,
    ), patch("backend.app.services.persistent_cache.time.time", return_value=100.0):
        save_cached_json("academic:arxiv", key, {"papers": [{"id": "one"}]})

    with patch(
        "backend.app.services.persistent_cache._cache_path",
        return_value=cache_path,
    ), patch("backend.app.services.persistent_cache.time.time", return_value=10_000.0):
        assert load_cached_json(
            "academic:arxiv",
            key,
            ttl_seconds=60,
            allow_expired=True,
        ) == {"papers": [{"id": "one"}]}
