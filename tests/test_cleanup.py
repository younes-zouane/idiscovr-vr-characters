"""
Tests for vr_service.py's cleanup sweep (Part 1.5): old output files get
deleted, idle sessions get evicted. Tests inject `now` and craft artificial
ages directly rather than sleeping for real minutes.
"""

import os
import time

import vr_service
from vr_service import OUTPUT_DIR, cleanup_once


def _touch_with_age(path, age_seconds, now):
    """Create a file and backdate its mtime to simulate age."""
    path.write_bytes(b"fake content")
    old_time = now - age_seconds
    os.utime(path, (old_time, old_time))


def test_cleanup_deletes_old_files_keeps_recent_ones():
    now = time.time()
    old_file = OUTPUT_DIR / "old_test_output.wav"
    recent_file = OUTPUT_DIR / "recent_test_output.wav"

    _touch_with_age(old_file, age_seconds=45 * 60, now=now)   # 45 min old — should go
    _touch_with_age(recent_file, age_seconds=5 * 60, now=now)  # 5 min old — should stay

    try:
        result = cleanup_once(now=now)
        assert result["deleted_files"] >= 1
        assert not old_file.exists()
        assert recent_file.exists()
    finally:
        recent_file.unlink(missing_ok=True)
        old_file.unlink(missing_ok=True)


def test_cleanup_evicts_idle_sessions_keeps_active_ones():
    now = time.time()
    vr_service._sessions.clear()
    vr_service._session_last_used.clear()

    vr_service._sessions["idle_session"] = {}
    vr_service._session_last_used["idle_session"] = now - (45 * 60)  # 45 min idle

    vr_service._sessions["active_session"] = {}
    vr_service._session_last_used["active_session"] = now - (5 * 60)  # 5 min idle

    result = cleanup_once(now=now)

    assert result["evicted_sessions"] == 1
    assert "idle_session" not in vr_service._sessions
    assert "active_session" in vr_service._sessions

    vr_service._sessions.clear()
    vr_service._session_last_used.clear()


def test_cleanup_no_op_when_nothing_is_stale():
    now = time.time()
    vr_service._sessions.clear()
    vr_service._session_last_used.clear()

    recent_file = OUTPUT_DIR / "brand_new_output.wav"
    _touch_with_age(recent_file, age_seconds=10, now=now)
    vr_service._sessions["fresh"] = {}
    vr_service._session_last_used["fresh"] = now - 10

    try:
        result = cleanup_once(now=now)
        assert result["deleted_files"] == 0
        assert result["evicted_sessions"] == 0
    finally:
        recent_file.unlink(missing_ok=True)
        vr_service._sessions.clear()
        vr_service._session_last_used.clear()


def test_get_or_create_session_records_last_used():
    vr_service._sessions.clear()
    vr_service._session_last_used.clear()

    session_id, _ = vr_service.get_or_create_session(None)
    assert session_id in vr_service._session_last_used

    first_seen = vr_service._session_last_used[session_id]
    vr_service.get_or_create_session(session_id)
    second_seen = vr_service._session_last_used[session_id]
    assert second_seen >= first_seen

    vr_service._sessions.clear()
    vr_service._session_last_used.clear()