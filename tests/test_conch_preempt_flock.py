"""Preemption must not leave a record-without-flock window open to a third party.

Regression tests for two residuals from PR #12:
  * a preempter can hold the lock RECORD without holding the flock, so a third
    process that wins the flock steals a live, fresh claim;
  * two concurrent preempters racing a holder neither can flock could both
    believe they hold the conch.
"""

import json
import threading
import time

import pytest

from voice_mode.conch import Conch


@pytest.fixture(autouse=True)
def _no_flock_wait(monkeypatch):
    """Default: the superseded holder never drops its flock (stuck holder)."""
    monkeypatch.setattr(Conch, "PREEMPT_FLOCK_WAIT", 0.0, raising=False)


def _record():
    return json.loads(Conch.LOCK_FILE.read_text())


def test_fresh_live_record_without_flock_is_not_stealable_by_try_acquire():
    holder = Conch(agent_name="holder")
    assert holder.try_acquire()
    preempter = Conch(agent_name="preempter")
    assert preempter.preempt_acquire("preempter")
    assert preempter._fd is None, "setup: epoch-only claim (holder still flocks)"

    holder.release()  # superseded holder stands down, drops its flock
    third = Conch(agent_name="third")
    assert third.try_acquire() is False, (
        "a third process won the flock and stole a live, fresh preempter record"
    )
    assert _record()["agent"] == "preempter"
    assert preempter.is_superseded() is False
    preempter.release()


def test_preempter_retakes_the_flock_when_the_holder_stands_down(monkeypatch):
    monkeypatch.setattr(Conch, "PREEMPT_FLOCK_WAIT", 3.0)
    holder = Conch(agent_name="holder")
    assert holder.try_acquire()
    threading.Timer(0.15, holder.release).start()

    preempter = Conch(agent_name="preempter")
    assert preempter.preempt_acquire("preempter")
    assert preempter._fd is not None, "preempter never re-attempted the flock"
    assert Conch(agent_name="x").try_acquire() is False
    preempter.release()


def test_second_preempter_refuses_a_fresh_preemption():
    holder = Conch(agent_name="holder")
    assert holder.try_acquire()
    a = Conch(agent_name="a")
    b = Conch(agent_name="b")
    assert a.preempt_acquire("a")
    assert b.preempt_acquire("b") is False
    assert b._acquired is False
    assert _record()["agent"] == "a" and a.is_superseded() is False


def test_preempt_allowed_again_after_the_settle_window(monkeypatch):
    monkeypatch.setattr(Conch, "PREEMPT_SETTLE_SECONDS", 0.0, raising=False)
    holder = Conch(agent_name="holder")
    assert holder.try_acquire()
    a, b = Conch(agent_name="a"), Conch(agent_name="b")
    assert a.preempt_acquire("a")
    assert b.preempt_acquire("b") is True
    assert a.is_superseded() is True


def test_concurrent_preempters_yield_exactly_one_winner():
    holder = Conch(agent_name="holder")
    assert holder.try_acquire()
    contenders = [Conch(agent_name=f"p{i}") for i in range(8)]
    results = []
    gate = threading.Barrier(len(contenders))

    def go(c):
        gate.wait()
        results.append(c.preempt_acquire(c.agent_name))

    threads = [threading.Thread(target=go, args=(c,)) for c in contenders]
    [t.start() for t in threads]
    [t.join(timeout=30) for t in threads]
    assert results.count(True) == 1, results
    assert sum(c._acquired for c in contenders) == 1
