"""Shared voicemode daemon: per-client identity, idle exit, serve --daemon.

One daemon serves every Claude Code session, so "who am I" can no longer be
this process's pid. These tests pin that the conch and the convomode floor
tell clients apart inside ONE process, that legacy (pid-only) records still
compare correctly, and that the daemon goes idle only with no tool call
running.
"""

import asyncio
import atexit
import json
import os
import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from voice_mode import client_identity
from voice_mode.client_identity import ClientIdentity, owner_for_pid
from voice_mode.conch import Conch
from voice_mode.daemon import ClientIdentityMiddleware, IdleExit


# Real, live client pids: a record whose owner pid is dead now counts as stale.
_LIVE = [subprocess.Popen(["sleep", "600"]) for _ in range(3)]
PID_A, PID_B, PID_C = (p.pid for p in _LIVE)
atexit.register(lambda: [p.kill() for p in _LIVE])


def _as(pid):
    """Context manager-ish: bind a client identity for pid."""
    return client_identity.bind(ClientIdentity(owner_for_pid(pid), pid))


@pytest.fixture(autouse=True)
def _reset_identity(monkeypatch):
    monkeypatch.setattr(client_identity, "SHARED_MODE", False)
    client_identity._peer_cache.clear()
    yield
    client_identity._peer_cache.clear()


class TestOwnerTokens:
    def test_unbound_owner_is_this_process(self):
        assert client_identity.current_owner() == f"pid:{os.getpid()}"
        assert client_identity.current_client_pid() is None

    def test_bound_owner_is_the_client(self):
        token = _as(4242)
        try:
            assert client_identity.current_owner() == "pid:4242"
            assert client_identity.current_client_pid() == 4242
        finally:
            client_identity.unbind(token)
        assert client_identity.current_owner() == f"pid:{os.getpid()}"

    def test_legacy_record_owner_comes_from_pid(self):
        assert client_identity.record_owner({"pid": 77}) == "pid:77"
        assert client_identity.record_owner({"pid": 77, "owner": "pid:9"}) == "pid:9"
        assert client_identity.record_owner({}) is None
        assert client_identity.record_owner({"pid": True}) is None


class TestIdentifyRequest:
    def _request(self, headers=None, client=("127.0.0.1", 50123), server=("127.0.0.1", 8765)):
        return SimpleNamespace(headers=headers or {},
                               scope={"client": client, "server": server})

    def test_header_wins(self):
        ident = client_identity.identify_request(
            self._request({client_identity.CLIENT_PID_HEADER: "5150"}))
        assert ident == ClientIdentity("pid:5150", 5150)

    def test_loopback_peer_resolved_and_cached(self, monkeypatch):
        calls = []

        def fake_lookup(client_port, server_port):
            calls.append((client_port, server_port))
            return PID_A
        monkeypatch.setattr(client_identity, "_lsof_peer_pid", fake_lookup)
        req = self._request()
        assert client_identity.identify_request(req) == ClientIdentity(f"pid:{PID_A}", PID_A)
        assert client_identity.identify_request(req).pid == PID_A
        assert len(calls) == 1 and calls[0] == (50123, 8765)

    def test_peer_cache_is_keyed_by_both_ports_and_expires(self, monkeypatch):
        answers = iter([PID_A, PID_B, PID_C])
        monkeypatch.setattr(client_identity, "_lsof_peer_pid", lambda c, s: next(answers))
        assert client_identity.peer_pid(50123, 8765) == PID_A
        assert client_identity.peer_pid(50123, 9999) == PID_B  # other server port
        clock = [client_identity.time.monotonic() + client_identity._PEER_CACHE_TTL + 1]
        monkeypatch.setattr(client_identity.time, "monotonic", lambda: clock[0])
        assert client_identity.peer_pid(50123, 8765) == PID_C  # port reuse after TTL

    def test_cached_peer_that_died_is_re_resolved(self, monkeypatch):
        dead = subprocess.Popen(["true"])
        dead.wait()
        answers = iter([dead.pid, 777])
        monkeypatch.setattr(client_identity, "_lsof_peer_pid", lambda c, s: next(answers))
        assert client_identity.peer_pid(50124, 8765) == dead.pid
        assert client_identity.peer_pid(50124, 8765) == 777

    def test_non_loopback_is_unidentified(self, monkeypatch):
        monkeypatch.setattr(client_identity, "_lsof_peer_pid", lambda *a: 1)
        assert client_identity.identify_request(
            self._request(client=("10.0.0.5", 4000))) is None

    def test_lsof_output_parsing_picks_the_client_end(self, monkeypatch):
        me = os.getpid()
        out = (f"p{me}\nn127.0.0.1:8765->127.0.0.1:50123\n"
               "p999\nn127.0.0.1:50123->127.0.0.1:8765\n")
        monkeypatch.setattr(subprocess, "run",
                            lambda *a, **k: SimpleNamespace(stdout=out))
        assert client_identity._lsof_peer_pid(50123, 8765) == 999


class TestConchAcrossClientsInOneProcess:
    def test_second_client_is_refused_and_can_ask_for_yield(self):
        t = _as(PID_A)
        try:
            a = Conch(agent_name="a")
            assert a.try_acquire()
        finally:
            client_identity.unbind(t)
        record = json.loads(Conch.LOCK_FILE.read_text())
        assert record["owner"] == f"pid:{PID_A}" and record["pid"] == os.getpid()

        t = _as(PID_B)
        try:
            b = Conch(agent_name="b")
            assert not b.try_acquire()
            Conch.request_yield("b")
        finally:
            client_identity.unbind(t)

        # The holder (checked from a thread with no bound client) sees B's ask;
        # B never sees its own.
        assert Conch.is_wanted(owner=a.owner) is True
        assert a.should_yield() is True
        assert Conch.is_wanted(owner=b.owner) is False
        assert a.is_superseded() is False

        # Only the requester clears its request.
        Conch.clear_yield_request(owner=a.owner)
        assert Conch.WANTED_FILE.exists()
        Conch.clear_yield_request(owner=b.owner)
        assert not Conch.WANTED_FILE.exists()
        a.release()

    def test_preempt_by_another_client_supersedes_holder(self):
        t = _as(PID_A)
        try:
            a = Conch(agent_name="a")
            assert a.try_acquire()
        finally:
            client_identity.unbind(t)
        t = _as(PID_B)
        try:
            b = Conch(agent_name="b")
            assert b.preempt_acquire()
        finally:
            client_identity.unbind(t)
        assert a.is_superseded() is True
        assert b.is_superseded() is False
        a.release()
        b.release()

    def test_legacy_pid_only_record_from_another_process_is_not_ours(self):
        a = Conch(agent_name="a")
        assert a.try_acquire()
        Conch.LOCK_FILE.write_text(json.dumps({"pid": os.getpid() + 1, "epoch": a._epoch}))
        assert a.is_superseded() is True
        a.release()

    def test_stdio_behaviour_unchanged_without_a_client(self):
        a = Conch(agent_name="a")
        assert a.owner == f"pid:{os.getpid()}"
        assert a.try_acquire()
        assert json.loads(Conch.LOCK_FILE.read_text())["owner"] == a.owner
        assert a.is_superseded() is False
        a.release()


class TestFloorIdentity:
    @pytest.fixture
    def floor(self, tmp_path, monkeypatch):
        from voice_mode import pause_resume
        path = tmp_path / "floor-queue.json"
        monkeypatch.setattr(pause_resume, "FLOOR_QUEUE_PATH", str(path))

        def write(pid):
            path.write_text(json.dumps({"floor_holder": {"agent": "x", "pid": pid}}))
        return write

    def test_bound_client_is_the_session(self, floor, monkeypatch):
        from voice_mode import pause_resume
        monkeypatch.setattr(pause_resume, "_ancestor_pids", lambda pid=None, limit=12: {pid})
        monkeypatch.setattr(client_identity, "SHARED_MODE", True)
        floor(PID_A)
        t = _as(PID_A)
        try:
            assert pause_resume.we_hold_floor() is True
        finally:
            client_identity.unbind(t)
        t = _as(PID_B)
        try:
            assert pause_resume.we_hold_floor() is False
            assert pause_resume.floor_handed_off(True) is True
        finally:
            client_identity.unbind(t)

    def test_shared_mode_without_client_is_unknown_not_handed_off(self, floor, monkeypatch):
        from voice_mode import pause_resume
        monkeypatch.setattr(client_identity, "SHARED_MODE", True)
        floor(PID_A)
        assert pause_resume.we_hold_floor() is None
        assert pause_resume.floor_handed_off(True) is False

    def test_stdio_still_walks_own_ancestors(self, floor, monkeypatch):
        from voice_mode import pause_resume
        seen = {}

        def fake_ancestors(pid=None, limit=12):
            seen["pid"] = pid
            return {os.getpid()}
        monkeypatch.setattr(pause_resume, "_ancestor_pids", fake_ancestors)
        floor(os.getpid())
        assert pause_resume.we_hold_floor() is True
        assert seen["pid"] is None


class TestIdleExit:
    def test_idle_only_with_nothing_in_flight(self):
        now = [0.0]
        idle = IdleExit(100, clock=lambda: now[0])
        now[0] = 99
        assert not idle.is_idle()
        idle.begin()
        now[0] = 500
        assert not idle.is_idle(), "a running tool call (a listen) keeps it alive"
        idle.end()
        now[0] = 599
        assert not idle.is_idle()
        now[0] = 600
        assert idle.is_idle()

    def test_zero_disables(self):
        idle = IdleExit(0, clock=lambda: 1e9)
        assert not idle.is_idle()

    def test_watch_fires_once_idle(self):
        now = [0.0]
        idle = IdleExit(10, clock=lambda: now[0])
        fired = []

        async def run():
            task = asyncio.ensure_future(idle.watch(lambda: fired.append(True), poll=0.01))
            await asyncio.sleep(0.05)
            assert not fired
            now[0] = 11
            await asyncio.wait_for(task, 1)
        asyncio.run(run())
        assert fired == [True]


class TestMiddleware:
    def test_binds_identity_for_the_call_and_counts_it(self):
        idle = IdleExit(100)
        mw = ClientIdentityMiddleware(idle)
        request = SimpleNamespace(headers={client_identity.CLIENT_PID_HEADER: "6060"},
                                  scope={})
        seen = {}

        async def call_next(ctx):
            seen["owner"] = client_identity.current_owner()
            seen["in_flight"] = idle.in_flight
            return "ok"

        with patch("voice_mode.daemon.get_http_request", return_value=request):
            assert asyncio.run(mw.on_call_tool(MagicMock(), call_next)) == "ok"
        assert seen == {"owner": "pid:6060", "in_flight": 1}
        assert idle.in_flight == 0
        assert client_identity.current() is None

    def test_unidentified_client_gets_unique_anon_owner_and_unknown_floor(self):
        mw = ClientIdentityMiddleware()
        seen = []

        async def call_next(ctx):
            seen.append((client_identity.current_owner(), client_identity.current_client_pid()))
            return "ok"

        with patch("voice_mode.daemon.get_http_request", side_effect=RuntimeError):
            asyncio.run(mw.on_call_tool(MagicMock(), call_next))
            asyncio.run(mw.on_call_tool(MagicMock(), call_next))
        assert all(o.startswith("anon:") and pid is None for o, pid in seen)
        assert seen[0][0] != seen[1][0]
        assert seen[0][0] != owner_for_pid(os.getpid())

    def test_client_death_mid_call_cancels_the_call_and_frees_the_conch(self, tmp_path, monkeypatch):
        import signal
        import time
        monkeypatch.setattr(Conch, "LOCK_FILE", tmp_path / "conch")
        victim = subprocess.Popen(["sleep", "60"])
        request = SimpleNamespace(
            headers={client_identity.CLIENT_PID_HEADER: str(victim.pid)}, scope={})
        mw = ClientIdentityMiddleware(poll=0.1)
        state = {}

        async def call_next(ctx):
            conch = Conch(agent_name="victim")
            assert conch.try_acquire()
            state["conch"] = conch
            try:
                await asyncio.sleep(60)  # a long listen
            finally:
                conch.release()
                state["released"] = True

        async def scenario():
            with patch("voice_mode.daemon.get_http_request", return_value=request):
                call = asyncio.ensure_future(mw.on_call_tool(MagicMock(), call_next))
                await asyncio.sleep(0.3)
                assert Conch.read_lock_state()[0] == Conch.STATE_HELD
                # While the client lives, another client cannot take the conch.
                tok = _as(os.getpid() + 1)
                try:
                    assert not Conch(agent_name="other").try_acquire()
                finally:
                    client_identity.unbind(tok)
                victim.send_signal(signal.SIGKILL)
                victim.wait()
                t0 = time.monotonic()
                with pytest.raises(asyncio.CancelledError):
                    await call
                tok = _as(os.getpid() + 1)
                try:
                    other = Conch(agent_name="other")
                    assert other.try_acquire()
                    other.release()
                finally:
                    client_identity.unbind(tok)
                return time.monotonic() - t0

        elapsed = asyncio.run(scenario())
        assert state.get("released") is True
        assert elapsed < 3

    def test_dead_owner_record_is_stale_even_though_daemon_pid_is_alive(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Conch, "LOCK_FILE", tmp_path / "conch")
        dead = subprocess.Popen(["true"])
        dead.wait()
        (tmp_path / "conch").write_text(json.dumps({
            "pid": os.getpid(), "owner": owner_for_pid(dead.pid), "agent": "x",
            "acquired": "2999-01-01T00:00:00"}))
        assert Conch.read_lock_state()[0] == Conch.STATE_FREE
        assert not Conch.is_active()
        tok = _as(os.getpid() + 1)
        try:
            c = Conch(agent_name="new")
            assert c.try_acquire()
            c.release()
        finally:
            client_identity.unbind(tok)

    def test_anon_owner_is_never_judged_dead(self):
        assert not client_identity.owner_is_dead("anon:abc")
        assert not client_identity.owner_is_dead(owner_for_pid(os.getpid()))


class TestServeDaemonFlag:
    def test_daemon_implies_stateless_idle_exit_and_identity(self, monkeypatch):
        from click.testing import CliRunner
        from voice_mode.cli import voice_mode_main_cli
        from voice_mode.server import mcp

        monkeypatch.delenv("VOICEMODE_DAEMON_IDLE_EXIT_SECONDS", raising=False)
        added = []
        monkeypatch.setattr(mcp, "add_middleware", added.append)
        http_app = MagicMock()
        monkeypatch.setattr(mcp, "http_app", http_app)
        server = MagicMock()

        async def serve(sockets=None):
            return None
        server.return_value.serve = serve
        exits = []

        def fake_exit(code):
            exits.append(code)
            raise SystemExit(code)
        monkeypatch.setattr(os, "_exit", fake_exit)
        with patch("uvicorn.Server", server), patch("uvicorn.Config"), \
                patch("voice_mode.cli._init_event_logging"):
            result = CliRunner().invoke(voice_mode_main_cli, ["serve", "--daemon", "--port", "1"])
        assert result.exit_code == 0, result.output
        assert exits == [0]  # F6: hard exit even though the idle watcher never fired
        assert http_app.call_args.kwargs["stateless_http"] is True
        assert len(added) == 1 and isinstance(added[0], ClientIdentityMiddleware)
        assert added[0].idle.seconds == 2700
        assert client_identity.SHARED_MODE is True
        assert "Idle exit after 2700s" in result.output
