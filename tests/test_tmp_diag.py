import faulthandler, sys, time
from tests.test_send_always_ends_listen import *  # noqa
from tests.test_send_always_ends_listen import sim, _listen_async, _MuteFlipStream
def test_diag(sim, monkeypatch):
    monkeypatch.setattr(C.sd, "InputStream", _MuteFlipStream)
    (sim / "hold-mode").touch()
    _MuteFlipStream.before = 10; _MuteFlipStream.effect = "stall"
    t0=time.monotonic()
    t, box = _listen_async(max_duration=30.0)
    time.sleep(0.4); (sim / "turn-end.signal").touch()
    for _ in range(3):
        time.sleep(0.4)
        print(f"\n--- t={time.monotonic()-t0:.1f} alive={t.is_alive()} stats={box['stats']}", file=sys.stderr)
        if t.is_alive():
            faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
    t.join(10)
