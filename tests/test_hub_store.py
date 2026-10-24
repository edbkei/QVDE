from qvde_common import protocol
from qvde_hub.buffer import HubStore

SAMPLE = {"accel_x": 0.1, "accel_y": 0.0, "accel_z": 9.8, "gyro_roll": 1, "gyro_pitch": 2, "gyro_yaw": 3}


def derived(store, house, ts):
    env = protocol.make_envelope("derivado", house, {"p": 0.1})
    store.enqueue(env, protocol.up_topic(house, "derivado"), ts_ref=ts)
    return env["msg_id"]


def test_samples_roundtrip():
    s = HubStore()
    for i in range(20):
        s.add_sample("h", "s1", 1000 + i, SAMPLE)
    assert s.count_since("h", "s1", 1010) == 10
    rows = s.recent("h", "s1", 5)
    assert [r[0] for r in rows] == [1015, 1016, 1017, 1018, 1019]
    assert len(s.window("h", "s1", 1000, 1004)) == 5


def test_cleanup_waits_for_cloud_confirmation():
    s = HubStore()
    for t in range(0, 100):
        s.add_sample("h", "s1", float(t), SAMPLE)
    first = derived(s, "h", 10.0)
    second = derived(s, "h", 60.0)
    # retention expired for everything (now=1000, retention=1s) but nothing confirmed
    assert s.cleanup(1.0, now=1000.0)["raw"] == 0
    s.confirm([first])
    # watermark = last confirmed record (t=10): samples with ts < 10 may go
    assert s.cleanup(1.0, now=1000.0)["raw"] == 10
    s.confirm([second])
    assert s.cleanup(1.0, now=1000.0)["raw"] == 50   # now ts < 60


def test_out_of_order_confirmation_keeps_gap():
    s = HubStore()
    for t in range(0, 100):
        s.add_sample("h", "s1", float(t), SAMPLE)
    _first, second = derived(s, "h", 10.0), derived(s, "h", 60.0)
    s.confirm([second])                      # the older record is still pending
    assert s.cleanup(1.0, now=1000.0)["raw"] == 0


def test_retention_respected_even_when_confirmed():
    s = HubStore()
    for t in range(0, 100):
        s.add_sample("h", "s1", float(t), SAMPLE)
    s.confirm([derived(s, "h", 99.0)])
    assert s.cleanup({"h": 50.0}, now=100.0)["raw"] == 50   # only ts < 50


def test_pending_and_resend():
    s = HubStore()
    m = derived(s, "h", 1.0)
    assert [r[0] for r in s.pending()] == [m]
    s.mark_sent([m])
    assert s.pending(resend_after_s=30) == []
    assert [r[0] for r in s.pending(resend_after_s=0)] == [m]   # resend if no ack
    assert s.confirm([m]) == 1 and s.backlog() == 0
