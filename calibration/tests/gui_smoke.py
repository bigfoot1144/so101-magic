"""Run the real Qt panel and multiprocessing worker with a toy plant only.

QT_QPA_PLATFORM=offscreen python tests/gui_smoke.py --config work/config.json --params work/seed_params.json
"""
import argparse
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from PySide6 import QtCore, QtWidgets
    from so101_calibrate import launch_gui
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--params", required=True)
    parser.add_argument("--screenshot")
    args = parser.parse_args()
    app = QtWidgets.QApplication(sys.argv[:1])
    started = time.perf_counter()
    state = {"stage": 0, "error": None, "hold_at": None, "done": False}

    def poll():
        windows = [w for w in app.topLevelWidgets() if hasattr(w, "send") and hasattr(w, "state")]
        if not windows: return
        w = windows[0]
        try:
            if time.perf_counter() - started > 30: raise AssertionError("GUI smoke test timed out")
            if w.state.get("faulted"): raise AssertionError(w.state["message"])
            stage, mode = state["stage"], w.state.get("mode")
            if stage == 0 and mode == "OFF" and w.state.get("feedback"):
                w.ready.setChecked(True)
                w.arm()
                state["stage"] = 1
            elif stage == 1 and mode == "HOLD":
                state.update(stage=2, hold_at=time.perf_counter())
            elif stage == 2 and time.perf_counter() - state["hold_at"] > 1:
                w.send("freeze")
                state["stage"] = 3
            elif stage == 3 and w.state.get("frozen"):
                w.editors[1].setValue(w.editors[1].value() + .5)
                w.capture()
                state["stage"] = 4
            elif stage == 4 and w.state.get("pair_count") == 1:
                w.send("off")
                state["stage"] = 5
            elif stage == 5 and mode == "OFF":
                w.send("lock_fit")
                state["stage"] = 6
            elif stage == 6 and mode == "OFF_FIT":
                assert not w.arm_button.isEnabled(), "arm button enabled during offline lock"
                w.send("arm")  # Worker also rejects, independent of disabled widget.
                state.update(stage=7, hold_at=time.perf_counter())
            elif stage == 7 and time.perf_counter() - state["hold_at"] > .4:
                assert not w.state["armed"], "hardware worker ignored fit lock"
                w.send("unlock_fit")
                state["stage"] = 8
            elif stage == 8 and mode == "OFF":
                if args.screenshot: w.grab().save(str(Path(args.screenshot).resolve()))
                state["done"] = True
                timer.stop()
                w.close()
        except Exception as exc:
            state["error"] = str(exc)
            timer.stop()
            w.emergency()
            w.closing = True
            w.close()

    timer = QtCore.QTimer()
    timer.timeout.connect(poll)
    timer.start(100)
    with tempfile.TemporaryDirectory() as tmp:
        launch_gui(SimpleNamespace(config=args.config, params=args.params, port=None,
                                   out=tmp, demo=True, no_viewer=True))
    if state["error"]: raise RuntimeError(state["error"])
    assert state["done"], "GUI closed before checks completed"
    print("PASS: Qt panel, worker process, hold, pose capture, torque-off and offline-fit lock (synthetic only)")


if __name__ == "__main__": main()
