#!/usr/bin/env python3
"""Desktop pose matching and automated SO-101 identification. See README.md."""
from __future__ import annotations

import argparse
import copy
import multiprocessing as mp
import os
import queue
import signal
import sys
import time
from pathlib import Path

import numpy as np
import so101_sysid as s
from so101_calibration_core import Mechanics, atomic_json, snapshot_model, stamp, worker


def launch_gui(args):
    from PySide6 import QtCore, QtGui, QtWidgets as W
    import mujoco
    import mujoco.viewer

    if not args.demo and not (args.port and args.port.strip()):
        raise ValueError("Set --port to the follower's actual /dev/serial/by-id/... path. FOLLOWER_PORT is empty.")
    if args.no_viewer and not args.demo:
        raise ValueError("--no-viewer is for software/demo testing only.")
    source_cfg, source_xml = s.load_config(args.config)
    if source_cfg.get("synthetic") and not args.demo:
        raise ValueError("A synthetic mapping cannot be used with hardware.")
    params_path = Path(args.params).resolve()
    params = s.read_json(params_path)
    if params.get("calibration_interface_fit", {}).get("synthetic") and not args.demo:
        raise ValueError("Synthetic fitted parameters cannot be used for a physical session.")
    session = Path(args.out).resolve() / (("DEMO-" if args.demo else "session-") + stamp())
    session.mkdir(parents=True, exist_ok=False)
    cfg = copy.deepcopy(source_cfg)
    if args.demo:
        cfg.update(mapping_reviewed=True, synthetic=True)
    cfg = snapshot_model(cfg, source_xml, session)
    atomic_json(session / "initial_params.json", params)
    atomic_json(session / "session_info.json", {"synthetic": args.demo, "source_config": str(Path(args.config).resolve()),
                                               "source_params": str(params_path), "created": stamp()})
    mechanics = Mechanics(session / cfg["xml"])
    context = mp.get_context("spawn")
    inbox, outbox = context.Queue(maxsize=100), context.Queue(maxsize=4)
    stop_event = context.Event()
    process = context.Process(target=worker,
                args=(str(session / "config.json"), args.port, str(session), args.demo, inbox, outbox, stop_event))
    app = W.QApplication.instance() or W.QApplication(sys.argv[:1])
    app.setApplicationName("SO-101 calibration")

    class Window(W.QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("SO-101 calibration" + (" | DEMO: no serial connection" if args.demo else " | physical robot"))
            self.resize(1160, 870)
            self.state = {}
            self.viewer = None
            self.had_viewer = False
            self.last_state_time = time.perf_counter()
            self.initialized = False
            self.frozen_time = None
            self.closing = False
            self.fit_process = None
            self.fit_output = None
            self.fit_requested_at = None
            central = W.QWidget()
            self.setCentralWidget(central)
            layout = W.QVBoxLayout(central)
            title = W.QLabel("SO-101 · Pose matching and motion identification")
            title.setStyleSheet("font-size: 21px; font-weight: 600;")
            layout.addWidget(title)
            self.status = W.QLabel("Connecting read-only...")
            self.status.setWordWrap(True)
            self.status.setMinimumHeight(50)
            self.status.setStyleSheet("padding: 9px; background: #e8f0fa; color: #182b43;")
            layout.addWidget(self.status)
            self.temperature_alert = W.QLabel()
            self.temperature_alert.setWordWrap(True)
            self.temperature_alert.setMinimumHeight(76)
            self.temperature_alert.setStyleSheet(
                "background: #a71919; color: white; font-size: 19px; font-weight: bold; padding: 12px;")
            self.temperature_alert.hide()
            layout.addWidget(self.temperature_alert)
            note = W.QLabel("Preview edits change only the rendered pose. Motor movement requires an explicit button click.\n"
                            "Faults, lost UI heartbeat, or the 5-minute hold timeout release torque. Keep a catch area and power switch accessible.")
            note.setWordWrap(True)
            layout.addWidget(note)
            toolbar = W.QHBoxLayout()
            self.ready = W.QCheckBox("Arm supported; workspace clear")
            toolbar.addWidget(self.ready)
            self.arm_button = self.button(toolbar, "Enable hold · P=16, I=D=0", self.arm)
            self.off_button = self.button(toolbar, "Supported: torque OFF", lambda: self.send("off"))
            self.stop_button = self.button(toolbar, "STOP / torque OFF", self.emergency)
            self.stop_button.setStyleSheet("background: #a92222; color: white; font-weight: bold; padding: 9px;")
            layout.addLayout(toolbar)

            self.table = W.QTableWidget(6, 6)
            self.table.setHorizontalHeaderLabels(["Joint", "Encoder (deg)", "Sent goal (deg)", "Edit / preview (deg)", "Gravity estimate (N·m)", "Temperature (°C)"])
            self.table.verticalHeader().setVisible(False)
            self.table.setEditTriggers(W.QAbstractItemView.EditTrigger.NoEditTriggers)
            self.table.horizontalHeader().setSectionResizeMode(W.QHeaderView.ResizeMode.Stretch)
            self.table.setMinimumHeight(260)
            self.editors = []
            for k, joint in enumerate(s.JOINTS):
                for column in (0, 1, 2, 4, 5):
                    self.table.setItem(k, column, W.QTableWidgetItem(joint if column == 0 else "..."))
                editor = W.QDoubleSpinBox()
                editor.setRange(float(np.rad2deg(mechanics.limits[k, 0])), float(np.rad2deg(mechanics.limits[k, 1])))
                editor.setDecimals(2)
                editor.setSingleStep(.25)
                editor.setKeyboardTracking(False)
                editor.setAccelerated(True)
                self.editors.append(editor)
                cell = W.QWidget()
                cell_layout = W.QVBoxLayout(cell)
                cell_layout.setContentsMargins(3, 1, 3, 1)
                cell_layout.setSpacing(0)
                slider = W.QSlider(QtCore.Qt.Orientation.Horizontal)
                slider.setRange(int(np.ceil(editor.minimum() * 100)), int(np.floor(editor.maximum() * 100)))
                slider.valueChanged.connect(lambda v, e=editor: e.setValue(v / 100))
                editor.valueChanged.connect(lambda v, bar=slider: bar.setValue(int(round(v * 100))))
                cell_layout.addWidget(editor)
                cell_layout.addWidget(slider)
                self.table.setCellWidget(k, 3, cell)
                self.table.setRowHeight(k, 48)
            self.table.setMinimumHeight(326)
            layout.addWidget(self.table)
            layout.addWidget(W.QLabel("Gravity column uses the EDITED pose and the frozen CAD model, with velocity zero. It is not measured motor torque."))
            viewbar = W.QHBoxLayout()
            self.button(viewbar, "Preview current encoder pose", self.copy_encoder)
            self.button(viewbar, "Preview current sent goal", self.copy_goal)
            self.move_button = self.button(viewbar, "MOVE ROBOT to preview", self.move)
            self.cancel_button = self.button(viewbar, "Cancel motion; keep holding", lambda: self.send("cancel"))
            self.view_button = self.button(viewbar, "Open MuJoCo viewer", self.open_viewer)
            layout.addLayout(viewbar)
            visibility = W.QHBoxLayout()
            visibility.addWidget(W.QLabel("Viewer displays:"))
            self.view_mode = W.QComboBox()
            self.view_mode.addItems(["Edited reference / move preview", "Live encoder pose", "Last sent motor goal"])
            visibility.addWidget(self.view_mode)
            visibility.addStretch()
            layout.addLayout(visibility)

            tabs = W.QTabWidget()
            layout.addWidget(tabs)
            pose_tab = W.QWidget()
            pose_layout = W.QVBoxLayout(pose_tab)
            pose_layout.addWidget(W.QLabel("1. Freeze a stable measurement.  2. Adjust the rendered joints to match the physical links.\n"
                                           "3. Save the pair. Repeat across different poses. Signs remain those reviewed in align."))
            pose_buttons = W.QHBoxLayout()
            self.freeze_button = self.button(pose_buttons, "Freeze measurement for matching", lambda: self.send("freeze"))
            self.capture_button = self.button(pose_buttons, "Save matched pose + gravity", self.capture)
            self.offset_button = self.button(pose_buttons, "Fit offsets and save candidate", lambda: self.send("offsets"))
            pose_layout.addLayout(pose_buttons)
            method_row = W.QHBoxLayout()
            method_row.addWidget(W.QLabel("Reference measurement:"))
            self.method = W.QComboBox()
            self.method.addItems(["Visual match", "Measured link angles", "External pose measurement converted to joint angles"])
            method_row.addWidget(self.method)
            pose_layout.addLayout(method_row)
            self.pose_count = W.QLabel("No matched poses saved yet.")
            pose_layout.addWidget(self.pose_count)
            pose_layout.addWidget(W.QLabel("Visual matching is approximate. A digital angle gauge or calibrated external tracking improves accuracy without disassembly.\n"
                                          "Fitting offsets writes a candidate config. It never changes your current mapping or motor EEPROM."))
            pose_layout.addStretch()
            tabs.addTab(pose_tab, "Pose calibration")

            motion_tab = W.QWidget()
            motion_layout = W.QVBoxLayout(motion_tab)
            motion_layout.addWidget(W.QLabel("One automatic run: two training motions, then a different held-out motion. About 47 seconds.\n"
                                            "All six joints move together by at most ±3° (gripper ±2°), with an 8°/s command speed limit."))
            motion_buttons = W.QHBoxLayout()
            self.suite_button = self.button(motion_buttons, "RECORD coordinated run", lambda: self.send("suite"))
            self.fit_button = self.button(motion_buttons, "FIT + validate all complete runs (torque OFF)", self.fit)
            self.cancel_fit_button = self.button(motion_buttons, "Cancel offline fit", self.cancel_fit)
            motion_layout.addLayout(motion_buttons)
            motion_layout.addWidget(W.QLabel("Repeat recording at several clear poses before fitting. Fitting may take tens of minutes or longer on a Jetson.\n"
                                            "The first fit uses a bounded search budget. Its report shows residuals and optimizer limits; it does not certify sim-to-real."))
            self.fit_log = W.QPlainTextEdit()
            self.fit_log.setReadOnly(True)
            self.fit_log.setMaximumBlockCount(1000)
            motion_layout.addWidget(self.fit_log)
            tabs.addTab(motion_tab, "Automatic dynamics")

            footer = W.QHBoxLayout()
            folder_label = W.QLabel(str(session))
            folder_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            footer.addWidget(folder_label, 1)
            self.button(footer, "Open results folder", lambda: QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(session))))
            layout.addLayout(footer)
            self.timer = QtCore.QTimer(self)
            self.timer.timeout.connect(self.refresh)
            self.timer.start(100)
            if not args.no_viewer:
                QtCore.QTimer.singleShot(200, self.open_viewer)

        def button(self, layout, label, fn):
            button = W.QPushButton(label)
            button.clicked.connect(fn)
            layout.addWidget(button)
            return button

        def send(self, action, **fields):
            if stop_event.is_set(): return
            try: inbox.put_nowait({"action": action, "sent": time.perf_counter(), **fields})
            except queue.Full:
                stop_event.set()
                self.status.setText("Command queue stalled. Stopping worker and disabling torque.")

        def q(self): return np.deg2rad([e.value() for e in self.editors])

        def set_q(self, q):
            q = np.rad2deg(q)
            for editor, value in zip(self.editors, q): editor.setValue(float(value))
            self.view_mode.setCurrentIndex(0)

        def copy_encoder(self):
            row = self.state.get("feedback")
            if row: self.set_q(s.raw_to_sim(row["raw_position"], cfg))

        def copy_goal(self):
            if self.state.get("target_q_rad") is not None: self.set_q(self.state["target_q_rad"])

        def arm(self):
            if not self.ready.isChecked():
                self.status.setText("Support the arm in a clear pose away from the stops and check the readiness box first.")
                return
            self.send("arm")
            self.ready.setChecked(False)

        def move(self):
            self.view_mode.setCurrentIndex(0)
            self.send("move", q=self.q().tolist())

        def capture(self): self.send("capture", q=self.q().tolist(), method=self.method.currentText())

        def emergency(self):
            stop_event.set()
            self.status.setText("STOP requested. Worker disables torque and restores settings. Restart the application before reconnecting.")

        def open_viewer(self):
            if stop_event.is_set() or (self.viewer is not None and self.viewer.is_running()): return
            try:
                self.viewer = mujoco.viewer.launch_passive(mechanics.model, mechanics.data,
                                                           show_left_ui=False, show_right_ui=False)
                with self.viewer.lock():
                    self.viewer.cam.lookat[:] = [.0, .0, .18]
                    self.viewer.cam.distance = .8
                    self.viewer.cam.azimuth = 130
                    self.viewer.cam.elevation = -20
                    self.viewer.opt.geomgroup[2] = 1
                    self.viewer.opt.geomgroup[3] = 0
                self.had_viewer = True
            except Exception as exc:
                stop_event.set()
                self.status.setText(f"Viewer failed: {exc}. Hardware stop requested. Check the local desktop/OpenGL setup.")

        def fit(self):
            if self.state.get("armed") or self.state.get("mode") != "OFF": return
            self.fit_requested_at = time.perf_counter()
            self.send("lock_fit")

        def start_locked_fit(self):
            self.fit_requested_at = None
            self.fit_output = session / ("fit-" + stamp())
            self.fit_process = QtCore.QProcess(self)
            self.fit_process.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels)
            self.fit_process.readyReadStandardOutput.connect(self.fit_text)
            self.fit_process.finished.connect(self.fit_finished)
            self.fit_process.errorOccurred.connect(self.fit_error)
            arguments = [str(Path(__file__).resolve()), "fit", "--session", str(session), "--params", str(session / "initial_params.json"),
                         "--out", str(self.fit_output)]
            if args.demo: arguments.append("--allow-synthetic")
            self.fit_log.appendPlainText("Starting offline fit; no motor commands will be sent.\n")
            self.fit_process.start(sys.executable, arguments)

        def fit_error(self, error):
            self.fit_log.appendPlainText(f"Fit process error: {error}")
            if error == QtCore.QProcess.ProcessError.FailedToStart: self.send("unlock_fit")

        def fit_text(self):
            text = bytes(self.fit_process.readAllStandardOutput()).decode(errors="replace")
            self.fit_log.moveCursor(QtGui.QTextCursor.MoveOperation.End)
            self.fit_log.insertPlainText(text)

        def fit_finished(self, code, status):
            self.fit_text()
            self.fit_log.appendPlainText(f"\nFit process ended with code {code}. Results: {self.fit_output}")
            self.send("unlock_fit")

        def cancel_fit(self):
            if self.fit_process and self.fit_process.state() != QtCore.QProcess.ProcessState.NotRunning:
                self.fit_process.terminate()
                self.fit_log.appendPlainText("Cancelling offline fit. A training checkpoint is not a validated parameter file.")

        def refresh(self):
            self.send("heartbeat")
            for _ in range(10):
                try: new_state = outbox.get_nowait()
                except queue.Empty: break
                self.state = new_state
                self.last_state_time = time.perf_counter()
                if new_state.get("message"):
                    prefix = "DEMO · " if args.demo else ""
                    remaining = f" Hold timeout in {new_state['remaining_hold_s']:.0f}s." if new_state.get("armed") else ""
                    self.status.setText(prefix + new_state["mode"] + ": " + new_state["message"] + remaining)
            state = self.state
            temperature_warning = state.get("temperature_warning")
            self.temperature_alert.setText(temperature_warning or "")
            self.temperature_alert.setVisible(bool(temperature_warning))
            if self.fit_requested_at is not None:
                if state.get("mode") == "OFF_FIT": self.start_locked_fit()
                elif time.perf_counter() - self.fit_requested_at > 3:
                    self.fit_requested_at = None
                    self.fit_log.appendPlainText("Fit did not receive the worker's torque-OFF lock. Request rejected.")
            if state.get("armed") and time.perf_counter() - self.last_state_time > 1:
                stop_event.set()
                self.status.setText("Worker feedback stale. Stop requested; support arm and cut power if it does not release.")
            if not process.is_alive() and not state.get("closed") and not self.closing:
                stop_event.set()
                self.status.setText("Motor worker exited. " + state.get("message", "") + " If shutdown was not confirmed, support the arm and cut motor power.")
            row = state.get("feedback")
            if row:
                q_actual = s.raw_to_sim(row["raw_position"], cfg)
                if not self.initialized:
                    self.set_q(q_actual)
                    self.initialized = True
                for k in range(6):
                    self.table.item(k, 1).setText(f"{np.rad2deg(q_actual[k]):.2f}")
                    self.table.item(k, 5).setText(str(row["temperature_c"][k]))
                    if state.get("target_q_rad") is not None:
                        self.table.item(k, 2).setText(f"{np.rad2deg(state['target_q_rad'][k]):.2f}")
            frozen = state.get("frozen")
            if frozen and frozen["host_time"] != self.frozen_time:
                self.frozen_time = frozen["host_time"]
                self.set_q(frozen["mapped_q_rad"])
            running_fit = self.fit_process is not None and self.fit_process.state() != QtCore.QProcess.ProcessState.NotRunning
            available = bool(row) and not state.get("faulted") and not stop_event.is_set() and process.is_alive()
            view_available = args.no_viewer or (self.viewer is not None and self.viewer.is_running())
            mode, armed = state.get("mode"), state.get("armed", False)
            self.arm_button.setEnabled(available and view_available and mode == "OFF" and not armed and not running_fit and self.fit_requested_at is None)
            self.off_button.setEnabled(available and armed)
            self.move_button.setEnabled(available and armed and mode == "HOLD" and not running_fit)
            self.cancel_button.setEnabled(available and armed and mode in ("MOVE", "RECORD"))
            self.freeze_button.setEnabled(available and mode in ("OFF", "HOLD") and not running_fit)
            self.capture_button.setEnabled(available and bool(frozen) and mode in ("OFF", "HOLD") and not running_fit)
            self.offset_button.setEnabled(available and not armed and state.get("pair_count", 0) >= 3 and not running_fit)
            self.suite_button.setEnabled(available and armed and mode == "HOLD" and not running_fit)
            self.fit_button.setEnabled(available and mode == "OFF" and not armed and not running_fit and self.fit_requested_at is None)
            self.cancel_fit_button.setEnabled(bool(running_fit))
            self.pose_count.setText(f"Saved matched poses: {state.get('pair_count', 0)}. Offsets need at least three; varied link orientations are better.")
            if self.had_viewer and not self.viewer.is_running() and not stop_event.is_set():
                self.emergency()
            # Every shared MjData access is under viewer.lock(). No physics step:
            # this viewer is a kinematic reference, NOT an actuator prediction.
            from contextlib import nullcontext
            with self.viewer.lock() if self.viewer is not None and self.viewer.is_running() else nullcontext():
                torque = mechanics.gravity(self.q())
                for k, value in enumerate(torque): self.table.item(k, 4).setText(f"{value:+.4f}")
                shown = self.q()
                if self.view_mode.currentIndex() == 1 and row: shown = s.raw_to_sim(row["raw_position"], cfg)
                if self.view_mode.currentIndex() == 2 and state.get("target_q_rad") is not None: shown = state["target_q_rad"]
                mechanics.data.qpos[mechanics.qadr] = shown
                mechanics.data.qvel[:] = 0
                mujoco.mj_forward(mechanics.model, mechanics.data)
            if self.viewer is not None and self.viewer.is_running(): self.viewer.sync()
            if self.closing and not process.is_alive(): self.close()

        def closeEvent(self, event):
            if not self.closing and self.state.get("armed"):
                answer = W.QMessageBox.question(self, "Support the arm", "Closing disables torque. Support the arm, then choose Yes to close.")
                if answer != W.QMessageBox.StandardButton.Yes:
                    event.ignore()
                    return
            self.closing = True
            stop_event.set()
            self.cancel_fit()
            if process.is_alive():
                self.status.setText("Waiting for torque-off and worker shutdown. If communication fails, support arm and cut power.")
                event.ignore()
                return
            if self.viewer is not None: self.viewer.close()
            self.timer.stop()
            event.accept()

    window = Window()
    process.start()
    signal.signal(signal.SIGINT, lambda *_: window.emergency())
    window.show()
    try:
        return app.exec()
    finally:
        stop_event.set()
        process.join(timeout=2)
        if process.is_alive():
            print("Worker shutdown not confirmed. Support arm and cut motor power. Software cannot stop a disconnected servo.", file=sys.stderr)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    gui = sub.add_parser("gui", help="Open desktop controls and MuJoCo reference viewer")
    gui.add_argument("--config", default="work/config.json")
    gui.add_argument("--params", default="work/seed_params.json")
    gui.add_argument("--port", help="Actual follower serial device; omit only in demo mode")
    gui.add_argument("--out", default="sessions")
    gui.add_argument("--demo", action="store_true", help="Toy plant, never opens serial; output marked SYNTHETIC")
    gui.add_argument("--no-viewer", action="store_true", help="Demo/offscreen GUI smoke tests only")
    fit = sub.add_parser("fit", help="Fit recorded sessions offline; never opens serial")
    fit.add_argument("--session", action="append", required=True)
    fit.add_argument("--params", default="work/seed_params.json")
    fit.add_argument("--out")
    fit.add_argument("--max-evals", type=int, default=48, help="Per joint per pass; default is an initial bounded search")
    fit.add_argument("--passes", type=int, default=1)
    fit.add_argument("--dt", type=float, default=.004, help="Fit timestep; validation always uses .002")
    fit.add_argument("--allow-synthetic", action="store_true", help="Software checks only; results marked synthetic")
    evaluate = sub.add_parser("evaluate", help="Evaluate NEW sessions with frozen parameters; no fitting or serial access")
    evaluate.add_argument("--session", action="append", required=True)
    evaluate.add_argument("--params", required=True)
    evaluate.add_argument("--baseline", default="work/seed_params.json")
    evaluate.add_argument("--out")
    evaluate.add_argument("--allow-synthetic", action="store_true")
    return p


def main():
    args = parser().parse_args()
    if args.command == "gui": return launch_gui(args)
    from so101_calibration_fit import fit_campaign, evaluate_campaign
    if args.command == "evaluate":
        evaluate_campaign(args.session, args.params, args.baseline, args.out, args.allow_synthetic)
        return 0
    fit_campaign(args.session, args.params, args.out, args.max_evals, args.passes, args.dt, args.allow_synthetic)
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted. Support the arm and cut motor power if communication was lost.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
