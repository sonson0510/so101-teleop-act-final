#!/usr/bin/env python3
"""Isolated, local SO-101 teleoperation prototype (25 Hz simulation ticks)."""
import argparse
import hashlib
import json
import time
import uuid
from pathlib import Path
import sys
import tkinter as tk
import os

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))
from so101_pick_place_ik import ARM, OPEN, CLOSED, grasp_point, solve_ik
from teleop_core import CommandLimiter, CommandRunner
from teleop_sim import SimBackend
from asset_fingerprint import fingerprint
from evaluation import score

JOINTS = (*ARM, "gripper")
DT = 0.04


class Teleop:
    def __init__(self, output):
        self.model = mujoco.MjModel.from_xml_path(str(ROOT / "scenes/so101_pick_place.xml"))
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=240, width=320)
        for flag in (mujoco.mjtRndFlag.mjRND_SHADOW, mujoco.mjtRndFlag.mjRND_REFLECTION,
                     mujoco.mjtRndFlag.mjRND_FOG, mujoco.mjtRndFlag.mjRND_SKYBOX):
            self.renderer.scene.flags[flag] = 0
        self.ids = np.array([self.model.actuator(n).id for n in JOINTS])
        self.qids = np.array([self.model.joint(n).qposadr[0] for n in JOINTS])
        self.dofs = [self.model.joint(n).dofadr[0] for n in ARM]
        limits = np.array([self.model.jnt_range[self.model.joint(n).id] for n in JOINTS])
        self.limiter = CommandLimiter(limits[:, 0], limits[:, 1], [0.04] * 5 + [0.08])
        self.backend = SimBackend(self.model, self.data, self.renderer, JOINTS)
        self.runner = CommandRunner(self.backend, self.limiter)
        self.gripper_body = self.model.body("gripper").id
        self.output = output
        home = json.loads((ROOT / "configs/home.json").read_text())
        if tuple(home["joint_order"]) != JOINTS:
            raise ValueError("Home joint order mismatch")
        self.home = np.asarray(home["qpos_rad"], dtype=float)
        self.output.mkdir(parents=True, exist_ok=True)
        self.saved_success, self.saved_failure = self.count_saved_labels()
        self.saved = self.saved_success + self.saved_failure
        self.root = tk.Tk()
        self._timer_resolution = False
        if os.name == "nt":
            import ctypes
            self._timer_resolution = ctypes.windll.winmm.timeBeginPeriod(1) == 0
        self.root.title("SO-101 手動操作")
        self.root.resizable(True, True)
        self.root.configure(bg="#e9eef4")
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.root.geometry(f"{min(1160, screen_width - 40)}x{min(780, screen_height - 80)}+20+20")
        heading = tk.Frame(self.root, bg="#e9eef4")
        heading.pack(fill="x", padx=16, pady=(12, 6))
        tk.Label(heading, text="SO-101  手動操作・実演記録", font=("Yu Gothic UI", 16, "bold"),
                 bg="#e9eef4", fg="#172334").pack(side="left")
        tk.Button(heading, text="最大化 / 戻す", command=self.toggle_maximize,
                  font=("Yu Gothic UI", 10), takefocus=False).pack(side="right")
        main = tk.Frame(self.root, bg="#e9eef4")
        main.pack(fill="both", expand=True, padx=16, pady=(0, 12))
        camera_panel = tk.Frame(main, bg="#e9eef4")
        camera_panel.pack(side="left", fill="both", expand=True)
        control_panel = tk.Frame(main, bg="#e9eef4", width=370)
        control_panel.pack(side="right", fill="y", padx=(14, 0))
        control_panel.pack_propagate(False)
        views = tk.Frame(camera_panel, bg="#e9eef4")
        views.pack(anchor="n", pady=(0, 8))
        fixed = tk.Frame(views, bg="#e9eef4")
        fixed.pack(side="left", padx=(0, 8))
        tk.Label(fixed, text="記録用カメラ（固定）  ↑ X− (W)   → Y+ (D)", font=("Yu Gothic UI", 10, "bold"),
                 bg="#e9eef4").pack()
        self.image = tk.Label(fixed)
        self.image.pack()
        operator = tk.Frame(views, bg="#e9eef4")
        operator.pack(side="left")
        tk.Label(operator, text="操作用ビュー（通常は同じ画像）", font=("Yu Gothic UI", 10, "bold"),
                 bg="#e9eef4").pack()
        self.operator_image = tk.Label(operator)
        self.operator_image.pack()
        self.operator_camera = mujoco.MjvCamera()
        self.operator_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.operator_camera.lookat[:] = [0.20, 0.0, 0.08]
        self.azimuth = tk.DoubleVar(value=150)
        self.elevation = tk.DoubleVar(value=-25)
        self.distance = tk.DoubleVar(value=0.8)
        self.aux_view = tk.BooleanVar(value=False)
        tk.Checkbutton(camera_panel, text="補助視点を使う（練習専用・学習データ対象外）",
                       variable=self.aux_view, bg="#e9eef4", takefocus=False).pack(anchor="w")
        camera_bar = tk.Frame(camera_panel, bg="#e9eef4")
        camera_bar.pack(anchor="w")
        for label, variable, low, high in (("方位", self.azimuth, 0, 360),
                                            ("仰角", self.elevation, -80, 10),
                                            ("距離", self.distance, 0.4, 1.4)):
            tk.Scale(camera_bar, label=label, variable=variable, from_=low, to=high,
                     resolution=1 if label != "距離" else 0.05, orient="horizontal",
                     length=175, bg="#e9eef4", takefocus=False).pack(side="left")
        status_panel = tk.LabelFrame(camera_panel, text="現在の状態", font=("Yu Gothic UI", 11, "bold"),
                                     bg="#e9eef4", fg="#172334", padx=8, pady=7)
        status_panel.pack(fill="x", pady=(10, 0))
        self.status = tk.StringVar()  # Combined text retained for diagnostics.
        self.status_main = tk.StringVar()
        self.status_record = tk.StringVar()
        self.status_position = tk.StringVar()
        self.status_warning = tk.StringVar()
        self.status_badge = tk.Label(status_panel, textvariable=self.status_main, anchor="w",
                                     font=("Yu Gothic UI", 12, "bold"), bg="#d9f2de", fg="#173b24",
                                     padx=10, pady=6)
        self.status_badge.pack(fill="x")
        tk.Label(status_panel, textvariable=self.status_record, justify="left", anchor="w",
                 font=("Yu Gothic UI", 10), bg="white", fg="#172334",
                 padx=10, pady=5).pack(fill="x", pady=(4, 0))
        tk.Label(status_panel, textvariable=self.status_position, justify="left", anchor="w",
                 font=("Consolas", 10), bg="white", fg="#172334",
                 padx=10, pady=5).pack(fill="x")
        self.warning_label = tk.Label(status_panel, textvariable=self.status_warning, justify="left",
                                      anchor="w", font=("Yu Gothic UI", 10, "bold"),
                                      bg="#eef2f6", fg="#536273", padx=10, pady=5)
        self.warning_label.pack(fill="x", pady=(4, 0))
        record = tk.LabelFrame(control_panel, text="実演データの記録", font=("Yu Gothic UI", 11, "bold"),
                               bg="#e9eef4", padx=8, pady=8)
        record.pack(fill="x", pady=(0, 10))
        record.grid_columnconfigure(0, weight=1)
        record.grid_columnconfigure(1, weight=0)
        record.grid_columnconfigure(2, weight=1)
        self.workflow_buttons = {}
        stage_style = {"bg": "#e9eef4", "fg": "#425466", "font": ("Yu Gothic UI", 9, "bold")}
        tk.Label(record, text="1  準備", anchor="w", **stage_style).grid(
            row=0, column=0, sticky="w", padx=3, pady=(0, 3))
        tk.Label(record, text="2  記録（左から右へ）", anchor="w", **stage_style).grid(
            row=2, column=0, columnspan=3, sticky="w", padx=3, pady=(5, 2))
        tk.Label(record, text="3  記録後：結果を選択", anchor="w", **stage_style).grid(
            row=4, column=0, columnspan=3, sticky="w", padx=3, pady=(5, 2))
        for name, title, fn, row, column, columnspan, color in (
            ("reset", "開始位置にリセット", self.reset, 0, 1, 2, "#ffffff"),
            ("record", "記録開始", self.record, 3, 0, 1, "#cfe8ff"),
            ("stop_record", "記録終了", self.stop_record, 3, 2, 1, "#ffffff"),
            ("save_success", "成功で保存", lambda: self.save("success"), 5, 0, 1, "#d9f2de"),
            ("save_failure", "失敗で保存", lambda: self.save("failure"), 5, 2, 1, "#fff0d9"),
            ("discard", "この記録を破棄", self.discard, 6, 0, 3, "#f1f3f5"),
        ):
            button = tk.Button(record, text=title, command=fn, bg=color, font=("Yu Gothic UI", 10),
                               height=1, takefocus=False, disabledforeground="#929aa3")
            button.grid(row=row, column=column, columnspan=columnspan, sticky="ew", padx=3, pady=2)
            self.workflow_buttons[name] = button
        tk.Label(record, text="→", bg="#e9eef4", fg="#425466",
                 font=("Yu Gothic UI", 12, "bold")).grid(row=3, column=1, padx=4)
        tk.Label(record, text="または", bg="#e9eef4", fg="#687786",
                 font=("Yu Gothic UI", 8)).grid(row=5, column=1, padx=4)
        safety = tk.LabelFrame(control_panel, text="停止と再開", font=("Yu Gothic UI", 11, "bold"),
                               bg="#e9eef4", padx=8, pady=8)
        safety.pack(fill="x", pady=(0, 10))
        self.hold_button = tk.Button(safety, text="停止・保持 (Space)", command=self.hold, bg="#ffd9d9",
                                     font=("Yu Gothic UI", 10, "bold"), height=2,
                                     takefocus=False, disabledforeground="#929aa3")
        self.hold_button.pack(fill="x", pady=(0, 5))
        self.resume_button = tk.Button(safety, text="操作再開", command=self.resume, bg="#d9f2de",
                                       font=("Yu Gothic UI", 10), height=2, takefocus=False,
                                       disabledforeground="#929aa3")
        self.resume_button.pack(fill="x")
        motion = tk.LabelFrame(control_panel, text="手先・グリッパー操作（画面方向と一致）",
                               font=("Yu Gothic UI", 11, "bold"), bg="#e9eef4", padx=8, pady=8)
        motion.pack(fill="x")
        self.held_buttons = set()
        self.tap_buttons = set()
        self.motion_buttons = {}
        xy = tk.Frame(motion, bg="#e9eef4")
        xy.pack(fill="x")
        tk.Label(xy, text="XY（画面内）", bg="#e9eef4", font=("Yu Gothic UI", 10, "bold")).grid(
            row=0, column=0, columnspan=3, pady=(0, 3))
        positions = {
            "w": (1, 1, "↑ 画面上\nX−  W"),
            "a": (2, 0, "← 画面左\nY−  A"),
            "d": (2, 2, "画面右 →\nY+  D"),
            "s": (3, 1, "↓ 画面下\nX+  S"),
        }
        for key, (row, column, label) in positions.items():
            button = self.make_motion_button(xy, label, key, width=10)
            button.grid(row=row, column=column, padx=3, pady=3)
        other = tk.Frame(motion, bg="#e9eef4")
        other.pack(fill="x", pady=(8, 0))
        for column, (label, key) in enumerate((("Z− 下降  Q", "q"), ("Z+ 上昇  E", "e"),
                                               ("閉じる  P", "p"), ("開く  O", "o"))):
            button = self.make_motion_button(other, label, key, width=9)
            button.grid(row=0, column=column, padx=2, pady=3)
        self.fine = tk.BooleanVar(value=False)
        self.fine_button = tk.Checkbutton(control_panel, text="微動（ゆっくり動かす）", variable=self.fine,
                                          bg="#e9eef4", font=("Yu Gothic UI", 10), takefocus=False)
        self.fine_button.pack(anchor="w", pady=(8, 3))
        tk.Label(control_panel, text="キーボード: A/D・W/S・E/Q  |  開閉: O/P\n微動: Shift  |  停止: Space",
                 justify="left", bg="#e9eef4", font=("Yu Gothic UI", 9)).pack(anchor="w")
        self.keys = set()
        self.last_input = time.perf_counter()
        self.root.bind_all("<KeyPress>", self.key_down)
        self.root.bind_all("<KeyRelease>", self.key_up)
        self.root.bind("<FocusOut>", self.on_focus_out)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.state = "DISCONNECTED"
        self.frames = []
        self.initial = None
        self.review = None
        self.assessment = None
        self.assisted_during_record = False
        self.warning = ""
        self.session_id = uuid.uuid4().hex
        self.last_tick = None
        self.overruns = 0
        self.tick_durations_ms = []
        self.tick_gaps_ms = []
        self.reset()
        self.runner.connect()
        self.update_button_states()
        self.root.after(40, self.tick)

    def count_saved_labels(self):
        counts = {"success": 0, "failure": 0}
        for path in self.output.glob("*.npz"):
            try:
                with np.load(path, allow_pickle=False) as data:
                    metadata = json.loads(str(data["metadata"].item()))
                label = metadata.get("label")
                if label in counts:
                    counts[label] += 1
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                # Ignore unrelated, partial, or damaged files; saving still works.
                continue
        return counts["success"], counts["failure"]

    def key_down(self, event):
        k = event.keysym.lower()
        if k == "space":
            if self.state in ("READY", "RECORDING") and not self.runner.stopped:
                self.hold()
            return
        if self.state not in ("READY", "RECORDING") or self.runner.stopped:
            return
        self.keys.add(k)
        self.last_input = time.perf_counter()

    def key_up(self, event):
        self.keys.discard(event.keysym.lower())

    def make_motion_button(self, parent, label, key, width):
        button = tk.Button(parent, text=label, width=width, height=2, bg="#ffffff",
                           font=("Yu Gothic UI", 10, "bold"), takefocus=False)
        button.bind("<ButtonPress-1>", lambda _, k=key: self.motion_press(k))
        button.bind("<ButtonRelease-1>", lambda _, k=key: self.motion_release(k))
        button.bind("<Leave>", lambda _, k=key: self.motion_release(k))
        self.motion_buttons[key] = button
        return button

    def motion_press(self, key):
        if self.state not in ("READY", "RECORDING") or self.runner.stopped:
            return
        self.held_buttons.add(key)
        self.tap_buttons.add(key)
        self.motion_buttons[key].configure(bg="#bcdcff", relief="sunken")

    def motion_release(self, key):
        self.held_buttons.discard(key)
        self.motion_buttons[key].configure(bg="#ffffff", relief="raised")

    def clear_motion_buttons(self):
        self.held_buttons.clear()
        self.tap_buttons.clear()
        for button in self.motion_buttons.values():
            button.configure(bg="#ffffff", relief="raised")

    @staticmethod
    def set_enabled(widget, enabled):
        widget.configure(state="normal" if enabled else "disabled")

    def update_button_states(self):
        ready = self.state == "READY"
        recording = self.state == "RECORDING"
        review = self.state == "REVIEW"
        movable = (ready or recording) and not self.runner.stopped
        self.set_enabled(self.workflow_buttons["reset"], ready)
        self.set_enabled(self.workflow_buttons["record"], ready and not self.runner.stopped)
        self.set_enabled(self.workflow_buttons["stop_record"], recording)
        self.set_enabled(self.workflow_buttons["discard"], review)
        saveable = review and not self.overruns
        self.set_enabled(self.workflow_buttons["save_success"], saveable)
        self.set_enabled(self.workflow_buttons["save_failure"], saveable)
        self.set_enabled(self.hold_button, movable)
        self.set_enabled(self.resume_button, ready and self.runner.stopped)
        self.set_enabled(self.fine_button, movable)
        for button in self.motion_buttons.values():
            self.set_enabled(button, movable)
        if not movable and (self.keys or self.held_buttons or self.tap_buttons):
            self.keys.clear()
            self.clear_motion_buttons()

    def on_focus_out(self, _event):
        # Moving focus between controls in this window is still valid input.
        self.root.after_idle(self.clear_if_window_inactive)

    def clear_if_window_inactive(self):
        if self.root.focus_displayof() is None:
            self.keys.clear()
            self.clear_motion_buttons()
            if self.state in ("READY", "RECORDING"):
                self.target = grasp_point(self.data, self.gripper_body).copy()
                self.goal = self.data.ctrl[self.ids].copy()
                self.warning = "画面のフォーカスが外れたため入力を解除しました"

    def toggle_maximize(self):
        self.root.state("normal" if self.root.state() == "zoomed" else "zoomed")

    def hold(self, reason="hold"):
        self.keys.clear()
        self.clear_motion_buttons()
        if self.state in ("READY", "RECORDING"):
            self.runner.stop()
            self.goal = self.data.ctrl[self.ids].copy()
            self.target = grasp_point(self.data, self.gripper_body).copy()
            self.warning = "現在姿勢で保持中"
            if self.state == "RECORDING":
                self.stop_record(reason)

    def resume(self):
        if self.state != "READY":
            return
        self.runner.resume()
        self.goal = self.data.ctrl[self.ids].copy()
        self.target = grasp_point(self.data, self.gripper_body).copy()
        self.warning = "操作を再開しました"

    def reset(self):
        if self.state in ("RECORDING", "REVIEW"):
            self.warning = "記録終了後、保存か破棄をしてからリセットしてください"
            return
        self.backend.reset(self.home)
        self.backend.step(100)
        self.target = grasp_point(self.data, self.gripper_body).copy()
        self.goal = self.data.ctrl[self.ids].copy()
        self.goal[-1] = OPEN
        self.keys.clear()
        self.clear_motion_buttons()
        self.state = "READY"
        self.warning = ""
        if self.runner.connected:
            self.runner.last_action = self.data.ctrl[self.ids].astype(np.float32).copy()
            self.runner.stopped = False

    def snapshot(self):
        return self.backend.snapshot()

    def record(self):
        if self.state != "READY" or self.runner.stopped:
            return
        self.initial = self.snapshot()
        self.frames = []
        self.overruns = 0
        self.tick_durations_ms = []
        self.tick_gaps_ms = []
        self.last_tick = None
        self.state = "RECORDING"
        self.end_reason = "operator_stop"
        self.assisted_during_record = bool(self.aux_view.get())

    def stop_record(self, reason="operator_stop"):
        if self.state == "RECORDING":
            self.end_reason = reason
            self.review = self.frames
            self.frames = []
            if len(self.review) >= 2:
                self.assessment = score([f["object_pos"] for f in self.review],
                                        [f["gripper_pos"] for f in self.review],
                                        gripper_open=bool(self.review[-1]["action"][-1] > 0.8))
            else:
                self.assessment = None
            self.state = "REVIEW"
            self.warning = (f"自動成功判定: {'成功' if self.assessment['success'] else '失敗'}"
                            if self.assessment else "記録が短すぎます")

    def discard(self):
        if self.state == "REVIEW":
            self.review = None
            self.assessment = None
            self.state = "READY"

    def save(self, label):
        if self.state != "REVIEW" or not self.review:
            return
        if self.overruns:
            self.warning = f"保存不可: 時間超過 {self.overruns} 回。破棄してください"
            return
        episode = uuid.uuid4().hex
        final = self.output / f"{episode}.npz"
        temp = self.output / f"{episode}.partial.npz"
        fields = {k: np.stack([f[k] for f in self.review]) for k in
                  ("image", "state", "action", "sim_time", "host_time", "send_time", "object_pos", "gripper_pos")}
        fields.update({"initial_" + k: v for k, v in self.initial.items()})
        meta = {"schema": "teleop_raw_v1", "label": label, "frequency": 25,
                "joint_order": JOINTS, "model": "scenes/so101_pick_place.xml", "input": "keyboard",
                "model_sha256": fingerprint(ROOT), "timing_overruns": self.overruns,
                "automatic_assessment": self.assessment, "session_id": self.session_id,
                "auxiliary_view_used": self.assisted_during_record,
                "episode_id": episode, "seed": 0, "joint_unit": "rad", "end_reason": self.end_reason,
                "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "wall_gap_max_ms": max(self.tick_gaps_ms, default=0.0),
                "wall_gap_p95_ms": float(np.percentile(self.tick_gaps_ms, 95)) if self.tick_gaps_ms else 0.0}
        try:
            np.savez_compressed(temp, **fields, metadata=json.dumps(meta))
            temp.rename(final)
        except Exception as exc:
            self.warning = f"保存失敗: {exc}"
            return
        self.saved += 1
        if label == "success":
            self.saved_success += 1
        else:
            self.saved_failure += 1
        self.review = None
        self.assessment = None
        self.state = "READY"
        self.warning = f"保存しました: {final.name}"

    def tick(self):
        if self.state not in ("READY", "RECORDING", "REVIEW"):
            return
        start = time.perf_counter()
        if self.last_tick is not None and start - self.last_tick > 0.25:
            had_input = bool(self.keys or self.held_buttons or self.tap_buttons)
            self.keys.clear()
            self.clear_motion_buttons()
            self.backend.stop()
            self.goal = self.data.ctrl[self.ids].copy()
            self.runner.last_action = self.goal.astype(np.float32).copy()
            self.target = grasp_point(self.data, self.gripper_body).copy()
            if self.state == "RECORDING":
                self.stop_record("input_timeout")
            elif had_input:
                self.warning = "入力タイムアウト: 移動を解除しました。操作は続けられます"
        if self.state == "RECORDING" and self.last_tick is not None:
            self.tick_gaps_ms.append((start - self.last_tick) * 1000)
        if self.state == "RECORDING" and self.last_tick is not None and start - self.last_tick > DT * 1.25:
            self.overruns += 1
            self.warning = f"処理時間超過: {self.overruns} 回"
        self.last_tick = start
        if self.state == "RECORDING" and self.aux_view.get():
            self.assisted_during_record = True
        # A local KeyPress is a held state until KeyRelease or FocusOut;
        # expiring it by event age would make motion depend on OS key repeat.
        if self.state != "REVIEW":
            active = self.keys | self.held_buttons | self.tap_buttons
            self.tap_buttons.clear()
            direction = np.array([int("s" in active)-int("w" in active),
                                  int("d" in active)-int("a" in active),
                                  int("e" in active)-int("q" in active)], dtype=float)
            speed = 0.015 if self.fine.get() or "shift_l" in active or "shift_r" in active else 0.06
            if np.any(direction):
                proposed = self.target + direction / np.linalg.norm(direction) * speed * DT
                seed, error = solve_ik(self.model, proposed, self.data.qpos.copy(), self.dofs,
                                       self.gripper_body, iterations=35)
                if np.isfinite(error) and error < 0.015:
                    joints = seed[self.qids[:5]]
                    self.goal[:5] += np.clip(joints - self.goal[:5], -0.04, 0.04)
                    self.target = proposed
                    self.warning = ""
                else:
                    self.warning = f"IK不成立: 位置誤差 {error:.3f} m"
            if "o" in active:
                self.goal[-1] = OPEN
            if "p" in active:
                self.goal[-1] = CLOSED
            self.goal[-1] = np.clip(self.goal[-1], CLOSED, OPEN)
            observation = self.backend.observe()
            rgb = observation.image
            observation_time = observation.acquired_at
            result = self.runner.submit(self.goal)
            if not result.accepted and not self.runner.stopped:
                self.state = "FAULT"
                self.warning = f"指令を拒否しました: {result.reason}"
                return
            if result.limited:
                self.warning = "関節目標を制限しました"
            action = result.action if result.accepted else self.data.ctrl[self.ids].astype(np.float32).copy()
            self.goal = action.astype(float)
            send_time = result.sent_at if result.accepted else time.perf_counter()
            if self.state == "RECORDING":
                if len(self.frames) >= 1500:
                    self.warning = "上限60秒に達しました。記録を確認してください"
                    self.stop_record("max_duration")
                else:
                    self.frames.append({"image": rgb, "state": observation.state,
                                        "action": action, "sim_time": self.data.time,
                                        "host_time": observation_time, "send_time": send_time,
                                        "object_pos": self.data.xpos[self.model.body("block").id].copy(),
                                        "gripper_pos": self.data.xpos[self.gripper_body].copy()})
            self.backend.step(20)
        else:
            self.renderer.update_scene(self.data, camera="observation")
            rgb = self.renderer.render().copy()
        if self.aux_view.get():
            self.operator_camera.azimuth = self.azimuth.get()
            self.operator_camera.elevation = self.elevation.get()
            self.operator_camera.distance = self.distance.get()
            self.renderer.update_scene(self.data, camera=self.operator_camera)
            operator_rgb = self.renderer.render().copy()
        else:
            operator_rgb = rgb
        display_scale = max(1, min(3, (self.root.winfo_width() - 430) // 640,
                                   (self.root.winfo_height() - 340) // 240))
        ppm = b"P6\n320 240\n255\n" + rgb.tobytes()
        self.photo = tk.PhotoImage(data=ppm, format="PPM")
        if display_scale > 1:
            self.photo = self.photo.zoom(display_scale, display_scale)
        self.image.configure(image=self.photo)
        operator_ppm = b"P6\n320 240\n255\n" + operator_rgb.tobytes()
        self.operator_photo = tk.PhotoImage(data=operator_ppm, format="PPM")
        if display_scale > 1:
            self.operator_photo = self.operator_photo.zoom(display_scale, display_scale)
        self.operator_image.configure(image=self.operator_photo)
        elapsed = time.perf_counter() - start
        if self.state == "RECORDING":
            self.tick_durations_ms.append(elapsed * 1000)
        if self.state == "RECORDING" and elapsed > DT:
            self.overruns += 1
            self.warning = f"処理時間超過: {self.overruns} 回"
        state_label = {"DISCONNECTED": "未接続", "READY": "操作可能", "RECORDING": "記録中",
                       "REVIEW": "確認中", "FAULT": "異常停止"}.get(self.state, self.state)
        input_label = "停止中（操作再開を押す）" if self.runner.stopped else "有効"
        view_label = ("補助視点使用済み（学習対象外）"
                      if self.state in ("RECORDING", "REVIEW") and self.assisted_during_record
                      else "補助視点（練習用）" if self.aux_view.get() else "固定視点のみ")
        active_names = ",".join(sorted(self.keys | self.held_buttons | self.tap_buttons)) or "なし"
        grasp = grasp_point(self.data, self.gripper_body)
        frames = len(self.frames) if self.state == "RECORDING" else len(self.review or [])
        stopped = self.runner.stopped
        main_text = ("【停止中】『操作再開』を押してください" if stopped else
                     "【記録中】操作を実演しています" if self.state == "RECORDING" else
                     "【操作可能】準備できています" if self.state == "READY" else
                     "【確認中】記録結果を確認してください" if self.state == "REVIEW" else
                     f"【要確認】{state_label}")
        badge_colors = (("#ffd9d9", "#721c24") if stopped or self.state == "FAULT" else
                        ("#ffe7ad", "#654500") if self.state == "RECORDING" else
                        ("#d9ecff", "#173f68") if self.state == "REVIEW" else
                        ("#d9f2de", "#173b24"))
        self.status_main.set(main_text)
        self.status_badge.configure(bg=badge_colors[0], fg=badge_colors[1])
        self.status_record.set(
            f"記録: {state_label}　 フレーム: {frames}\n"
            f"保存済み: 成功 {self.saved_success} 件　失敗 {self.saved_failure} 件　合計 {self.saved} 件\n"
            f"入力: {input_label}　 押下中: {active_names}　 視点: {view_label}")
        self.status_position.set(
            f"現在 XYZ  X {grasp[0]:+.3f}  Y {grasp[1]:+.3f}  Z {grasp[2]:+.3f} m\n"
            f"目標 XYZ  X {self.target[0]:+.3f}  Y {self.target[1]:+.3f}  Z {self.target[2]:+.3f} m")
        warning_text = f"注意: {self.warning}" if self.warning else f"正常動作中　処理時間 {elapsed*1000:.1f} ms"
        self.status_warning.set(warning_text)
        self.warning_label.configure(bg="#ffd9d9" if self.warning else "#eef2f6",
                                     fg="#721c24" if self.warning else "#536273")
        self.status.set(f"{main_text}\n{self.status_record.get()}\n{self.status_position.get()}\n{warning_text}")
        self.update_button_states()
        deadline = start + DT - 0.001
        remaining = deadline - time.perf_counter()
        if remaining > 0.008:
            time.sleep(remaining - 0.008)
        while time.perf_counter() < deadline:
            pass
        self.root.after(1, self.tick)

    def close(self):
        self.runner.disconnect()
        self.renderer.close()
        self.root.destroy()
        if self._timer_resolution:
            import ctypes
            ctypes.windll.winmm.timeEndPeriod(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=ROOT / "datasets")
    args = p.parse_args()
    Teleop(args.output).root.mainloop()

