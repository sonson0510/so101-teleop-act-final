"""MuJoCo backend with simulator-only reset, step and snapshot methods."""
import time
import mujoco
import numpy as np
from teleop_core import Observation, SendResult


class SimBackend:
    def __init__(self, model, data, renderer, joint_names):
        self.model = model
        self.data = data
        self.renderer = renderer
        self.joint_names = joint_names
        self.qids = np.array([model.joint(n).qposadr[0] for n in joint_names])
        self.ids = np.array([model.actuator(n).id for n in joint_names])
        self.connected = False

    def connect(self):
        self.connected = True

    def observe(self):
        if not self.connected:
            raise ConnectionError("MuJoCo backend disconnected")
        self.renderer.update_scene(self.data, camera="observation")
        image = self.renderer.render().copy()
        return Observation(self.data.qpos[self.qids].astype(np.float32).copy(), image,
                           time.perf_counter())

    def send_action(self, action):
        if not self.connected:
            return SendResult(False, None, time.perf_counter(), "disconnected")
        self.data.ctrl[self.ids] = action
        return SendResult(True, np.asarray(action, dtype=np.float32).copy(), time.perf_counter())

    def stop(self):
        if self.connected:
            self.data.ctrl[self.ids] = self.data.qpos[self.qids]

    def disconnect(self):
        self.connected = False

    def reset(self, home):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qids] = home
        self.data.ctrl[self.ids] = home
        mujoco.mj_forward(self.model, self.data)

    def step(self, steps=20):
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)

    def snapshot(self):
        return {k: np.array(getattr(self.data, k)).copy() for k in ("qpos", "qvel", "act", "ctrl")} | {"time": float(self.data.time)}
