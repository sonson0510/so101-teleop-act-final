#!/usr/bin/env python3
"""Replay a teleop raw episode from its complete MuJoCo state."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from asset_fingerprint import fingerprint
from evaluation import score
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("episode", type=Path)
    args = p.parse_args()
    with np.load(args.episode, allow_pickle=False) as episode:
        meta = json.loads(str(episode["metadata"]))
        if meta["schema"] != "teleop_raw_v1" or tuple(meta["joint_order"]) != JOINTS:
            raise ValueError("Unsupported episode schema or joint order")
        if "model_sha256" in meta and meta["model_sha256"] != fingerprint(ROOT):
            raise ValueError("Model assets differ from recorded episode")
        images, states, actions = (episode[k] for k in ("image", "state", "action"))
        n = len(actions)
        if n == 0:
            raise ValueError("Empty episode")
        if not (images.shape == (n, 240, 320, 3) and states.shape == actions.shape == (n, 6)):
            raise ValueError("Frame shape mismatch")
        if not all(np.isfinite(episode[k]).all() for k in ("state", "action", "sim_time", "host_time", "send_time")):
            raise ValueError("Nonfinite sample")
        if np.any(np.diff(episode["sim_time"]) <= 0):
            raise ValueError("Simulation time is not increasing")
        if not np.allclose(np.diff(episode["sim_time"]), 0.04, atol=1e-5):
            raise ValueError("Simulation tick interval is not 40 ms")
        model = mujoco.MjModel.from_xml_path(str(ROOT / meta["model"]))
        data = mujoco.MjData(model)
        for k in ("qpos", "qvel", "act", "ctrl"):
            getattr(data, k)[:] = episode["initial_" + k]
        data.time = float(episode["initial_time"])
        mujoco.mj_forward(model, data)
        qids = [model.joint(j).qposadr[0] for j in JOINTS]
        aids = [model.actuator(j).id for j in JOINTS]
        errors = []
        object_errors = []
        replay_objects = []
        replay_grippers = []
        recorded_objects = episode["object_pos"]
        for state, action in zip(states, actions):
            errors.append(float(np.linalg.norm(data.qpos[qids] - state)))
            replay_objects.append(data.xpos[model.body("block").id].copy())
            replay_grippers.append(data.xpos[model.body("gripper").id].copy())
            data.ctrl[aids] = action
            for _ in range(20):
                mujoco.mj_step(model, data)
        replay_objects = np.asarray(replay_objects)
        replay_grippers = np.asarray(replay_grippers)
        if recorded_objects.shape == replay_objects.shape:
            object_errors = np.linalg.norm(replay_objects - recorded_objects, axis=1).tolist()
        assessment = score(np.vstack([replay_objects, data.xpos[model.body("block").id]]),
                           np.vstack([replay_grippers, data.xpos[model.body("gripper").id]]),
                           gripper_open=bool(actions[-1, -1] > 0.8))
        result = {"frames": n, "label": meta["label"], "state_rmse_rad": float(np.sqrt(np.mean(np.square(errors)))),
                  "max_state_error_rad": max(errors), "max_object_error_m": max(object_errors) if object_errors else None,
                  **assessment,
                  "final_object_position": data.xpos[model.body("block").id].tolist()}
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
