#!/usr/bin/env python3
"""Scripted Cartesian Pick & Place baseline for SO-101 in MuJoCo.

The grasp-point definition and damped least-squares IK follow the approach in
desmondzee/ICL-so101 (MIT), adapted here to add transport, placement, success
checks, and an optional interactive viewer.
"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


ARM = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
GRASP_OFFSET = np.array([0.010, 0.0, -0.080])
OPEN = 1.20
CLOSED = -0.17
TARGET_XY = np.array([0.20, 0.12])
KEY_SPACE = 32
KEY_ESCAPE = 256
KEY_R = ord("R")


class ReplayRequested(Exception):
    """Restart the current episode from the model's initial state."""


class QuitRequested(Exception):
    """Close the interactive viewer."""


def grasp_point(data: mujoco.MjData, gripper_body: int) -> np.ndarray:
    rotation = data.xmat[gripper_body].reshape(3, 3)
    return data.xpos[gripper_body] + rotation @ GRASP_OFFSET


def solve_ik(model, target, seed, arm_dofs, gripper_body, iterations=200):
    work = mujoco.MjData(model)
    work.qpos[:] = seed
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    error = np.zeros(6)
    for _ in range(iterations):
        mujoco.mj_kinematics(model, work)
        mujoco.mj_comPos(model, work)
        rotation = work.xmat[gripper_body].reshape(3, 3)
        approach = -rotation[:, 2]
        error = np.concatenate(
            [target - grasp_point(work, gripper_body), np.cross(approach, [0, 0, -1])]
        )
        mujoco.mj_jac(
            model,
            work,
            jacp,
            jacr,
            grasp_point(work, gripper_body),
            gripper_body,
        )
        jacobian = np.vstack([jacp, jacr])[:, arm_dofs]
        delta = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + 1e-4 * np.eye(6), error
        )
        for index, name in enumerate(ARM):
            joint = model.joint(name)
            low, high = model.jnt_range[joint.id]
            address = joint.qposadr[0]
            work.qpos[address] = np.clip(work.qpos[address] + delta[index], low, high)
    return work.qpos.copy(), float(np.linalg.norm(error[:3]))


def set_viewer_text(viewer, controls, phase) -> None:
    if viewer is None:
        return
    status = "PAUSED" if controls["paused"] else phase
    viewer.set_texts(
        (
            mujoco.mjtFontScale.mjFONTSCALE_150,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            "SO-101 Pick & Place",
            f"Phase: {status}\nR: replay\nSpace: pause/resume\nEsc: quit",
        )
    )


def handle_controls(viewer, controls, phase) -> None:
    if viewer is None:
        return
    if controls["quit_requested"] or not viewer.is_running():
        raise QuitRequested
    if controls["replay_requested"]:
        controls["replay_requested"] = False
        controls["paused"] = False
        raise ReplayRequested
    while controls["paused"]:
        set_viewer_text(viewer, controls, phase)
        viewer.sync()
        time.sleep(0.05)
        if controls["quit_requested"] or not viewer.is_running():
            raise QuitRequested
        if controls["replay_requested"]:
            controls["replay_requested"] = False
            controls["paused"] = False
            raise ReplayRequested


def physics_step(model, data, viewer, controls, phase) -> None:
    handle_controls(viewer, controls, phase)
    controls["phase"] = phase
    simulation_step = controls.setdefault("simulation_step", 0)
    frame_callback = controls.get("frame_callback")
    record_every_steps = controls.get("record_every_steps", 20)
    if frame_callback is not None and simulation_step % record_every_steps == 0:
        frame_callback(model, data, controls)
    started = time.monotonic()
    mujoco.mj_step(model, data)
    controls["simulation_step"] = simulation_step + 1
    if viewer is not None:
        viewer.sync()
        remaining = model.opt.timestep - (time.monotonic() - started)
        if remaining > 0:
            time.sleep(remaining)


def move_grasp_point(
    model,
    data,
    viewer,
    target,
    gripper_value,
    arm_dofs,
    arm_actuators,
    gripper_actuator,
    gripper_body,
    controls,
    phase,
    waypoints=50,
    steps_per_waypoint=20,
):
    start = grasp_point(data, gripper_body).copy()
    seed = data.qpos.copy()
    data.ctrl[gripper_actuator] = gripper_value
    final_error = 0.0
    for alpha in np.linspace(0.0, 1.0, waypoints)[1:]:
        desired = start + alpha * (target - start)
        seed, final_error = solve_ik(
            model, desired, seed, arm_dofs, gripper_body
        )
        data.ctrl[arm_actuators] = [seed[model.joint(name).qposadr[0]] for name in ARM]
        for _ in range(steps_per_waypoint):
            physics_step(model, data, viewer, controls, phase)
    return final_error


def settle(model, data, viewer, controls, phase, steps):
    for _ in range(steps):
        physics_step(model, data, viewer, controls, phase)


def run_episode(
    model, data, viewer, controls, arm_dofs, arm_actuators,
    gripper_actuator, gripper_body, block_body,
):
    mujoco.mj_resetData(model, data)
    object_initial_xy = controls.get("object_initial_xy")
    if object_initial_xy is not None:
        block_joint_qpos = model.joint("block_freejoint").qposadr[0]
        data.qpos[block_joint_qpos:block_joint_qpos + 2] = object_initial_xy
    controls["simulation_step"] = 0
    controls["phase"] = "reset"
    mujoco.mj_forward(model, data)
    block_start = data.xpos[block_body].copy()
    phase_waypoints = controls.get("phase_waypoints", {})
    phase_steps = controls.get("phase_steps", {})

    def waypoints(name, default=50):
        return int(phase_waypoints.get(name, default))

    def steps(name, default):
        return int(phase_steps.get(name, default))

    seed = data.qpos.copy()
    initial_guess = [np.arctan2(-block_start[1], block_start[0]), 0.6, -0.4, 1.35, 0.0]
    for name, value in zip(ARM, initial_guess, strict=True):
        seed[model.joint(name).qposadr[0]] = value
    ready, ready_error = solve_ik(
        model, block_start + [0, 0, 0.07], seed, arm_dofs, gripper_body
    )
    data.qpos[:] = ready
    data.qpos[model.joint("gripper").qposadr[0]] = OPEN
    data.ctrl[arm_actuators] = [ready[model.joint(name).qposadr[0]] for name in ARM]
    data.ctrl[gripper_actuator] = OPEN
    mujoco.mj_forward(model, data)

    phase_positions = {}
    ik_errors = {"ready": ready_error}

    def mark(name):
        position = data.xpos[block_body].copy()
        phase_positions[name] = position.tolist()
        print(f"{name}: block={position.round(4)}", flush=True)

    def phase(name):
        print(f"phase: {name}", flush=True)
        set_viewer_text(viewer, controls, name)

    phase("settling")
    settle(model, data, viewer, controls, "settling", steps("settling", 300))
    mark("initial")

    phase("descend")
    ik_errors["descend"] = move_grasp_point(
        model, data, viewer, block_start + [0, 0, 0.012], OPEN,
        arm_dofs, arm_actuators, gripper_actuator, gripper_body,
        controls, "descend", waypoints=waypoints("descend"),
    )
    mark("descend")

    phase("close")
    data.ctrl[gripper_actuator] = CLOSED
    settle(model, data, viewer, controls, "close", steps("close", 1000))
    mark("close")

    phase("lift")
    ik_errors["lift"] = move_grasp_point(
        model, data, viewer, block_start + [0, 0, 0.10], CLOSED,
        arm_dofs, arm_actuators, gripper_actuator, gripper_body,
        controls, "lift", waypoints=waypoints("lift"),
    )
    mark("lift")

    target_above = np.array([TARGET_XY[0], TARGET_XY[1], 0.115])
    phase("transport")
    ik_errors["transport"] = move_grasp_point(
        model, data, viewer, target_above, CLOSED,
        arm_dofs, arm_actuators, gripper_actuator, gripper_body,
        controls, "transport", waypoints=waypoints("transport"),
    )
    mark("transport")

    phase("lower")
    ik_errors["lower"] = move_grasp_point(
        model, data, viewer, [TARGET_XY[0], TARGET_XY[1], 0.027], CLOSED,
        arm_dofs, arm_actuators, gripper_actuator, gripper_body,
        controls, "lower", waypoints=waypoints("lower"),
    )
    mark("lower")

    phase("release")
    data.ctrl[gripper_actuator] = OPEN
    settle(model, data, viewer, controls, "release", steps("release", 800))
    mark("release")

    phase("retreat")
    ik_errors["retreat"] = move_grasp_point(
        model, data, viewer, target_above, OPEN,
        arm_dofs, arm_actuators, gripper_actuator, gripper_body,
        controls, "retreat", waypoints=waypoints("retreat"),
    )
    settle(model, data, viewer, controls, "retreat", steps("retreat", 500))
    mark("final")

    final = data.xpos[block_body].copy()
    heights = [position[2] for position in map(np.asarray, phase_positions.values())]
    xy_error = float(np.linalg.norm(final[:2] - TARGET_XY))
    moved = float(np.linalg.norm(final[:2] - block_start[:2]))
    lifted = float(max(heights) - block_start[2])
    success = bool(xy_error < 0.045 and final[2] < 0.05 and lifted > 0.06 and moved > 0.08)
    result = {
        "initial_block_position": block_start.tolist(),
        "final_block_position": final.tolist(),
        "target_xy": TARGET_XY.tolist(),
        "target_xy_error": xy_error,
        "lift_height": lifted,
        "horizontal_distance_moved": moved,
        "ik_errors": ik_errors,
        "phase_positions": phase_positions,
        "success": success,
    }
    print(json.dumps(result, indent=2), flush=True)
    return success


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--object-x", type=float, default=0.25)
    parser.add_argument("--object-y", type=float, default=0.0)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
    data = mujoco.MjData(model)
    arm_dofs = [model.joint(name).dofadr[0] for name in ARM]
    arm_actuators = [model.actuator(name).id for name in ARM]
    gripper_actuator = model.actuator("gripper").id
    gripper_body = model.body("gripper").id
    block_body = model.body("block").id

    controls = {
        "paused": False,
        "replay_requested": False,
        "quit_requested": False,
        "object_initial_xy": [args.object_x, args.object_y],
    }

    def key_callback(keycode):
        if keycode in (KEY_R, ord("r")):
            controls["replay_requested"] = True
        elif keycode == KEY_SPACE:
            controls["paused"] = not controls["paused"]
        elif keycode == KEY_ESCAPE:
            controls["quit_requested"] = True

    context = (
        mujoco.viewer.launch_passive(model, data, key_callback=key_callback)
        if args.viewer else nullcontext(None)
    )
    success = False
    with context as viewer:
        if viewer is not None:
            viewer.cam.lookat[:] = (0.16, 0.05, 0.12)
            viewer.cam.distance = 0.85
            viewer.cam.azimuth = 140
            viewer.cam.elevation = -25

        while True:
            try:
                success = run_episode(
                    model, data, viewer, controls, arm_dofs, arm_actuators,
                    gripper_actuator, gripper_body, block_body,
                )
                if viewer is None:
                    break
                set_viewer_text(viewer, controls, "finished - press R to replay")
                print("Press R to replay, Space to pause/resume, or Esc to quit.", flush=True)
                while viewer.is_running():
                    handle_controls(viewer, controls, "finished - press R to replay")
                    viewer.sync()
                    time.sleep(0.05)
                break
            except ReplayRequested:
                print("Replay requested: resetting simulation.", flush=True)
                continue
            except QuitRequested:
                controls["quit_requested"] = True
                if viewer is not None:
                    viewer.close()
                break

    if not success and not controls["quit_requested"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
