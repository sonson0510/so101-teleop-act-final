#!/usr/bin/env python3
"""Evaluate an isolated teleop ACT policy from the same fixed home as collection."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

import mujoco
import numpy as np
import torch
import av
from PIL import Image, ImageDraw, ImageFont
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act import ACTPolicy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from teleop_core import CommandLimiter
from asset_fingerprint import fingerprint
from evaluation import score


JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
TARGET_XY = np.array([0.20, 0.12])


def write_video(frames, output: Path, label: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    font_path = Path("/mnt/c/Windows/Fonts/NotoSansJP-VF.ttf")
    font = ImageFont.truetype(str(font_path), 17) if font_path.exists() else ImageFont.load_default()
    container = av.open(str(output), mode="w")
    stream = container.add_stream("libx264", rate=25)
    stream.width = frames[0].shape[1]
    stream.height = frames[0].shape[0]
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "21", "preset": "medium"}
    for index, array in enumerate(frames):
        image = Image.fromarray(array)
        draw = ImageDraw.Draw(image, "RGBA")
        draw.rectangle((0, 0, image.width, 33), fill=(8, 20, 38, 205))
        draw.text((9, 5), f"{label} | {index / 25:.2f} s", font=font, fill="white")
        frame = av.VideoFrame.from_ndarray(np.asarray(image), format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--allow-diagnostic-synthetic", action="store_true")
    parser.add_argument(
        "--n-action-steps",
        type=int,
        help="Override how many predicted actions are executed before replanning.",
    )
    parser.add_argument("--video-output", type=Path)
    parser.add_argument("--video-label", default="SO-101 ACT A0")
    parser.add_argument(
        "--direct-sim-control",
        action="store_true",
        help="Match the prior simulation evaluation: clip to actuator limits without the per-tick delta limiter.",
    )
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    metrics_path = args.artifact_dir / "metrics.json"
    if not metrics_path.is_file() and args.artifact_dir.parent.name == "checkpoints":
        metrics_path = args.artifact_dir.parent.parent / "metrics.json"
    train_metrics = json.loads(metrics_path.read_text())
    manifest = json.loads((Path(train_metrics["dataset_root"]) / "teleop_manifest.json").read_text())
    if manifest["model_sha256"] != fingerprint(ROOT):
        raise ValueError("Evaluation model assets differ from training dataset")
    if manifest.get("diagnostic_synthetic") and (
        not args.allow_diagnostic_synthetic or "diagnostic" not in str(args.output_dir).lower() or args.steps > 5
    ):
        raise ValueError("Synthetic policy evaluation is limited to short diagnostic output")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifact = args.artifact_dir.resolve()
    policy = ACTPolicy.from_pretrained(artifact / "policy").to(device).eval()
    if args.n_action_steps is not None:
        if not 1 <= args.n_action_steps <= policy.config.chunk_size:
            raise ValueError("--n-action-steps must be between 1 and chunk_size")
        policy.config.n_action_steps = args.n_action_steps
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(artifact),
    )
    policy.reset()

    model = mujoco.MjModel.from_xml_path(str(ROOT / "scenes/so101_pick_place.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    home = json.loads((ROOT / "configs/home.json").read_text())
    if tuple(home["joint_order"]) != JOINTS:
        raise ValueError("Home joint order mismatch")
    block_body = model.body("block").id
    actuator_ids = [model.actuator(name).id for name in JOINTS]
    joint_addresses = [model.joint(name).qposadr[0] for name in JOINTS]
    data.qpos[joint_addresses] = home["qpos_rad"]
    data.ctrl[actuator_ids] = home["qpos_rad"]
    mujoco.mj_forward(model, data)
    # Simulator stabilization is outside the learned episode.
    for _ in range(100):
        mujoco.mj_step(model, data)
    block_start = data.xpos[block_body].copy()
    renderer = mujoco.Renderer(model, height=240, width=320)
    for flag in (mujoco.mjtRndFlag.mjRND_SHADOW, mujoco.mjtRndFlag.mjRND_REFLECTION,
                 mujoco.mjtRndFlag.mjRND_FOG, mujoco.mjtRndFlag.mjRND_SKYBOX):
        renderer.scene.flags[flag] = 0
    limits = np.array([model.jnt_range[model.joint(n).id] for n in JOINTS])
    limiter = CommandLimiter(limits[:, 0], limits[:, 1], [0.04] * 5 + [0.08])
    limited_count = 0
    positions, gripper_positions, actions, raw_actions, inference_ms, video_frames = [], [], [], [], [], []
    try:
        for step in range(args.steps):
            renderer.update_scene(data, camera="observation")
            image = renderer.render().copy()
            if args.video_output is not None:
                video_frames.append(image)
            state = data.qpos[joint_addresses].astype(np.float32).copy()
            batch = {
                "observation.images.observation": (
                    torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
                ),
                "observation.state": torch.from_numpy(state).unsqueeze(0),
            }
            started = time.perf_counter()
            batch = preprocessor(batch)
            action = policy.select_action(batch)
            action = postprocessor(action).squeeze(0).cpu().numpy()
            if device.type == "cuda":
                torch.cuda.synchronize()
            inference_ms.append((time.perf_counter() - started) * 1000)
            raw_actions.append(action.astype(np.float32).copy())
            if args.direct_sim_control:
                action = np.clip(action, limits[:, 0], limits[:, 1])
            else:
                action, limited = limiter.apply(action, data.ctrl[actuator_ids])
                limited_count += int(limited)
            data.ctrl[actuator_ids] = action
            actions.append(action.astype(np.float32))
            positions.append(data.xpos[block_body].astype(np.float32).copy())
            gripper_positions.append(data.xpos[model.body("gripper").id].astype(np.float32).copy())
            for _ in range(20):
                mujoco.mj_step(model, data)
    finally:
        renderer.close()

    positions = np.asarray(positions)
    final = data.xpos[block_body].copy()
    assessment = score(np.vstack([positions, final]),
                       np.vstack([gripper_positions, data.xpos[model.body("gripper").id]]),
                       gripper_open=bool(actions[-1][-1] > 0.8))
    legacy_moved_m = float(np.linalg.norm(final[:2] - block_start[:2]))
    legacy_success = bool(
        assessment["max_lift_m"] > 0.06
        and assessment["final_xy_error_m"] < 0.045
        and legacy_moved_m > 0.08
        and final[2] < 0.05
    )
    metrics = {
        "diagnostic_synthetic": bool(manifest.get("diagnostic_synthetic")),
        "model_sha256": fingerprint(ROOT),
        "chunk_size": policy.config.chunk_size,
        "n_action_steps": policy.config.n_action_steps,
        "control_mode": "direct_sim_clip" if args.direct_sim_control else "delta_limited",
        **assessment,
        "legacy_a100_success": legacy_success,
        "legacy_moved_m": legacy_moved_m,
        "final_block_z_m": float(final[2]),
        "mean_inference_ms": float(np.mean(inference_ms)),
        "limited_actions": limited_count,
        "raw_action_min": np.min(np.asarray(raw_actions), axis=0).tolist(),
        "raw_action_max": np.max(np.asarray(raw_actions), axis=0).tolist(),
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        output / "trajectory.npz",
        block_position=positions,
        action=np.asarray(actions),
        raw_action=np.asarray(raw_actions),
    )
    if args.video_output is not None:
        write_video(video_frames, args.video_output.resolve(), args.video_label)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
