#!/usr/bin/env python3
"""Convert validated human teleop raw episodes to a separate LeRobot v3 dataset."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from asset_fingerprint import fingerprint
from evaluation import score

NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
IMAGE = "observation.images.observation"
TASK = "Pick up the red cube and place it on the green target."


def validate(path: Path, allow_synthetic=False):
    with np.load(path, allow_pickle=False) as raw:
        meta = json.loads(str(raw["metadata"]))
        if meta.get("schema") != "teleop_raw_v1" or tuple(meta["joint_order"]) != NAMES:
            raise ValueError(f"Schema/joint mismatch: {path}")
        if meta.get("model_sha256") != fingerprint(ROOT):
            raise ValueError(f"Model fingerprint mismatch: {path}")
        if meta.get("timing_overruns", 1):
            raise ValueError(f"Timing quality failure: {path}")
        if meta.get("input") != "keyboard" and not allow_synthetic:
            raise ValueError(f"Not a human keyboard demonstration: {path}")
        if meta.get("auxiliary_view_used", True) and not allow_synthetic:
            raise ValueError(f"Auxiliary operator view was used: {path}")
        if meta.get("label") != "success":
            raise ValueError(f"Not labeled successful: {path}")
        if not allow_synthetic:
            if "object_pos" not in raw or "gripper_pos" not in raw:
                raise ValueError(f"Missing success trajectories: {path}")
            assessment = score(raw["object_pos"], raw["gripper_pos"],
                               gripper_open=bool(raw["action"][-1, -1] > 0.8))
            if not assessment["success"] or not meta.get("automatic_assessment", {}).get("success"):
                raise ValueError(f"Automatic success check failed: {path}")
        images, states, actions = (raw[k].copy() for k in ("image", "state", "action"))
        n = len(images)
        if n < 2 or images.shape != (n, 240, 320, 3) or images.dtype != np.uint8:
            raise ValueError(f"Image shape/dtype mismatch: {path}")
        if states.shape != (n, 6) or actions.shape != (n, 6) or not np.isfinite(states).all() or not np.isfinite(actions).all():
            raise ValueError(f"State/action mismatch: {path}")
        if not np.allclose(np.diff(raw["sim_time"]), 0.04, atol=1e-5):
            raise ValueError(f"Simulation cadence mismatch: {path}")
        if not np.all(np.diff(raw["host_time"]) > 0) or not np.all(raw["send_time"] >= raw["host_time"]):
            raise ValueError(f"Host timestamp mismatch: {path}")
        return images, states.astype(np.float32), actions.astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("raw_dir", type=Path)
    p.add_argument("output_dir", type=Path)
    p.add_argument("--repo-id", default="local/so101_teleop")
    p.add_argument("--episode-list", type=Path,
                   help="Text file of exact .npz basenames to convert, one per line")
    p.add_argument("--split", choices=("train", "validation"), default="train")
    p.add_argument("--diagnostic-synthetic", action="store_true",
                   help="Allow synthetic input only for conversion API tests; output path must contain diagnostic")
    args = p.parse_args()
    if args.episode_list:
        names = [line.strip() for line in args.episode_list.read_text().splitlines() if line.strip()]
        if not names or len(names) != len(set(names)) or any(Path(n).name != n or not n.endswith(".npz") for n in names):
            raise ValueError("Invalid episode list")
        files = [args.raw_dir / name for name in names]
        if not all(path.is_file() for path in files):
            raise FileNotFoundError("Listed episode is missing")
    else:
        files = sorted(args.raw_dir.glob("*.npz"))
    if not files:
        raise ValueError("No raw episodes")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.diagnostic_synthetic and "diagnostic" not in str(args.output_dir).lower():
        raise ValueError("Diagnostic output path must contain 'diagnostic'")
    validated = [(path, *validate(path, allow_synthetic=args.diagnostic_synthetic)) for path in files]
    from lerobot.datasets import LeRobotDataset
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id, root=args.output_dir, fps=25, robot_type="so101_mujoco",
        features={
            IMAGE: {"dtype": "video", "shape": (240, 320, 3), "names": ["height", "width", "channel"]},
            "observation.state": {"dtype": "float32", "shape": (6,), "names": list(NAMES)},
            "action": {"dtype": "float32", "shape": (6,), "names": list(NAMES)},
        }, use_videos=True, image_writer_threads=4)
    manifest = []
    for path, images, states, actions in validated:
        for image, state, action in zip(images, states, actions, strict=True):
            dataset.add_frame({IMAGE: image, "observation.state": state, "action": action, "task": TASK})
        dataset.save_episode()
        manifest.append({"raw": path.name, "raw_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "frames": len(images)})
    dataset.finalize()
    (args.output_dir / "teleop_manifest.json").write_text(json.dumps({
        "episodes": manifest, "model_sha256": fingerprint(ROOT),
        "diagnostic_synthetic": args.diagnostic_synthetic,
        "split": "diagnostic" if args.diagnostic_synthetic else args.split,
    }, indent=2))
    print(json.dumps({"episodes": len(manifest), "frames": sum(x["frames"] for x in manifest)}, indent=2))


if __name__ == "__main__":
    main()
