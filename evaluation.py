"""Fixed-scene pick-and-place success criteria for diagnostics and evaluation."""
import numpy as np


TARGET_XY = np.array([0.20, 0.12])


def score(object_positions, gripper_positions, *, gripper_open: bool, fps=25):
    obj = np.asarray(object_positions, dtype=float)
    grip = np.asarray(gripper_positions, dtype=float)
    if obj.ndim != 2 or obj.shape[1] != 3 or grip.shape != obj.shape or len(obj) < 2:
        raise ValueError("Expected matching N x 3 trajectories")
    lift = float(obj[:, 2].max() - obj[0, 2])
    final_xy_error = float(np.linalg.norm(obj[-1, :2] - TARGET_XY))
    retreat = float(np.linalg.norm(grip[-1] - obj[-1]))
    window = min(len(obj), max(2, round(0.5 * fps)))
    final_speed = float(np.max(np.linalg.norm(np.diff(obj[-window:], axis=0), axis=1)) * fps)
    stable_window_present = len(obj) >= round(0.5 * fps)
    success = bool(lift >= 0.03 and final_xy_error <= 0.03 and obj[-1, 2] <= 0.05
                   and retreat >= 0.05 and gripper_open and stable_window_present and final_speed <= 0.015)
    return {"success": success, "max_lift_m": lift, "final_xy_error_m": final_xy_error,
            "retreat_m": retreat, "final_speed_m_s": final_speed,
            "stable_window_present": stable_window_present, "gripper_open": bool(gripper_open)}
