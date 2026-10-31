"""Fingerprint every local asset that defines the isolated MuJoCo model."""
import hashlib
from pathlib import Path


def fingerprint(root: Path) -> str:
    paths = [root / "scenes/so101_pick_place.xml", root / "configs/home.json",
             root / "third_party/ICL-so101/sim/so101/so101_new_calib_camera.xml"]
    paths.extend(sorted((root / "third_party/ICL-so101/sim/so101/assets").glob("*")))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()
