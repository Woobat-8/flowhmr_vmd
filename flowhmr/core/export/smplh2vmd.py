from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

try:
    from flowhmr.core.bodymodels.smpl_skeleton import SMPLH_JOINT_NAMES_52 as _SMPLH_JOINT_NAMES_52
except Exception:  # pragma: no cover - fallback for standalone tooling without the full ML stack
    _SMPLH_JOINT_NAMES_52 = [
        "pelvis",
        "left_hip",
        "right_hip",
        "spine1",
        "left_knee",
        "right_knee",
        "spine2",
        "left_ankle",
        "right_ankle",
        "spine3",
        "left_foot",
        "right_foot",
        "neck",
        "left_collar",
        "right_collar",
        "head",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_index1",
        "left_index2",
        "left_index3",
        "left_middle1",
        "left_middle2",
        "left_middle3",
        "left_pinky1",
        "left_pinky2",
        "left_pinky3",
        "left_ring1",
        "left_ring2",
        "left_ring3",
        "left_thumb1",
        "left_thumb2",
        "left_thumb3",
        "right_index1",
        "right_index2",
        "right_index3",
        "right_middle1",
        "right_middle2",
        "right_middle3",
        "right_pinky1",
        "right_pinky2",
        "right_pinky3",
        "right_ring1",
        "right_ring2",
        "right_ring3",
        "right_thumb1",
        "right_thumb2",
        "right_thumb3",
    ]

SMPLH_JOINT_NAMES_52 = _SMPLH_JOINT_NAMES_52


# --- Canonical MMD bone names used by the retargeter ---
MMD_CORE_BONES = {
    "全ての親": "全ての親",
    "センター": "センター",
    "グルーブ": "グルーブ",
    "腰": "腰",
    "下半身": "下半身",
    "上半身": "上半身",
    "上半身2": "上半身2",
    "上半身3": "上半身3",
    "首": "首",
    "頭": "頭",
    "左肩": "左肩",
    "右肩": "右肩",
    "左腕": "左腕",
    "右腕": "右腕",
    "左ひじ": "左ひじ",
    "右ひじ": "右ひじ",
    "左手首": "左手首",
    "右手首": "右手首",
    "左足": "左足",
    "右足": "右足",
    "左ひざ": "左ひざ",
    "右ひざ": "右ひざ",
    "左足首": "左足首",
    "右足首": "右足首",
    "左つま先": "左つま先",
    "右つま先": "右つま先",
}

MMD_FINGER_BONES = {
    "left_thumb1": "左親指１",
    "left_thumb2": "左親指２",
    "left_thumb3": "左親指３",
    "left_index1": "左人指１",
    "left_index2": "左人指２",
    "left_index3": "左人指３",
    "left_middle1": "左中指１",
    "left_middle2": "左中指２",
    "left_middle3": "左中指３",
    "left_ring1": "左薬指１",
    "left_ring2": "左薬指２",
    "left_ring3": "左薬指３",
    "left_pinky1": "左小指１",
    "left_pinky2": "左小指２",
    "left_pinky3": "左小指３",
    "right_thumb1": "右親指１",
    "right_thumb2": "右親指２",
    "right_thumb3": "右親指３",
    "right_index1": "右人指１",
    "right_index2": "右人指２",
    "right_index3": "右人指３",
    "right_middle1": "右中指１",
    "right_middle2": "右中指２",
    "right_middle3": "右中指３",
    "right_ring1": "右薬指１",
    "right_ring2": "右薬指２",
    "right_ring3": "右薬指３",
    "right_pinky1": "右小指１",
    "right_pinky2": "右小指２",
    "right_pinky3": "右小指３",
}

# Build an explicit mapping from the actual SMPL-H joint names used in the repo (52-joint layout)
# to the conventional target MMD humanoid skeleton.
# This is intentionally conservative: only bones that have a clear MMD equivalent are mapped.
SMPLH_TO_MMD_BONE_MAP = {
    "pelvis": "下半身",
    "left_hip": "左足",
    "right_hip": "右足",
    "spine1": "上半身",
    "left_knee": "左ひざ",
    "right_knee": "右ひざ",
    "spine2": "上半身2",
    "left_ankle": "左足首",
    "right_ankle": "右足首",
    "spine3": "上半身3",
    "left_foot": "左つま先",
    "right_foot": "右つま先",
    "neck": "首",
    "left_collar": "左肩",
    "right_collar": "右肩",
    "head": "頭",
    "left_shoulder": "左腕",
    "right_shoulder": "右腕",
    "left_elbow": "左ひじ",
    "right_elbow": "右ひじ",
    "left_wrist": "左手首",
    "right_wrist": "右手首",
    **MMD_FINGER_BONES,
}

# Additional MMD names that are often present but not required for a functional animation.
MMD_OPTIONAL_BONES = {
    "左肩P": "左肩P",
    "右肩P": "右肩P",
    "左腕捩": "左腕捩",
    "右腕捩": "右腕捩",
    "左手捩": "左手捩",
    "右手捩": "右手捩",
    "左足ＩＫ": "左足ＩＫ",
    "右足ＩＫ": "右足ＩＫ",
    "左つま先ＩＫ": "左つま先ＩＫ",
    "右つま先ＩＫ": "右つま先ＩＫ",
}


def build_smplh_to_mmd_mapping() -> dict[str, str]:
    """Return the explicit, repo-derived SMPL-H to MMD mapping used by the retargeter."""
    mapping = dict(SMPLH_TO_MMD_BONE_MAP)
    # The canonical MMD hierarchy expects a pelvis/waist to root relation; we keep the mapping explicit.
    mapping["root"] = "全ての親"
    mapping["center"] = "センター"
    mapping["groove"] = "グルーブ"
    mapping["waist"] = "腰"
    return mapping


def convert_cartesian_to_mmd(vec: Sequence[float] | np.ndarray) -> np.ndarray:
    """Map a Cartesian vector from SMPL-H space to the conventional MMD basis.

    This is a conservative basis conversion intended for an MMD humanoid rig:
    - keep X as the horizontal right axis
    - move the original Y component into the MMD Z axis
    - flip the sign of the original Y component into the MMD depth axis
    - preserve the original X axis

    This is not a model-specific rig solver; it is a stable basis-conversion helper used to
    steer the exported motion into the target MMD coordinate convention.
    """
    arr = np.asarray(vec, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(f"Expected a 3D vector, got shape {arr.shape}")
    return np.array([arr[0], arr[2], -arr[1]], dtype=np.float64)


def quaternion_to_rotation_matrix(quat: Sequence[float]) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64)
    if q.shape != (4,):
        raise ValueError(f"Quaternion must have shape (4,), got {q.shape}")
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotation_matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 rotation matrix, got shape {m.shape}")
    tr = np.trace(m)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s

    q = np.array([w, x, y, z], dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm <= 1e-12:
        raise ValueError("Quaternion conversion failed: zero norm")
    q /= norm
    if q[0] < 0:
        q = -q
    return q


def convert_quaternion_to_mmd(quat: Sequence[float]) -> np.ndarray:
    """Convert an SMPL-H quaternion to the MMD basis using a stable rotation-basis transform.

    The repo already rotates exported SMPL-H root transforms around Y by 180° before saving.
    The MMD conversion here applies a further basis adjustment to account for the left-handed
    MMD coordinate convention. This is a standard retargeting basis transform rather than naively
    copying the source quaternion.
    """
    q = np.asarray(quat, dtype=np.float64)
    if q.shape != (4,):
        raise ValueError(f"Expected quaternion (4,), got {q.shape}")
    matrix = quaternion_to_rotation_matrix(q)
    basis = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    converted = basis @ matrix @ basis.T
    return rotation_matrix_to_quaternion(converted)


def get_mmd_joint_mapping() -> dict[str, str]:
    """Return the standard MMD mapping with the repo’s actual SMPL-H joint names as the keys."""
    return build_smplh_to_mmd_mapping()


# Backwards-compatible helper names for the retargeting layer.
MMD_TO_SMPLH_MAP = build_smplh_to_mmd_mapping()
MMD_STANDARD_BONES = tuple(sorted(MMD_CORE_BONES.keys()))


# VMD file generation code continues below.
VMD_HEADER_MAGIC = b"Vocaloid Motion Data 0002"
VMD_HEADER_SIZE = 30
VMD_MODEL_NAME_SIZE = 20
VMD_BONE_NAME_SIZE = 15
VMD_INTERPOLATION_BYTES = 64


def _as_float_array(values: Sequence[float], length: int, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != (length,):
        raise ValueError(f"{name} must have length {length}, got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _normalize_quaternion(quat: Sequence[float]) -> tuple[float, float, float, float]:
    q = np.asarray(quat, dtype=np.float64)
    if q.shape != (4,):
        raise ValueError(f"Quaternion must be length 4, got shape {q.shape}")
    if not np.all(np.isfinite(q)):
        raise ValueError("Quaternion contains non-finite values")
    norm = float(np.linalg.norm(q))
    if norm <= 1e-12:
        raise ValueError("Quaternion norm is zero")
    q = q / norm
    if q[3] < 0:
        q = -q
    return tuple(float(v) for v in q)


def _encode_fixed_text(value: str, width: int, *, name: str) -> bytes:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a str")
    data = value.encode("shift_jis")
    if len(data) > width:
        raise ValueError(f"{name} encoded length {len(data)} exceeds {width} bytes: {value!r}")
    return data.ljust(width, b"\x00")


def _default_interpolation_bytes() -> bytes:
    # A conventional, valid VMD interpolation block.
    pattern = b"\x14\x14\x6b\x6b"
    return (pattern * 16)[: VMD_INTERPOLATION_BYTES]


@dataclass(frozen=True)
class BoneKeyframe:
    bone_name: str
    frame: int
    translation: tuple[float, float, float]
    quaternion: tuple[float, float, float, float]
    interpolation: bytes | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.frame, (int, np.integer)):
            raise TypeError("frame must be an integer")
        if self.frame < 0:
            raise ValueError("frame must be non-negative")

        _as_float_array(self.translation, 3, name="translation")
        quat = _normalize_quaternion(self.quaternion)
        object.__setattr__(self, "quaternion", quat)

        if self.interpolation is None:
            object.__setattr__(self, "interpolation", _default_interpolation_bytes())
        elif len(self.interpolation) != VMD_INTERPOLATION_BYTES:
            raise ValueError(
                f"interpolation must be exactly {VMD_INTERPOLATION_BYTES} bytes, got {len(self.interpolation)}"
            )

    def serialize(self) -> bytes:
        encoded_name = _encode_fixed_text(self.bone_name, VMD_BONE_NAME_SIZE, name="bone_name")
        trans = _as_float_array(self.translation, 3, name="translation")
        quat = np.asarray(_normalize_quaternion(self.quaternion), dtype=np.float64)

        payload = bytearray()
        payload.extend(encoded_name)
        payload.extend(struct.pack("<I", int(self.frame)))
        payload.extend(struct.pack("<fff", float(trans[0]), float(trans[1]), float(trans[2])))
        payload.extend(struct.pack("<ffff", float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])))
        payload.extend(self.interpolation)
        return bytes(payload)


class VMDWriter:
    def __init__(self, model_name: str = "Model"):
        self.model_name = model_name

    def serialize(self, bone_keyframes: Iterable[BoneKeyframe]) -> bytes:
        ordered = sorted(bone_keyframes, key=lambda k: (k.bone_name, int(k.frame)))
        header = VMD_HEADER_MAGIC.ljust(VMD_HEADER_SIZE, b"\x00")
        model_name = _encode_fixed_text(self.model_name, VMD_MODEL_NAME_SIZE, name="model_name")

        payload = bytearray()
        payload.extend(header)
        payload.extend(model_name)
        payload.extend(struct.pack("<I", len(ordered)))

        for frame in ordered:
            payload.extend(frame.serialize())

        # Additional VMD sections: morphs, cameras, lights, self-shadows, IK info.
        # For a bone-only animation, the counts are zero.
        for _ in range(5):
            payload.extend(struct.pack("<I", 0))

        return bytes(payload)

    def write(self, path: str | Path, bone_keyframes: Iterable[BoneKeyframe]) -> Path:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(self.serialize(bone_keyframes))
        return out_path


def serialize_vmd(
    bone_keyframes: Iterable[BoneKeyframe], *,
    model_name: str = "Model",
) -> bytes:
    return VMDWriter(model_name=model_name).serialize(bone_keyframes)


__all__ = [
    "VMD_HEADER_MAGIC",
    "VMD_HEADER_SIZE",
    "VMD_BONE_NAME_SIZE",
    "BoneKeyframe",
    "VMDWriter",
    "serialize_vmd",
]
