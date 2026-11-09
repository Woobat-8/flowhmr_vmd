import numpy as np

from flowhmr.core.export.smplh2vmd import (
    SMPLH_TO_MMD_BONE_MAP,
    convert_cartesian_to_mmd,
    convert_quaternion_to_mmd,
)


def test_mapping_contains_expected_targets():
    assert SMPLH_TO_MMD_BONE_MAP["pelvis"] == "下半身"
    assert SMPLH_TO_MMD_BONE_MAP["spine1"] == "上半身"
    assert SMPLH_TO_MMD_BONE_MAP["spine2"] == "上半身2"
    assert SMPLH_TO_MMD_BONE_MAP["spine3"] == "上半身3"
    assert SMPLH_TO_MMD_BONE_MAP["head"] == "頭"
    assert SMPLH_TO_MMD_BONE_MAP["left_shoulder"] == "左腕"
    assert SMPLH_TO_MMD_BONE_MAP["left_wrist"] == "左手首"
    assert SMPLH_TO_MMD_BONE_MAP["left_knee"] == "左ひざ"
    assert SMPLH_TO_MMD_BONE_MAP["left_ankle"] == "左足首"
    assert SMPLH_TO_MMD_BONE_MAP["left_foot"] == "左つま先"
    assert SMPLH_TO_MMD_BONE_MAP["left_index1"] == "左人指１"
    assert SMPLH_TO_MMD_BONE_MAP["right_index1"] == "右人指１"


def test_vector_conversion_changes_basis_consistently():
    vec = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    converted = convert_cartesian_to_mmd(vec)
    assert np.allclose(converted, np.array([1.0, 3.0, -2.0]))


def test_quaternion_conversion_preserves_rotation_norm():
    quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    converted = convert_quaternion_to_mmd(quat)
    assert np.isclose(np.linalg.norm(converted), 1.0)
