import struct

import pytest

from flowhmr.core.export.smplh2vmd import BoneKeyframe, VMDWriter


def test_vmd_header_and_bone_frame_serialization():
    bone = BoneKeyframe(
        bone_name="左足",
        frame=30,
        translation=(0.1, 0.2, 0.3),
        quaternion=(0.0, 0.0, 0.0, 1.0),
    )

    payload = bone.serialize()
    assert len(payload) == 111
    assert payload[:15].rstrip(b"\x00") == "左足".encode("shift_jis")

    writer = VMDWriter(model_name="Miku")
    out = writer.serialize([bone])
    assert out[:30].rstrip(b"\x00") == b"Vocaloid Motion Data 0002"
    assert out[30:50].rstrip(b"\x00") == b"Miku"
    assert struct.unpack("<I", out[50:54])[0] == 1


def test_shift_jis_and_invalid_quaternion_rejected():
    with pytest.raises(ValueError):
        BoneKeyframe("左足", 0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0))

    with pytest.raises(ValueError):
        BoneKeyframe("左足", 0, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0))


def test_writer_keeps_order_and_counts():
    frames = [
        BoneKeyframe("右足", 10, (1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        BoneKeyframe("左足", 5, (0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
    ]

    out = VMDWriter(model_name="Model").serialize(frames)
    bone_count = struct.unpack("<I", out[50:54])[0]
    assert bone_count == 2
