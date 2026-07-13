import csv

import numpy as np
import pytest

from bts_nvs.data.manifest import (
    TEST_POSE_COLUMNS,
    output_name_from_test_image_name,
    read_test_poses_csv,
)
from bts_nvs.data.validation import DataContractError


def _row(**overrides):
    values = {
        "image_name": "test_001.JPG",
        "qw": "1",
        "qx": "0",
        "qy": "0",
        "qz": "0",
        "tx": "1.5",
        "ty": "-2",
        "tz": "3",
        "fx": "926",
        "fy": "926",
        "cx": "660",
        "cy": "494.5",
        "width": "1320",
        "height": "989",
    }
    values.update(overrides)
    return values


def _write_csv(path, rows, columns=TEST_POSE_COLUMNS):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def test_read_test_poses_csv_preserves_order_and_colmap_convention(tmp_path):
    path = tmp_path / "test_poses.csv"
    _write_csv(path, [_row(), _row(image_name="test_002.JPG", tx="4")])

    poses = read_test_poses_csv(path)

    assert [pose.image_name for pose in poses] == ["test_001.JPG", "test_002.JPG"]
    assert [pose.output_name for pose in poses] == ["test_001.png", "test_002.png"]
    np.testing.assert_allclose(poses[0].world_to_camera[:3, 3], [1.5, -2.0, 3.0])
    assert poses[0].intrinsics.width == 1320
    assert poses[0].intrinsics.height == 989
    assert poses[0].intrinsics.fx == 926.0


def test_negative_quaternion_represents_same_rotation(tmp_path):
    path = tmp_path / "test_poses.csv"
    _write_csv(path, [_row(qw="-1")])

    pose = read_test_poses_csv(path)[0]

    np.testing.assert_allclose(pose.world_to_camera[:3, :3], np.eye(3))


@pytest.mark.parametrize(
    "name",
    [
        "../test.JPG",
        "folder/test.JPG",
        "folder\\test.JPG",
        "",
        ".",
        "..",
        "CON.JPG",
        "com1.jpg",
        "bad:name.JPG",
        "bad\x00name.JPG",
        "trailing. ",
    ],
)
def test_output_name_rejects_non_basename_input(name):
    with pytest.raises(DataContractError):
        output_name_from_test_image_name(name)


def test_output_name_replaces_source_suffix_with_png():
    assert output_name_from_test_image_name("DJI_0001.JPG") == "DJI_0001.png"


def test_csv_requires_exact_column_order(tmp_path):
    path = tmp_path / "test_poses.csv"
    columns = list(TEST_POSE_COLUMNS)
    columns[1], columns[2] = columns[2], columns[1]
    _write_csv(path, [_row()], tuple(columns))

    with pytest.raises(DataContractError, match="columns"):
        read_test_poses_csv(path)


def test_csv_rejects_duplicate_names_and_output_collisions(tmp_path):
    duplicate_path = tmp_path / "duplicate.csv"
    _write_csv(duplicate_path, [_row(), _row()])
    with pytest.raises(DataContractError, match="duplicate"):
        read_test_poses_csv(duplicate_path)

    collision_path = tmp_path / "collision.csv"
    _write_csv(
        collision_path,
        [_row(image_name="same.JPG"), _row(image_name="same.jpeg")],
    )
    with pytest.raises(DataContractError, match="output"):
        read_test_poses_csv(collision_path)

    case_collision_path = tmp_path / "case_collision.csv"
    _write_csv(
        case_collision_path,
        [_row(image_name="same.JPG"), _row(image_name="SAME.jpeg")],
    )
    with pytest.raises(DataContractError, match="output"):
        read_test_poses_csv(case_collision_path)


def test_csv_rejects_rows_with_more_than_fourteen_values(tmp_path):
    path = tmp_path / "test_poses.csv"
    values = [_row()[column] for column in TEST_POSE_COLUMNS]
    path.write_text(
        ",".join(TEST_POSE_COLUMNS) + "\n" + ",".join(values + ["UNEXPECTED"]) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(DataContractError, match="14"):
        read_test_poses_csv(path)


@pytest.mark.parametrize(
    "overrides",
    [
        {"qw": "0", "qx": "0", "qy": "0", "qz": "0"},
        {"fx": "nan"},
        {"width": "1320.5"},
        {"height": "0"},
    ],
)
def test_csv_rejects_invalid_numeric_contract(tmp_path, overrides):
    path = tmp_path / "test_poses.csv"
    _write_csv(path, [_row(**overrides)])

    with pytest.raises(DataContractError):
        read_test_poses_csv(path)
