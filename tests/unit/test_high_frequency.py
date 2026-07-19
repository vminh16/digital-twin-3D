from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PIL import Image

from bts_nvs.cameras.distortion import CameraDistortion
from bts_nvs.cameras.intrinsics import CameraIntrinsics
from bts_nvs.data.dataset import CameraSample
from bts_nvs.evaluation.high_frequency import (
    evaluate_render_directory,
    high_frequency_metrics,
)


def _step_image() -> np.ndarray:
    image = np.zeros((32, 32, 3), dtype=np.float64)
    image[:, 16:] = 1.0
    return image


def _sample(name: str = "validation.JPG") -> CameraSample:
    image = (_step_image() * 255.0).astype(np.uint8)
    mask = np.ones((32, 32), dtype=bool)
    mask[0] = False
    return CameraSample(
        image=image,
        world_to_camera=np.eye(4),
        intrinsics=CameraIntrinsics(32, 32, 16.0, 16.0, 16.0, 16.0),
        distortion=CameraDistortion("PINHOLE", ()),
        valid_mask=mask,
        image_name=name,
    )


class _Dataset:
    def __init__(self, samples: tuple[CameraSample, ...]) -> None:
        self.samples = samples
        self.manifest = SimpleNamespace(scene_id="scene")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> CameraSample:
        return self.samples[index]


def test_identical_images_have_zero_high_frequency_error() -> None:
    image = _step_image()

    assert high_frequency_metrics(image, image) == {
        "hf_l1": 0.0,
        "missing_edge": 0.0,
        "spurious_edge": 0.0,
    }


def test_blurred_edge_increases_missing_edge() -> None:
    target = _step_image()
    blurred = cv2.GaussianBlur(target, (9, 9), 2.0)

    assert high_frequency_metrics(blurred, target)["missing_edge"] > 0.0


def test_noise_in_flat_region_increases_spurious_edge() -> None:
    target = _step_image()
    noisy = target.copy()
    checker = np.indices((8, 8)).sum(axis=0) % 2
    noisy[:8, :8] = checker[..., None]

    assert high_frequency_metrics(noisy, target)["spurious_edge"] > 0.0


@pytest.mark.parametrize(
    "prediction,target",
    [
        (np.zeros((4, 4)), np.zeros((4, 4))),
        (np.zeros((4, 4, 3)), np.zeros((5, 4, 3))),
        (np.full((4, 4, 3), np.nan), np.zeros((4, 4, 3))),
        (np.full((4, 4, 3), 2.0), np.zeros((4, 4, 3))),
    ],
)
def test_high_frequency_rejects_invalid_images(prediction, target) -> None:
    with pytest.raises(ValueError):
        high_frequency_metrics(prediction, target)


def test_render_directory_uses_dataset_names_and_masks_invalid_pixels(tmp_path) -> None:
    sample = _sample()
    rendered = sample.image.copy()
    rendered[0] = 255 - rendered[0]
    Image.fromarray(rendered).save(tmp_path / "validation.png")

    report = evaluate_render_directory(_Dataset((sample,)), tmp_path)

    assert report["image_count"] == 1
    assert set(report["images"]) == {"validation.JPG"}
    assert report["hf_l1_mean"] == pytest.approx(0.0)
    assert report["missing_edge_mean"] == pytest.approx(0.0)
    assert report["spurious_edge_mean"] == pytest.approx(0.0)


def test_render_directory_rejects_missing_and_colliding_png_names(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="missing validation render"):
        evaluate_render_directory(_Dataset((_sample(),)), tmp_path)

    samples = (_sample("same.JPG"), _sample("same.png"))
    with pytest.raises(ValueError, match="collide"):
        evaluate_render_directory(_Dataset(samples), tmp_path)
