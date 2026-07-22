from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PIL import Image

from bts_nvs.evaluation.detail_metrics import (
    detail_metrics,
    evaluate_detail_directory,
)


def _step_image(offset: int = 16) -> np.ndarray:
    image = np.zeros((32, 32, 3), dtype=np.float64)
    image[:, offset:] = 1.0
    return image


def _sample(name: str = "validation.JPG") -> SimpleNamespace:
    image = (_step_image() * 255.0).astype(np.uint8)
    mask = np.ones((32, 32), dtype=bool)
    mask[0] = False
    return SimpleNamespace(image=image, valid_mask=mask, image_name=name)


class _Dataset:
    def __init__(self, samples: tuple[SimpleNamespace, ...]) -> None:
        self.samples = samples
        self.manifest = SimpleNamespace(scene_id="scene")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> SimpleNamespace:
        return self.samples[index]


def test_identical_images_have_zero_detail_error() -> None:
    image = _step_image()

    assert detail_metrics(image, image) == {
        "hf_l1": 0.0,
        "missing_edge": 0.0,
        "spurious_edge": 0.0,
        "symmetric_edge_distance": 0.0,
    }


def test_blur_increases_missing_edge_error() -> None:
    target = _step_image()
    blurred = cv2.GaussianBlur(target, (9, 9), 2.0)

    assert detail_metrics(blurred, target)["missing_edge"] > 0.0


def test_flat_region_noise_increases_spurious_edge_error() -> None:
    target = _step_image()
    noisy = target.copy()
    checker = np.indices((8, 8)).sum(axis=0) % 2
    noisy[:8, :8] = checker[..., None]

    assert detail_metrics(noisy, target)["spurious_edge"] > 0.0


def test_shifted_edge_increases_symmetric_edge_distance() -> None:
    target = _step_image()
    shifted = _step_image(offset=20)

    assert detail_metrics(shifted, target)["symmetric_edge_distance"] > 0.0


def test_valid_mask_removes_invalid_pixel_error() -> None:
    target = _step_image()
    prediction = target.copy()
    prediction[0] = 1.0 - prediction[0]
    mask = np.ones((32, 32), dtype=bool)
    mask[0] = False

    assert detail_metrics(prediction, target, mask) == {
        "hf_l1": 0.0,
        "missing_edge": 0.0,
        "spurious_edge": 0.0,
        "symmetric_edge_distance": 0.0,
    }


@pytest.mark.parametrize(
    "prediction,target,valid_mask",
    [
        (np.zeros((4, 4)), np.zeros((4, 4)), None),
        (np.zeros((4, 4, 3)), np.zeros((5, 4, 3)), None),
        (np.full((4, 4, 3), np.nan), np.zeros((4, 4, 3)), None),
        (np.full((4, 4, 3), 2.0), np.zeros((4, 4, 3)), None),
        (np.zeros((4, 4, 3)), np.zeros((4, 4, 3)), np.ones((3, 4))),
        (np.zeros((4, 4, 3)), np.zeros((4, 4, 3)), np.zeros((4, 4))),
    ],
)
def test_detail_metrics_reject_invalid_inputs(
    prediction: np.ndarray,
    target: np.ndarray,
    valid_mask: np.ndarray | None,
) -> None:
    with pytest.raises(ValueError):
        detail_metrics(prediction, target, valid_mask)


def test_detail_directory_uses_dataset_names_and_masks_invalid_pixels(
    tmp_path,
) -> None:
    sample = _sample()
    rendered = sample.image.copy()
    rendered[0] = 255 - rendered[0]
    Image.fromarray(rendered).save(tmp_path / "validation.png")

    report = evaluate_detail_directory(_Dataset((sample,)), tmp_path)

    assert report["scene_id"] == "scene"
    assert report["image_count"] == 1
    assert set(report["images"]) == {"validation.JPG"}
    assert report["hf_l1_mean"] == pytest.approx(0.0)
    assert report["missing_edge_mean"] == pytest.approx(0.0)
    assert report["spurious_edge_mean"] == pytest.approx(0.0)
    assert report["symmetric_edge_distance_mean"] == pytest.approx(0.0)


def test_detail_directory_rejects_missing_extra_and_colliding_names(tmp_path) -> None:
    dataset = _Dataset((_sample(),))
    with pytest.raises(FileNotFoundError, match="missing validation render"):
        evaluate_detail_directory(dataset, tmp_path)

    Image.fromarray(_sample().image).save(tmp_path / "validation.png")
    Image.fromarray(_sample().image).save(tmp_path / "extra.png")
    with pytest.raises(ValueError, match="filenames mismatch"):
        evaluate_detail_directory(dataset, tmp_path)

    collision = _Dataset((_sample("same.JPG"), _sample("same.png")))
    with pytest.raises(ValueError, match="collide"):
        evaluate_detail_directory(collision, tmp_path)


def test_detail_directory_rejects_non_rgb_and_wrong_resolution(tmp_path) -> None:
    dataset = _Dataset((_sample(),))
    Image.fromarray(np.zeros((32, 32), dtype=np.uint8)).save(
        tmp_path / "validation.png"
    )
    with pytest.raises(ValueError, match="RGB"):
        evaluate_detail_directory(dataset, tmp_path)

    Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(
        tmp_path / "validation.png"
    )
    with pytest.raises(ValueError, match="resolution"):
        evaluate_detail_directory(dataset, tmp_path)
