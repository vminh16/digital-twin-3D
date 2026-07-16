import numpy as np
import pytest
import torch

from bts_nvs.cameras.distortion import CameraDistortion
from bts_nvs.cameras.intrinsics import CameraIntrinsics
from bts_nvs.data.dataset import CameraSample
from bts_nvs.training.input_pipeline import TrainingInputPipeline


def _sample(value: int = 128) -> CameraSample:
    return CameraSample(
        image=np.full((4, 5, 3), value, dtype=np.uint8),
        world_to_camera=np.eye(4, dtype=np.float64),
        intrinsics=CameraIntrinsics(5, 4, 3.0, 3.0, 2.5, 2.0),
        distortion=CameraDistortion("PINHOLE", ()),
        valid_mask=np.ones((4, 5), dtype=bool),
        image_name="a.png",
    )


def test_cpu_pipeline_preserves_values_and_converts_rgb_to_unit_float():
    pipeline = TrainingInputPipeline(torch.device("cpu"), pinned_transfer=True)

    output = pipeline.transfer(_sample())

    assert pipeline.slot_count == 0
    assert output.rgb.dtype == torch.float32
    torch.testing.assert_close(
        output.rgb,
        torch.full((4, 5, 3), 128.0 / 255.0, dtype=torch.float32),
    )
    assert output.mask.dtype == torch.bool
    assert output.world_to_camera.dtype == torch.float64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_pipeline_uses_exactly_two_pinned_slots():
    pipeline = TrainingInputPipeline(torch.device("cuda"), pinned_transfer=True)

    first = pipeline.transfer(_sample(10))
    second = pipeline.transfer(_sample(20))
    third = pipeline.transfer(_sample(30))
    torch.cuda.synchronize()

    assert pipeline.slot_count == 2
    assert all(slot.image.is_pinned() and slot.mask.is_pinned() for slot in pipeline.slots)
    assert first.rgb.device.type == "cuda"
    assert second.rgb.device.type == "cuda"
    assert third.rgb.device.type == "cuda"
