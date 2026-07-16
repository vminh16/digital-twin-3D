from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import torch

from bts_nvs.cameras.distortion import CameraDistortion
from bts_nvs.cameras.intrinsics import CameraIntrinsics
from bts_nvs.data.dataset import CameraSample


@dataclass(frozen=True)
class DeviceCameraSample:
    rgb: torch.Tensor
    mask: torch.Tensor
    world_to_camera: torch.Tensor
    intrinsics: CameraIntrinsics
    distortion: CameraDistortion
    image_name: str


@dataclass
class _PinnedSlot:
    image: torch.Tensor
    mask: torch.Tensor
    world_to_camera: torch.Tensor
    transfer_done: torch.cuda.Event | None = None


class TrainingInputPipeline:
    """Copies one camera sample to the training device using at most two pins."""

    def __init__(self, device: torch.device, *, pinned_transfer: bool = False) -> None:
        self.device = torch.device(device)
        self._pinned = bool(pinned_transfer and self.device.type == "cuda")
        self.slots: list[_PinnedSlot] = []
        self._next_slot = 0

    @property
    def slot_count(self) -> int:
        return 2 if self._pinned else 0

    def _new_slot(self, sample: CameraSample) -> _PinnedSlot:
        return _PinnedSlot(
            image=torch.empty(sample.image.shape, dtype=torch.uint8, pin_memory=True),
            mask=torch.empty(sample.valid_mask.shape, dtype=torch.bool, pin_memory=True),
            world_to_camera=torch.empty((4, 4), dtype=torch.float64, pin_memory=True),
        )

    def transfer(
        self,
        sample: CameraSample,
        *,
        world_to_camera: np.ndarray | None = None,
    ) -> DeviceCameraSample:
        pose_array = (
            sample.world_to_camera
            if world_to_camera is None
            else np.asarray(world_to_camera, dtype=np.float64)
        )
        if pose_array.shape != (4, 4):
            raise ValueError("world_to_camera must have shape (4, 4)")
        if not self._pinned:
            image = torch.from_numpy(sample.image).to(self.device)
            mask = torch.from_numpy(sample.valid_mask).to(self.device)
            pose = torch.from_numpy(pose_array).to(self.device)
        else:
            slot_index = self._next_slot
            self._next_slot = (self._next_slot + 1) % 2
            if len(self.slots) <= slot_index:
                self.slots.append(self._new_slot(sample))
            slot = self.slots[slot_index]
            if slot.image.shape != sample.image.shape:
                if slot.transfer_done is not None:
                    slot.transfer_done.synchronize()
                slot = self._new_slot(sample)
                self.slots[slot_index] = slot
            elif slot.transfer_done is not None:
                slot.transfer_done.synchronize()

            slot.image.copy_(torch.from_numpy(sample.image))
            slot.mask.copy_(torch.from_numpy(sample.valid_mask))
            slot.world_to_camera.copy_(torch.from_numpy(pose_array))
            image = slot.image.to(self.device, non_blocking=True)
            mask = slot.mask.to(self.device, non_blocking=True)
            pose = slot.world_to_camera.to(self.device, non_blocking=True)
            slot.transfer_done = torch.cuda.Event()
            slot.transfer_done.record(torch.cuda.current_stream(self.device))

        return DeviceCameraSample(
            rgb=image.to(dtype=torch.float32).div_(255.0),
            mask=mask,
            world_to_camera=pose,
            intrinsics=sample.intrinsics,
            distortion=sample.distortion,
            image_name=sample.image_name,
        )
