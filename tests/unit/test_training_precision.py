from __future__ import annotations

from contextlib import nullcontext

import pytest
import torch


class _FakeScaler:
    def __init__(self, scale: float = 8.0) -> None:
        self.value = scale
        self.unscaled: list[torch.optim.Optimizer] = []
        self.stepped: list[torch.optim.Optimizer] = []
        self.update_calls = 0
        self.loaded_state = None

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss * self.value

    def get_scale(self) -> float:
        return self.value

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    parameter.grad.div_(self.value)
        self.unscaled.append(optimizer)

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        optimizer.step()
        self.stepped.append(optimizer)

    def update(self) -> None:
        self.update_calls += 1

    def state_dict(self) -> dict:
        return {"scale": self.value}

    def load_state_dict(self, state: dict) -> None:
        self.loaded_state = state


def test_fp32_precision_preserves_ordinary_backward_and_step() -> None:
    from bts_nvs.training.precision import TrainingPrecision

    parameter = torch.nn.Parameter(torch.tensor([2.0]))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    projected = parameter * 3.0
    projected.retain_grad()
    controller = TrainingPrecision("fp32", torch.device("cpu"))

    assert isinstance(controller.autocast(), nullcontext)
    assert controller.backward_and_unscale(
        projected.sum(), {"p": optimizer}, projected
    ) == 1.0
    assert torch.equal(projected.grad, torch.ones_like(projected))
    controller.step({"p": optimizer})

    assert torch.allclose(parameter, torch.tensor([1.7]))


def test_amp_unscales_all_optimizers_and_non_leaf_projected_gradient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bts_nvs.training.precision as precision_module
    from bts_nvs.training.precision import TrainingPrecision

    fake = _FakeScaler(scale=8.0)
    monkeypatch.setattr(
        precision_module.torch.amp,
        "GradScaler",
        lambda *args, **kwargs: fake,
    )
    first = torch.nn.Parameter(torch.tensor([2.0]))
    second = torch.nn.Parameter(torch.tensor([3.0]))
    optimizers = {
        "first": torch.optim.SGD([first], lr=0.1),
        "second": torch.optim.SGD([second], lr=0.1),
    }
    projected = first * second
    projected.retain_grad()
    controller = TrainingPrecision("amp-fp16", torch.device("cuda"))

    scale = controller.backward_and_unscale(projected.sum(), optimizers, projected)

    assert scale == 8.0
    assert fake.unscaled == list(optimizers.values())
    assert torch.equal(projected.grad, torch.ones_like(projected))
    assert torch.equal(first.grad, torch.tensor([3.0]))
    assert torch.equal(second.grad, torch.tensor([2.0]))

    controller.step(optimizers)
    assert fake.stepped == list(optimizers.values())
    assert fake.update_calls == 1


def test_amp_rejects_cpu_and_missing_projected_gradient() -> None:
    from bts_nvs.training.precision import TrainingPrecision

    with pytest.raises(ValueError, match="requires CUDA"):
        TrainingPrecision("amp-fp16", torch.device("cpu"))

    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    projected = parameter.detach().clone()
    controller = TrainingPrecision.__new__(TrainingPrecision)
    controller.mode = "amp-fp16"
    controller.scaler = _FakeScaler()
    with pytest.raises(RuntimeError, match="projected means gradient"):
        controller.backward_and_unscale(parameter.sum(), {"p": optimizer}, projected)


def test_precision_state_round_trip_is_required_only_for_amp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bts_nvs.training.precision as precision_module
    from bts_nvs.training.precision import TrainingPrecision

    fp32 = TrainingPrecision("fp32", torch.device("cpu"))
    assert fp32.state_dict() == {}
    fp32.load_state_dict({})

    fake = _FakeScaler(scale=16.0)
    monkeypatch.setattr(
        precision_module.torch.amp,
        "GradScaler",
        lambda *args, **kwargs: fake,
    )
    amp = TrainingPrecision("amp-fp16", torch.device("cuda"))
    assert amp.state_dict() == {"scale": 16.0}
    amp.load_state_dict({"scale": 32.0})
    assert fake.loaded_state == {"scale": 32.0}
    with pytest.raises(ValueError, match="precision state"):
        amp.load_state_dict({})


def test_amp_skips_optimizer_without_a_gradient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bts_nvs.training.precision as precision_module
    from bts_nvs.training.precision import TrainingPrecision

    fake = _FakeScaler()
    monkeypatch.setattr(
        precision_module.torch.amp,
        "GradScaler",
        lambda *args, **kwargs: fake,
    )
    active = torch.nn.Parameter(torch.tensor([2.0]))
    inactive = torch.nn.Parameter(torch.tensor([3.0]))
    optimizers = {
        "active": torch.optim.SGD([active], lr=0.1),
        "inactive": torch.optim.SGD([inactive], lr=0.1),
    }
    projected = active * 2.0
    projected.retain_grad()
    controller = TrainingPrecision("amp-fp16", torch.device("cuda"))

    controller.backward_and_unscale(projected.sum(), optimizers, projected)
    controller.step(optimizers)

    assert fake.unscaled == [optimizers["active"]]
    assert fake.stepped == [optimizers["active"]]
    assert torch.equal(inactive, torch.tensor([3.0]))
