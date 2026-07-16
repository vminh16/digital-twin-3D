from pathlib import Path

import pytest
import torch

from bts_nvs.data.dataset import SceneDataset
from bts_nvs.data.manifest import (
    build_scene_manifest,
    load_scene_manifest,
    save_scene_manifest,
)
from bts_nvs.models.initialization import initialize_from_manifest
from bts_nvs.training.trainer import Trainer


pytestmark = [pytest.mark.real_data, pytest.mark.cuda]


def test_smoke_hcm0181_real_cuda(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("real HCM0181 smoke requires CUDA-enabled PyTorch")
    pytest.importorskip("gsplat")

    scene_root = Path("data/bts_scenes/HCM0181")
    if not scene_root.is_dir():
        pytest.skip(f"HCM0181 scene not found at {scene_root}")

    source_manifest = scene_root / "manifest" / "manifest.json"
    if source_manifest.is_file():
        manifest_json = source_manifest
        manifest = load_scene_manifest(manifest_json, scene_root)
    else:
        manifest = build_scene_manifest(scene_root)
        manifest_dir = tmp_path / "manifest"
        save_scene_manifest(manifest, manifest_dir)
        manifest_json = manifest_dir / "manifest.json"

    intrinsics = manifest.train_intrinsics[0]
    resize = (intrinsics.width // 4, intrinsics.height // 4)
    dataset = SceneDataset(
        manifest,
        scene_root,
        undistort=True,
        resize=resize,
    )
    gaussians = initialize_from_manifest(manifest)
    initial_sh0 = gaussians.sh0.detach().cpu().clone()
    config = {
        "scene_id": manifest.scene_id,
        "resize_factor": 4,
        "resize_width": resize[0],
        "resize_height": resize[1],
        "undistort": True,
        "seed": 0,
        "max_steps": 10,
        "prune_opa": 0.005,
        "grow_grad2d": 0.0002,
        "grow_scale3d": 0.01,
        "refine_start_step": 5,
        "refine_stop_step": 15,
        "refine_every": 2,
        "reset_every": 5,
        "lambda_dssim": 0.2,
        "data_range": 1.0,
    }
    output = tmp_path / "run"
    trainer = Trainer(
        gaussians=gaussians,
        dataset=dataset,
        output_dir=output,
        config=config,
        manifest_json_path=manifest_json,
        device=torch.device("cuda"),
    )
    initial = trainer.evaluate_train_view(
        0,
        render_path=output / "train_previews" / "initial.png",
        reference_path=output / "train_previews" / "reference.png",
    )
    trainer.train(stop_step=10, checkpoint_every=5)
    final = trainer.evaluate_train_view(
        0,
        render_path=output / "train_previews" / "final.png",
    )

    assert not torch.equal(trainer.gaussians.sh0.detach().cpu(), initial_sh0)
    assert initial["psnr_db"] is not None
    assert final["psnr_db"] is not None
    assert final["alpha_coverage"] > 0.0
    assert (output / "checkpoints" / "step_000000010.pt").is_file()
    assert (output / "summary.json").is_file()
