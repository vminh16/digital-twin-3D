import os
import sys

try:
    import torch
except ImportError as e:
    print("FAILED: PyTorch is not installed in the current environment.")
    print("Please install PyTorch matching your CUDA version first.")
    sys.exit(1)

# Add src to the path so we can import bts_nvs
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

def test_package_import():
    print("=== Test 1: Package Import ===")
    try:
        import bts_nvs
        print("Success: bts_nvs imported successfully!")
    except ImportError as e:
        print(f"FAILED: Cannot import bts_nvs. Error: {e}")
        sys.exit(1)

def test_gpu_cuda():
    print("\n=== Test 2: GPU and CUDA Status ===")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    if cuda_available:
        device_count = torch.cuda.device_count()
        print(f"CUDA Device Count: {device_count}")
        for i in range(device_count):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("WARNING: CUDA is not available. Running on CPU.")

def test_gsplat_smoke():
    print("\n=== Test 3: gsplat Forward & Backward Smoke Test ===")
    try:
        import gsplat
        print(f"gsplat version: {gsplat.__version__}")
    except ImportError as e:
        print("WARNING: gsplat is not installed in the current environment.")
        print("Skipping gsplat smoke test. (This is expected if run locally on CPU-only machines.)")
        return

    if not torch.cuda.is_available():
        print("Skipping gsplat forward/backward smoke test because CUDA is not available.")
        return

    try:
        from bts_nvs.cameras.intrinsics import CameraIntrinsics
        from bts_nvs.models.gaussian_parameters import GaussianParameters
        from bts_nvs.rendering.gsplat_renderer import render_gaussians

        # Setup small dummy Gaussian properties
        N = 10
        means = torch.randn(N, 3, dtype=torch.float32, device="cuda")
        means[:, 2] += 5.0
        # Quaternions [w, x, y, z]
        quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * N, dtype=torch.float32, device="cuda")
        scales = torch.full((N, 3), -3.0, dtype=torch.float32, device="cuda")
        opacities = torch.zeros(N, dtype=torch.float32, device="cuda")
        sh0 = torch.zeros((N, 1, 3), dtype=torch.float32, device="cuda")
        shN = torch.zeros((N, 15, 3), dtype=torch.float32, device="cuda")
        gaussians = GaussianParameters(means, scales, quats, opacities, sh0, shN)

        # Setup dummy Camera Extrinsics (World-to-Camera, 4x4 matrix)
        viewmat = torch.eye(4, dtype=torch.float32, device="cuda")

        # Setup dummy Camera Intrinsics (3x3 matrix)
        # fx=500, fy=500, cx=256, cy=256
        width = 512
        height = 512
        intrinsics = CameraIntrinsics(width, height, 500.0, 500.0, 256.0, 256.0)

        print("Running gsplat.rasterization forward pass...")
        # Render colors [1, H, W, 3] or [H, W, 3], alphas [1, H, W, 1] or [H, W, 1]
        # depending on version of gsplat. Let's run and catch outputs.
        result = render_gaussians(
            gaussians=gaussians,
            viewmat=viewmat,
            intrinsics=intrinsics,
            active_sh_degree=0,
        )
        render_colors = result.rgb
        print(f"Forward pass completed. Render output shape: {render_colors.shape}")

        print("Running backward pass...")
        loss = render_colors.sum()
        loss.backward()
        print("Backward pass completed successfully!")
        print(f"Gradients computed: means.grad is not None: {gaussians.means.grad is not None}")

    except Exception as e:
        print(f"FAILED: gsplat smoke test encountered an error:\n{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_package_import()
    test_gpu_cuda()
    test_gsplat_smoke()
    print("\n=== Smoke Tests Finished ===")
