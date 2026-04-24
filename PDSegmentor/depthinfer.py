import argparse
import time
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from .depth_anything_v2.dpt import DepthAnythingV2
except ImportError:
    from depth_anything_v2.dpt import DepthAnythingV2


ROOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT_DIR.parent
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "depth_anything_v2_vits.pth"
DEFAULT_OUTDIR = PROJECT_ROOT / "outputs" / "depth"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(description="Run the released Depth Anything V2 depth inference.")
    parser.add_argument(
        "--img-path",
        type=Path,
        required=True,
        help="A single image path or a directory containing images.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Path to the depth checkpoint.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=518,
        help="Square inference size used by the released depth checkpoint.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help="Directory used to save depth predictions.",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="vits",
        choices=["vits", "vitb", "vitl", "vitg"],
        help="Backbone encoder type.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Inference device.",
    )
    parser.add_argument(
        "--grayscale",
        dest="grayscale",
        action="store_true",
        help="Save the preview using a grayscale colormap instead of Spectral_r.",
    )
    parser.add_argument(
        "--show",
        dest="show",
        action="store_true",
        help="Display each predicted depth map in a window.",
    )
    parser.add_argument(
        "--save-npy",
        dest="save_npy",
        action="store_true",
        help="Save the raw float depth map as a .npy file.",
    )
    return parser.parse_args()


def is_cuda_compatible() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        major, minor = torch.cuda.get_device_capability(0)
        arch = f"sm_{major}{minor}"
        arch_list = torch.cuda.get_arch_list()
        return not arch_list or arch in arch_list
    except Exception:
        return True


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        if not is_cuda_compatible():
            raise RuntimeError(
                "CUDA was requested, but the installed PyTorch build does not support this GPU architecture."
            )
        return torch.device("cuda")

    if device_arg == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        if is_cuda_compatible():
            return torch.device("cuda")
        print("Warning: CUDA is visible, but this PyTorch build does not support the current GPU. Falling back to CPU.")
    return torch.device("cpu")


def build_model(encoder: str, checkpoint: Path, device: torch.device) -> DepthAnythingV2:
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
    }

    model = DepthAnythingV2(**model_configs[encoder])
    state_dict = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()
    return model


def list_images(img_path: Path) -> list[Path]:
    if img_path.is_file():
        if img_path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image format: {img_path}")
        return [img_path]

    if not img_path.exists():
        raise FileNotFoundError(f"Input path not found: {img_path}")

    images = sorted(
        path for path in img_path.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise FileNotFoundError(f"No images found under: {img_path}")
    return images


def save_depth_outputs(
    outdir: Path,
    image_path: Path,
    depth: np.ndarray,
    grayscale: bool,
    save_npy: bool,
):
    outdir.mkdir(parents=True, exist_ok=True)

    depth_min = float(depth.min())
    depth_max = float(depth.max())
    depth_norm = (depth - depth_min) / (max(depth_max - depth_min, 1e-8))
    depth_uint8 = (depth_norm * 255).astype(np.uint8)

    cmap = matplotlib.colormaps.get_cmap("gray" if grayscale else "Spectral_r")
    color_map = (cmap(depth_norm)[:, :, :3] * 255).astype(np.uint8)

    gray_path = outdir / f"{image_path.stem}_depth_gray.png"
    color_path = outdir / f"{image_path.stem}_depth_color.png"
    cv2.imwrite(str(gray_path), depth_uint8)
    cv2.imwrite(str(color_path), cv2.cvtColor(color_map, cv2.COLOR_RGB2BGR))

    if save_npy:
        np.save(outdir / f"{image_path.stem}_depth.npy", depth.astype(np.float32))


def run_depth_inference(
    img_path: Path,
    checkpoint: Path = DEFAULT_CHECKPOINT,
    outdir: Path = DEFAULT_OUTDIR,
    encoder: str = "vits",
    device_arg: str = "auto",
    input_size: int = 518,
    grayscale: bool = False,
    show: bool = False,
    save_npy: bool = False,
    log_output: bool = True,
):
    device = resolve_device(device_arg)
    image_paths = list_images(img_path)

    if log_output:
        print(f"Device: {device}")
        print(f"Checkpoint: {checkpoint}")
        print(f"Input images: {len(image_paths)} from {img_path}")
        print(f"Saving depth outputs to: {outdir}")

    model = build_model(encoder, checkpoint, device)
    cmap = matplotlib.colormaps.get_cmap("gray" if grayscale else "Spectral_r")

    for idx, image_path in enumerate(image_paths, start=1):
        if log_output:
            print(f"Progress {idx}/{len(image_paths)}: {image_path}")
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")

        start = time.perf_counter()
        depth = model.infer_image(image, input_size)
        elapsed = time.perf_counter() - start
        if log_output:
            print(f"Inference time: {elapsed:.2f}s | Depth range: {depth.min():.3f} ~ {depth.max():.3f}")

        save_depth_outputs(outdir, image_path, depth, grayscale, save_npy)

        if show:
            plt.imshow(depth, cmap=cmap)
            plt.title("Predicted Depth")
            plt.axis("off")
            plt.show()


def main():
    args = parse_args()
    run_depth_inference(
        img_path=args.img_path,
        checkpoint=args.checkpoint,
        outdir=args.outdir,
        encoder=args.encoder,
        device_arg=args.device,
        input_size=args.input_size,
        grayscale=args.grayscale,
        show=args.show,
        save_npy=args.save_npy,
    )


if __name__ == "__main__":
    main()
