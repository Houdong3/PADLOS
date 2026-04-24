import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose

try:
    from .depth_anything_v2.pdfusedpt import PDDPT
    from .depth_anything_v2.util.transform import NormalizeImage, PrepareForNet, Resize
except ImportError:
    from depth_anything_v2.pdfusedpt import PDDPT
    from depth_anything_v2.util.transform import NormalizeImage, PrepareForNet, Resize


ROOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT_DIR.parent
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "pgm.pth"
DEFAULT_SAVE_DIR = PROJECT_ROOT / "outputs" / "masks"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run PGM binary-mask inference and benchmark the forward speed."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Path to the PGM checkpoint.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing input images.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=DEFAULT_SAVE_DIR,
        help="Directory used to save predicted binary masks.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Inference device.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=518,
        help="Square inference size used by the released PGM checkpoint.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Threshold applied after sigmoid to get binary masks.",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="vits",
        choices=["vits", "vitb", "vitl", "vitg"],
        help="Backbone encoder type.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=1,
        help="Number of segmentation output channels.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup images before timing.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only process the first N images. Use 0 to process all images.",
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


def list_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image format: {input_path}")
        return [input_path]

    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    images = sorted(
        path for path in input_path.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise FileNotFoundError(f"No images found under: {input_path}")
    return images


def build_transform(input_size: int):
    return Compose(
        [
            Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method="lower_bound",
                image_interpolation_method=cv2.INTER_LINEAR,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ]
    )


def load_model(checkpoint: Path, device: torch.device, encoder: str, num_classes: int) -> torch.nn.Module:
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    model = PDDPT(num_classes=num_classes, encoder=encoder)
    state_dict = torch.load(checkpoint, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path: Path, transform, device: torch.device):
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Failed to read image: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    orig_size = image_rgb.shape[:2]
    image = image_rgb.astype(np.float32) / 255.0
    image = transform({"image": image})["image"]
    image_tensor = torch.from_numpy(image).unsqueeze(0).to(device).float()
    return image_tensor, orig_size


def postprocess(pred: torch.Tensor, orig_size: tuple[int, int], threshold: float) -> np.ndarray:
    prob = torch.sigmoid(pred)
    binary = (prob > threshold).float()
    binary = F.interpolate(binary, size=orig_size, mode="nearest").squeeze(0).squeeze(0)
    return (binary.cpu().numpy().astype(np.uint8)) * 255


def synchronize_if_needed(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def summarize_times(times_ms: list[float], label: str):
    mean_ms = float(np.mean(times_ms))
    median_ms = float(np.median(times_ms))
    min_ms = float(np.min(times_ms))
    max_ms = float(np.max(times_ms))
    fps = 1000.0 / mean_ms if mean_ms > 0 else float("inf")
    print(
        f"{label}: mean={mean_ms:.2f} ms, median={median_ms:.2f} ms, "
        f"min={min_ms:.2f} ms, max={max_ms:.2f} ms, FPS={fps:.2f}"
    )


def save_mask(save_dir: Path, image_path: Path, mask: np.ndarray):
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{image_path.stem}_mask.png"
    cv2.imwrite(str(save_path), mask)


def run_mask_inference(
    input_path: Path,
    checkpoint: Path = DEFAULT_CHECKPOINT,
    save_dir: Path = DEFAULT_SAVE_DIR,
    device_arg: str = "auto",
    input_size: int = 518,
    threshold: float = 0.05,
    encoder: str = "vits",
    num_classes: int = 1,
    warmup: int = 5,
    limit: int = 0,
    log_output: bool = True,
    benchmark: bool = True,
):
    device = resolve_device(device_arg)
    image_paths = list_images(input_path)
    if limit > 0:
        image_paths = image_paths[:limit]

    transform = build_transform(input_size)

    if log_output:
        print(f"Device: {device}")
        print(f"Checkpoint: {checkpoint}")
        print(f"Input images: {len(image_paths)} from {input_path}")
        print(f"Saving masks to: {save_dir}")
        print(f"Threshold: {threshold}")

    model = load_model(checkpoint, device, encoder, num_classes)

    warmup_count = min(warmup, len(image_paths))
    with torch.inference_mode():
        for image_path in image_paths[:warmup_count]:
            image_tensor, orig_size = preprocess_image(image_path, transform, device)
            pred = model(image_tensor)
            _ = postprocess(pred, orig_size, threshold)
    synchronize_if_needed(device)

    if benchmark:
        cached_inputs = []
        with torch.inference_mode():
            for image_path in image_paths:
                image_tensor, orig_size = preprocess_image(image_path, transform, device)
                cached_inputs.append((image_path, image_tensor, orig_size))

        forward_times_ms = []
        with torch.inference_mode():
            for idx, (_, image_tensor, _) in enumerate(cached_inputs, start=1):
                synchronize_if_needed(device)
                start = time.perf_counter()
                _ = model(image_tensor)
                synchronize_if_needed(device)
                forward_times_ms.append((time.perf_counter() - start) * 1000.0)
                if log_output and (idx % 10 == 0 or idx == len(cached_inputs)):
                    print(f"Forward benchmark progress: {idx}/{len(cached_inputs)}")

        end_to_end_times_ms = []

    with torch.inference_mode():
        for idx, image_path in enumerate(image_paths, start=1):
            if benchmark:
                synchronize_if_needed(device)
                start = time.perf_counter()
            image_tensor, orig_size = preprocess_image(image_path, transform, device)
            pred = model(image_tensor)
            mask = postprocess(pred, orig_size, threshold)
            if benchmark:
                synchronize_if_needed(device)
                end_to_end_times_ms.append((time.perf_counter() - start) * 1000.0)

            save_mask(save_dir, image_path, mask)
            if benchmark and log_output and (idx % 10 == 0 or idx == len(image_paths)):
                print(f"End-to-end benchmark progress: {idx}/{len(image_paths)}")

    if benchmark and log_output:
        print()
        summarize_times(forward_times_ms, "Forward-only")
        summarize_times(end_to_end_times_ms, "End-to-end (read + preprocess + forward + binarize)")


def main():
    args = parse_args()
    run_mask_inference(
        input_path=args.input_dir,
        checkpoint=args.checkpoint,
        save_dir=args.save_dir,
        device_arg=args.device,
        input_size=args.input_size,
        threshold=args.threshold,
        encoder=args.encoder,
        num_classes=args.num_classes,
        warmup=args.warmup,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
