import argparse
from pathlib import Path

from PDSegmentor.benchmark_pgm import list_images, run_mask_inference
from PDSegmentor.depthinfer import run_depth_inference


ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT_DIR / "outputs"
DEFAULT_MASK_DIR = OUTPUT_ROOT / "masks"
DEFAULT_DEPTH_DIR = OUTPUT_ROOT / "depth"
DEFAULT_PGM_CHECKPOINT = ROOT_DIR / "PDSegmentor" / "checkpoints" / "pgm.pth"
DEFAULT_DEPTH_CHECKPOINT = ROOT_DIR / "PDSegmentor" / "checkpoints" / "depth_anything_v2_vits.pth"


def parse_args():
    parser = argparse.ArgumentParser(description="Unified PADLOS PGM inference entrypoint.")
    parser.add_argument(
        "--img-path",
        type=Path,
        required=True,
        help="A single image path or a directory containing images.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="both",
        choices=["both", "mask", "depth"],
        help="Which PGM output(s) to generate.",
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
        help="Square inference size used by the released checkpoints.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only process the first N images for mask inference. Use 0 to process all images.",
    )
    parser.add_argument(
        "--pgm-checkpoint",
        type=Path,
        default=DEFAULT_PGM_CHECKPOINT,
        help="Path to the released PGM checkpoint.",
    )
    parser.add_argument(
        "--depth-checkpoint",
        type=Path,
        default=DEFAULT_DEPTH_CHECKPOINT,
        help="Path to the released depth checkpoint.",
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=DEFAULT_MASK_DIR,
        help="Directory used to save predicted binary masks.",
    )
    parser.add_argument(
        "--depth-dir",
        type=Path,
        default=DEFAULT_DEPTH_DIR,
        help="Directory used to save depth predictions.",
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
        help="Backbone encoder type for both released checkpoints.",
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
        help="Number of warmup images before mask timing.",
    )
    parser.add_argument(
        "--grayscale",
        action="store_true",
        help="Save depth previews using a grayscale colormap instead of Spectral_r.",
    )
    parser.add_argument(
        "--save-npy",
        action="store_true",
        help="Save raw float depth maps as .npy files.",
    )
    parser.add_argument(
        "--show-depth",
        action="store_true",
        help="Display each predicted depth map in a window.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    image_paths = list_images(args.img_path)
    if args.limit > 0:
        image_paths = image_paths[: args.limit]

    print(f"Input images: {len(image_paths)} from {args.img_path}")
    if args.task in {"both", "mask"}:
        print(f"Saving masks to: {args.mask_dir}")
    if args.task in {"both", "depth"}:
        print(f"Saving depth outputs to: {args.depth_dir}")

    if args.task in {"both", "mask"}:
        run_mask_inference(
            input_path=args.img_path,
            checkpoint=args.pgm_checkpoint,
            save_dir=args.mask_dir,
            device_arg=args.device,
            input_size=args.input_size,
            threshold=args.threshold,
            encoder=args.encoder,
            num_classes=args.num_classes,
            warmup=args.warmup,
            limit=args.limit,
            log_output=False,
            benchmark=False,
        )

    if args.task in {"both", "depth"}:
        run_depth_inference(
            img_path=args.img_path,
            checkpoint=args.depth_checkpoint,
            outdir=args.depth_dir,
            encoder=args.encoder,
            device_arg=args.device,
            input_size=args.input_size,
            grayscale=args.grayscale,
            show=args.show_depth,
            save_npy=args.save_npy,
            log_output=False,
        )


if __name__ == "__main__":
    main()
