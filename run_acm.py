import argparse
from pathlib import Path

from instance_extractor.acm import run_instance_extraction


ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT_DIR / "outputs"
DEFAULT_MASK_DIR = OUTPUT_ROOT / "masks"
DEFAULT_DEPTH_DIR = OUTPUT_ROOT / "depth"
DEFAULT_INSTANCE_DIR = OUTPUT_ROOT / "instances"


def parse_args():
    parser = argparse.ArgumentParser(description="Unified PADLOS ACM instance extraction entrypoint.")
    parser.add_argument(
        "--img-path",
        type=Path,
        required=True,
        help="A single image path or a directory containing images.",
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=DEFAULT_MASK_DIR,
        help="Directory containing binary masks produced by PGM.",
    )
    parser.add_argument(
        "--depth-dir",
        type=Path,
        default=DEFAULT_DEPTH_DIR,
        help="Directory containing depth maps produced by PGM.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=DEFAULT_INSTANCE_DIR,
        help="Directory used to save ACM instance extraction outputs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only process the first N images. Use 0 to process all images.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_instance_extraction(
        img_path=args.img_path,
        mask_dir=args.mask_dir,
        depth_dir=args.depth_dir,
        save_dir=args.save_dir,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
