import argparse
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import skeletonize

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.pyplot as plt

try:
    from .core import WireSegmenter, build_neighbors_for_points
except ImportError:
    from core import WireSegmenter, build_neighbors_for_points


ROOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT_DIR.parent
DEFAULT_MASK_DIR = PROJECT_ROOT / "outputs" / "masks"
DEFAULT_DEPTH_DIR = PROJECT_ROOT / "outputs" / "depth"
DEFAULT_SAVE_DIR = PROJECT_ROOT / "outputs" / "instances"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class LeeCCNumbaNeighborWireSegmenter(WireSegmenter):
    """Released ACM variant with component-wise Lee skeletonization."""

    @staticmethod
    def _componentwise_lee_skeleton(mask: np.ndarray, pad: int = 1) -> np.ndarray:
        mask_u8 = (mask > 0).astype(np.uint8)
        if mask_u8.sum() == 0:
            return np.zeros_like(mask_u8, dtype=np.uint8)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
        skeleton = np.zeros_like(mask_u8, dtype=np.uint8)

        for label_id in range(1, num_labels):
            x, y, w, h, area = stats[label_id]
            if area <= 0:
                continue

            roi_mask = (labels[y : y + h, x : x + w] == label_id).astype(np.uint8)
            roi_padded = np.pad(roi_mask, pad, mode="constant", constant_values=0)
            roi_skeleton = skeletonize(roi_padded, method="lee").astype(np.uint8)
            roi_skeleton = roi_skeleton[pad:-pad, pad:-pad]
            skeleton[y : y + h, x : x + w] = np.maximum(skeleton[y : y + h, x : x + w], roi_skeleton)

        return skeleton

    def _get_neighbors_map(self):
        sk = (self.skeleton > 0).astype(np.uint8)
        ys, xs = np.nonzero(sk)
        neighbors_dict = defaultdict(list)

        if ys.size == 0:
            return neighbors_dict

        h, w = sk.shape
        neigh_y, neigh_x, counts = build_neighbors_for_points(
            ys.astype(np.int32),
            xs.astype(np.int32),
            sk,
            h,
            w,
        )

        for idx in range(ys.shape[0]):
            y = int(ys[idx])
            x = int(xs[idx])
            cnt = int(counts[idx])
            if cnt <= 0:
                continue

            key = (y, x)
            for j in range(cnt):
                neighbors_dict[key].append((int(neigh_y[idx, j]), int(neigh_x[idx, j])))

        return neighbors_dict

    def preprocess(self):
        self.mask = self._process_mask(self.mask)
        self.skeleton = self._componentwise_lee_skeleton(self.mask)
        self.distmap = cv2.distanceTransform(self.mask, cv2.DIST_L2, 3)
        self.mean_dist = self.distmap[self.skeleton.astype(bool)].mean() if np.any(self.skeleton) else 0
        self.neighbors = self._get_neighbors_num()
        self.neighbors_map = self._get_neighbors_map()

        endpoints = self._get_ends(self.skeleton, self.neighbors)
        junctions = self._get_intersections(self.skeleton, self.neighbors)

        self.pruned_junctions = self._prune_split_ends(self.skeleton, endpoints, junctions)
        self.original_junctions = set((y, x) for y, x in self.pruned_junctions)
        (
            self.intersections,
            self.junction_to_intersection,
            self.intersection_to_neighbors,
        ) = self._cluster_intersections(self.pruned_junctions)
        self.original_junctions = set((y, x) for y, x in self.pruned_junctions)
        self.neighbors = self._get_neighbors_num()
        self.endpoints = self._get_ends(self.skeleton, self.neighbors)


def warmup_numba():
    sk = np.zeros((8, 8), dtype=np.uint8)
    sk[2:6, 3] = 1
    ys, xs = np.nonzero(sk)
    build_neighbors_for_points(
        ys.astype(np.int32),
        xs.astype(np.int32),
        sk,
        sk.shape[0],
        sk.shape[1],
    )


def initialize_endpoint_nodes(segmenter: WireSegmenter) -> None:
    segmenter.endpoint_nodes = [n for n in segmenter.graph.nodes if n.startswith("EP")]


def reconstruct_instances(segmenter: WireSegmenter, optimal_paths: list[dict]) -> np.ndarray:
    h, w = segmenter.distmap.shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    num_paths = len(optimal_paths)
    colors = [tuple(int(c * 255) for c in color[:3]) for color in plt.cm.tab20(np.linspace(0, 1, num_paths))]

    crossed_edges = {}
    for p_id, nodes in enumerate([p["node_path"] for p in optimal_paths]):
        for i in range(len(nodes) - 1):
            key = tuple(sorted((nodes[i], nodes[i + 1])))
            crossed_edges.setdefault(key, []).append(p_id)

    all_segments = []

    for path_idx, path_info in enumerate(optimal_paths):
        node_path = path_info["node_path"]
        avg_width = max(10, int(round(path_info["avg_width"] * 2)))
        color = colors[path_idx]

        if len(node_path) == 2:
            edge_data = segmenter.graph.get_edge_data(node_path[0], node_path[1])
            pts = [(p[0], p[1]) for p in edge_data["path"]]
            pixel_path_segments = [np.array(pts, dtype=np.int32)]
        else:
            pixel_path_segments = segmenter.get_midpoint_path(node_path)

        full_pixel_path = np.vstack(pixel_path_segments)
        curve_points = segmenter.fit_smooth_curve(full_pixel_path)
        curve_xy = curve_points[:, [1, 0]]

        cross_pairs = []
        for e_idx, (u, v) in enumerate(zip(node_path[:-1], node_path[1:])):
            key = tuple(sorted((u, v)))
            if len(crossed_edges.get(key, [])) > 1:
                seg = pixel_path_segments[e_idx]
                mid_pt = seg[len(seg) // 2]

                distances = np.linalg.norm(curve_xy - mid_pt[::-1], axis=1)
                mid_idx = int(np.argmin(distances))

                prev_pt = seg[0] if e_idx == 0 else pixel_path_segments[e_idx - 1][-1]
                next_pt = seg[-1] if e_idx == len(node_path) - 2 else pixel_path_segments[e_idx + 1][0]
                prev_depth = float(segmenter.depth[int(prev_pt[0]), int(prev_pt[1])])
                next_depth = float(segmenter.depth[int(next_pt[0]), int(next_pt[1])])
                avg_depth = (prev_depth + next_depth) / 2.0
                cross_pairs.append((mid_idx, avg_depth))

        cross_pairs.sort(key=lambda t: t[0])
        cross_indices = [t[0] for t in cross_pairs]
        cross_depths = [t[1] for t in cross_pairs]

        segment_points = [0]
        for i in range(len(cross_indices) - 1):
            mid_split = (cross_indices[i] + cross_indices[i + 1]) // 2
            segment_points.append(mid_split)
        segment_points += [len(curve_xy)]
        segment_points = sorted(list(set(segment_points)))

        def depth_at_xy(pt_xy):
            x = int(np.clip(pt_xy[0], 0, w - 1))
            y = int(np.clip(pt_xy[1], 0, h - 1))
            return float(segmenter.depth[y, x])

        for seg_i in range(len(segment_points) - 1):
            s_start = segment_points[seg_i]
            s_end = segment_points[seg_i + 1]
            if s_start >= s_end:
                continue
            segment_curve = curve_xy[s_start:s_end]

            if len(cross_depths) > 0 and seg_i < len(cross_depths):
                segment_depth = float(cross_depths[seg_i])
            else:
                mid_pt = segment_curve[len(segment_curve) // 2]
                segment_depth = depth_at_xy(mid_pt)

            all_segments.append(
                {
                    "path_idx": path_idx,
                    "curve": segment_curve,
                    "depth": segment_depth,
                    "width": avg_width,
                    "color": color,
                }
            )

    all_segments.sort(key=lambda x: x["depth"])

    canvas_step = np.zeros_like(canvas)
    for seg in all_segments:
        pts = seg["curve"].astype(np.int32)
        for j in range(len(pts) - 1):
            cv2.line(canvas_step, tuple(pts[j]), tuple(pts[j + 1]), seg["color"], int(seg["width"]))

    return canvas_step[1:-1, 1:-1, :]


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


def resolve_mask_path(image_path: Path, mask_dir: Path) -> Path | None:
    candidates = [
        mask_dir / f"{image_path.stem}_mask.png",
        mask_dir / f"{image_path.stem}.png",
        mask_dir / image_path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def resolve_depth_path(image_path: Path, depth_dir: Path) -> Path | None:
    candidates = [
        depth_dir / f"{image_path.stem}_depth_gray.png",
        depth_dir / f"{image_path.stem}.png",
        depth_dir / f"{image_path.stem}_depth.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def save_instance_output(save_dir: Path, image_path: Path, output: np.ndarray):
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{image_path.stem}_instances.png"
    cv2.imwrite(str(save_path), output)


def run_instance_extraction(
    img_path: Path,
    mask_dir: Path = DEFAULT_MASK_DIR,
    depth_dir: Path = DEFAULT_DEPTH_DIR,
    save_dir: Path = DEFAULT_SAVE_DIR,
    limit: int = 0,
):
    image_paths = list_images(img_path)
    if limit > 0:
        image_paths = image_paths[:limit]

    warmup_numba()

    print(f"Input images: {len(image_paths)} from {img_path}")
    print(f"Using masks from: {mask_dir}")
    print(f"Using depth from: {depth_dir}")
    print(f"Saving instances to: {save_dir}")

    for image_path in image_paths:
        mask_path = resolve_mask_path(image_path, mask_dir)
        depth_path = resolve_depth_path(image_path, depth_dir)

        if mask_path is None or depth_path is None:
            continue

        try:
            segmenter = LeeCCNumbaNeighborWireSegmenter(str(image_path), str(mask_path), str(depth_path))
            segmenter.preprocess()
            segmenter.build_graph()
            initialize_endpoint_nodes(segmenter)
            _, optimal_paths = segmenter.optimize_wires_global(
                w_bend=0.4,
                w_color=0.3,
                w_width=0.1,
                w_len=0.1,
                k_paths=100,
            )
            final_output = reconstruct_instances(segmenter, optimal_paths)

            save_instance_output(save_dir, image_path, final_output)
        except Exception:
            continue


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Run ACM with component-wise Lee skeletonization and Numba neighbor mapping."
    )
    parser.add_argument("--img-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, default=DEFAULT_MASK_DIR)
    parser.add_argument("--depth-dir", type=Path, default=DEFAULT_DEPTH_DIR)
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N images. 0 means all.")
    return parser


if __name__ == "__main__":
    args = build_argparser().parse_args()
    run_instance_extraction(
        img_path=args.img_dir,
        mask_dir=args.mask_dir,
        depth_dir=args.depth_dir,
        save_dir=args.save_dir,
        limit=args.limit,
    )
