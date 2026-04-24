import cv2
import logging
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pulp
from networkx.drawing.nx_pydot import graphviz_layout
from scipy import ndimage
from scipy.ndimage import label, center_of_mass
from scipy.spatial import KDTree
from scipy.spatial import cKDTree
from scipy.interpolate import splprep, splev
from scipy.spatial.distance import pdist, squareform
from skimage.morphology import skeletonize
from skimage import draw
from skimage.draw import line
from skimage.color import lab2rgb
from skimage.color import rgb2lab
from skimage.color import deltaE_ciede2000
from sklearn.cluster import DBSCAN
from itertools import combinations, islice
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie2000
from collections import defaultdict
import time
import math
from adjustText import adjust_text
from numba import njit


if not hasattr(np, 'asscalar'):
    np.asscalar = lambda a: a.item()

visualize = False

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@njit
def build_neighbors_for_points(ys, xs, skel, h, w):
    n = ys.shape[0]
    neigh_y = -np.ones((n, 8), dtype=np.int32)
    neigh_x = -np.ones((n, 8), dtype=np.int32)
    cnt = np.zeros(n, dtype=np.int32)

    for idx in range(n):
        y = ys[idx]
        x = xs[idx]

        c = 0
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                if dy==0 and dx==0:
                    continue
                ny = y + dy
                nx = x + dx
                if 0 <= ny < h and 0 <= nx < w and skel[ny, nx]:
                    neigh_y[idx, c] = ny
                    neigh_x[idx, c] = nx
                    c += 1
        cnt[idx] = c

    return neigh_y, neigh_x, cnt


class WireSegmenter:
    def __init__(self, image_path, mask_path,depth_path,pad = 1):
        raw_image = cv2.imread(image_path)
        raw_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        raw_depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

        raw_image_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)

        self.image = self._pad_image(raw_image_rgb, pad,constant_values=255)
        self.mask = self._pad_image(raw_mask, pad,constant_values=0)
        self.depth = self._pad_image(raw_depth, pad,constant_values=0)
        self.skeleton = None
        self.distmap = None
        self.neighbors = None
        
        self.endpoints = []
        self.intersections = []
        self.edges= []

        self.fullpoint_adjacency = None
        self.endpoint_adjacency = None
        self.endpoint_connectivity = None

        self.graph = None

        self.max_reuse_count =2  
        self.mean_dist = 0

    def _pad_image(self, image, pad, constant_values=0):
        """Pad an image with a constant border."""

        if len(image.shape) == 2:
            return np.pad(image, pad, mode='constant', constant_values=constant_values)
        elif len(image.shape) == 3:
            return np.pad(
                image, ((pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=constant_values)
        else:
            raise ValueError(f"Unsupported image dimensions: {image.shape}")

    def _process_mask(self,original_mask):
        blurred = cv2.GaussianBlur(original_mask, (5, 5), 0)
        binary_mask = ((blurred / 255.0) > 0.5).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        return (closed > 0).astype(np.uint8)

    def _get_boundary_pixels(self, mask):

        inner_mask = mask[1:-1, 1:-1]
        inner_mask[1:-1, 1:-1] = 0
        inner_mask_bin = (inner_mask > 0).astype(np.uint8)
    
        kernel = np.ones((3, 3), dtype=np.uint8)
        neighbor_count = cv2.filter2D(inner_mask_bin, -1, kernel)
    
        neighbor_count_no_center = neighbor_count - inner_mask_bin

        boundary_mask = (inner_mask_bin == 1) & (neighbor_count_no_center <= 1)
        points = np.column_stack(np.nonzero(boundary_mask))
        
        return points.astype(np.int16)

    def _get_neighbors_num(self):
        sk = (self.skeleton > 0).astype(np.uint8)
        neighbors = (
            np.roll(sk,  1, axis=0) + np.roll(sk, -1, axis=0) +
            np.roll(sk,  1, axis=1) + np.roll(sk, -1, axis=1) +
            np.roll(np.roll(sk,  1, axis=0),  1, axis=1) +
            np.roll(np.roll(sk,  1, axis=0), -1, axis=1) +
            np.roll(np.roll(sk, -1, axis=0),  1, axis=1) +
            np.roll(np.roll(sk, -1, axis=0), -1, axis=1))

        return neighbors
    
    def _get_neighbors_map(self):
        sk = (self.skeleton > 0).astype(np.uint8)
        y_idxs, x_idxs = np.nonzero(sk)
        neighbors_dict = defaultdict(list)
        h, w = sk.shape

        shifts = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),           (0, 1),
                  (1, -1),  (1, 0),  (1, 1)]
        
        for dy, dx in shifts:
            shifted = np.roll(sk, (-dy, -dx), axis=(0, 1))
            
            if dy == 1: shifted[-1, :] = 0
            elif dy == -1: shifted[0, :] = 0
            if dx == 1: shifted[:, -1] = 0
            elif dx == -1: shifted[:, 0] = 0
            
            common = (sk & shifted)
            ys, xs = np.nonzero(common)
            
            for y, x in zip(ys, xs):
                neighbors_dict[(y, x)].append((y + dy, x + dx))
                
        return neighbors_dict

    def _get_ends(self,skeleton,neighbors):
        ends = np.column_stack(np.nonzero((skeleton == 1) & (neighbors == 1)))
        return ends
    
    def _get_intersections(self,skeleton,neighbors):
        intersections = np.column_stack(np.nonzero((skeleton  == 1) & (neighbors > 2)))
        return intersections
    
    def _cluster_intersections(self, junctions: np.ndarray) -> tuple[np.ndarray, dict, dict]:
        """Cluster raw junctions into intersection centroids."""
        if junctions is None or junctions.shape[0] == 0:
            return np.empty((0, 2), dtype=np.uint16), {}, {}

        inter_map = np.zeros_like(self.skeleton, dtype=np.uint8)
        for y, x in junctions:
            inter_map[y, x] = 1

        labeled, num_clusters = label(inter_map)
        
        centroids = []
        junction_to_centroid = {}
        centroid_to_junctions = defaultdict(list)

        for cluster_id in range(1, num_clusters + 1):
            cluster_coords = np.argwhere(labeled == cluster_id)
            if len(cluster_coords) == 0:
                continue
            
            center_y = int(round(cluster_coords[:, 0].mean()))
            center_x = int(round(cluster_coords[:, 1].mean()))
            centroid = (center_y, center_x)
            centroids.append(centroid)

            for (y, x) in cluster_coords:
                junction_to_centroid[(y, x)] = centroid
                centroid_to_junctions[centroid].append((y, x))

        intersection_to_neighbors = defaultdict(list)
        
        for centroid, junction_points in centroid_to_junctions.items():
            all_neighbors = []
            for (y, x) in junction_points:
                neighbors = self.neighbors_map.get((y, x), [])
                all_neighbors.extend(neighbors)
            
            intersection_to_neighbors[centroid] = all_neighbors

        return np.array(centroids, dtype=np.uint16), junction_to_centroid, intersection_to_neighbors


    def _plot_reconnect_debug(self, skeleton, p1, p2, d1, d2, angle, 
                            vec_actual, consis1, consis2):
        plt.figure(figsize=(9, 9))
        plt.imshow(skeleton, cmap='gray')
        
        plt.scatter(p1[1], p1[0], color='red', s=120, label=f'p1 ({p1[0]},{p1[1]})')
        plt.scatter(p2[1], p2[0], color='blue', s=120, label=f'p2 ({p2[0]},{p2[1]})')
        
        scale = min(skeleton.shape) * 0.015
        plt.arrow(p1[1], p1[0], d1[1]*scale, d1[0]*scale, 
                color='red', width=1, head_width=scale*0.2)
        plt.arrow(p2[1], p2[0], d2[1]*scale, d2[0]*scale,
                color='blue', width=1, head_width=scale*0.2)
        
        plt.plot([p1[1], p2[1]], [p1[0], p2[0]], 'g--', lw=1.5, 
                label='Actual vector')
        
        plt.text(p1[1]+5, p1[0]-5, f"d1=({d1[1]:.2f},{d1[0]:.2f})", color='red')
        plt.text(p2[1]+5, p2[0]-5, f"d2=({d2[1]:.2f},{d2[0]:.2f})", color='blue')
        plt.text((p1[1]+p2[1])/2, (p1[0]+p2[0])/2+10, 
                f"Vec_actual=({vec_actual[1]:.0f},{vec_actual[0]:.0f})", color='green')
        
        plt.title(f"Reconnect Check Angle={angle:.1f}°\n"
                f"C1={consis1:.2f}, C2={consis2:.2f}")
        plt.legend()
        plt.axis('equal')
        plt.tight_layout()
        plt.show()


    def _trace_path_from_end(self, skeleton: np.ndarray, start: tuple) -> tuple[list, bool]:
        """Trace from an endpoint to the next dead-end or intersection."""
        h, w = skeleton.shape
        visited = set()
        path = [start]
        curr = start

        neighbor_count = np.zeros_like(skeleton, dtype=np.uint8)
        coords = np.argwhere(skeleton > 0)
        for x, y in coords:
            count = np.sum(skeleton[max(0, x-1):min(h, x+2), max(0, y-1):min(w, y+2)]) - 1
            neighbor_count[x, y] = count

        while True:
            visited.add(curr)
            x, y = curr

            if neighbor_count[x, y] > 1:
                return path, True

            next_pixel = None
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and skeleton[nx, ny] > 0 and (nx, ny) not in visited:
                        next_pixel = (nx, ny)
                        break
                if next_pixel:
                    break

            if next_pixel is None:
                return path, False

            path.append(next_pixel)
            curr = next_pixel

            if len(path) > self.distmap[skeleton.astype(bool)].mean():
                break

        return path, False


    def _prune_path(self, skeleton: np.ndarray, path: list) -> None:
        """Remove a path from the skeleton."""
        path = np.asarray(path)
        skeleton[path[:, 0], path[:, 1]] = 0


    def _compute_direction(self, skeleton: np.ndarray, start_point: tuple, steps: int = 10) -> np.ndarray | None:
        path = [start_point]
        current = start_point
        visited = set([start_point])
        prev_direction = None

        for _ in range(steps):
            y, x = current
            candidates = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny_, nx_ = y + dy, x + dx
                    if (0 <= ny_ < skeleton.shape[0] and 0 <= nx_ < skeleton.shape[1]
                            and skeleton[ny_, nx_] > 0 and (ny_, nx_) not in visited):
                        candidates.append((ny_, nx_))

            if not candidates:
                break
            elif len(candidates) > 1:
                vectors = [np.array(nb) - np.array(current) for nb in candidates]
                if prev_direction is None:
                    best_idx = np.argmax([np.linalg.norm(v) for v in vectors])
                else:
                    best_idx = np.argmin([np.linalg.norm(v - prev_direction) for v in vectors])
                next_point = candidates[best_idx]
            else:
                next_point = candidates[0]

            visited.add(next_point)
            path.append(next_point)
            prev_direction = np.array(next_point) - np.array(current)
            current = next_point

        if len(path) > 1:
            diffs = [np.array(path[i]) - np.array(path[i - 1]) for i in range(1, len(path))]
            avg_direction = np.mean(diffs, axis=0)
            return avg_direction / (np.linalg.norm(avg_direction) + 1e-8)
        return None

    def _prune_split_ends(
            self,
            skeleton: np.ndarray,
            ends: np.ndarray,
            intersections: np.ndarray,
            visualize: bool = True
        ):
        """Prune short split branches and reconnect them when appropriate."""
        visited = np.zeros_like(skeleton, dtype=bool)
        
        end_set = set(tuple(e) for e in ends)
        
        for e in end_set:
            if e not in self.neighbors_map:
                continue
                
            curr_pixel = e
            path = [curr_pixel]
            prune = False
            found_nothing = False

            while True:
                y, x = curr_pixel
                visited[y, x] = True

                neighbors = self.neighbors_map.get(curr_pixel, [])
                nbs = [nb for nb in neighbors if not visited[nb[0], nb[1]]]

                if len(nbs) == 1:
                    curr_pixel = nbs[0]
                    path.append(curr_pixel)
                elif len(nbs) == 0:
                    found_nothing = True
                    break
                else:
                    prune = True
                    break

                if len(path) > 5 * self.mean_dist:
                    break

            if found_nothing or not prune:
                continue

            for pixel in path:
                py, px = pixel
                
                if pixel not in self.neighbors_map:
                    continue

                for neighbor in self.neighbors_map[pixel]:
                    if neighbor in self.neighbors_map:
                        if pixel in self.neighbors_map[neighbor]:
                            self.neighbors_map[neighbor].remove(pixel)
                
                del self.neighbors_map[pixel]
                skeleton[py, px] = 0

            if len(path) > 0:
                y_end, x_end = path[-1]
                vecs = self._get_boundary_pixels(skeleton[y_end-4:y_end+5, x_end-4:x_end+5])

                if len(vecs) == 2:
                    vecs[:, 0] += y_end - 3
                    vecs[:, 1] += x_end - 3
                    p1, p2 = vecs
                    
                    d1 = self._compute_direction(skeleton, tuple(p1))
                    d2 = self._compute_direction(skeleton, tuple(p2))
                    reconnect = False
                    if d1 is not None and d2 is not None:
                        cos_theta = np.clip(np.dot(d1, d2), -1.0, 1.0)
                        angle = np.degrees(np.arccos(cos_theta))
                        vec_actual = np.array([p2[0]-p1[0], p2[1]-p1[1]])
                        norm_vec_actual = vec_actual / (np.linalg.norm(vec_actual)+1e-8)
                        consistency1 = -np.dot(d1, norm_vec_actual)
                        consistency2 = -np.dot(d2, -norm_vec_actual)
                        reconnect = (angle > 120) and (consistency1 > 0.7) and (consistency2 > 0.7)
                    
                    if reconnect:
                        ys, xs = line(p1[0], p1[1], p2[0], p2[1])
                        skeleton[ys, xs] = 1
                        
                        for y_new, x_new in zip(ys, xs):
                            new_pixel = (y_new, x_new)
                            self.neighbors_map[new_pixel] = []
                            
                            for dy in (-1,0,1):
                                for dx in (-1,0,1):
                                    if dy == 0 and dx == 0:
                                        continue
                                    ny = y_new + dy
                                    nx = x_new + dx
                                    neighbor_pixel = (ny, nx)
                                    
                                    if 0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1] and skeleton[ny, nx]:
                                        self.neighbors_map[new_pixel].append(neighbor_pixel)
                                        if neighbor_pixel in self.neighbors_map:
                                            self.neighbors_map[neighbor_pixel].append(new_pixel)

                elif len(vecs) == 1:
                    vec = vecs.squeeze()
                    y_new, x_new = vec
                    new_pixel = (y_new, x_new)
                    self.neighbors_map[new_pixel] = []
                    for dy in (-1,0,1):
                        for dx in (-1,0,1):
                            if dy == 0 and dx == 0:
                                continue
                            ny = y_new + dy
                            nx = x_new + dx
                            neighbor_pixel = (ny, nx)
                            if 0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1] and skeleton[ny, nx]:
                                self.neighbors_map[new_pixel].append(neighbor_pixel)
                                if neighbor_pixel in self.neighbors_map:
                                    self.neighbors_map[neighbor_pixel].append(new_pixel)

        pruned_junctions = [j for j in intersections if tuple(j) in self.neighbors_map]
        
        return np.array(pruned_junctions)


    def preprocess(self):
        self.mask  = self._process_mask(self.mask)
        self.skeleton = skeletonize(self.mask, method='lee').astype(np.uint8)
        self.distmap  = cv2.distanceTransform(self.mask, cv2.DIST_L2, 3)
        self.mean_dist = self.distmap[self.skeleton.astype(bool)].mean() if np.any(self.skeleton) else 0
        self.neighbors = self._get_neighbors_num()
        self.neighbors_map = self._get_neighbors_map()

        endpoints = self._get_ends(self.skeleton,self.neighbors)
        junctions = self._get_intersections(self.skeleton,self.neighbors)

        self.pruned_junctions = self._prune_split_ends(self.skeleton, endpoints, junctions)
        self.original_junctions = set()
        for y, x in self.pruned_junctions:
            self.original_junctions.add((y, x))
        
        self.intersections, self.junction_to_intersection, self.intersection_to_neighbors = self._cluster_intersections(self.pruned_junctions)
        self.original_junctions = set((y, x) for y, x in self.pruned_junctions)
        
        self.neighbors= self._get_neighbors_num()
        endpoints = self._get_ends(self.skeleton,self.neighbors)

        self.endpoints = endpoints


    @staticmethod
    def _crop_edge_path(path: list[tuple[int, int]],
                        crop: int = 5,
                        node_positions: list[tuple[int, int]] = []) -> list[tuple[int, int]]:
        if len(path) < 5:
            return []

        start = np.array(path[0])  # (y, x)
        end = np.array(path[-1])

        def in_crop_area(node_yx: tuple[int, int], center_yx: np.ndarray) -> bool:
            return abs(node_yx[0] - center_yx[0]) <= crop and abs(node_yx[1] - center_yx[1]) <= crop

        has_node_near_start = any(in_crop_area(n, start) for n in node_positions)
        has_node_near_end = any(in_crop_area(n, end) for n in node_positions)

        if has_node_near_start or has_node_near_end:
            return path  # 保留原路径

        trimmed = []
        for p in path:
            if (abs(p[0] - start[0]) > crop or abs(p[1] - start[1]) > crop) and \
            (abs(p[0] - end[0]) > crop or abs(p[1] - end[1]) > crop):
                trimmed.append(tuple(p))

        return trimmed
    
    def build_graph(self):
        """Build a graph from the skeleton."""
        self.debug_paths = []
        self.graph = nx.MultiGraph()
        skel = (self.skeleton > 0).astype(np.uint8)
        h, w = skel.shape
        pos_to_node = {}
        self.max_ip_id = -1

        if not hasattr(self, 'neighbors_map') or self.neighbors_map is None or not isinstance(self.neighbors_map, dict):
            print("[错误] self.neighbors_map 未初始化或非字典格式！")
            return

        for i, (y, x) in enumerate(self.endpoints):
            name = f"EP_{i}"
            self.graph.add_node(name, pos=(x, y), type='endpoint')
            pos_to_node[(y, x)] = name

        for i, (y, x) in enumerate(self.intersections):
            name = f"IP_{i}"
            self.graph.add_node(name, pos=(x, y), type='intersection')
            pos_to_node[(y, x)] = name
            if i > self.max_ip_id:
                self.max_ip_id = i

        intersection_to_junctions = defaultdict(list)
        if hasattr(self, 'junction_to_intersection') and isinstance(self.junction_to_intersection, dict):
            for junction_coord, centroid_coord in self.junction_to_intersection.items():
                if not isinstance(junction_coord, tuple):
                    junction_coord = tuple(junction_coord)
                if not isinstance(centroid_coord, tuple):
                    centroid_coord = tuple(centroid_coord)
                intersection_to_junctions[centroid_coord].append(junction_coord)
        else:
            print("[警告] self.junction_to_intersection 未定义或非字典！")

        visited_edges = set()



        all_real_starts = []
        for (start_node_coord, node_start) in pos_to_node.items():
            node_type = self.graph.nodes[node_start]['type']
            if node_type == 'intersection':
                real_start_points = intersection_to_junctions.get(start_node_coord, [])
            else:
                real_start_points = [start_node_coord]
            for real_start in real_start_points:
                all_real_starts.append( (real_start, node_start) )

        for (current_start_coord, node_start) in all_real_starts:
            initial_neighbors = self.neighbors_map.get(current_start_coord, [])
            
            for neigh_coord in initial_neighbors:
                edge_key = tuple(sorted([current_start_coord, neigh_coord]))
                if edge_key in visited_edges:
                    continue
                
                path = [current_start_coord, neigh_coord]
                visited_edges.add(edge_key)
                prev_coord = current_start_coord
                curr_coord = neigh_coord
                hit_node = None
                path_edges = {edge_key}

                while True:
                    if curr_coord in pos_to_node:
                        candidate_node = pos_to_node[curr_coord]
                        if candidate_node != node_start:
                            hit_node = candidate_node
                            path.append(curr_coord)
                        break

                    if curr_coord in self.junction_to_intersection:
                        centroid_coord = self.junction_to_intersection[curr_coord]
                        hit_node = pos_to_node.get(centroid_coord, None)
                        if hit_node != node_start and hit_node is not None:
                            path.append(curr_coord)
                        break
                    
                    curr_neighbors = self.neighbors_map.get(curr_coord, [])
                    possible_next = [nb for nb in curr_neighbors if nb != prev_coord]

                    if not possible_next:
                        path_edges = set()
                        break

                    next_coord = None
                    if len(possible_next) == 1:
                        next_coord = possible_next[0]
                    else:
                        best_dot_product = -float('inf')
                        dir_vec = (curr_coord[0] - prev_coord[0], curr_coord[1] - prev_coord[1])
                        for nb in possible_next:
                            if nb in path[-5:]:
                                continue
                            cand_dir = (nb[0] - curr_coord[0], nb[1] - curr_coord[1])
                            dot_product = dir_vec[0] * cand_dir[0] + dir_vec[1] * cand_dir[1]
                            if dot_product > best_dot_product:
                                best_dot_product = dot_product
                                next_coord = nb
                    
                    if next_coord is None:
                        path_edges = set()
                        break

                    new_edge = tuple(sorted([curr_coord, next_coord]))
                    path_edges.add(new_edge)
                    visited_edges.add(new_edge)
                    path.append(next_coord)
                    prev_coord = curr_coord
                    curr_coord = next_coord

                if hit_node and path_edges and len(path) >= 2:
                    if node_start == hit_node:
                        continue
                    u, v = (node_start, hit_node) if node_start < hit_node else (hit_node, node_start)
                    visited_edges.update(path_edges)

                    trimmed_path = path.copy()
                    pixel_path = np.array(trimmed_path)
                    self.debug_paths.append(pixel_path)

                    path_len = len(pixel_path)
                    start_is_EP = self.graph.nodes[node_start]['type'] == 'endpoint'
                    hit_is_EP = self.graph.nodes[hit_node]['type'] == 'endpoint'
                    if start_is_EP and hit_is_EP and path_len < self.mean_dist * 10:
                        print(f"Discard short EP-EP edge: {node_start} <-> {hit_node} (len: {path_len})")
                        continue

                    midpoints = []
                    if path_len > 0:
                        if path_len <= self.mean_dist * 10:
                            mid_idx = path_len // 2
                            midpoints = [tuple(pixel_path[mid_idx])]
                        else:
                            indices = np.linspace(0, path_len - 1, 10, dtype=int)
                            midpoints = [tuple(pixel_path[i]) for i in indices[1:-1]]

                    color, width, length = self.get_edge_features(pixel_path)
                    self.graph.add_edge(
                        u, v,
                        path=trimmed_path,
                        color=color,
                        width=width,
                        length=path_len,
                        midpoints=midpoints,
                        direction=(node_start, hit_node)
                    )


        edges_to_process = defaultdict(list)
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            edge_pair = frozenset([u, v])
            edges_to_process[edge_pair].append((u, v, key, data))

        new_nodes = []
        new_edges = []
        edges_to_remove = []
        next_ip_num = self.max_ip_id + 1

        for edge_pair, edges in edges_to_process.items():
            if len(edges) <= 1:
                continue
            u, v = tuple(edge_pair)
            for (orig_u, orig_v, key, data) in edges:
                orig_path = data.get("path", [])
                if len(orig_path) < 2:
                    continue
                mid_idx = len(orig_path) // 2
                path1 = orig_path[:mid_idx + 1]
                path2 = orig_path[mid_idx:]
                mid_coord = orig_path[mid_idx]
                mid_y, mid_x = mid_coord

                new_node_name = f"IP_{next_ip_num}"
                next_ip_num += 1
                self.max_ip_id = next_ip_num - 1
                new_nodes.append((new_node_name, {
                    "pos": (mid_x, mid_y),
                    "type": "intersection",
                    "is_mid_node": True
                }))

                edges_to_remove.append((orig_u, orig_v, key))
                orig_color = data.get("color")
                orig_width = data.get("width")

                new_edges.append((orig_u, new_node_name, {
                    "path": path1, "color": orig_color, "width": orig_width,
                    "length": len(path1), "midpoints": data["midpoints"][:len(data["midpoints"])//2],
                    "direction": (orig_u, new_node_name)
                }))
                new_edges.append((new_node_name, orig_v, {
                    "path": path2, "color": orig_color, "width": orig_width,
                    "length": len(path2), "midpoints": data["midpoints"][len(data["midpoints"])//2:],
                    "direction": (new_node_name, orig_v)
                }))

        if edges_to_remove:
            self.graph.remove_edges_from(edges_to_remove)
        for node_name, attrs in new_nodes:
            self.graph.add_node(node_name, **attrs)
        for u, v, attrs in new_edges:
            self.graph.add_edge(u, v, **attrs)
        

        self.graph = nx.Graph(self.graph)
        


    def compute_adjacency_matrices(self):
        """Compute full-graph and endpoint adjacency matrices."""
        self.endpoint_nodes = [n for n in self.graph.nodes if n.startswith("EP")]
        self.fullpoint_adjacency = nx.adjacency_matrix(self.graph).todense()
        
        if self.endpoint_nodes:
            self.endpoint_adjacency = nx.adjacency_matrix(
                self.graph.subgraph(self.endpoint_nodes), nodelist=self.endpoint_nodes
            ).todense()
        else:
            self.endpoint_adjacency = np.zeros((0, 0), dtype=int)  


    def compute_endpoint_connectivity(self):
        """Compute endpoint connectivity."""
        n = len(self.endpoint_nodes)
        self.endpoint_connectivity = np.zeros((n, n), dtype=int)
        length_dict = dict(nx.all_pairs_shortest_path_length(self.graph))

        for i, node_i in enumerate(self.endpoint_nodes):
            for j, node_j in enumerate(self.endpoint_nodes):
                if node_j in length_dict.get(node_i, {}):
                    self.endpoint_connectivity[i, j] = 1


    def get_edge_features(self, pixel_path):
        """Return average Lab color, width, and length for an edge."""

        if not hasattr(self, "lab_image"):
            img_float = self.image.astype(np.float32) / 255.0
            self.lab_image = cv2.cvtColor(img_float, cv2.COLOR_RGB2Lab)

        pixel_colors = self.lab_image[pixel_path[:, 0], pixel_path[:, 1], :]
        pixel_widths = self.distmap[pixel_path[:, 0], pixel_path[:, 1]]

        avg_color = pixel_colors.mean(axis=0)
        avg_width = pixel_widths.mean()
        edge_length = len(pixel_path)

        return avg_color, avg_width, edge_length

    def extract_endpoint_features(self):
        """Extract average color and width around each endpoint."""
        colors = []
        widths = []

        for node in self.endpoint_nodes:
            path_pixels = []
            for _, _, data in self.graph.edges(node, data=True):
                path = data.get('path', [])
                path_pixels.extend(path)
            path_pixels = np.array(path_pixels)

            if len(path_pixels) == 0:
                colors.append(np.zeros(3))
                widths.append(0)
                continue

            pixel_colors = self.image[path_pixels[:, 0], path_pixels[:, 1], :]
            pixel_widths = self.distmap[path_pixels[:, 0], path_pixels[:, 1]]

            pixel_colors_rgb = pixel_colors.astype(np.uint8).reshape(-1, 1, 3)
            pixel_colors_lab = cv2.cvtColor(pixel_colors_rgb, cv2.COLOR_BGR2Lab).reshape(-1, 3)
            avg_lab = pixel_colors_lab.mean(axis=0)

            colors.append(avg_lab)
            widths.append(pixel_widths.mean())

        self.endpoint_colors = np.array(colors)
        self.endpoint_widths = np.array(widths)

    def compute_ciede2000_similarity(self,edge_colors: np.ndarray, edge_widths: np.ndarray, edge_lengths: np.ndarray, sigma_width: float = 5.0,sigma_color:float=20) -> tuple[float, float]:
        n = len(edge_colors)
        if n < 2:
            return 1.0, 1.0

        c1 = edge_colors[:, np.newaxis, :]
        c2 = edge_colors[np.newaxis, :, :]
        dist_matrix = deltaE_ciede2000(c1, c2)
        
        idx = np.triu_indices(n, k=1)
        color_dists = dist_matrix[idx]

        color_sims = np.exp(- (color_dists ** 2) / (2 * sigma_color ** 2))

        width_dists = pdist(edge_widths.reshape(-1, 1), metric='euclidean')
        width_sims = np.exp(-width_dists**2 / (2 * sigma_width**2))

        color_sim_mat = squareform(color_sims)
        width_sim_mat = squareform(width_sims)

        length_mat = np.outer(edge_lengths, edge_lengths)
        np.fill_diagonal(length_mat, 0)
        np.fill_diagonal(color_sim_mat, 0)
        np.fill_diagonal(width_sim_mat, 0)

        total_weight = length_mat.sum()
        if total_weight == 0:
            return 1.0, 1.0

        color_similarity = np.sum(color_sim_mat * length_mat) / total_weight
        width_similarity = np.sum(width_sim_mat * length_mat) / total_weight

        return color_similarity, width_similarity

    def compute_path_similarity(self,
        edge_colors: np.ndarray,
        edge_widths: np.ndarray,
        edge_lengths: np.ndarray,
        sigma_width: float = 5.0
    ) -> tuple[float, float]:
        """Compute path-level color and width similarity."""

        n = len(edge_colors)
        if n < 2:
            return 1.0, 1.0

        color_dists = pdist(edge_colors, metric='cosine')
        color_sims = 1.0 - color_dists

        width_dists = pdist(edge_widths.reshape(-1, 1), metric='euclidean')
        width_sims = np.exp(-width_dists**2 / (2 * sigma_width**2))

        color_sim_mat = squareform(color_sims)
        width_sim_mat = squareform(width_sims)

        length_mat = np.outer(edge_lengths, edge_lengths)
        np.fill_diagonal(length_mat, 0)
        np.fill_diagonal(color_sim_mat, 0)
        np.fill_diagonal(width_sim_mat, 0)

        total_weight = length_mat.sum()
        if total_weight == 0:
            return 1.0, 1.0

        color_similarity = np.sum(color_sim_mat * length_mat) / total_weight
        width_similarity = np.sum(width_sim_mat * length_mat) / total_weight

        return color_similarity, width_similarity
    

    def convert_node_path_to_pixel_path(self, node_path: list) -> list:
        """Convert node path to pixel path with direction correction."""
        pixel_path = []

        for i in range(len(node_path) - 1):
            u, v = node_path[i], node_path[i + 1]
            edge_data = self.graph.get_edge_data(u, v)

            if not edge_data or 'path' not in edge_data:
                continue

            edge_pixels = edge_data['path']

            if edge_data.get('direction') != (u, v):
                edge_pixels = edge_pixels[::-1]

            if i == 0:
                pixel_path.extend(edge_pixels)
            else:
                pixel_path.extend(edge_pixels[1:])

        return pixel_path

    def _plot_reconnect_debug(self, skeleton, p1, p2, d1, d2, angle, 
                            vec_actual, consis1, consis2):
        plt.figure(figsize=(9, 9))
        plt.imshow(skeleton, cmap='gray')
        
        plt.scatter(p1[1], p1[0], color='red', s=120, label=f'p1 ({p1[0]},{p1[1]})')
        plt.scatter(p2[1], p2[0], color='blue', s=120, label=f'p2 ({p2[0]},{p2[1]})')
        
        scale = min(skeleton.shape) * 0.015
        plt.arrow(p1[1], p1[0], d1[1]*scale, d1[0]*scale, 
                color='red', width=1, head_width=scale*0.2)
        plt.arrow(p2[1], p2[0], d2[1]*scale, d2[0]*scale,
                color='blue', width=1, head_width=scale*0.2)
        
        plt.plot([p1[1], p2[1]], [p1[0], p2[0]], 'g--', lw=1.5, 
                label='Actual vector')
        
        plt.text(p1[1]+5, p1[0]-5, f"d1=({d1[1]:.2f},{d1[0]:.2f})", color='red')
        plt.text(p2[1]+5, p2[0]-5, f"d2=({d2[1]:.2f},{d2[0]:.2f})", color='blue')
        plt.text((p1[1]+p2[1])/2, (p1[0]+p2[0])/2+10, 
                f"Vec_actual=({vec_actual[1]:.0f},{vec_actual[0]:.0f})", color='green')
        
        plt.title(f"Reconnect Check Angle={angle:.1f}°\n"
                f"C1={consis1:.2f}, C2={consis2:.2f}")
        plt.legend()
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

    
    def get_midpoint_path(self, node_path: list, crossed_edges=None) -> list[np.ndarray]:
        """Split a node path into per-edge midpoint segments."""
        segments: list[np.ndarray] = []

        start_pos = self.graph.nodes[node_path[0]]["pos"]
        start_pt = np.array([(start_pos[1], start_pos[0])], dtype=np.float64)

        for i, (u, v) in enumerate(zip(node_path[:-1], node_path[1:])):
            edge_data = self.graph.get_edge_data(u, v)
            if not edge_data or "midpoints" not in edge_data:
                continue

            midpoints = edge_data["midpoints"]
            if edge_data.get("direction") != (u, v):
                midpoints = midpoints[::-1]

            pts = np.array(midpoints, dtype=np.float64)

            is_crossed = False
            if crossed_edges is not None:
                key = tuple(sorted((u, v)))
                if key in crossed_edges and len(crossed_edges[key]) >= 2:
                    is_crossed = True

            if is_crossed:
                if len(pts) > 0:
                    mid_idx = len(pts) // 2
                    pts = pts[mid_idx:mid_idx+1]   # 保留 1 点
                else:
                    pts = pts  # 没点就算了
            else:
                pts = pts  # 全部 midpoints

            if i == 0:
                pts = np.vstack((start_pt, pts))

            segments.append(pts)

        if segments:
            end_pos = self.graph.nodes[node_path[-1]]["pos"]
            end_pt = np.array([(end_pos[1], end_pos[0])], dtype=np.float64)
            segments[-1] = np.vstack((segments[-1], end_pt))

        return segments
    
    def compute_bend_energy_from_points(self, points: np.ndarray) -> float:
        if isinstance(points, (list, tuple)):
            points = np.vstack(points).astype(np.float64)
        total_energy = 0.0
        prev_sign = None
        flip_penalty_factor = 2

        for i in range(1, len(points) - 1):
            v1 = points[i] - points[i - 1]
            v2 = points[i + 1] - points[i]
            norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if norm1 == 0 or norm2 == 0:
                continue

            dot = np.dot(v1, v2)
            angle = np.arccos(np.clip(dot / (norm1 * norm2), -1.0, 1.0))
            deg_angle = np.degrees(angle)

            cross_z = v1[0] * v2[1] - v1[1] * v2[0]
            sign = np.sign(cross_z)

            if prev_sign is not None and sign != 0 and prev_sign != 0 and sign != prev_sign:
                total_energy += deg_angle * flip_penalty_factor
            else:
                total_energy += deg_angle

            prev_sign = sign

        return total_energy
    
    def generate_all_candidate_paths(self, w_bend, w_color, w_width, w_len, k_paths):
        candidate_paths = []
        path_id_counter = 0
        endpoints = self.endpoint_nodes
        raw_paths = []

        for ep1, ep2 in combinations(endpoints, 2):
            if self.graph.has_edge(ep1, ep2):
                path = [ep1, ep2]
                pixel_path = self.convert_node_path_to_pixel_path(path)
                length = len(pixel_path)
                if length < self.mean_dist:
                    continue

                bend_energy = 0.0
                color_sim, width_sim = 1.0, 1.0
                avg_width = self.graph[ep1][ep2]['width']
                avg_color = self.graph[ep1][ep2]['color']
                raw_paths.append({
                    'id': path_id_counter,
                    'pair': (ep1, ep2),
                    'node_path': path,
                    'pixel_path': pixel_path,
                    'bend_energy': bend_energy,
                    'color_sim': color_sim,
                    'width_sim': width_sim,
                    'length': length,
                    'avg_width': avg_width,
                    'avg_color': avg_color
                })
                path_id_counter += 1

            else:
                try:
                    paths = islice(nx.shortest_simple_paths(self.graph, ep1, ep2), k_paths)
                    for path in paths:
                        pixel_path = self.convert_node_path_to_pixel_path(path)
                        if len(pixel_path) < 3:
                            continue
                        midpoint_path = self.get_midpoint_path(path)
                        bend_energy = self.compute_bend_energy_from_points(midpoint_path)
                        length = len(pixel_path)

                        edge_colors, edge_widths, edge_lengths = [], [], []
                        for u, v in zip(path[:-1], path[1:]):
                            data = self.graph[u][v]
                            edge_colors.append(data['color'])
                            edge_widths.append(data['width'])
                            edge_lengths.append(data['length'])

                        color_sim, width_sim = self.compute_ciede2000_similarity(
                            np.array(edge_colors), np.array(edge_widths), np.array(edge_lengths)
                        )

                        avg_width = np.average(edge_widths, weights=edge_lengths) if np.sum(edge_lengths) > 0 else np.mean(edge_widths)
                        edge_colors_arr = np.array(edge_colors)         # (n_edges,3)
                        edge_lens_arr   = np.array(edge_lengths)
                        avg_color = np.average(edge_colors_arr, axis=0, weights=edge_lens_arr)
                        raw_paths.append({
                            'id': path_id_counter,
                            'pair': (ep1, ep2),
                            'node_path': path,
                            'pixel_path': pixel_path,
                            'midpoint_path': midpoint_path,
                            'bend_energy': bend_energy,
                            'color_sim': color_sim,
                            'width_sim': width_sim,
                            'length': length,
                            'avg_width': avg_width,
                            'avg_color ':avg_color 
                        })
                        path_id_counter += 1

                except nx.NetworkXNoPath:
                    continue

        if not raw_paths:
            logger.warning("No candidate paths generated")
            return []

        max_bend = max(p['bend_energy'] for p in raw_paths) or 1
        max_len = max(p['length'] for p in raw_paths) or 1
        self.max_bend_energy = max_bend
        self.max_path_length = max_len

        for p in raw_paths:
            bend_norm = p['bend_energy'] / max_bend
            color_cost = 1 - p['color_sim']
            width_cost = 1 - p['width_sim']
            length_cost =  (p['length'] / max_len)

            total_cost = (
                w_bend * bend_norm +
                w_color * color_cost +
                w_width * width_cost +
                w_len * length_cost
            )

            candidate_paths.append({
                **p,
                'bend_norm': bend_norm,
                'color_cost': color_cost,
                'width_cost': width_cost,
                'length_cost': length_cost,
                'total_cost': total_cost
            })

        return candidate_paths


    def get_path_edges(self, node_path):
        """Return normalized edges from a node path."""
        edges = []
        for i in range(len(node_path)-1):
            u, v = sorted([node_path[i], node_path[i+1]])
            edges.append((u, v))
        return edges
    
    def optimize_wires_global(self, w_bend=0.2, w_color=0.4, w_width=0.1, w_len=0.3, k_paths=20):
        """Solve the global wire assignment problem."""
        all_candidates = self.generate_all_candidate_paths(w_bend, w_color, w_width, w_len, k_paths)
        model = pulp.LpProblem("Global_Wire_Optimization", pulp.LpMinimize)
        
        y = {p['id']: pulp.LpVariable(f"y_{p['id']}", cat=pulp.LpBinary) 
            for p in all_candidates}
        
        model += pulp.lpSum(p['total_cost'] * y[p['id']] for p in all_candidates)
        
        endpoint_coverage = defaultdict(list)
        for p in all_candidates:
            ep1, ep2 = p['pair']
            endpoint_coverage[ep1].append(y[p['id']])
            endpoint_coverage[ep2].append(y[p['id']])
        
        for ep, vars in endpoint_coverage.items():
            model += pulp.lpSum(vars) >= 1, f"Coverage_{ep}"
        
        edge_usage = defaultdict(list)
        for p in all_candidates:
            path_id = p['id']
            for edge in self.get_path_edges(p['node_path']):
                edge_usage[edge].append(y[path_id])
        
        for edge, vars in edge_usage.items():
            model += pulp.lpSum(vars) >= 1, f"Reuse_Coverage_{edge}"    
            model += pulp.lpSum(vars) <= 3, f"Reuse_Limit_{edge}"  # 限制最大复用次数
        
        solver = pulp.PULP_CBC_CMD(msg=False)
        model.solve(solver)
        selected_paths = []
        selected_paths_with_width = []  # 新增：包含 avg_width

        for p in all_candidates:
            if y[p['id']].value() == 1:
                selected_paths.append(p['node_path'])
                selected_paths_with_width.append({
                    'node_path': p['node_path'],
                    'pixel_path': p['pixel_path'],
                    'avg_width': p['avg_width']  
                })

        return selected_paths,selected_paths_with_width

    def visualize(self, paths=None):
        pad = 50  
        def pad_image(img, channels=False, value=255):
            if channels:
                return np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=value)
            return np.pad(img, pad, mode='constant', constant_values=value)

        images = {
            "original": pad_image(self.image, channels=True, value=255),
            "skeleton": pad_image(self.skeleton, value=0),
            "mask": pad_image(self.mask, value=0),
            "blank": pad_image(np.ones_like(self.mask)*255, value=255)
        }

        def shift_coords(coords):
            return [(x+pad, y+pad) for (x,y) in coords]

        points = {
            "endpoints": shift_coords(self.endpoints),
            "intersections": shift_coords(self.intersections)
        }

        pos_padded = {node: (x+pad, y+pad) for node, (x,y) in self.graph.nodes(data='pos')}

        edges = list(self.graph.edges(data=True))
        color_avgs = []
        width_avgs = []
        for u,v,d in edges:
            path_pixels = np.array(d.get('path', []))
            if len(path_pixels) == 0:
                color_avgs.append(np.array([255,255,255]))
                width_avgs.append(1)
                continue

            pixel_colors = self.image[path_pixels[:,0], path_pixels[:,1], :]
            avg_color = pixel_colors.mean(axis=0)
            color_avgs.append(avg_color)
            pixel_widths = self.distmap[path_pixels[:,0], path_pixels[:,1]]
            avg_width = pixel_widths.mean()
            width_avgs.append(avg_width)

        color_avgs = np.array(color_avgs) / 255.0
        width_avgs = np.array(width_avgs)
        min_w, max_w = width_avgs.min(), width_avgs.max()
        width_norm = (width_avgs - min_w) / (max_w - min_w + 1e-6)
        widths = 1 + 4 * width_norm

        def save_single_plot(ax_num, title):
            fig_single = plt.figure(figsize=(10, 8))
            ax_single = fig_single.add_subplot(111)
            
            if ax_num == 1:
                ax_single.imshow(images["original"])
            elif ax_num == 2:
                ax_single.imshow(images["skeleton"], cmap='gray_r')
                if points["endpoints"]:
                    ep_x, ep_y = zip(*points["endpoints"])
                    ax_single.scatter(ep_y, ep_x, c='red', s=40, label='Endpoints')
                if points["intersections"]:
                    ip_x, ip_y = zip(*points["intersections"])
                    ax_single.scatter(ip_y, ip_x, c='blue', s=60, label='Intersections')
                ax_single.legend(fontsize=20)
            elif ax_num == 3:
                ax_single.imshow(images["mask"], cmap='gray_r')
            elif ax_num == 4:
                ax_single.imshow(images["blank"], cmap='gray', vmin=0, vmax=255)
                
                node_colors = ['orange' if 'EP' in n else 'skyblue' for n in self.graph.nodes]
                nx.draw_networkx_nodes(self.graph, pos_padded, ax=ax_single, node_color=node_colors, node_size=120)
                
                for (u,v,d), c, w in zip(edges, color_avgs, widths):
                    x0,y0 = pos_padded[u]
                    x1,y1 = pos_padded[v]
                    ax_single.plot([x0,x1], [y0,y1], color=c, linewidth=w, alpha=0.9)
                
                texts = []
                for node, (x, y) in pos_padded.items():
                    texts.append(
                        ax_single.text(
                            x, y, node,
                            fontsize=20,
                            ha='center',
                            va='center',
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="grey", alpha=0.7)
                        )
                    )

                adjust_text(
                    texts,
                    ax=ax_single,
                    expand_points=(1.2, 1.4),     
                    expand_text=(1.2, 1.4),
                    force_points=(0.5, 0.5),     
                    force_text=(0.7, 0.7),
                    lim=1000,                    
                    only_move={'points': 'xy', 'text': 'xy'}, 
                    arrowprops=dict(arrowstyle='-', color='gray', lw=0.5)
                )
                
            
            ax_single.axis('off')
            plt.tight_layout()
            
            filename = f"{title.replace(' ', '_').lower()}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.1)
            print(f"Saved: {filename}")
            plt.close(fig_single)

        save_single_plot(1, "Original Image")
        save_single_plot(2, "Skeleton with Key Points") 
        save_single_plot(3, "Wire Segmentation")
        save_single_plot(4, "Graph Structure")

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        ax1, ax2, ax3, ax4 = axes.flatten()

        ax1.imshow(images["original"])
        ax1.set_title("Padded Original Image")
        ax1.axis('off')

        ax2.imshow(images["skeleton"], cmap='gray_r')
        if points["endpoints"]:
            ep_x, ep_y = zip(*points["endpoints"])
            ax2.scatter(ep_y, ep_x, c='red', s=40, label='Endpoints')
        if points["intersections"]:
            ip_x, ip_y = zip(*points["intersections"])
            ax2.scatter(ip_y, ip_x, c='blue', s=60, label='Intersections')
        ax2.set_title("Skeleton with Key Points")
        ax2.legend()
        ax2.axis('off')

        ax3.imshow(images["mask"], cmap='gray_r')
        ax3.set_title("Wire Segmentation")
        ax3.axis('off')

        ax4.imshow(images["blank"], cmap='gray', vmin=0, vmax=255)
        
        node_colors = ['orange' if 'EP' in n else 'skyblue' for n in self.graph.nodes]
        nx.draw_networkx_nodes(self.graph, pos_padded, ax=ax4, node_color=node_colors, node_size=120)
        
        for (u,v,d), c, w in zip(edges, color_avgs, widths):
            x0,y0 = pos_padded[u]
            x1,y1 = pos_padded[v]
            ax4.plot([x0,x1], [y0,y1], color=c, linewidth=w, alpha=0.9)
        
        for node, (x,y) in pos_padded.items():
            ax4.text(x, y-15, node, fontsize=9, ha='center', va='bottom',
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="grey", alpha=0.7))
        
        ax4.set_title("Graph with Edge Color and Width")
        ax4.axis('off')

        plt.tight_layout()
        plt.show()
            

    def ciede2000(self,c1, c2):
        c1 = np.array(c1, dtype=np.float32).reshape(1, 1, 3)
        c2 = np.array(c2, dtype=np.float32).reshape(1, 1, 3)
        lab1 = c1
        lab2 = c2
        return deltaE_ciede2000(lab1, lab2)[0, 0]

    def assign_by_distance_only(self, mask: np.ndarray, path_list: list):
        h, w = mask.shape
        labeled_mask = np.zeros((h, w), dtype=np.uint8)

        all_pts = []
        path_ids = []

        for pid, p in enumerate(path_list):
            for (x, y) in p['pixel_path']:
                if 0 <= y < h and 0 <= x < w:
                    all_pts.append([x, y])
                    path_ids.append(pid + 1)  # 1-based

        if not all_pts:
            return labeled_mask

        tree = KDTree(all_pts)

        yy, xx = np.where(mask > 0)
        for y, x in zip(yy, xx):
            d, idx = tree.query([x, y])
            labeled_mask[y, x] = path_ids[idx]

        return labeled_mask

    
    def visualize_labeled_mask(labeled_mask):
        import matplotlib.colors as mcolors
        from matplotlib import cm

        N = labeled_mask.max()
        color_map = cm.get_cmap('tab20', N+1)

        color_mask = np.zeros((*labeled_mask.shape, 3), dtype=np.uint8)
        for i in range(1, N+1):
            color = (np.array(color_map(i)[:3]) * 255).astype(np.uint8)
            color_mask[labeled_mask == i] = color

        plt.figure(figsize=(10, 10))
        plt.imshow(color_mask)
        plt.title("Wire Instances (Color-coded)")
        plt.axis("off")
        plt.show()

    def is_edge_crossing(self,edge, edge_to_paths):
        """Check if an edge is used by more than one path."""
        u, v = edge
        return len(edge_to_paths.get((u, v), [])) + len(edge_to_paths.get((v, u), [])) > 1


    def node_path_to_edges(self,node_path):
        return [(node_path[i], node_path[i + 1]) for i in range(len(node_path) - 1)]


    def fit_bspline(self,points, s=2.0):
        """Fit B-spline curve to 2D points."""
        if len(points) < 4:
            return np.array(points)
        points = np.array(points)
        x, y = points[:, 0], points[:, 1]
        tck, _ = splprep([x, y], s=s)
        u = np.linspace(0, 1, len(points))
        x_fine, y_fine = splev(u, tck)
        return np.stack([x_fine, y_fine], axis=1)
    
    def identify_crossing_segments(self, optimal_paths):
        """Split paths into crossing and non-crossing segments."""
        edge_to_paths = defaultdict(list)

        for pid, path in enumerate(optimal_paths):
            node_edges = self.node_path_to_edges(path)
            for edge in node_edges:
                edge_to_paths[edge].append(pid)

        per_path_segments = []
        for path in optimal_paths:
            segments = []
            node_path = path
            current_type = None
            current_nodes = []

            for i in range(len(node_path) - 1):
                edge = (node_path[i], node_path[i + 1])
                crossing = self.is_edge_crossing(edge, edge_to_paths)
                segment_type = 'crossing' if crossing else 'non-crossing'

                if current_type is None:
                    current_type = segment_type

                if segment_type != current_type:
                    segments.append({'type': current_type, 'nodes': current_nodes})
                    current_nodes = [node_path[i]]
                    current_type = segment_type

                current_nodes.append(node_path[i + 1])

            if current_nodes:
                segments.append({'type': current_type, 'nodes': current_nodes})

            per_path_segments.append(segments)

        return per_path_segments, edge_to_paths

    def fit_and_resolve_crossings(self, per_path_segments):
        """对每条路径的non-crossing段进行拟合，并分析交叉段"""
        fitted_results = []

        for path_id, segments in enumerate(per_path_segments):
            path_fits = []
            for seg in segments:
                nodes = seg['nodes']
                edges = self.node_path_to_edges(nodes)
                pixels = []
                for u, v in edges:
                    if self.graph.has_edge(u, v):
                        pixels.extend(self.graph[u][v]['path'])

                if not pixels:
                    continue
                fitted_curve = self.fit_bspline(pixels)

                widths = []
                for x, y in fitted_curve:
                    xi, yi = int(round(x)), int(round(y))
                    if 0 <= yi < self.distmap.shape[0] and 0 <= xi < self.distmap.shape[1]:
                        widths.append(2 * self.distmap[yi, xi])

                path_fits.append({
                    'type': seg['type'],
                    'nodes': nodes,
                    'fitted_curve': fitted_curve,
                    'widths': widths
                })
            fitted_results.append(path_fits)

        return fitted_results
    
    def reconstruct_instance_mask(self, fitted_splines, crossing_orders, image_shape):
        """Render an instance mask from fitted curves."""
        h, w = image_shape
        instance_mask = np.zeros((h, w), dtype=np.uint8)

        rendered = set()

        for pid, spline_info in fitted_splines.items():
            for seg in spline_info:
                pts = seg['curve']
                widths = seg['widths']
                for (x, y), w_ in zip(pts, widths):
                    rr, cc = draw.disk((y, x), radius=w_ / 2, shape=instance_mask.shape)
                    instance_mask[rr, cc] = pid + 1

        for edge, pid_list in crossing_orders.items():
            for order, pid in enumerate(reversed(pid_list)):
                spline_info = fitted_splines[pid]
                for seg in spline_info:
                    if seg['type'] != 'crossing':
                        continue
                    u, v = seg['nodes'][0], seg['nodes'][-1]
                    if (u, v) != edge and (v, u) != edge:
                        continue

                    pts = seg['curve']
                    widths = seg['widths']
                    for (x, y), w_ in zip(pts, widths):
                        rr, cc = draw.disk((y, x), radius=w_ / 2, shape=instance_mask.shape)
                        instance_mask[rr, cc] = pid + 1  # 重叠时后绘制会覆盖前面

        return instance_mask

    def fit_non_crossing_segments(self, segments, edge_to_paths):

        fitted_splines = {}

        h, w = self.distmap.shape

        for path_id, path_segments in enumerate(segments):
            all_pixels = []
            all_nodes = []

            for segment in path_segments:
                if segment['type'] == 'non-crossing':
                    nodes = segment['nodes']
                    edges = self.node_path_to_edges(nodes)

                    for u, v in edges:
                        if self.graph.has_edge(u, v):
                            all_pixels.extend(self.graph[u][v]['path'])
                            all_nodes.extend([u, v])

            if not all_pixels:
                continue

            seen = set()
            all_nodes_unique = []
            for node in all_nodes:
                if node not in seen:
                    all_nodes_unique.append(node)
                    seen.add(node)

            fitted_curve = self.fit_bspline(all_pixels)

            widths = []
            for x, y in fitted_curve:
                ix = min(max(int(round(x)), 0), w - 1)
                iy = min(max(int(round(y)), 0), h - 1)
                widths.append(self.distmap[iy, ix] * 2)

            fitted_splines[path_id] = {
                'curve': fitted_curve,
                'widths': widths,
                'nodes': all_nodes_unique,
            }

        return fitted_splines

    def fit_smooth_curve(self, points, smooth=0.5, num_points=200):
        """Fit a smooth curve and fall back to the input on failure."""
        try:
            if isinstance(points, (list, tuple)):
                original_points = np.vstack(points).astype(np.float64)
            else:
                original_points = np.asarray(points, dtype=np.float64)

            if original_points.ndim != 2 or original_points.shape[1] != 2:
                print("[严重警告] 输入的原始点集不是有效的二维坐标数组！")
                return np.array([[0.0, 0.0]])

            n_points = original_points.shape[0]
            if n_points < 4:
                print(f"[警告] 进行B样条拟合至少需要 4 个点，但实际得到 {n_points} 个。将返回原始路径。")
                return original_points

            if np.any(np.isnan(original_points)) or np.any(np.isinf(original_points)):
                print("[警告] 输入的点集中包含 NaN 或无穷大 (Inf) 值。将返回原始路径。")
                return original_points
                
            if np.allclose(original_points, original_points[0]):
                print("[警告] 输入的所有点都重合。将返回原始路径。")
                return original_points

            tck, _ = splprep(original_points.T, s=smooth)
            u_new = np.linspace(0, 1, num_points)
            x_new, y_new = splev(u_new, tck)
            smooth_curve = np.vstack([x_new, y_new]).T
            return smooth_curve

        except Exception as e:
            print(f"[错误] 在执行B样条曲线拟合时发生未知错误: {e}。将返回原始路径。")
            if 'original_points' in locals() and original_points.ndim == 2 and original_points.shape[1] == 2:
                return original_points
            else:
                return np.array([[0.0, 0.0]])


    
    def draw_paths_with_radius(self,paths_pixel, distmap, canvas_shape, color=(255, 255, 255)):
        """Render thick paths using the distance map as radius."""
        canvas = np.zeros(canvas_shape, dtype=np.uint8)
        
        for path in paths_pixel:
            for x, y in path:
                radius = int(distmap[x, y])
                if radius > 0:
                    cv2.circle(canvas, (y, x), radius, color, -1)

        return canvas
    def extract_path_color_non_crossing(self,graph, path_nodes, edge_to_paths):
        """Average path color using only non-crossing edges."""
        colors = []
        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i + 1]
            if not graph.has_edge(u, v):
                continue
            edge_key = tuple(map(tuple, graph[u][v]['path']))
            if len(edge_to_paths.get(edge_key, [])) > 1:
                continue  # 是交叉边，跳过
            color = graph[u][v].get("color", None)
            if color is not None:
                colors.append(np.array(color))
        if colors:
            mean_color = np.mean(colors, axis=0)
            return tuple(int(c) for c in mean_color)
        else:
            return (255, 255, 255)  # fallback color if全部是交叉边

    def get_all_path_colors(self,graph, paths, edge_to_paths):
        return [self.extract_path_color_non_crossing(graph, node_path, edge_to_paths) for node_path in paths]
    def segments_to_edge_paths(self,graph, per_path_segments):
        """Convert segmented node paths into edge pixel paths."""
        all_paths_edges = []

        for segments in per_path_segments:
            edges_list = []
            for segment in segments:
                nodes = segment['nodes']
                for i in range(len(nodes) - 1):
                    u, v = nodes[i], nodes[i+1]
                    if graph.has_edge(u, v):
                        edge_path = graph[u][v]['path']
                    elif graph.has_edge(v, u):
                        edge_path = graph[v][u]['path']
                    else:
                        raise RuntimeError(f"Edge ({u},{v}) not found in graph")
                    edges_list.append(edge_path)
            all_paths_edges.append(edges_list)
        return all_paths_edges

    def draw_non_crossing_paths(
        self,
        paths_pixel,
        distmap: np.ndarray,
        path_colors: list,
        per_path_segments: list
    ) -> np.ndarray:
        """Draw only the non-crossing parts of each path."""
        
        canvas = np.zeros((*distmap.shape, 3), dtype=np.uint8)

        for color, segments in zip(path_colors, per_path_segments):
            color = tuple(int(c) for c in self.lab_to_bgr(color))
            for segment in segments:
                if segment['type'] != 'non-crossing':
                    continue

                for y, x in segment['nodes']:
                    if 0 <= y < distmap.shape[0] and 0 <= x < distmap.shape[1]:
                        radius = int(distmap[y, x])
                        if radius > 0:
                            cv2.circle(canvas, (x, y), radius, color, -1)

        return canvas
    
    def draw_crossing_over(self, canvas, edge_to_paths, path_colors, distmap):
        from skimage.color import rgb2lab

        canvas_lab = rgb2lab(canvas / 255.0)

        for edge, paths in edge_to_paths.items():
            if len(paths) < 2:
                continue

            color_labs = [path_colors[i] for i in paths]

            for pt in edge:
                x, y = pt
                if not (0 <= x < canvas.shape[0] and 0 <= y < canvas.shape[1]):
                    continue
                radius = int(distmap[x, y])
                if radius <= 0:
                    continue

                canvas_pixel_lab = canvas_lab[x, y]
                distances = [np.linalg.norm(canvas_pixel_lab - c) for c in color_labs]
                top_idx = np.argmin(distances)
                top_lab = path_colors[paths[top_idx]]

                top_color_bgr = tuple(int(c) for c in self.lab_to_bgr(top_lab))
                cv2.circle(canvas, (y, x), radius, top_color_bgr, -1)

        return canvas
    
    def lab_to_bgr(self, lab_color):
        """Convert a Lab color to OpenCV BGR."""
        try:
            if not isinstance(lab_color, (tuple, list, np.ndarray)):
                raise ValueError(f"lab_color 必须是元组、列表或数组，实际是 {type(lab_color)}")
            
            lab_arr = np.array(lab_color, dtype=np.float64)
            if lab_arr.size != 3:
                raise ValueError(f"lab_color 必须是3个元素，实际有 {lab_arr.size} 个元素")
            
            lab_arr = lab_arr.reshape(1, 1, 3)
            
            if lab_arr[0, 0, 0] < 0 or lab_arr[0, 0, 0] > 100:
                print(f"警告: L 值超出范围 [0,100]: {lab_arr[0,0,0]}")
                lab_arr[0,0,0] = np.clip(lab_arr[0,0,0], 0, 100)
            
            rgb = lab2rgb(lab_arr)
            
            rgb_255 = np.round(rgb * 255).astype(np.uint8)
            
            r, g, b = rgb_255[0, 0, 0], rgb_255[0, 0, 1], rgb_255[0, 0, 2]
            return (b, g, r)
        
        except Exception as e:
            print(f"LAB 转 BGR 错误: {e}，使用默认红色 (0,0,255)")
            return (0, 0, 255)



def assign_mask_pixels_to_paths_with_color(mask,
                                           curve_points_list,
                                           path_colors_bgr,
                                           path_widths,
                                           rgb_image=None,
                                           alpha=0.7,
                                           k=3):
    H,W = mask.shape
    assignment = -1*np.ones((H,W), dtype=np.int32)

    path_colors_lab = [rgb2lab(np.uint8([[bgr[::-1]]])/255.0)[0,0] for bgr in path_colors_bgr]

    all_centers = []
    all_ids     = []
    for i, pts in enumerate(curve_points_list):
        all_centers.append(pts)
        all_ids.append(np.full(len(pts), i, dtype=np.int32))
    all_centers = np.vstack(all_centers)
    all_ids     = np.hstack(all_ids)

    tree = cKDTree(all_centers.astype(np.float32))

    if rgb_image is None:
        lab_img = np.ones((H,W,3),dtype=np.float32)*200
    else:
        lab_img = rgb2lab(rgb_image/255.0)

    ys,xs = np.nonzero(mask)
    pixel_coords = np.stack([xs, ys], axis=1).astype(np.float32)

    dists, idxs = tree.query(pixel_coords, k=min(k, len(all_centers)))

    for i,(x,y) in enumerate(zip(xs,ys)):
        best_score = 1e9
        best_pid   = -1
        for dist_val, node_idx in zip(np.atleast_1d(dists[i]), np.atleast_1d(idxs[i])):
            pid = all_ids[node_idx]

            geo_score = dist_val / max(1.0, path_widths[pid])

            color_score = np.linalg.norm(lab_img[y,x] - path_colors_lab[pid])

            score = alpha*geo_score + (1-alpha)*color_score
            if score < best_score:
                best_score = score
                best_pid = pid
        assignment[y,x] = best_pid

    return assignment


import numpy as np
import cv2
import matplotlib.pyplot as plt
import os 
import glob

if __name__ == "__main__":

    img_dir = "/media/user/drive11/padlos/evaluation/data/imgs/"
    mask_dir = "/media/user/drive11/padlos/evaluation/data/binlabels/"
    depth_dir = "/media/user/drive11/padlos/evaluation/data/depths/"
    save_dir = "/media/user/drive11/padlos/evaluation/exp_ACM/ACM/PADLOS"

    os.makedirs(save_dir, exist_ok=True)

    img_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))

    total_time = 0  # 仅保留总耗时统计
    total_graph_time = 0
    total_opt_time = 0
    total_recon_time = 0
    
    processed_count = 0
    total_images = len(img_files)

    for img_path in img_files:
        fname = os.path.basename(img_path)        # c1_13.jpg
        name_no_ext = os.path.splitext(fname)[0]  # c1_13

        mask_path  = os.path.join(mask_dir,  fname)
        depth_path = os.path.join(depth_dir, name_no_ext + ".png")

        if not os.path.exists(mask_path) or not os.path.exists(depth_path):
            continue
        
        single_start_time = time.time()

        segmenter = WireSegmenter(img_path, mask_path, depth_path)

        graph_start_time = time.time()
        segmenter.preprocess()
        segmenter.build_graph()
        segmenter.compute_adjacency_matrices()
        segmenter.compute_endpoint_connectivity()
        graph_stage_time = time.time() - graph_start_time

        opt_start_time = time.time()
        node_paths, optimal_paths = segmenter.optimize_wires_global(
            w_bend=0.4,
            w_color=0.3,
            w_width=0.1,
            w_len=0.1,
            k_paths=100
        )
        opt_stage_time = time.time() - opt_start_time

        recon_start_time = time.time()
        h, w = segmenter.distmap.shape
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        point_canvas = canvas.copy()

        num_paths = len(optimal_paths)
        colors = [tuple(int(c*255) for c in color[:3]) for color in plt.cm.tab20(np.linspace(0,1,num_paths))]

        crossed_edges = {}
        for p_id, nodes in enumerate([p['node_path'] for p in optimal_paths]):
            for i in range(len(nodes)-1):
                key = tuple(sorted((nodes[i], nodes[i+1])))
                crossed_edges.setdefault(key, []).append(p_id)

        all_segments = []

        for path_idx, path_info in enumerate(optimal_paths):
            node_path = path_info['node_path']
            avg_width = max(10, int(round(path_info['avg_width'] * 2)))
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
                    mid_pt = seg[len(seg)//2]

                    distances = np.linalg.norm(curve_xy - mid_pt[::-1], axis=1)
                    mid_idx = int(np.argmin(distances))

                    prev_pt = seg[0] if e_idx == 0 else pixel_path_segments[e_idx-1][-1]
                    next_pt = seg[-1] if e_idx == len(node_path)-2 else pixel_path_segments[e_idx+1][0]
                    prev_depth = float(segmenter.depth[int(prev_pt[0]), int(prev_pt[1])])
                    next_depth = float(segmenter.depth[int(next_pt[0]), int(next_pt[1])])
                    avg_depth = (prev_depth + next_depth) / 2.0

                    cross_pairs.append((mid_idx, avg_depth))

            cross_pairs.sort(key=lambda t: t[0])
            cross_indices = [t[0] for t in cross_pairs]
            cross_depths = [t[1] for t in cross_pairs]

            segment_points = [0]
            for i in range(len(cross_indices)-1):
                mid_split = (cross_indices[i] + cross_indices[i+1]) // 2
                segment_points.append(mid_split)
            segment_points += [len(curve_xy)]
            segment_points = sorted(list(set(segment_points)))

            def depth_at_xy(pt_xy):
                x = int(np.clip(pt_xy[0], 0, w-1))
                y = int(np.clip(pt_xy[1], 0, h-1))
                return float(segmenter.depth[y, x])

            for seg_i in range(len(segment_points)-1):
                s_start = segment_points[seg_i]
                s_end = segment_points[seg_i+1]
                if s_start >= s_end:
                    continue
                segment_curve = curve_xy[s_start:s_end]

                if len(cross_depths) > 0 and seg_i < len(cross_depths):
                    segment_depth = float(cross_depths[seg_i])
                else:
                    mid_pt = segment_curve[len(segment_curve)//2]
                    segment_depth = depth_at_xy(mid_pt)

                all_segments.append({
                    'path_idx': path_idx,
                    'curve': segment_curve,
                    'depth': segment_depth,
                    'width': avg_width,
                    'color': color
                })
        all_segments.sort(key=lambda x: x['depth'])

        canvas_step = np.zeros_like(canvas)

        for seg in all_segments:
            pts = seg['curve'].astype(np.int32)
            for j in range(len(pts)-1):
                cv2.line(canvas_step, tuple(pts[j]), tuple(pts[j+1]),
                         seg['color'], int(seg['width']))

        final_output = canvas_step[1:-1, 1:-1, :]
        recon_stage_time = time.time() - recon_start_time
        
        single_total_time = time.time() - single_start_time
        total_time += single_total_time
        total_graph_time += graph_stage_time
        total_opt_time += opt_stage_time
        total_recon_time += recon_stage_time
        
        save_path = os.path.join(save_dir, f"{name_no_ext}.png")
        cv2.imwrite(save_path, final_output)
        
        processed_count += 1

        print(
            f"[Timing] {name_no_ext}: "
            f"Graph Generation={graph_stage_time:.6f}s, "
            f"Global Path Optimization={opt_stage_time:.6f}s, "
            f"Pseudo-depth-guided Instance Reconstruction={recon_stage_time:.6f}s, "
            f"Total={single_total_time:.6f}s"
        )
    
    print("\n=== ACM Algorithm Post-processing Summary ===")
    print(f"processed images: {total_images}")
    print(f"successful images: {processed_count}")
    print(f"total post time: {total_time:.6f}s")
    
    if processed_count > 0:
        avg_post_time = total_time / processed_count
        avg_fps = processed_count / total_time
        avg_graph_time = total_graph_time / processed_count
        avg_opt_time = total_opt_time / processed_count
        avg_recon_time = total_recon_time / processed_count

        print("\nStage-wise timing:")
        print(f"Graph Generation total time: {total_graph_time:.6f}s")
        print(f"Global Path Optimization total time: {total_opt_time:.6f}s")
        print(f"Pseudo-depth-guided Instance Reconstruction total time: {total_recon_time:.6f}s")
        print(f"Graph Generation avg time: {avg_graph_time:.6f}s/image")
        print(f"Global Path Optimization avg time: {avg_opt_time:.6f}s/image")
        print(f"Pseudo-depth-guided Instance Reconstruction avg time: {avg_recon_time:.6f}s/image")

        print(f"avg post time: {avg_post_time:.6f}s/image")
        print(f"avg FPS: {avg_fps:.2f}")
    else:
        print("avg post time: 0.000000s/image")
        print("avg FPS: 0.00")
