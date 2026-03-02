import cv2
import numpy as np
from collections import defaultdict


class CircleDetector:
    """Circle detector for matching puzzle games."""

    def __init__(self, image_path: str, config: dict):
        self.image_path = image_path
        self.config = config
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.circles = None
        self.color_data = []
        self.color_clusters = {}

    def detect_circles(self):
        """Detect circles using Hough transform with config parameters."""
        det = self.config['detection']
        hough = det['hough']
        circles = cv2.HoughCircles(
            self.gray, cv2.HOUGH_GRADIENT,
            dp=hough['dp'],
            minDist=hough['min_dist'],
            param1=hough['param1'],
            param2=hough['param2'],
            minRadius=det['min_radius'],
            maxRadius=det['max_radius']
        )

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            x_range = det.get('x_range')
            filtered = []
            for (x, y, r) in circles:
                if x_range and not (x_range[0] <= x <= x_range[1]):
                    continue
                filtered.append([x, y, r])
            if filtered:
                self.circles = np.array(filtered)
                return self.circles

        self.circles = None
        return None

    def extract_ball_colors(self):
        """Extract average color from the center of each detected ball."""
        if self.circles is None:
            return None

        ratio = self.config['detection']['sample_radius_ratio']
        self.color_data = []
        for i, (x, y, r) in enumerate(self.circles):
            mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
            sample_radius = int(r * ratio)
            cv2.circle(mask, (x, y), sample_radius, 255, -1)
            pixels = self.image[mask > 0]

            if len(pixels) > 0:
                avg_color_bgr = np.mean(pixels, axis=0).astype(int)
                avg_color_rgb = avg_color_bgr[::-1]
                hsv_color = cv2.cvtColor(
                    np.uint8([[avg_color_bgr]]), cv2.COLOR_BGR2HSV
                )[0][0]
                is_locked = self._detect_lock(x, y, r)
                self.color_data.append({
                    'id': i,
                    'color_bgr': tuple(avg_color_bgr.tolist()),
                    'color_rgb': tuple(avg_color_rgb.tolist()),
                    'color_hsv': tuple(hsv_color.tolist()),
                    'center': (x, y),
                    'radius': r,
                    'locked': is_locked
                })

        return self.color_data

    def _detect_lock(self, x, y, r):
        """Return True if the ball has a gray lock icon at its center."""
        lock_ratio = self.config['detection']['lock_sample_ratio']
        lock_radius = int(r * lock_ratio)
        mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (x, y), lock_radius, 255, -1)
        center_pixels = self.image[mask > 0]

        if len(center_pixels) == 0:
            return False

        gray_pixels = cv2.cvtColor(
            center_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2GRAY
        ).flatten()
        avg_gray = np.mean(gray_pixels)

        hsv_pixels = cv2.cvtColor(
            center_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV
        )
        avg_saturation = np.mean(hsv_pixels[:, 0, 1])

        threshold = self.config['detection']['lock_detect_threshold']
        return avg_gray < threshold and avg_saturation < 50

    def classify_colors(self):
        """Assign a color type index to each ball using nearest-neighbor clustering."""
        if not self.color_data:
            return None

        threshold = self.config['detection']['color_threshold']
        colors_rgb = np.array([c['color_rgb'] for c in self.color_data])
        color_types = []
        type_representatives = []

        for color in colors_rgb:
            assigned = False
            for type_idx, rep_color in enumerate(type_representatives):
                if np.linalg.norm(color - rep_color) < threshold:
                    color_types.append(type_idx)
                    assigned = True
                    break
            if not assigned:
                type_representatives.append(color)
                color_types.append(len(type_representatives) - 1)

        for i, color_type in enumerate(color_types):
            self.color_data[i]['color_type'] = color_type

        return color_types

    def find_optimal_paths(self):
        """Find all connectable paths grouped by color type."""
        if not self.color_data:
            return None

        threshold = self.config['detection']['distance_threshold']
        color_graphs = defaultdict(lambda: defaultdict(list))

        locked_count = sum(1 for c in self.color_data if c.get('locked', False))
        if locked_count > 0:
            print(f"Detected {locked_count} locked ball(s), skipping them")

        for i in range(len(self.color_data)):
            if self.color_data[i].get('locked', False):
                continue
            for j in range(i + 1, len(self.color_data)):
                if self.color_data[j].get('locked', False):
                    continue
                c1, c2 = self.color_data[i], self.color_data[j]
                if c1.get('color_type') == c2.get('color_type'):
                    x1, y1 = c1['center']
                    x2, y2 = c2['center']
                    dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    if dist <= threshold:
                        ct = c1['color_type']
                        color_graphs[ct][i].append((j, dist))
                        color_graphs[ct][j].append((i, dist))

        all_paths = []
        for color_type, graph in color_graphs.items():
            all_paths.extend(self._find_paths_for_color(graph, color_type))
        return all_paths

    def _find_paths_for_color(self, graph, color_type):
        if not graph:
            return []

        visited_global = set()
        paths = []
        nodes = list(graph.keys())
        end_nodes = [n for n in nodes if len(graph[n]) == 1]
        other_nodes = [n for n in nodes if len(graph[n]) > 1]

        for start_node in end_nodes + other_nodes:
            if start_node in visited_global:
                continue
            path = self._dfs_longest_path(graph, start_node, visited_global, set())
            if len(path) >= 2:
                total_distance = 0
                edges = []
                for i in range(len(path) - 1):
                    for neighbor, dist in graph[path[i]]:
                        if neighbor == path[i + 1]:
                            total_distance += dist
                            edges.append((path[i], path[i + 1], dist))
                            break
                paths.append({
                    'color_type': color_type,
                    'nodes': path,
                    'edges': edges,
                    'total_distance': total_distance,
                    'length': len(path)
                })
                visited_global.update(path)

        return paths

    def _dfs_longest_path(self, graph, start, visited_global, visited_current):
        visited_current.add(start)
        valid_neighbors = [
            (n, d) for n, d in graph[start]
            if n not in visited_current and n not in visited_global
        ]
        if not valid_neighbors:
            return [start]
        next_node, _ = min(valid_neighbors, key=lambda x: x[1])
        return [start] + self._dfs_longest_path(graph, next_node, visited_global, visited_current)

    def draw_connections(self, paths=None):
        """Draw detected paths and circle markers onto the image."""
        if paths is None:
            paths = self.find_optimal_paths()
        if not paths:
            return self.image.copy()

        result_img = self.image.copy()
        line_colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 128, 0)
        ]

        for path in paths:
            nodes = path['nodes']
            line_color = line_colors[path['color_type'] % len(line_colors)]
            for i in range(len(nodes) - 1):
                pos1 = self.color_data[nodes[i]]['center']
                pos2 = self.color_data[nodes[i + 1]]['center']
                cv2.line(result_img, pos1, pos2, line_color, 3)

        for color_info in self.color_data:
            x, y = color_info['center']
            r = color_info['radius']
            if color_info.get('locked', False):
                cv2.line(result_img, (x - 10, y - 10), (x + 10, y + 10), (0, 0, 255), 2)
                cv2.line(result_img, (x - 10, y + 10), (x + 10, y - 10), (0, 0, 255), 2)
                cv2.circle(result_img, (x, y), r, (0, 0, 255), 2)
            else:
                color_type = color_info.get('color_type', 0)
                cv2.circle(result_img, (x, y), r, line_colors[color_type % len(line_colors)], 1)

        return result_img
