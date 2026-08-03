import hashlib
from dataclasses import dataclass

import cv2
import numpy as np

TRAVERSABILITY_ONTOLOGY = {
    "likely_traversable": {"label": "Wahrscheinlich befahrbar", "color": "#55d96f", "value": 1},
    "limited": {"label": "Eingeschränkt oder unsicher", "color": "#e7c84d", "value": 2},
    "not_traversable": {"label": "Wahrscheinlich nicht befahrbar", "color": "#e05b52", "value": 3},
    "unknown": {"label": "Nicht bewertbar", "color": "#737c78", "value": 0},
}


def encode_mask(mask, output_width=96):
    height, width = mask.shape
    output_height = max(1, round(height * output_width / width))
    small = cv2.resize(mask.astype(np.uint8), (output_width, output_height), interpolation=cv2.INTER_NEAREST)
    flat = small.reshape(-1)
    rle = []
    if len(flat):
        previous, count = int(flat[0]), 1
        for value in flat[1:]:
            value = int(value)
            if value == previous:
                count += 1
            else:
                rle.extend([previous, count])
                previous, count = value, 1
        rle.extend([previous, count])
    return {"width": output_width, "height": output_height, "rle": rle}


def decode_mask(encoded):
    values = []
    rle = encoded["rle"]
    for index in range(0, len(rle), 2):
        values.extend([rle[index]] * rle[index + 1])
    return np.asarray(values, np.uint8).reshape(encoded["height"], encoded["width"])


def _normalised_polygon(contour, width, height, max_points=64):
    epsilon = max(1.0, 0.008 * cv2.arcLength(contour, True))
    polygon = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    while len(polygon) > max_points:
        epsilon *= 1.35
        polygon = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    return [[round(float(x / width), 5), round(float(y / height), 5)] for x, y in polygon]


def mask_regions(mask, class_id, confidence, reasons, min_area_ratio=0.0015, limit=18):
    height, width = mask.shape
    contours = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    regions = []
    for contour in contours:
        area_ratio = cv2.contourArea(contour) / max(1, width * height)
        if area_ratio < min_area_ratio:
            continue
        regions.append(
            {
                "region_id": f"{class_id}-{len(regions) + 1:02d}",
                "class_id": class_id,
                "polygon": _normalised_polygon(contour, width, height),
                "confidence": round(float(confidence), 3),
                "area_ratio": round(float(area_ratio), 5),
                "reasons": reasons,
            }
        )
        if len(regions) >= limit:
            break
    return regions


def _polygon_mask(shape, points):
    height, width = shape
    mask = np.zeros(shape, np.uint8)
    if len(points) >= 3:
        polygon = np.asarray([[round(x * width), round(y * height)] for x, y in points], np.int32)
        cv2.fillPoly(mask, [polygon], 255)
    return mask


def _component_connected_to_seed(mask, seed):
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    best_label, best_score, anchored = 0, -1.0, False
    for label in range(1, count):
        component = labels == label
        area = int(stats[label, cv2.CC_STAT_AREA])
        seed_overlap = int(np.count_nonzero(component & (seed > 0)))
        centre_x = float(centroids[label, 0]) / max(1, mask.shape[1])
        score = area + seed_overlap * 24 - abs(centre_x - 0.5) * area * 0.45
        if seed_overlap > 0 and score > best_score:
            best_label, best_score, anchored = label, score, True
    if best_label == 0:
        return np.zeros_like(mask), False
    return np.where(labels == best_label, 255, 0).astype(np.uint8), anchored


def _horizontal_clearance(mask):
    """Half-width to the nearest lateral boundary inside each free row."""
    free = mask > 0
    height, width = free.shape
    left = np.zeros((height, width), np.float32)
    right = np.zeros((height, width), np.float32)
    running = np.zeros(height, np.float32)
    for x in range(width):
        running = np.where(free[:, x], running + 1, 0)
        left[:, x] = running
    running.fill(0)
    for x in range(width - 1, -1, -1):
        running = np.where(free[:, x], running + 1, 0)
        right[:, x] = running
    return np.minimum(left, right)


@dataclass
class VehicleConfiguration:
    width_m: float = 0.35
    safety_margin_per_side_m: float = 0.20
    source: str = "configured_default"

    @property
    def required_width_m(self):
        return self.width_m + 2 * self.safety_margin_per_side_m


class TerrainAnalyzer:
    def __init__(self, vehicle=None, near_field_width_m=3.2, metric_calibration="perspective_estimate"):
        self.vehicle = vehicle or VehicleConfiguration()
        self.near_field_width_m = near_field_width_m
        self.metric_calibration = metric_calibration
        self.previous_centerline = []
        self.stable_frames = 0

    def _roi(self, shape):
        height, width = shape
        mask = np.zeros(shape, np.uint8)
        bottom_left, bottom_right = (0.04, 0.96) if height > width else (0.08, 0.92)
        polygon = np.asarray(
            [
                [width * 0.42, height * 0.34],
                [width * 0.58, height * 0.34],
                [width * bottom_right, height * 0.98],
                [width * bottom_left, height * 0.98],
            ],
            np.int32,
        )
        cv2.fillConvexPoly(mask, polygon, 255)
        return mask

    def _centre_prior(self, shape):
        height, width = shape
        mask = np.zeros(shape, np.uint8)
        polygon = np.asarray(
            [
                [width * 0.46, height * 0.40],
                [width * 0.54, height * 0.40],
                [width * 0.76, height],
                [width * 0.24, height],
            ],
            np.int32,
        )
        cv2.fillConvexPoly(mask, polygon, 255)
        return mask

    def _obstacle_mask(self, shape, vegetation, detections, balanced):
        height, width = shape
        objects = np.zeros(shape, np.uint8)
        for detection in detections:
            if detection["class_id"] not in {"tree", "shrub", "vegetation_cluster", "unknown_obstacle"}:
                continue
            objects = cv2.bitwise_or(objects, _polygon_mask(shape, detection.get("polygon", [])))
        objects = cv2.dilate(objects, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        gray = cv2.cvtColor(balanced, cv2.COLOR_BGR2GRAY)
        vertical_gradient = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
        threshold = max(45, int(np.percentile(vertical_gradient, 88)))
        vertical = np.where(vertical_gradient >= threshold, 255, 0).astype(np.uint8)
        vertical = cv2.morphologyEx(
            vertical, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, max(11, round(height * 0.035))))
        )
        vertical = cv2.dilate(vertical, np.ones((7, 7), np.uint8))
        dense_vegetation = cv2.morphologyEx(vegetation, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
        structured_vegetation = cv2.bitwise_and(dense_vegetation, vertical)
        obstacle = cv2.bitwise_or(objects, structured_vegetation)
        return obstacle, vertical_gradient

    def _ground_mask(self, balanced, vegetation, obstacle, vertical_gradient, roi, source_image=None):
        height, width = roi.shape
        source_image = source_image if source_image is not None and source_image.shape == balanced.shape else balanced
        gray = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
        saturation = cv2.cvtColor(source_image, cv2.COLOR_BGR2HSV)[:, :, 1]
        lab = cv2.cvtColor(balanced, cv2.COLOR_BGR2LAB).astype(np.float32)
        # Direct clipping and broad low-chroma highlights are not observable
        # terrain. Detect them on the unenhanced source frame so CLAHE cannot
        # turn glare into apparently detailed ground evidence.
        glare = (gray >= 170) & (saturation <= 28)
        extreme = np.where((gray < 16) | (gray > 248) | glare, 255, 0).astype(np.uint8)
        extreme = cv2.dilate(extreme, np.ones((5, 5), np.uint8))
        centre_prior = self._centre_prior((height, width))
        seed = np.zeros_like(roi)
        seed[round(height * 0.69) : round(height * 0.96)] = centre_prior[round(height * 0.69) : round(height * 0.96)]
        valid_seed = (seed > 0) & (obstacle == 0) & (extreme == 0)
        prototypes = []
        bottom_samples = lab[valid_seed & (np.indices(roi.shape)[0] > height * 0.80)]
        middle_samples = lab[valid_seed & (np.indices(roi.shape)[0] <= height * 0.80)]
        if len(bottom_samples) >= 40:
            prototypes.append(np.median(bottom_samples, axis=0))
        if len(middle_samples) >= 40:
            prototypes.append(np.median(middle_samples, axis=0))
        if not prototypes:
            samples = lab[(roi > 0) & (obstacle == 0) & (extreme == 0)]
            prototypes.append(np.median(samples, axis=0) if len(samples) else np.asarray([128, 128, 128], np.float32))
        distances = []
        scales = np.asarray([1.45, 1.0, 1.0], np.float32)
        for prototype in prototypes:
            distances.append(np.linalg.norm((lab - prototype) / scales, axis=2))
        colour_distance = np.min(np.stack(distances), axis=0)
        colour_score = np.exp(-np.square(colour_distance / 58.0))
        gradient_scale = max(55.0, float(np.percentile(vertical_gradient, 95)))
        non_vertical = 1 - np.clip(vertical_gradient.astype(np.float32) / gradient_scale, 0, 1)
        position_score = np.where(centre_prior > 0, 1.0, 0.52)
        visibility_pixel = np.where(extreme > 0, 0.0, 1.0)
        vegetation_penalty = np.where(vegetation > 0, 0.72, 1.0)
        score = (
            0.47 * colour_score
            + 0.20 * non_vertical
            + 0.17 * position_score
            + 0.10 * visibility_pixel
            + 0.06 * vegetation_penalty
        )
        candidate = np.where((score >= 0.40) & (roi > 0) & (obstacle == 0) & (extreme == 0), 255, 0).astype(np.uint8)
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        ground, anchored = _component_connected_to_seed(candidate, valid_seed.astype(np.uint8) * 255)
        connectivity = cv2.countNonZero(ground) / max(1, cv2.countNonZero(candidate))
        return ground, score, extreme, valid_seed.astype(np.uint8) * 255, anchored, float(connectivity)

    def _horizontal_step_mask(self, balanced, ground):
        gray = cv2.cvtColor(balanced, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 65, 155)
        width = gray.shape[1]
        horizontal = cv2.morphologyEx(
            edges, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(17, round(width * 0.055)), 3))
        )
        horizontal = cv2.dilate(horizontal, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 7)))
        return cv2.bitwise_and(horizontal, ground)

    def _warp_previous_line(self, motion, shape, motion_inliers):
        if not self.previous_centerline or motion_inliers < 30:
            return []
        points = np.asarray([self.previous_centerline], np.float32)
        warped = cv2.transform(points, motion)[0]
        height, width = shape
        return [(float(np.clip(x, 0, width - 1)), float(np.clip(y, 0, height - 1))) for x, y in warped]

    def _corridor(
        self, drive_component, distance, motion, motion_inliers, visibility_score, ground_confidence, timestamp_ms
    ):
        height, width = drive_component.shape
        previous = self._warp_previous_line(motion, (height, width), motion_inliers)

        def previous_x(y):
            if not previous:
                return None
            return min(previous, key=lambda point: abs(point[1] - y))[0]

        # Dense rows make the polygon follow current-frame free-space edges
        # instead of cutting across an obstacle between sparse samples.
        rows = list(range(round(height * 0.92), round(height * 0.43), -max(3, round(height * 0.012))))
        points, half_widths, clearance_ratios, raw_shifts = [], [], [], []
        spatial_x = width * 0.5
        for y in rows:
            y1, y2 = max(0, y - 2), min(height, y + 3)
            xs = np.where(np.any(drive_component[y1:y2] > 0, axis=0))[0]
            if not len(xs):
                if len(points) >= 3:
                    break
                continue
            perspective = 0.24 + 0.76 * np.clip((y / height - 0.34) / 0.64, 0, 1)
            pixels_per_m = width / self.near_field_width_m * perspective
            required_half = self.vehicle.required_width_m * pixels_per_m / 2
            clearances = distance[y, xs]
            eligible = clearances >= max(2.0, required_half * 0.62)
            if not np.any(eligible):
                if len(points) >= 3:
                    break
                continue
            xs, clearances = xs[eligible], clearances[eligible]
            temporal_x = previous_x(y)
            scores = clearances - 0.18 * np.abs(xs - spatial_x)
            if temporal_x is not None:
                scores -= 0.28 * np.abs(xs - temporal_x)
            chosen_index = int(np.argmax(scores))
            raw_x = float(xs[chosen_index])
            if temporal_x is not None:
                chosen_x = 0.65 * raw_x + 0.35 * temporal_x
            else:
                chosen_x = raw_x
            spatial_weight = 0.10 if temporal_x is not None else 0.28
            chosen_x = (1 - spatial_weight) * chosen_x + spatial_weight * spatial_x
            if temporal_x is not None:
                raw_shifts.append(abs(chosen_x - temporal_x))
            clearance = float(distance[y, int(np.clip(round(chosen_x), 0, width - 1))])
            clearance_ratio = clearance / max(1.0, required_half)
            # The corridor may end where the reliable near-field becomes too
            # narrow. Do not let that single terminal/horizon row downgrade an
            # otherwise sufficiently long, connected corridor; an actual
            # bottleneck inside the accepted corridor still remains limiting.
            if clearance_ratio < 1.0 and len(points) >= 7:
                break
            # Render an inner envelope; the remaining 4% is a rasterisation
            # guard so polygon edges cannot touch a red/unknown boundary after
            # normalisation and client-side scaling.
            usable_half = min(required_half * 0.96, clearance * 0.84)
            points.append((chosen_x, float(y)))
            half_widths.append(usable_half)
            clearance_ratios.append(clearance_ratio)
            spatial_x = chosen_x
        if len(points) < 4:
            self.previous_centerline = []
            self.stable_frames = 0
            return {
                "status": "unavailable",
                "polygon": [],
                "centerline": [],
                "confidence": 0.0,
                "minimum_width_ratio": 0.0,
                "minimum_width_m": None,
                "stability_px": 0.0,
                "stable_frames": 0,
                "source_frame_timestamp_ms": timestamp_ms,
                "temporally_smoothed": False,
                "reasons": ["no_connected_free_corridor"],
            }
        mean_shift = float(np.mean(raw_shifts)) if raw_shifts else 0.0
        if previous and mean_shift <= width * 0.045:
            self.stable_frames += 1
        else:
            self.stable_frames = 1
        self.previous_centerline = points
        minimum_ratio = float(min(clearance_ratios))
        minimum_width_estimate = self.vehicle.required_width_m * minimum_ratio
        confidence = float(
            np.clip(
                0.30 * visibility_score
                + 0.30 * ground_confidence
                + 0.28 * min(1, minimum_ratio)
                + 0.12 * min(1, self.stable_frames / 3),
                0,
                1,
            )
        )
        available = minimum_ratio >= 1.0 and confidence >= 0.58 and len(points) >= 7
        status = "available" if available else "uncertain"
        ordered = list(reversed(points))
        ordered_widths = list(reversed(half_widths))
        left = [
            [round(max(0, x - half) / width, 5), round(y / height, 5)] for (x, y), half in zip(ordered, ordered_widths)
        ]
        right = [
            [round(min(width - 1, x + half) / width, 5), round(y / height, 5)]
            for (x, y), half in reversed(list(zip(ordered, ordered_widths)))
        ]
        reasons = []
        if minimum_ratio < 1:
            reasons.append("width_below_required")
        if visibility_score < 0.7:
            reasons.append("limited_visibility")
        if self.metric_calibration != "calibrated":
            reasons.append("metric_scale_estimated")
        if not reasons:
            reasons.append("connected_clear_ground")
        return {
            "status": status,
            "polygon": left + right,
            "centerline": [[round(x / width, 5), round(y / height, 5)] for x, y in ordered],
            "confidence": round(confidence, 3),
            "minimum_width_ratio": round(minimum_ratio, 3),
            "minimum_width_m": round(minimum_width_estimate, 3),
            "stability_px": round(mean_shift, 2),
            "stable_frames": self.stable_frames,
            "source_frame_timestamp_ms": timestamp_ms,
            "temporally_smoothed": bool(previous),
            "reasons": reasons,
        }

    def analyze(self, balanced, vegetation, detections, motion, motion_inliers, timestamp_ms, source_image=None):
        height, width = vegetation.shape
        source_image = source_image if source_image is not None and source_image.shape == balanced.shape else balanced
        roi = self._roi((height, width))
        obstacle, vertical_gradient = self._obstacle_mask((height, width), vegetation, detections, balanced)
        ground, ground_score, extreme, seed, anchored, connectivity = self._ground_mask(
            balanced, vegetation, obstacle, vertical_gradient, roi, source_image
        )
        gray = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        blur_score = float(np.clip((1800 - sharpness) / 1350, 0, 1))
        exposure_bad = cv2.countNonZero(cv2.bitwise_and(extreme, roi)) / max(1, cv2.countNonZero(roi))
        visibility_score = float(np.clip(0.78 * (1 - min(1, exposure_bad * 2.4)) + 0.22 * (1 - blur_score), 0, 1))
        step_mask = self._horizontal_step_mask(balanced, ground)
        red = cv2.bitwise_and(
            cv2.bitwise_or(obstacle, step_mask), cv2.bitwise_or(roi, cv2.dilate(obstacle, np.ones((3, 3), np.uint8)))
        )
        free_ground = cv2.bitwise_and(ground, cv2.bitwise_not(red))
        # A structureless frame cannot support a traversability decision. Keep
        # independently visible obstacles red, but make the remaining driving
        # ROI unknown and suppress the corridor entirely.
        severe_blur = blur_score >= 0.82
        if severe_blur:
            free_ground[:] = 0
        drive_component, drive_anchored = _component_connected_to_seed(free_ground, seed)
        # Vehicle fit is a lateral-width question. A regular 2D distance
        # transform also measures distance to the lower image boundary and can
        # therefore invent a bottleneck directly in front of the camera.
        distance = _horizontal_clearance(drive_component)
        local_mean = cv2.boxFilter(gray.astype(np.float32), -1, (11, 11))
        local_sq = cv2.boxFilter(np.square(gray.astype(np.float32)), -1, (11, 11))
        roughness = np.clip((np.sqrt(np.maximum(0, local_sq - np.square(local_mean))) - 6) / 28, 0, 1)
        vegetation_density = cv2.blur((vegetation > 0).astype(np.float32), (25, 25))
        rows = np.indices((height, width))[0]
        perspective = 0.24 + 0.76 * np.clip((rows / height - 0.34) / 0.64, 0, 1)
        required_half = self.vehicle.required_width_m * (width / self.near_field_width_m) * perspective / 2
        width_ratio_map = distance / np.maximum(1.0, required_half)
        ground_values = ground_score[ground > 0]
        ground_confidence = (
            float(np.mean(ground_values) * visibility_score * (0.55 + 0.45 * connectivity))
            if len(ground_values)
            else 0.0
        )
        current_supported = (ground_score >= 0.50) & (extreme == 0) & (drive_component > 0)
        green_gate = (
            current_supported
            & (width_ratio_map >= 1.0)
            & (roughness <= 0.70)
            & (vegetation_density <= 0.60)
            & drive_anchored
        )
        if ground_confidence < 0.46 or visibility_score < 0.62 or blur_score > 0.72:
            green_gate[:] = False
        green = np.where(green_gate, 255, 0).astype(np.uint8)
        # Green must itself be one continuous, near-field-anchored area. Any
        # detached high-score island remains yellow instead of suggesting an
        # isolated patch can be reached safely.
        green, _ = _component_connected_to_seed(green, seed)
        yellow = cv2.bitwise_and(drive_component, cv2.bitwise_not(green))
        class_map = np.zeros((height, width), np.uint8)
        class_map[green > 0] = TRAVERSABILITY_ONTOLOGY["likely_traversable"]["value"]
        class_map[yellow > 0] = TRAVERSABILITY_ONTOLOGY["limited"]["value"]
        class_map[red > 0] = TRAVERSABILITY_ONTOLOGY["not_traversable"]["value"]
        corridor_component = cv2.erode(drive_component, np.ones((5, 5), np.uint8))
        corridor_clearance = _horizontal_clearance(corridor_component)
        corridor = self._corridor(
            corridor_component,
            corridor_clearance,
            motion,
            motion_inliers,
            visibility_score,
            ground_confidence,
            timestamp_ms,
        )
        corridor_samples = []
        for x, y in corridor["centerline"]:
            pixel_x = int(np.clip(round(x * width), 0, width - 1))
            pixel_y = int(np.clip(round(y * height), 0, height - 1))
            corridor_samples.append(
                class_map[pixel_y, pixel_x] == TRAVERSABILITY_ONTOLOGY["likely_traversable"]["value"]
            )
        green_support = float(np.mean(corridor_samples)) if corridor_samples else 0.0
        corridor["green_support_fraction"] = round(green_support, 3)
        if corridor["status"] == "available" and green_support < 0.70:
            corridor["status"] = "uncertain"
            corridor["reasons"].append("insufficient_green_surface_support")
        if corridor["status"] == "available" and (ground_confidence < 0.50 or visibility_score < 0.68):
            corridor["status"] = "uncertain"
            corridor["reasons"].append("confidence_gate")
        roi_pixels = max(1, cv2.countNonZero(roi))
        # Coverage is a property of the assessed driving ROI. Obstacles are still
        # rendered outside it for context, but must never inflate an ROI fraction
        # beyond 100 percent.
        roi_bool = roi > 0
        coverages = {
            class_id: round(float(np.count_nonzero((class_map == item["value"]) & roi_bool) / roi_pixels), 4)
            for class_id, item in TRAVERSABILITY_ONTOLOGY.items()
        }
        obstacle_ratio = coverages["not_traversable"]
        ground_roughness = float(np.mean(roughness[ground > 0])) if cv2.countNonZero(ground) else 1.0
        minimum_ratio = corridor["minimum_width_ratio"]
        factors = {
            "free_width_score": round(float(np.clip(minimum_ratio / 1.15, 0, 1)), 3),
            "obstacle_clearance_score": round(float(np.clip(1 - obstacle_ratio * 3.0, 0, 1)), 3),
            "connectivity_score": round(float(connectivity if anchored else 0), 3),
            "smoothness_score": round(float(1 - ground_roughness), 3),
            "bottleneck_clearance_score": round(float(np.clip(minimum_ratio, 0, 1)), 3),
            "visibility_score": round(visibility_score, 3),
            "calibration_score": 1.0 if self.metric_calibration == "calibrated" else 0.45,
            "temporal_stability_score": round(
                float(np.clip(1 - corridor["stability_px"] / max(1, width * 0.08), 0, 1)), 3
            ),
        }
        if corridor["status"] == "available":
            overall_class = "likely_traversable"
        elif corridor["status"] == "uncertain" or coverages["limited"] >= 0.035:
            overall_class = "limited"
        elif obstacle_ratio >= 0.08:
            overall_class = "not_traversable"
        else:
            overall_class = "unknown"
        overall_confidence = float(
            np.clip(
                np.mean(
                    [
                        ground_confidence,
                        visibility_score,
                        factors["connectivity_score"],
                        factors["obstacle_clearance_score"],
                    ]
                ),
                0,
                1,
            )
        )
        reason_map = {
            "likely_traversable": ["connected_ground", "clearance_above_required", "current_frame_support"],
            "limited": ["limited_clearance_or_surface_uncertainty"],
            "not_traversable": ["visible_obstacle_or_step_evidence"],
            "unknown": ["insufficient_visible_ground_evidence"],
        }
        regions = []
        for class_id, item in TRAVERSABILITY_ONTOLOGY.items():
            class_mask = np.where(class_map == item["value"], 255, 0).astype(np.uint8)
            if class_id == "unknown":
                class_mask = cv2.bitwise_and(class_mask, roi)
            regions.extend(mask_regions(class_mask, class_id, overall_confidence, reason_map[class_id]))
        source_hash = hashlib.sha256(source_image.tobytes()).hexdigest()[:20]
        result = {
            "source_frame_timestamp_ms": timestamp_ms,
            "source_frame_hash": source_hash,
            "ground": {
                "mask": encode_mask(np.where(ground > 0, 1, 0).astype(np.uint8)),
                "regions": mask_regions(ground, "ground", ground_confidence, ["frame_colour_texture_connectivity"]),
                "confidence": round(ground_confidence, 3),
                "visible_ratio": round(float(cv2.countNonZero(ground) / max(1, width * height)), 4),
                "source": "current_video_frame_inference",
            },
            "traversability": {
                "mask": encode_mask(class_map),
                "regions": regions,
                "class_coverage": coverages,
                "overall_class": overall_class,
                "overall_confidence": round(overall_confidence, 3),
            },
            "corridor": corridor,
            "factors": factors,
            "quality": {
                "blur_score": round(blur_score, 3),
                "exposure_score": round(float(1 - exposure_bad), 3),
                "motion_inliers": motion_inliers,
                "unknown_ratio": coverages["unknown"],
            },
            "evidence": {"representative": False, "reasons": []},
        }
        return result, {"class_map": class_map, "ground": ground}


def render_evidence(image, terrain, class_map):
    height, width = image.shape[:2]
    resized = cv2.resize(class_map, (width, height), interpolation=cv2.INTER_NEAREST)
    colours = {
        0: (120, 124, 120),
        1: (80, 215, 105),
        2: (70, 200, 235),
        3: (70, 75, 225),
    }
    overlay = image.copy()
    for value, colour in colours.items():
        overlay[resized == value] = colour
    composed = cv2.addWeighted(image, 0.55, overlay, 0.45, 0)
    corridor = terrain["corridor"]
    if corridor["centerline"]:
        points = np.asarray([[round(x * width), round(y * height)] for x, y in corridor["centerline"]], np.int32)
        colour = (235, 245, 80) if corridor["status"] == "available" else (30, 210, 245)
        cv2.polylines(composed, [points], False, colour, max(3, round(width / 220)), cv2.LINE_AA)
    label = f"{terrain['traversability']['overall_class']}  conf={terrain['traversability']['overall_confidence']:.2f}"
    cv2.rectangle(composed, (12, 12), (min(width - 12, 560), 52), (8, 12, 10), -1)
    cv2.putText(composed, label, (22, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (240, 245, 240), 2, cv2.LINE_AA)
    return composed
