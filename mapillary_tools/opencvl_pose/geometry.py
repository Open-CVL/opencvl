from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from scipy.spatial import cKDTree


def rotation_matrix_from_angles(
    roll: float, pitch: float, yaw: float, order: str = "XYZ"
) -> np.ndarray:
    """Construct a 3D rotation matrix from Euler angles in radians."""
    cx, sx = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(yaw), np.sin(yaw)
    rotations = {
        "X": np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]),
        "Y": np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]),
        "Z": np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]),
    }
    if sorted(order) != ["X", "Y", "Z"]:
        raise ValueError("order must contain X, Y, and Z exactly once")
    result = np.eye(3)
    for axis in order:
        result = result @ rotations[axis]
    return result


def approximate_meridian_convergence(
    latitude_deg: float, longitude_deg: float, zone_number: int
) -> float:
    """Approximate the UTM grid convergence angle in degrees."""
    latitude = np.radians(latitude_deg)
    longitude = np.radians(longitude_deg)
    central_meridian = np.radians((zone_number - 1) * 6 - 180 + 3)
    return float(
        np.degrees(np.arctan(np.tan(longitude - central_meridian) * np.sin(latitude)))
    )


def resize_with_intrinsics(
    image: np.ndarray, intrinsics: np.ndarray, scale: float
) -> tuple[np.ndarray, np.ndarray]:
    resized = cv2.resize(
        image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR
    )
    scaled = np.asarray(intrinsics, dtype=np.float64).copy()
    scaled[0, :] *= scale
    scaled[1, :] *= scale
    return resized, scaled


def center_crop_with_intrinsics(
    image: np.ndarray,
    intrinsics: np.ndarray,
    crop_height: int,
    crop_width: int,
    depth_map: np.ndarray | None = None,
    scene_map: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Center-crop an image and optional LiDAR maps while updating the principal point."""
    height, width = image.shape[:2]
    if height < crop_height or width < crop_width:
        raise ValueError(
            f"Cannot crop {crop_width}x{crop_height} from image of size {width}x{height}"
        )
    start_y = (height - crop_height) // 2
    start_x = (width - crop_width) // 2
    cropped_intrinsics = np.asarray(intrinsics, dtype=np.float64).copy()
    cropped_intrinsics[0, 2] -= start_x
    cropped_intrinsics[1, 2] -= start_y
    crop = np.s_[start_y : start_y + crop_height, start_x : start_x + crop_width]
    return (
        image[crop],
        cropped_intrinsics,
        None if depth_map is None else depth_map[crop],
        None if scene_map is None else scene_map[crop],
    )


def generate_depth_and_scene_maps(
    points_camera: np.ndarray, intrinsics: np.ndarray, image_shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize camera-frame LiDAR with a nearest-depth z-buffer."""
    height, width = image_shape
    points = np.asarray(points_camera, dtype=np.float64)
    positive = points[:, 2] > 0
    points = points[positive]
    depth_map = np.full((height, width), np.inf, dtype=np.float32)
    scene_map = np.full((height, width, 3), np.nan, dtype=np.float32)
    if len(points) == 0:
        return depth_map, scene_map

    projected = (intrinsics @ points.T).T
    uv = np.rint(projected[:, :2] / projected[:, 2:3]).astype(np.int32)
    inside = (
        (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    )
    uv, points = uv[inside], points[inside]
    if len(points) == 0:
        return depth_map, scene_map

    linear_pixel = uv[:, 1] * width + uv[:, 0]
    depth_order = np.argsort(points[:, 2])
    _, first_in_depth_order = np.unique(linear_pixel[depth_order], return_index=True)
    selected = depth_order[first_in_depth_order]
    selected_uv = uv[selected]
    selected_points = points[selected]
    depth_map[selected_uv[:, 1], selected_uv[:, 0]] = selected_points[:, 2]
    scene_map[selected_uv[:, 1], selected_uv[:, 0]] = selected_points
    return depth_map, scene_map


def associate_lidar_to_matches(
    anchor_uv: np.ndarray,
    depth_map: np.ndarray,
    scene_map: np.ndarray,
    max_distance_px: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Associate each anchor match with its nearest projected LiDAR sample."""
    valid_v, valid_u = np.where(np.isfinite(depth_map))
    if len(valid_u) == 0 or len(anchor_uv) == 0:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
            np.empty(0, dtype=np.int64),
        )
    lidar_uv = np.column_stack([valid_u, valid_v]).astype(np.float64)
    distances, nearest = cKDTree(lidar_uv).query(
        np.asarray(anchor_uv, dtype=np.float64),
        distance_upper_bound=max_distance_px,
    )
    matched = np.isfinite(distances) & (nearest < len(lidar_uv))
    match_indices = np.flatnonzero(matched)
    nearest_uv = lidar_uv[nearest[matched]].astype(np.int32)
    points_3d = scene_map[nearest_uv[:, 1], nearest_uv[:, 0]]
    return points_3d.astype(np.float32), nearest_uv.astype(np.float32), match_indices


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value
