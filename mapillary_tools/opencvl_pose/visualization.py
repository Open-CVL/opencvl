from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def save_correspondences(
    anchor: np.ndarray,
    mapillary: np.ndarray,
    anchor_uv: np.ndarray,
    mapillary_uv: np.ndarray,
    output_path: Path,
    max_matches: int = 24,
) -> None:
    """Save a compact visualization of a spatially distributed match subset."""
    canvas = np.concatenate([anchor, mapillary], axis=1)
    if len(anchor_uv):
        selected = np.linspace(
            0, len(anchor_uv) - 1, min(max_matches, len(anchor_uv)), dtype=int
        )
        for order, index in enumerate(selected):
            hue = int(179 * order / max(1, len(selected) - 1))
            color_bgr = cv2.cvtColor(np.uint8([[[hue, 190, 255]]]), cv2.COLOR_HSV2BGR)[
                0, 0
            ]
            color = tuple(int(value) for value in color_bgr[::-1])
            start = tuple(np.rint(anchor_uv[index]).astype(int))
            end_array = np.rint(mapillary_uv[index]).astype(int)
            end = (int(end_array[0] + anchor.shape[1]), int(end_array[1]))
            cv2.line(canvas, start, end, color, 2, cv2.LINE_AA)
            cv2.circle(canvas, start, 4, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, end, 4, color, -1, cv2.LINE_AA)
    Image.fromarray(canvas).save(output_path)
