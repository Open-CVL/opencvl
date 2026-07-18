from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .dataset import Sample
from .pose import GroundTruthPose
from .watermark import source_label, watermark_image


PANEL_WIDTH = 720


def render_sample_pair(
    sample: Sample,
    output: str | Path,
) -> Path:
    """Render a ground/aerial pair with its ground-truth pose."""

    Image, ImageDraw, ImageOps = _require_pillow()
    if sample.ground_image is None or sample.aerial_image is None:
        raise ValueError(f"sample {sample.sample_id} is missing an image path")

    with Image.open(sample.ground_image) as source:
        ground = ImageOps.exif_transpose(source).convert("RGB")
    with Image.open(sample.aerial_image) as source:
        aerial = ImageOps.exif_transpose(source).convert("RGB")
    ground = watermark_image(ground, source_label(sample, "ground"))
    aerial = watermark_image(aerial, source_label(sample, "aerial"))
    pose = sample.pose()
    if pose is None:
        raise ValueError(f"sample {sample.sample_id} has no complete GT pose")

    canvas = _render_pair(
        Image,
        ImageDraw,
        ground,
        aerial,
        pose,
    )

    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def _render_pair(
    Image: Any,
    ImageDraw: Any,
    ground: Any,
    aerial: Any,
    pose: GroundTruthPose,
) -> Any:
    ground_panel = _resize_to_width(Image, ground, PANEL_WIDTH)
    aerial_panel = _resize_to_width(Image, aerial, PANEL_WIDTH)
    x, y = pose.pixel(*aerial.size)
    x *= aerial_panel.width / aerial.width
    y *= aerial_panel.height / aerial.height
    _draw_pose(
        ImageDraw.Draw(aerial_panel),
        pose,
        x,
        y,
    )

    border = max(1, int(round(PANEL_WIDTH / 360)))
    gap = max(2, int(round(PANEL_WIDTH * 0.012)))
    padding = max(5, int(round(PANEL_WIDTH * 0.022)))
    ground_frame = _add_border(Image, ground_panel, border)
    aerial_frame = _add_border(Image, aerial_panel, border)
    width = max(ground_frame.width, aerial_frame.width) + padding * 2
    height = ground_frame.height + aerial_frame.height + gap + padding * 2
    canvas = Image.new("RGB", (width, height), "white")
    ground_left = (width - ground_frame.width) // 2
    aerial_left = (width - aerial_frame.width) // 2
    canvas.paste(ground_frame, (ground_left, padding))
    canvas.paste(aerial_frame, (aerial_left, padding + ground_frame.height + gap))
    return canvas


def _resize_to_width(Image: Any, image: Any, width: int) -> Any:
    height = max(1, int(round(image.height * width / image.width)))
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    return image.resize((width, height), resampling)


def _add_border(Image: Any, image: Any, width: int) -> Any:
    framed = Image.new(
        "RGB",
        (image.width + width * 2, image.height + width * 2),
        (20, 20, 20),
    )
    framed.paste(image, (width, width))
    return framed


def _draw_pose(
    draw: Any,
    pose: GroundTruthPose,
    x: float,
    y: float,
) -> None:
    theta = math.radians(pose.heading_degrees)
    direction_x = math.sin(theta)
    direction_y = -math.cos(theta)
    length = max(27, int(round(PANEL_WIDTH * 0.135)))
    line_width = max(3, int(round(PANEL_WIDTH * 0.011)))
    marker_border_width = max(2, int(round(PANEL_WIDTH * 0.005)))
    marker_radius = max(7, int(round(PANEL_WIDTH * 0.022)))
    head_length = max(9, int(round(PANEL_WIDTH * 0.033)))
    head_width = head_length * 0.38
    end_x = x + direction_x * length
    end_y = y + direction_y * length
    shaft_end_x = end_x - direction_x * head_length
    shaft_end_y = end_y - direction_y * head_length
    color = (255, 20, 20)
    draw.line((x, y, shaft_end_x, shaft_end_y), fill=color, width=line_width)
    draw.polygon(
        _arrowhead_points(
            end_x,
            end_y,
            direction_x,
            direction_y,
            head_length,
            head_width,
        ),
        fill=color,
    )
    draw.polygon(
        _location_triangle(x, y, marker_radius + marker_border_width),
        fill="white",
    )
    draw.polygon(_location_triangle(x, y, marker_radius), fill=color)


def _arrowhead_points(
    end_x: float,
    end_y: float,
    direction_x: float,
    direction_y: float,
    length: float,
    half_width: float,
) -> tuple[tuple[float, float], ...]:
    base_x = end_x - direction_x * length
    base_y = end_y - direction_y * length
    perpendicular_x = -direction_y
    perpendicular_y = direction_x
    return (
        (end_x, end_y),
        (
            base_x + perpendicular_x * half_width,
            base_y + perpendicular_y * half_width,
        ),
        (
            base_x - perpendicular_x * half_width,
            base_y - perpendicular_y * half_width,
        ),
    )


def _location_triangle(
    x: float,
    y: float,
    radius: float,
) -> tuple[tuple[float, float], ...]:
    return (
        (x, y - radius),
        (x - radius * 0.88, y + radius * 0.72),
        (x + radius * 0.88, y + radius * 0.72),
    )


def _require_pillow() -> tuple[Any, Any, Any]:
    try:
        from PIL import Image, ImageDraw, ImageOps
    except ImportError as exc:
        raise ImportError(
            'Plotting requires Pillow. Run: python3 -m pip install "Pillow>=9.2"'
        ) from exc
    return Image, ImageDraw, ImageOps
