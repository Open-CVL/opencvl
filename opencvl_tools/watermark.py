from __future__ import annotations

from pathlib import Path
from typing import Any

from .dataset import Sample, find_dataset_root


AERIAL_SOURCE_BY_COUNTRY = {
    "SE": "Lantmäteriet",
    "NL": "Beeldmateriaal.nl / PDOK",
    "PL": "GUGiK",
    "NO": "Kartverket",
}

MAPILLARY_USERNAME_FIELDS = (
    "mapillary_uploader",
    "mapillary_creator_username",
    "creator_username",
)


def source_label(sample: Sample, image_type: str) -> str:
    if image_type == "ground":
        if sample.dataset == "zod":
            return "Zenseact AB"
        for field in MAPILLARY_USERNAME_FIELDS:
            username = sample.label.get(field)
            if isinstance(username, str) and username.strip():
                username = username.strip().lstrip("@").strip()
                if username:
                    return f"Mapillary / {username}"
        return "Mapillary"
    if image_type != "aerial":
        raise ValueError(f"unknown image type: {image_type}")
    for key in ("aerial_attribution", "aerial_provider", "aerial_source"):
        value = sample.label.get(key)
        if isinstance(value, str):
            attribution = value.removeprefix("Aerial source:").strip()
            if attribution:
                return attribution
    return AERIAL_SOURCE_BY_COUNTRY.get(
        (sample.country or "").upper(),
        "National open mapping data",
    )


def watermark_sample(
    root: str | Path,
    sample: Sample,
    output: str | Path,
) -> tuple[Path, ...]:
    """Write watermarked copies for one OpenCVL sample."""

    root_path = find_dataset_root(root)
    output_path = Path(output).expanduser().resolve()
    if output_path == root_path:
        raise ValueError("refusing to watermark in place; choose a separate output directory")
    try:
        output_path.relative_to(root_path)
    except ValueError:
        pass
    else:
        raise ValueError("watermark output must be outside the source dataset tree")
    output_path.mkdir(parents=True, exist_ok=True)

    seen: set[Path] = set()
    written: list[Path] = []
    for image_type in ("ground", "aerial"):
        source = sample.ground_image if image_type == "ground" else sample.aerial_image
        if source is None:
            continue
        source = source.resolve()
        if source in seen:
            continue
        seen.add(source)
        try:
            relative = source.relative_to(root_path)
        except ValueError as exc:
            raise ValueError(f"referenced image is outside the dataset root: {source}") from exc
        target = output_path / relative
        written.append(
            add_watermark(
                source,
                target,
                source_label(sample, image_type),
            )
        )
    return tuple(written)


def add_watermark(
    source: str | Path,
    output: str | Path,
    text: str,
) -> Path:
    """Write one source image with an attribution watermark."""

    Image, _, _, ImageOps = _require_pillow()

    source_path = Path(source).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    with Image.open(source_path) as opened:
        source_format = opened.format
        icc_profile = opened.info.get("icc_profile")
        transposed = ImageOps.exif_transpose(opened)
        exif = transposed.getexif()
        exif_bytes = exif.tobytes() if exif else None
        base = transposed.convert("RGBA")

    result = watermark_image(base, text)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_format = _format_for_output(Image, output_path, source_format)
    save_args: dict[str, Any] = {}
    if exif_bytes:
        save_args["exif"] = exif_bytes
    if icc_profile:
        save_args["icc_profile"] = icc_profile
    if output_format == "JPEG":
        save_args.update(quality=95, optimize=True)
    result.save(output_path, format=output_format, **save_args)
    return output_path


def watermark_image(
    image: Any,
    text: str,
) -> Any:
    """Return an RGB image with a source label."""

    Image, ImageDraw, ImageFont, _ = _require_pillow()
    text = text.strip()
    if not text:
        raise ValueError("watermark text must not be empty")

    base = image.convert("RGBA")
    margin = max(3, int(round(base.width * 0.008)))
    max_width = base.width - margin * 2
    start_size = max(14, int(round(base.width * 0.045)))
    font, font_size = _fit_font(
        Image,
        ImageFont,
        ImageDraw,
        text,
        max_width,
        start_size,
    )
    stroke_width = max(1, int(round(font_size * 0.045)))
    probe = ImageDraw.Draw(base)
    left, top, right, bottom = probe.textbbox(
        (0, 0),
        text,
        font=font,
        stroke_width=stroke_width,
    )
    x = base.width - margin - right
    y = base.height - margin - bottom

    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.text(
        (x, y),
        text,
        font=font,
        fill=(255, 255, 255, 255),
        stroke_width=stroke_width,
        stroke_fill=(0, 0, 0, 255),
    )
    return Image.alpha_composite(base, overlay).convert("RGB")


def _fit_font(
    Image: Any,
    ImageFont: Any,
    ImageDraw: Any,
    text: str,
    max_width: int,
    start_size: int,
) -> tuple[Any, int]:
    for size in range(start_size, 9, -1):
        font = _load_font(ImageFont, size)
        probe = ImageDraw.Draw(Image.new("L", (1, 1)))
        stroke_width = max(1, int(round(size * 0.045)))
        left, _, right, _ = probe.textbbox(
            (0, 0),
            text,
            font=font,
            stroke_width=stroke_width,
        )
        if right - left <= max_width:
            return font, size
    return _load_font(ImageFont, 9), 9


def _load_font(ImageFont: Any, size: int) -> Any:
    for path in (
        "DejaVuSerif.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/urw-base35/NimbusRoman-Regular.otf",
        "/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        "/System/Library/Fonts/NewYork.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _format_for_output(Image: Any, path: Path, source_format: str | None) -> str:
    if path.suffix:
        output_format = Image.registered_extensions().get(path.suffix.lower())
        if output_format is None:
            raise ValueError(f"unsupported output image extension: {path.suffix}")
        return output_format
    if source_format:
        return source_format
    raise ValueError(f"cannot infer image format for {path}")


def _require_pillow() -> tuple[Any, Any, Any, Any]:
    try:
        from PIL import Image, ImageDraw, ImageFont, ImageOps
    except ImportError as exc:
        raise ImportError(
            'Watermarking requires Pillow. Run: python3 -m pip install "Pillow>=9.2"'
        ) from exc
    return Image, ImageDraw, ImageFont, ImageOps
