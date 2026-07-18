from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator, Mapping

from .pose import GroundTruthPose, select_pose


LABEL_LIST_KEYS = ("labels", "images", "samples", "data")
GROUND_FIELD_CANDIDATES = ("ground_image", "mapil_image")
AERIAL_FIELD = "aerial_image"


JsonRecord = dict[str, Any]


@dataclass(frozen=True)
class LabelFile:
    """A discovered OpenCVL labels.json file and its inferred location metadata."""

    root: Path
    path: Path
    dataset: str
    split: str | None = None
    country: str | None = None
    group: str | None = None

    @property
    def relative_path(self) -> Path:
        return self.path.relative_to(self.root)


@dataclass(frozen=True)
class Sample:
    """One labeled OpenCVL ground/aerial pair."""

    label_file: LabelFile
    index: int
    label: JsonRecord
    sample_id: str
    dataset: str
    split: str | None
    country: str | None
    group: str | None
    ground_field: str | None
    ground_image: Path | None
    aerial_image: Path | None

    @property
    def label_path(self) -> Path:
        return self.label_file.path

    def pose(self) -> GroundTruthPose | None:
        return select_pose(self.label)

    def reference(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "subset": self.split,
            "country": self.country,
            "group": self.group,
            "sample_id": self.sample_id,
            "label_file": self.label_file.relative_path.as_posix(),
            "label_index": self.index,
        }


def find_dataset_root(path: str | Path) -> Path:
    """Return the OpenCVL dataset root for a root or parent-directory input."""

    root = Path(path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    if not root.is_dir():
        raise NotADirectoryError(root)

    nested = root / "OpenCVL"
    for candidate in (root, nested):
        if (candidate / "mapillary").is_dir() or (candidate / "zod").is_dir():
            return candidate
    raise ValueError(f"not an OpenCVL dataset root: {root}")


def iter_label_files(root: str | Path) -> Iterator[LabelFile]:
    root_path = find_dataset_root(root)
    paths = [
        path
        for path in sorted(root_path.rglob("labels.json"))
        if not any(part.startswith(".") for part in path.relative_to(root_path).parts)
    ]
    if not paths:
        raise FileNotFoundError(f"no labels.json files found under {root_path}")
    for path in paths:
        yield _infer_label_file(root_path, path)


def read_label_records(path: str | Path) -> list[JsonRecord]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = None
        for key in LABEL_LIST_KEYS:
            value = payload.get(key)
            if isinstance(value, list):
                records = value
                break
        if records is None:
            raise ValueError(f"unsupported labels.json object keys: {sorted(payload)}")
    else:
        raise ValueError(f"unsupported labels.json root type: {type(payload).__name__}")

    bad = [index for index, row in enumerate(records) if not isinstance(row, dict)]
    if bad:
        raise ValueError(f"labels.json contains non-object records at indexes {bad[:5]}")
    return list(records)


def iter_samples(
    root: str | Path,
    *,
    resolve_paths: bool = True,
) -> Iterator[Sample]:
    """Iterate samples from every OpenCVL labels.json file."""

    for label_file in iter_label_files(root):
        records = read_label_records(label_file.path)
        for index, record in enumerate(records):
            yield _sample_from_record(
                label_file,
                index,
                record,
                resolve_paths=resolve_paths,
            )


def find_sample_by_ground_image(root: str | Path, ground_image: str) -> Sample:
    """Find one sample by ground-image filename, stem, or relative path."""

    root_path = find_dataset_root(root)
    query = _normalize_lookup_text(ground_image)
    if not query:
        raise ValueError("ground image must not be empty")

    matches: list[tuple[LabelFile, int, JsonRecord, str]] = []
    for label_file in iter_label_files(root_path):
        for index, record in enumerate(read_label_records(label_file.path)):
            ground_field = _first_string_field(record, GROUND_FIELD_CANDIDATES)
            if ground_field is None:
                continue
            value = record[ground_field]
            if _ground_image_matches(value, query):
                matches.append((label_file, index, record, value))

    if not matches:
        raise ValueError(f"no sample matched ground image: {ground_image}")
    if len(matches) > 1:
        examples = ", ".join(
            f"{label_file.relative_path}:{value}"
            for label_file, _, _, value in matches[:5]
        )
        raise ValueError(
            f"ground image {ground_image!r} matched {len(matches)} samples; "
            f"use a longer relative path. Matches include: {examples}"
        )

    label_file, index, record, _ = matches[0]
    return _sample_from_record(label_file, index, record)


def _infer_label_file(root: Path, path: Path) -> LabelFile:
    rel = path.relative_to(root)
    parts = rel.parts
    dataset = parts[0] if parts else "unknown"
    split = parts[1] if len(parts) > 1 else None
    country = None
    group = None

    if dataset == "mapillary":
        if split == "cities" and len(parts) >= 5:
            country = parts[2]
            group = parts[3]
        elif split == "in_the_wild_test" and len(parts) >= 4:
            country = parts[2]
    elif dataset == "zod":
        if len(parts) >= 4:
            country = parts[2]
    return LabelFile(
        root=root,
        path=path,
        dataset=dataset,
        split=split,
        country=country,
        group=group,
    )


def _sample_from_record(
    label_file: LabelFile,
    index: int,
    record: JsonRecord,
    *,
    resolve_paths: bool = True,
) -> Sample:
    ground_field = _first_string_field(record, GROUND_FIELD_CANDIDATES)
    ground_path = (
        _resolve_image_path(label_file, record[ground_field])
        if resolve_paths and ground_field is not None
        else None
    )
    aerial_value = _string(record.get(AERIAL_FIELD))
    aerial_path = (
        _resolve_image_path(label_file, aerial_value)
        if resolve_paths and aerial_value
        else None
    )

    country = (_string(record.get("country")) or label_file.country)
    group = (
        label_file.group
        or _string(record.get("city"))
        or _string(record.get("drive_idx"))
        or _string(record.get("sequence_idx"))
    )
    sample_id = (
        _string(record.get("mapil_image_id"))
        or _stem_from_path(_string(record.get(ground_field)) if ground_field else None)
        or f"{label_file.relative_path}:{index}"
    )

    return Sample(
        label_file=label_file,
        index=index,
        label=record,
        sample_id=sample_id,
        dataset=label_file.dataset,
        split=label_file.split,
        country=country,
        group=group,
        ground_field=ground_field,
        ground_image=ground_path,
        aerial_image=aerial_path,
    )


def _resolve_image_path(label_file: LabelFile, value: str) -> Path:
    rel = _safe_relative_path(value)
    candidates = _path_candidates(label_file, rel)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _path_candidates(label_file: LabelFile, rel: Path) -> list[Path]:
    root = label_file.root
    parent = label_file.path.parent
    if label_file.dataset == "zod":
        raw = [root / "zod" / rel, root / rel, parent / rel]
    elif label_file.dataset == "mapillary":
        raw = [parent / rel, root / "mapillary" / rel, root / rel]
    else:
        raw = [parent / rel, root / rel]

    candidates: list[Path] = []
    seen: set[Path] = set()
    for candidate in raw:
        normalized = candidate.resolve(strict=False)
        if normalized not in seen:
            candidates.append(normalized)
            seen.add(normalized)
    return candidates


def _safe_relative_path(value: str) -> Path:
    if "\\" in value:
        raise ValueError(f"image paths must use forward slashes: {value!r}")
    posix = PurePosixPath(value)
    if posix.is_absolute() or ".." in posix.parts:
        raise ValueError(f"unsafe image path: {value!r}")
    if not posix.parts:
        raise ValueError("empty image path")
    return Path(*posix.parts)


def _first_string_field(record: Mapping[str, Any], fields: Iterable[str]) -> str | None:
    for field in fields:
        if _string(record.get(field)):
            return field
    return None


def _string(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _stem_from_path(value: str | None) -> str | None:
    if not value:
        return None
    return PurePosixPath(value).stem


def _normalize_lookup_text(value: str) -> str:
    return str(value).strip().replace("\\", "/")


def _ground_image_matches(value: str, query: str) -> bool:
    normalized = _normalize_lookup_text(value)
    if "/" in query:
        return normalized == query or normalized.endswith("/" + query)
    path = PurePosixPath(normalized)
    return query == path.name or query == path.stem
