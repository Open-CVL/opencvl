from __future__ import annotations

import json
import os
from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .dataset import Sample, find_dataset_root, iter_samples
from .release import ReleaseProfile, load_release_profile


@dataclass(frozen=True)
class SplitBuildSummary:
    output: Path
    counts: Counter[str]
    by_source: Counter[str]
    expected_official_counts: dict[str, int]
    files: dict[str, Path]

    def to_dict(self) -> dict[str, Any]:
        return {
            "counts": dict(sorted(self.counts.items())),
            "by_source": dict(sorted(self.by_source.items())),
            "expected_official_counts": self.expected_official_counts,
            "files": {key: path.name for key, path in self.files.items()},
        }


class SplitAssigner:
    """Assign release label rows to the official OpenCVL benchmark splits."""

    def __init__(self, profile: ReleaseProfile | None = None) -> None:
        self.profile = profile or load_release_profile()
        self._zod_groups: dict[tuple[str, str], str] = {}
        for subset, split_map in self.profile.zod_group_memberships.items():
            for split, groups in split_map.items():
                for group in groups:
                    key = (subset, _normalize_zod_group(group))
                    previous = self._zod_groups.get(key)
                    if previous and previous != split:
                        raise ValueError(f"{subset} group {group} belongs to two splits")
                    self._zod_groups[key] = split

    def assign(self, sample: Sample) -> str | None:
        source = f"{sample.dataset}/{sample.split or ''}"
        direct = self.profile.rules.get(source)
        if direct:
            return direct
        if sample.dataset != "zod" or sample.split not in {"drives", "sequences"}:
            return None
        field = "drive_idx" if sample.split == "drives" else "sequence_idx"
        value = sample.label.get(field)
        if value is None:
            return None
        return self._zod_groups.get((sample.split, _normalize_zod_group(value)))


def create_split_manifests(
    root: str | Path,
    output: str | Path,
    *,
    profile: ReleaseProfile | None = None,
    overwrite: bool = False,
) -> SplitBuildSummary:
    """Write official split JSONL files and return their count summary."""

    profile = profile or load_release_profile()
    root_path = find_dataset_root(root)
    output_path = Path(output).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    final_paths = {
        name: output_path / f"{name}.jsonl"
        for name in profile.official_splits
    }
    summary_path = output_path / "summary.json"
    temporary_paths = {
        name: path.with_suffix(path.suffix + ".tmp")
        for name, path in final_paths.items()
    }
    temporary_summary = summary_path.with_suffix(".json.tmp")

    collisions = [
        path
        for path in (*final_paths.values(), summary_path)
        if path.exists()
    ]
    if collisions and not overwrite:
        names = ", ".join(path.name for path in collisions)
        raise FileExistsError(f"split files already exist: {names}; pass overwrite=True")

    assigner = SplitAssigner(profile)
    counts: Counter[str] = Counter()
    by_source: Counter[str] = Counter()
    try:
        with ExitStack() as stack:
            handles = {
                name: stack.enter_context(path.open("w", encoding="utf-8"))
                for name, path in temporary_paths.items()
            }
            for sample in iter_samples(root_path, resolve_paths=False):
                assigned = assigner.assign(sample)
                if assigned is None:
                    raise ValueError(
                        f"sample is not assigned to an official split: "
                        f"{sample.label_file.relative_path}:{sample.index}"
                    )
                entry = sample.reference()
                entry["official_split"] = assigned
                serialized = json.dumps(entry, ensure_ascii=True, separators=(",", ":"))
                handles[assigned].write(serialized + "\n")
                counts[assigned] += 1
                source = f"{assigned}/{sample.dataset}/{sample.split or 'unknown'}"
                by_source[source] += 1

        official_match = _counts_match_profile(counts, profile)
        if not official_match:
            differences = ", ".join(_count_differences(counts, profile))
            raise ValueError(
                f"generated split counts do not match {profile.name}: {differences}"
            )

        summary = SplitBuildSummary(
            output=output_path,
            counts=counts,
            by_source=by_source,
            expected_official_counts=dict(profile.expected_counts["official_splits"]),
            files=final_paths,
        )
        summary_text = json.dumps(summary.to_dict(), indent=2, sort_keys=False) + "\n"
        temporary_summary.write_text(summary_text, encoding="utf-8")
        for name, temporary in temporary_paths.items():
            os.replace(temporary, final_paths[name])
        os.replace(temporary_summary, summary_path)
        return summary
    finally:
        for path in temporary_paths.values():
            path.unlink(missing_ok=True)
        temporary_summary.unlink(missing_ok=True)


def _counts_match_profile(counts: Mapping[str, int], profile: ReleaseProfile) -> bool:
    expected = profile.expected_counts["official_splits"]
    if any(
        int(counts.get(split, 0)) != int(expected[split])
        for split in profile.official_splits
    ):
        return False
    return True


def _count_differences(
    counts: Mapping[str, int],
    profile: ReleaseProfile,
) -> list[str]:
    expected = profile.expected_counts["official_splits"]
    differences = [
        f"{split}={int(counts.get(split, 0))} (expected {int(expected[split])})"
        for split in profile.official_splits
        if int(counts.get(split, 0)) != int(expected[split])
    ]
    return differences


def _normalize_zod_group(value: Any) -> str:
    text = str(value)
    return text.zfill(6) if text.isdigit() else text
