from __future__ import annotations

import hashlib
import re
import tarfile
import time
from pathlib import Path, PurePosixPath

from .release import ReleaseProfile, load_release_profile


CHUNK_SIZE = 8 * 1024 * 1024
CHECKSUM_FILENAME = "SHA256SUMS"


def sha256_file(path: str | Path, *, progress: bool = False) -> str:
    digest = hashlib.sha256()
    input_path = Path(path)
    total = input_path.stat().st_size
    transferred = 0
    last_report = 0.0
    with input_path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
            transferred += len(chunk)
            now = time.monotonic()
            if progress and now - last_report >= 5.0:
                print(
                    _progress_line(f"verifying {input_path.name}", transferred, total),
                    flush=True,
                )
                last_report = now
    return digest.hexdigest()


def verify_archive(path: str | Path, expected_sha256: str, *, progress: bool = False) -> bool:
    """Return whether a local archive matches an expected SHA-256 digest."""

    expected = expected_sha256.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError(f"invalid SHA-256 digest: {expected_sha256!r}")
    return sha256_file(path, progress=progress) == expected


def prepare_archives(
    archive_directory: str | Path,
    output: str | Path = "OpenCVL",
    *,
    progress: bool = True,
    profile: ReleaseProfile | None = None,
) -> Path:
    """Verify and extract a complete set of downloaded OpenCVL archives."""

    archive_root = Path(archive_directory).expanduser().resolve()
    if not archive_root.exists():
        raise FileNotFoundError(archive_root)
    if not archive_root.is_dir():
        raise NotADirectoryError(archive_root)

    profile = profile or load_release_profile()
    missing = tuple(
        filename
        for filename in profile.archive_filenames
        if not (archive_root / filename).is_file()
    )
    if missing:
        names = ", ".join(missing)
        present_count = len(profile.archive_filenames) - len(missing)
        raise FileNotFoundError(
            "downloaded archive set is incomplete: "
            f"found {present_count}/{len(profile.archive_filenames)}; missing {names}"
        )
    checksums = _read_checksum_manifest(archive_root / CHECKSUM_FILENAME)
    missing_checksums = [
        filename
        for filename in profile.archive_filenames
        if filename not in checksums
    ]
    if missing_checksums:
        names = ", ".join(missing_checksums)
        raise ValueError(f"{CHECKSUM_FILENAME} is missing checksums for: {names}")

    if progress:
        count = len(profile.archive_filenames)
        print(f"archive set complete: {count}/{count}")

    output_path = Path(output).expanduser().resolve()
    verified: list[Path] = []
    for filename in profile.archive_filenames:
        archive_path = archive_root / filename
        if progress:
            print(f"verifying: {archive_path}")
        if not verify_archive(archive_path, checksums[filename], progress=progress):
            raise ValueError(f"SHA-256 verification failed for {archive_path}")
        if progress:
            print(f"SHA-256 OK: {filename}")
        verified.append(archive_path)

    output_path.mkdir(parents=True, exist_ok=True)
    for archive_path in verified:
        extract_archive(archive_path, output_path)
        if progress:
            print(f"extracted: {archive_path.name} -> {output_path}")
    return output_path


def _read_checksum_manifest(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"missing checksum manifest: {path}")

    checksums: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            match = re.fullmatch(r"([0-9A-Fa-f]{64})\s+\*?([^\s]+)", line)
            if match is None:
                raise ValueError(f"{path}:{line_number}: invalid checksum entry")
            digest, filename = match.groups()
            if filename in checksums:
                raise ValueError(f"{path}:{line_number}: duplicate entry for {filename}")
            checksums[filename] = digest.lower()
    return checksums


def extract_archive(path: str | Path, destination: str | Path) -> None:
    """Safely extract a local tar archive into a dataset directory."""

    archive_path = Path(path)
    destination_path = Path(destination).expanduser().resolve()
    destination_path.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, mode="r:*") as tar:
        members = tar.getmembers()
        for member in members:
            relative = PurePosixPath(member.name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe tar member: {member.name!r}")
            if member.issym() or member.islnk():
                raise ValueError(f"links are not permitted in OpenCVL archives: {member.name!r}")
            if member.isdev() or member.isfifo():
                raise ValueError(
                    "special files are not permitted in OpenCVL archives: "
                    f"{member.name!r}"
                )
            target = (destination_path / Path(*relative.parts)).resolve(strict=False)
            try:
                target.relative_to(destination_path)
            except ValueError as exc:
                raise ValueError(f"tar member escapes destination: {member.name!r}") from exc
        data_filter = getattr(tarfile, "data_filter", None)
        if data_filter is None:
            tar.extractall(destination_path, members=members)
        else:
            tar.extractall(destination_path, members=members, filter=data_filter)


def _progress_line(filename: str, transferred: int, total: int | None) -> str:
    gib = transferred / (1024**3)
    if total:
        percent = transferred / total * 100.0
        return f"  {filename}: {gib:.2f} GiB / {total / (1024**3):.2f} GiB ({percent:.1f}%)"
    return f"  {filename}: {gib:.2f} GiB"
