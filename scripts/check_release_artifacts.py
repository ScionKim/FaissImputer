"""Validate release artifacts before they are uploaded to PyPI."""

from __future__ import annotations

import argparse
from email.parser import BytesParser
from pathlib import Path
from pathlib import PurePosixPath
import re
import tarfile
from zipfile import ZipFile


PROJECT_NAME = "faiss-imputer"
BANNED_DESCRIPTION_PATTERNS = (
    re.compile(r"\bis in progress\b", re.IGNORECASE),
    re.compile(r"\[!WARNING\]", re.IGNORECASE),
)
MARKDOWN_LINK_PATTERN = re.compile(r"\]\(([^)]+)\)")
ABSOLUTE_LINK_PREFIXES = ("https://", "http://", "mailto:", "#")


def read_wheel_metadata(path: Path):
    with ZipFile(path) as archive:
        candidates = [
            name
            for name in archive.namelist()
            if name.endswith(".dist-info/METADATA")
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"{path.name} must contain exactly one METADATA file"
            )
        return BytesParser().parsebytes(archive.read(candidates[0]))


def read_sdist_metadata(path: Path):
    with tarfile.open(path, "r:gz") as archive:
        candidates = [
            member
            for member in archive.getmembers()
            if PurePosixPath(member.name).name == "PKG-INFO"
            and len(PurePosixPath(member.name).parts) == 2
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"{path.name} must contain exactly one PKG-INFO file"
            )
        stream = archive.extractfile(candidates[0])
        if stream is None:
            raise ValueError(f"Could not read metadata from {path.name}")
        return BytesParser().parsebytes(stream.read())


def validate_description(
    description: str,
    artifact_name: str,
    version: str,
) -> None:
    for banned_pattern in BANNED_DESCRIPTION_PATTERNS:
        if banned_pattern.search(description):
            raise ValueError(
                f"{artifact_name} contains banned text matching "
                f"{banned_pattern.pattern!r}"
            )

    relative_links = [
        target
        for target in MARKDOWN_LINK_PATTERN.findall(description)
        if not target.startswith(ABSOLUTE_LINK_PREFIXES)
    ]
    if relative_links:
        raise ValueError(
            f"{artifact_name} contains relative Markdown links: "
            + ", ".join(relative_links)
        )

    required_text = (
        f"## What's new in {version}",
        f"/releases/tag/v{version}",
        f"/blob/v{version}/",
        f"faiss-imputer>={version}",
    )
    for text in required_text:
        if text not in description:
            raise ValueError(
                f"{artifact_name} is missing current-version text: {text!r}"
            )


def validate_artifacts(dist_dir: Path, release_tag: str | None) -> str:
    wheels = sorted(dist_dir.glob("*.whl"))
    sdists = sorted(dist_dir.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError(
            f"{dist_dir} must contain exactly one wheel and one source archive"
        )

    artifacts = (
        (wheels[0], read_wheel_metadata(wheels[0])),
        (sdists[0], read_sdist_metadata(sdists[0])),
    )
    versions = set()
    descriptions = set()

    for path, metadata in artifacts:
        if metadata["Name"] != PROJECT_NAME:
            raise ValueError(
                f"{path.name} has unexpected project name {metadata['Name']!r}"
            )

        version = metadata["Version"]
        if not version:
            raise ValueError(f"{path.name} is missing its Version metadata")
        versions.add(version)

        if metadata["Requires-Python"] != ">=3.10":
            raise ValueError(
                f"{path.name} has unexpected Requires-Python metadata"
            )
        content_type = metadata["Description-Content-Type"] or ""
        if not content_type.startswith("text/markdown"):
            raise ValueError(
                f"{path.name} does not declare a Markdown description"
            )

        description = metadata.get_payload()
        validate_description(description, path.name, version)
        descriptions.add(description)

    if len(versions) != 1:
        raise ValueError("Wheel and source archive versions do not match")
    if len(descriptions) != 1:
        raise ValueError("Wheel and source archive descriptions do not match")

    version = versions.pop()
    if release_tag is not None:
        expected_tag = f"v{version}"
        if release_tag != expected_tag:
            raise ValueError(
                f"Release tag must be {expected_tag!r}, got {release_tag!r}"
            )

    return version


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    parser.add_argument("--tag")
    args = parser.parse_args()

    version = validate_artifacts(args.dist_dir, args.tag)
    print(f"Validated {PROJECT_NAME} {version} release artifacts")


if __name__ == "__main__":
    main()
