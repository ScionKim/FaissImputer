"""Release metadata checks using small, synthetic archive fixtures."""

from io import BytesIO
import tarfile
from zipfile import ZipFile

import pytest

from scripts.check_release_artifacts import validate_artifacts


VERSION = "1.2.3"

DESCRIPTION = (
    "# FaissImputer\n\n"
    "KNN imputation with explicit donor policies.\n\n"
    "Install with: python -m pip install --upgrade faiss-imputer\n\n"
    "[API reference]"
    "(https://github.com/ScionKim/FaissImputer/blob/main/docs/api.md)\n"
)


def _metadata(version, description):
    return (
        "Metadata-Version: 2.4\n"
        "Name: faiss-imputer\n"
        f"Version: {version}\n"
        "Requires-Python: >=3.10\n"
        "Description-Content-Type: text/markdown\n"
        "\n"
        f"{description}"
    ).encode("utf-8")


def _write_artifacts(
    directory,
    *,
    sdist_version=VERSION,
    wheel_description=DESCRIPTION,
    sdist_description=DESCRIPTION,
):
    wheel = directory / f"faiss_imputer-{VERSION}-py3-none-any.whl"

    with ZipFile(wheel, "w") as archive:
        archive.writestr(
            f"faiss_imputer-{VERSION}.dist-info/METADATA",
            _metadata(VERSION, wheel_description),
        )

    sdist = directory / f"faiss_imputer-{sdist_version}.tar.gz"
    payload = _metadata(sdist_version, sdist_description)

    member = tarfile.TarInfo(
        f"faiss_imputer-{sdist_version}/PKG-INFO"
    )
    member.size = len(payload)

    with tarfile.open(sdist, "w:gz") as archive:
        archive.addfile(member, BytesIO(payload))


@pytest.mark.parametrize("release_tag", [None, f"v{VERSION}"])
def test_version_independent_description_is_accepted(tmp_path, release_tag):
    _write_artifacts(tmp_path)

    assert validate_artifacts(tmp_path, release_tag) == VERSION


@pytest.mark.parametrize(
    "description, message",
    [
        ("", "empty project description"),
        (" \n\t", "empty project description"),
        (
            "# FaissImputer\n\n[Usage](docs/usage.md)\n",
            "relative Markdown links",
        ),
        (
            "# FaissImputer\n\n[!WARNING]\n",
            "banned text",
        ),
        (
            "# FaissImputer\n\nDevelopment is in progress.\n",
            "banned text",
        ),
    ],
)
def test_invalid_description_is_rejected(tmp_path, description, message):
    _write_artifacts(
        tmp_path,
        wheel_description=description,
        sdist_description=description,
    )

    with pytest.raises(ValueError, match=message):
        validate_artifacts(tmp_path, None)


def test_sdist_description_is_also_validated(tmp_path):
    _write_artifacts(
        tmp_path,
        sdist_description="# FaissImputer\n\n[Usage](docs/usage.md)\n",
    )

    with pytest.raises(ValueError, match="relative Markdown links"):
        validate_artifacts(tmp_path, None)


def test_wheel_and_sdist_versions_must_match(tmp_path):
    _write_artifacts(tmp_path, sdist_version="1.2.4")

    with pytest.raises(
        ValueError,
        match="Wheel and source archive versions do not match",
    ):
        validate_artifacts(tmp_path, None)


def test_wheel_and_sdist_descriptions_must_match(tmp_path):
    _write_artifacts(
        tmp_path,
        sdist_description=DESCRIPTION + "\nUnexpected extra text.\n",
    )

    with pytest.raises(
        ValueError,
        match="Wheel and source archive descriptions do not match",
    ):
        validate_artifacts(tmp_path, None)


@pytest.mark.parametrize("release_tag", ["v1.2.4", VERSION])
def test_release_tag_must_match_artifact_version(tmp_path, release_tag):
    _write_artifacts(tmp_path)

    with pytest.raises(ValueError, match="Release tag must be"):
        validate_artifacts(tmp_path, release_tag)