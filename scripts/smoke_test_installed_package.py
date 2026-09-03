"""Smoke-test an installed FaissImputer distribution outside the source tree."""

from __future__ import annotations

import argparse
import importlib
import sys
from importlib.metadata import distribution, version
from pathlib import Path

import numpy as np

from faiss_imputer import FaissImputer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag")
    args = parser.parse_args()

    installed = distribution("faiss-imputer")
    recorded_files = {str(path) for path in installed.files or []}
    environment_root = Path(sys.prefix).resolve()
    modules = (
        ("faiss_imputer", "faiss_imputer/__init__.py"),
        ("faiss_imputer.faiss_imputer", "faiss_imputer/faiss_imputer.py"),
        ("faiss_imputer._matrix", "faiss_imputer/_matrix.py"),
    )
    for module_name, relative_path in modules:
        assert relative_path in recorded_files, relative_path
        module = importlib.import_module(module_name)
        actual = Path(module.__file__).resolve()
        expected_path = Path(installed.locate_file(relative_path)).resolve()
        assert actual == expected_path, (actual, expected_path)
        assert actual.is_relative_to(environment_root), actual
        print(f"{module_name}: {actual}")

    installed_version = version("faiss-imputer")
    if args.tag is not None:
        expected_tag = f"v{installed_version}"
        if args.tag != expected_tag:
            raise ValueError(
                f"Release tag must be {expected_tag!r}, got {args.tag!r}"
            )

    train = np.array([[0.0, 10.0], [2.0, 20.0]], dtype=np.float32)
    incomplete = np.array([[1.0, np.nan]], dtype=np.float32)
    expected = np.array([[1.0, 15.0]], dtype=np.float32)

    # Exercise the public fit/transform path with separate complete donors.
    transformed = FaissImputer(n_neighbors=2).fit(train).transform(incomplete)
    np.testing.assert_allclose(transformed, expected)
    partial_train = np.array(
        [
            [0.0, 10.0, np.nan],
            [2.0, 30.0, np.nan],
            [1.0, np.nan, 20.0],
            [3.0, np.nan, 40.0],
        ],
        dtype=np.float32,
    )
    partial_query = np.array(
        [[0.1, np.nan, np.nan], [np.nan, np.nan, np.nan]],
        dtype=np.float32,
    )
    train_before = partial_train.copy()
    query_before = partial_query.copy()
    partial_expected = np.array(
        [[0.1, 10.0, 20.0], [1.5, 20.0, 30.0]],
        dtype=np.float32,
    )

    partial_result = FaissImputer(
        n_neighbors=1, donor_policy="available"
    ).fit(partial_train).transform(partial_query)

    np.testing.assert_array_equal(partial_result, partial_expected)
    assert partial_result.dtype == np.float32
    np.testing.assert_array_equal(partial_train, train_before)
    np.testing.assert_array_equal(partial_query, query_before)
    print(f"Smoke-tested installed faiss-imputer {installed_version}")


if __name__ == "__main__":
    main()
