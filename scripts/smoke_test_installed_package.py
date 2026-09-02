"""Smoke-test an installed FaissImputer distribution outside the source tree."""

from __future__ import annotations

import argparse
from importlib.metadata import version

import numpy as np

from faiss_imputer import FaissImputer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag")
    args = parser.parse_args()

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
    print(f"Smoke-tested installed faiss-imputer {installed_version}")


if __name__ == "__main__":
    main()
