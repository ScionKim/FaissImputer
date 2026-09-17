"""Prepare held-out real-data imputation cases without query leakage."""

import hashlib
import json
from numbers import Integral

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler


TRAIN_SIZES = (10_000, 15_000)
QUERY_SIZE = 3_000
MAR_REFERENCE_ROWS = 10_000
MISSING_RATE = 0.10
N_NEIGHBORS = 5
DTYPES = ("float32", "float64")
MECHANISMS = ("MCAR", "MAR")


def array_digest(array):
    """Hash logical C-order values together with shape and dtype."""
    array = np.ascontiguousarray(array)
    header = json.dumps(
        {"shape": array.shape, "dtype": array.dtype.str},
        sort_keys=True,
    ).encode("ascii")
    result = hashlib.sha256()
    result.update(header)
    result.update(b"\0")
    result.update(memoryview(array).cast("B"))
    return result.hexdigest()


def load_dataset(data_home, *, download_if_missing=False):
    """Load cached California Housing features; ignore the target."""
    dataset = fetch_california_housing(
        data_home=data_home,
        download_if_missing=download_if_missing,
    )
    data = np.ascontiguousarray(dataset.data, dtype=np.float64)
    names = list(dataset.feature_names)

    if data.shape != (20_640, 8):
        raise ValueError("Unexpected California Housing feature shape")
    if not np.isfinite(data).all():
        raise ValueError("Dataset features must contain finite values")

    metadata = {
        "dataset": "California Housing",
        "loader": "sklearn.datasets.fetch_california_housing",
        "source": (
            "https://scikit-learn.org/stable/modules/generated/"
            "sklearn.datasets.fetch_california_housing.html"
        ),
        "rows": int(data.shape[0]),
        "features": int(data.shape[1]),
        "feature_names": names,
        "dataset_sha256": array_digest(data),
        "target_used": False,
    }
    return data, names, metadata


def mask_summary(mask, eligible):
    return {
        "overall_missing_rate": float(mask.mean()),
        "eligible_missing_rate": float(mask[:, eligible].mean()),
        "missing_per_feature": mask.sum(axis=0).tolist(),
        "rows_with_missing": int(mask.any(axis=1).sum()),
        "complete_rows": int((~mask.any(axis=1)).sum()),
    }


def prepare_case(
    data,
    feature_names,
    seed,
    mechanism,
    *,
    train_size=10_000,
    query_size=QUERY_SIZE,
    dtype="float32",
    missing_rate=MISSING_RATE,
    mar_reference_rows=MAR_REFERENCE_ROWS,
):
    """Prepare one case; scoring truth remains float64.

    Query rows are fixed across training sizes. Training rows form
    nested prefixes of a separate shuffled pool.

    MAR uses the MedInc median of a common training prefix. Scalers
    are fitted separately for each case, using observed training
    values only.
    """
    for name, value, minimum in (
        ("seed", seed, 0),
        ("train_size", train_size, 1),
        ("query_size", query_size, 1),
        ("mar_reference_rows", mar_reference_rows, 1),
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, Integral)
            or value < minimum
        ):
            raise ValueError(
                f"{name} must be an integer >= {minimum}"
            )

    seed = int(seed)
    train_size = int(train_size)
    query_size = int(query_size)
    mar_reference_rows = int(mar_reference_rows)

    if mechanism not in MECHANISMS:
        raise ValueError(f"Unknown mechanism: {mechanism}")

    output_dtype = np.dtype(dtype)
    if output_dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("dtype must be float32 or float64")

    data = np.asarray(data, dtype=np.float64)
    names = list(feature_names)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("data must be a two-dimensional feature matrix")
    if not np.isfinite(data).all():
        raise ValueError("Unmasked source data must contain finite values")
    if (
        len(names) != data.shape[1]
        or not all(isinstance(name, str) for name in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("Feature names must be unique strings matching data")
    if "MedInc" not in names:
        raise ValueError("The always-observed MedInc feature is required")
    if train_size + query_size > len(data):
        raise ValueError("Insufficient rows for disjoint training and query sets")
    if mar_reference_rows > train_size:
        raise ValueError("MAR reference rows must be within the training set")

    missing_rate = float(missing_rate)
    if not 0 < missing_rate < 1:
        raise ValueError("missing_rate must be between zero and one")

    condition_column = names.index("MedInc")
    eligible = [
        index for index in range(data.shape[1])
        if index != condition_column
    ]
    base_probability = missing_rate * data.shape[1] / len(eligible)
    maximum_probability = base_probability * (
        1.5 if mechanism == "MAR" else 1.0
    )
    if maximum_probability > 1:
        raise ValueError("Requested missingness produces a probability above one")

    # Reserving queries first keeps their row IDs fixed across train sizes.
    order = np.random.default_rng([seed, 0]).permutation(len(data))
    query_ids = order[:query_size]
    train_ids = order[query_size:query_size + train_size]
    train_raw = data[train_ids].copy()
    query_raw = data[query_ids].copy()

    # Both primary training sizes share these reference rows.
    # No query values or excluded training rows determine the threshold.
    cutoff = float(np.median(
        train_raw[:mar_reference_rows, condition_column]
    ))

    def make_mask(raw, stream):
        probability = np.full(len(raw), base_probability)
        if mechanism == "MAR":
            probability *= np.where(
                raw[:, condition_column] > cutoff,
                1.5,
                0.5,
            )

        mask = np.zeros(raw.shape, dtype=bool)
        draws = np.random.default_rng([seed, stream]).random(
            (len(raw), len(eligible))
        )
        mask[:, eligible] = draws < probability[:, None]
        return mask

    # Separate streams prevent training size from shifting query draws.
    train_mask = make_mask(train_raw, 1)
    query_mask = make_mask(query_raw, 2)

    if train_mask.all(axis=0).any():
        raise ValueError("Every training feature needs observed values")
    if not query_mask.any():
        raise ValueError("The query case contains no scored missing entries")

    observed_train = train_raw.copy()
    observed_train[train_mask] = np.nan

    scaler = StandardScaler().fit(observed_train)
    train = np.ascontiguousarray(
        scaler.transform(observed_train),
        dtype=output_dtype,
    )

    # Preserve a float64 reference for scoring both output dtypes.
    truth = np.ascontiguousarray(
        scaler.transform(query_raw),
        dtype=np.float64,
    )
    query = truth.astype(output_dtype, order="C", copy=True)
    query[query_mask] = np.nan

    if (
        not np.isfinite(train[~train_mask]).all()
        or not np.isfinite(query[~query_mask]).all()
        or not np.isfinite(truth).all()
    ):
        raise ValueError("Standardized values must remain finite")

    metadata = {
        "seed": seed,
        "mechanism": mechanism,
        "input_dtype": str(output_dtype),
        "truth_dtype": str(truth.dtype),
        "n_train": train_size,
        "n_query": query_size,
        "feature_names": names,
        "nominal_overall_missing_rate": missing_rate,
        "eligible_base_probability": base_probability,
        "always_observed": ["MedInc"],
        "mar_reference_rows": (
            mar_reference_rows if mechanism == "MAR" else None
        ),
        "mar_cutoff": cutoff if mechanism == "MAR" else None,
        "mar_low_probability": (
            0.5 * base_probability if mechanism == "MAR" else None
        ),
        "mar_high_probability": (
            1.5 * base_probability if mechanism == "MAR" else None
        ),
        "train_mask": mask_summary(train_mask, eligible),
        "query_mask": mask_summary(query_mask, eligible),
        "complete_donors": int((~train_mask.any(axis=1)).sum()),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "quality_scope": "masked held-out query entries",
        "quality_units": "standardized using observed training values",
        "fingerprints": {
            "dataset": array_digest(data),
            "train_row_ids": array_digest(train_ids),
            "query_row_ids": array_digest(query_ids),
            "raw_train": array_digest(train_raw),
            "raw_query": array_digest(query_raw),
            "train_mask": array_digest(train_mask),
            "query_mask": array_digest(query_mask),
            "train": array_digest(train),
            "query": array_digest(query),
            "truth": array_digest(truth),
        },
    }
    return train, query, truth, query_mask, metadata