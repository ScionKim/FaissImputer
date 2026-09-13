from numbers import Integral

import numpy as np
import faiss
from sklearn.impute import MissingIndicator
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data
from ._matrix import MatrixNaNIndex

class FaissImputer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Impute missing values using faiss."""

    def __init__(
        self,
        n_neighbors=3,
        metric="l2",
        strategy="mean",
        index_factory="Flat",
        donor_policy="complete",
        weights="uniform",
        add_indicator=False,
        keep_empty_features=False,
        missing_values=np.nan,
        copy=True,
    ):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.strategy = strategy
        self.index_factory = index_factory
        self.donor_policy = donor_policy
        self.weights = weights
        self.add_indicator = add_indicator
        self.keep_empty_features = keep_empty_features
        self.missing_values = missing_values
        self.copy = copy

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = (
            isinstance(self.missing_values, (float, np.floating))
            and bool(np.isnan(self.missing_values))
        )
        tags.transformer_tags.preserves_dtype = ["float32", "float64"]
        return tags

    @staticmethod
    def _numeric_missing_mask(X, marker):
        """Compare without rounding the marker to a different observed value."""
        if X.dtype.kind in "iu":
            integer_marker = int(marker)
            bounds = np.iinfo(X.dtype)
            if (
                integer_marker.as_integer_ratio() != marker.as_integer_ratio()
                or not bounds.min <= integer_marker <= bounds.max
            ):
                return np.zeros(X.shape, dtype=bool)
            return X == X.dtype.type(integer_marker)

        if X.dtype.kind in "fb":
            try:
                with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                    converted = X.dtype.type(marker)
            except OverflowError:
                return np.zeros(X.shape, dtype=bool)
            if not np.isfinite(converted) or (
                converted.item().as_integer_ratio() != marker.as_integer_ratio()
            ):
                return np.zeros(X.shape, dtype=bool)
            return X == converted

        # Object arrays retain the individual numeric values in mixed lists
        # and DataFrames, including integers too large for exact float64.
        marker_ratio = marker.as_integer_ratio()

        def matches(value):
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, Integral):
                return int(value).as_integer_ratio() == marker_ratio
            if isinstance(value, (float, np.floating)):
                return np.isfinite(value) and value.as_integer_ratio() == marker_ratio
            return value == marker

        return np.fromiter(
            (matches(value) for value in X.flat), dtype=bool, count=X.size,
        ).reshape(X.shape)

    def _validate_input(self, X, *, reset):
        if not isinstance(self.copy, (bool, np.bool_)):
            raise ValueError("copy must be a boolean")

        marker = self.missing_values
        if isinstance(marker, (bool, np.bool_)) or not isinstance(
            marker, (Integral, float, np.floating)
        ):
            raise ValueError(
                "missing_values must be np.nan or a finite real number"
            )

        if isinstance(marker, Integral):
            marker = int(marker)
            nan_marker = False
        else:
            nan_marker = bool(np.isnan(marker))
            if not nan_marker and not np.isfinite(marker):
                raise ValueError(
                    "missing_values must be np.nan or a finite real number"
                )
            marker = marker.item() if isinstance(marker, np.generic) else marker

        original = X
        X = validate_data(
            self,
            X,
            dtype=None,
            ensure_all_finite="allow-nan" if nan_marker else True,
            reset=reset,
        )

        # Preserve float64, including non-native byte order.
        # Other input dtypes retain the existing float32 conversion.
        output_dtype = np.dtype(
            np.float64
            if X.dtype.kind == "f" and X.dtype.itemsize == 8
            else np.float32
        )

        if nan_marker:
            with np.errstate(over="ignore", invalid="ignore"):
                normalized = np.asarray(X, dtype=output_dtype)
            if np.isinf(normalized).any():
                raise ValueError(
                    "Observed values must be finite and representable as "
                    f"{output_dtype.name}"
                )
            return normalized

        if hasattr(original, "iloc") and hasattr(original, "to_numpy"):
            # Preserve individual values from mixed pandas columns.
            X = original.to_numpy(dtype=object)
        elif not isinstance(original, np.ndarray):
            # Preserve exact values when matching mixed numeric lists.
            X = np.asarray(original, dtype=object)

        missing = self._numeric_missing_mask(X, marker)

        # Select missing cells before converting observed values.
        normalized = np.zeros(X.shape, dtype=output_dtype)
        with np.errstate(over="ignore", invalid="ignore"):
            np.copyto(normalized, X, where=~missing, casting="unsafe")

        if not np.isfinite(normalized).all():
            raise ValueError(
                "Observed values must be finite and representable as "
                f"{output_dtype.name}"
            )

        normalized[missing] = np.nan
        return normalized

    def _aggregate(self, values, *, axis, ignore_nan):
        """Reduce in the input dtype, repairing overflowing intermediates."""
        if self.strategy == "mean":
            aggregate = np.nanmean if ignore_nan else np.mean
        else:
            aggregate = np.nanmedian if ignore_nan else np.median

        with np.errstate(over="ignore", invalid="ignore"):
            result = aggregate(values, axis=axis)

        for position in np.flatnonzero(~np.isfinite(result)):
            selected = values[:, position] if axis == 0 else values[position, :]
            if ignore_nan and np.isnan(selected).all():
                continue

            selected64 = selected.astype(np.float64)
            with np.errstate(over="ignore", invalid="ignore"):
                repaired = aggregate(selected64)
                if not np.isfinite(repaired):
                    scale = np.nanmax(np.abs(selected64))
                    repaired = aggregate(selected64 / scale) * scale

            if not np.isfinite(repaired):
                raise ValueError(
                    "Aggregate must be finite and representable as "
                    f"{values.dtype.name}"
                )
            result[position] = repaired

        return result

    def _uses_uniform_weights(self):
        return self.weights is None or (
            isinstance(self.weights, str) and self.weights == "uniform"
        )

    def _weighted_mean(self, values, squared_distances):
        """Convert built-in squared distances before weighting."""
        distances = np.sqrt(
            np.maximum(
                np.asarray(squared_distances, dtype=np.float64), 0.0
            )
        )
        return self._weighted_mean_from_distances(values, distances)

    def _weighted_mean_from_distances(self, values, distances):
        """Average selected donors using actual, unsquared distances."""
        distances = np.array(distances, dtype=np.float64, copy=True)
        valid = ~np.isnan(values) & np.isfinite(distances)
        distances[~valid] = np.nan

        if callable(self.weights):
            raw_weights = np.asarray(self.weights(distances))
            if raw_weights.shape != values.shape:
                raise ValueError(
                    "weights callable must return an array with "
                    "the same shape as the distances"
                )
            if np.iscomplexobj(raw_weights):
                raise ValueError(
                    "weights callable must return real numeric weights"
                )
            try:
                weights = np.array(
                    raw_weights, dtype=np.float64, copy=True
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "weights callable must return real numeric weights"
                ) from exc

            # Undefined or unavailable neighbors cannot contribute.
            weights[~valid | np.isnan(weights)] = 0.0
            if not np.isfinite(weights).all():
                raise ValueError("weights must be finite")
        else:
            weights = np.zeros_like(distances)

            # Exact matches exclude all nonzero-distance neighbors.
            zero_distance = valid & (distances == 0)
            zero_rows = zero_distance.any(axis=1)

            # Proportional to 1 / distance, without reciprocal overflow.
            positive = valid & (distances > 0)
            minimum = np.min(
                np.where(positive, distances, np.inf),
                axis=1,
                keepdims=True,
            )
            np.divide(
                minimum,
                distances,
                out=weights,
                where=positive & ~zero_rows[:, None],
            )
            weights[zero_rows] = zero_distance[zero_rows]

        # Scaling avoids overflow from large finite callable weights.
        scale = np.max(np.abs(weights), axis=1, keepdims=True)
        if (scale == 0).any():
            raise ValueError(
                "weights must have a nonzero sum for every imputed value"
            )
        weights /= scale
        totals = weights.sum(axis=1)
        if (totals == 0).any():
            raise ValueError(
                "weights must have a nonzero sum for every imputed value"
            )

        safe_values = np.where(valid, values, 0.0).astype(np.float64)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            result = np.sum(safe_values * weights, axis=1) / totals

        # Repair overflowing intermediate sums without changing ordinary
        # reductions. Zero-weight values do not determine the scale.
        repair_rows = np.flatnonzero(~np.isfinite(result))
        if repair_rows.size:
            repair_weights = weights[repair_rows]
            repair_values = np.where(
                repair_weights != 0, safe_values[repair_rows], 0.0
            )
            scales = np.max(np.abs(repair_values), axis=1)
            scaled_values = np.divide(
                repair_values,
                scales[:, None],
                out=np.zeros_like(repair_values),
                where=scales[:, None] != 0,
            )
            numerators = np.sum(
                scaled_values * repair_weights, axis=1
            )

            # Combine multiplication and division through their exponents
            # so only an unrepresentable final result overflows.
            numerator_m, numerator_e = np.frexp(numerators)
            scale_m, scale_e = np.frexp(scales)
            total_m, total_e = np.frexp(totals[repair_rows])
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                result[repair_rows] = np.ldexp(
                    numerator_m * scale_m / total_m,
                    numerator_e + scale_e - total_e,
                )

        if (
            not np.isfinite(result).all()
            or (np.abs(result) > np.finfo(values.dtype).max).any()
        ):
            raise ValueError(
                "weighted mean must be finite and representable as "
                f"{values.dtype.name}"
            )

        return result

    def fit(self, X, y=None):
        """Fit the imputer; leave it unfitted if fitting fails."""
        self._clear_fitted_state()
        fit_succeeded = False

        try:
            self._fit(X, y)
            fit_succeeded = True
        finally:
            if not fit_succeeded:
                self._clear_fitted_state()

        return self

    def _clear_fitted_state(self):
        for name in (
            "n_features_in_",
            "feature_names_in_",
            "statistics_",
            "donors_",
            "metric_type_",
            "metric_callable_",
            "index_",
            "indicator_",
            "valid_features_",
            "donor_policy_",
            "donor_groups_",
            "available_index_",
            "all_donors_complete_",
        ):
            self.__dict__.pop(name, None)

    def _fit(self, X, y=None):
        """
        Fit the FaissImputer to the provided data.

        Parameters:
        - X (array-like): The input data with missing values to fit the imputer on.
        - y: Ignored.

        Returns:
        - self: Returns an instance of the fitted FaissImputer.
        """
        # Check input data
        X = self._validate_input(X, reset=True)

        # Check parameters
        if (
            isinstance(self.n_neighbors, (bool, np.bool_))
            or not isinstance(self.n_neighbors, Integral)
            or self.n_neighbors <= 0
        ):
            raise ValueError("n_neighbors must be a positive integer")

        is_callable_metric = callable(self.metric)
        if not (
            is_callable_metric
            or (
                isinstance(self.metric, str)
                and self.metric in ("l2", "nan_euclidean", "ip")
            )
        ):
            raise ValueError(
                "metric must be 'l2', 'nan_euclidean', 'ip', or a callable"
            )

        is_l2 = (
            not is_callable_metric
            and self.metric in ("l2", "nan_euclidean")
        )

        if self.strategy not in ('mean', 'median'):
            raise ValueError("strategy must be either 'mean' or 'median'")

        if not isinstance(self.index_factory, str):
            raise ValueError("index_factory must be a string")

        if not (
            self._uses_uniform_weights()
            or callable(self.weights)
            or (
                isinstance(self.weights, str)
                and self.weights == "distance"
            )
        ):
            raise ValueError(
                "weights must be 'uniform', 'distance', None, or a callable"
            )

        if not self._uses_uniform_weights():
            if self.strategy != "mean":
                raise ValueError(
                    "non-uniform weights require strategy='mean'"
                )
            if not (is_l2 or is_callable_metric):
                raise ValueError(
                    "non-uniform weights require "
                    "metric='l2' or 'nan_euclidean' or a callable"
                )

        if self.donor_policy not in ("complete", "available"):
            raise ValueError(
                "donor_policy must be either 'complete' or 'available'"
            )

        if not isinstance(self.add_indicator, (bool, np.bool_)):
            raise ValueError("add_indicator must be a boolean")

        if not isinstance(self.keep_empty_features, (bool, np.bool_)):
            raise ValueError("keep_empty_features must be a boolean")

        if is_callable_metric and self.index_factory != "Flat":
            raise ValueError(
                "callable metric requires index_factory='Flat'"
            )

        if self.donor_policy == "available" and (
            not (is_l2 or is_callable_metric)
            or self.index_factory != "Flat"
        ):
            raise ValueError(
                "donor_policy='available' requires "
                "metric='l2' or 'nan_euclidean' or a callable, "
                "and index_factory='Flat'"
            )

        self.metric_callable_ = (
            self.metric if is_callable_metric else None
        )

        # Learn missingness from all training rows before donor filtering.
        self.indicator_ = None
        if self.add_indicator:
            indicator = MissingIndicator(
                features="missing-only",
                sparse=False,
                error_on_new=False,
            )
            self.indicator_ = indicator.set_output(
                transform="default"
            ).fit(X)
        
        self.donor_policy_ = self.donor_policy
        # Fit-time empty columns never participate in donor selection or
        # imputation. Keep the original schema and indicator above intact.
        self.valid_features_ = ~np.isnan(X).all(axis=0)
        if not self.valid_features_.all():
            X = X[:, self.valid_features_]

        if X.shape[1] == 0:
            if self.donor_policy_ == "complete" and self.index_factory != "Flat":
                # Validate a custom factory without training or storing a
                # donor index. A malformed description must still fail fit.
                metric_type = (
                    faiss.METRIC_L2 if is_l2
                    else faiss.METRIC_INNER_PRODUCT
                )
                faiss.index_factory(
                    self.n_features_in_, self.index_factory, metric_type
                )
            # There is no value to estimate and no distance index to build.
            self.statistics_ = np.empty(0, dtype=X.dtype)
            self.donors_ = np.empty((0, 0), dtype=X.dtype)
            return self

        if self.donor_policy_ == "available":
            return self._fit_available(X)

        self.statistics_ = self._aggregate(X, axis=0, ignore_nan=True)

        # Extract non-missing data
        mask = ~np.isnan(X).any(axis=1)
        self.donors_ = X[mask].copy(order="F")
        
        if self.donors_.shape[0] == 0:
            raise ValueError(
                "X must contain at least one complete row to use as a donor"
            )

        if self.n_neighbors > self.donors_.shape[0]:
            raise ValueError(
                "n_neighbors cannot exceed the number of complete donors"
            )

        if self.metric_callable_ is not None:
            return self

        # Build faiss index
        self.metric_type_ = (
            faiss.METRIC_L2
            if is_l2
            else faiss.METRIC_INNER_PRODUCT
        )
        index = faiss.index_factory(
            self.donors_.shape[1],
            self.index_factory,
            self.metric_type_,
        )
        # Flat donor storage is unused: transform builds projected indexes.
        # Other factories retain their training and insertion validation.
        if self.index_factory != "Flat":
            donor_vectors = self._as_faiss_vectors(self.donors_)
            index.train(donor_vectors)
            index.add(donor_vectors)

        # Store the index as an attribute
        self.index_ = index

        return self

    def _fit_available(self, X):
        observed = ~np.isnan(X)

        self.statistics_ = self._aggregate(X, axis=0, ignore_nan=True)

        nonempty_rows = observed.any(axis=1)
        self.donors_ = X[nonempty_rows].copy()
        if self.metric_callable_ is not None:
            return self
        self.metric_type_ = faiss.METRIC_L2
        self.available_index_ = MatrixNaNIndex(
            self.donors_, n_features=self.n_features_in_
        )

        return self

    def get_feature_names_out(self, input_features=None):
        names = super().get_feature_names_out(input_features)
        output_names = (
            names if self.keep_empty_features else names[self.valid_features_]
        )
        if self.indicator_ is None:
            return output_names

        indicator_names = self.indicator_.get_feature_names_out(names)
        return np.concatenate((output_names, indicator_names))

    def _copy_or_reuse(self, X):
        """Reuse writable contiguous input only when copying is disabled."""
        if (
            not self.copy
            and X.flags.writeable
            and (X.flags.c_contiguous or X.flags.f_contiguous)
        ):
            return X
        return X.copy()

    def _format_output(self, imputed, original, indicators):
        """Restore empty columns and append previously captured indicators."""
        if not np.isfinite(imputed).all():
            raise ValueError(
                "Imputed values must be finite and representable as "
                f"{imputed.dtype.name}"
            )

        if self.keep_empty_features and not self.valid_features_.all():
            restored = np.zeros(original.shape, dtype=imputed.dtype)
            restored[:, self.valid_features_] = imputed
            imputed = restored

        if indicators is None or indicators.shape[1] == 0:
            return imputed

        return np.concatenate((imputed, indicators), axis=1)

    @staticmethod
    def _as_faiss_vectors(values):
        """Prepare search vectors without changing the original values."""
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            vectors = np.ascontiguousarray(values, dtype=np.float32)

        if values.dtype.itemsize > vectors.dtype.itemsize:
            if not np.isfinite(vectors).all():
                raise ValueError(
                    "FAISS search values must be finite and "
                    "representable as float32"
                )

        return vectors

    def _callable_distances(self, query):
        """Evaluate the metric on independent rows in the original schema."""
        full_query = np.full(
            self.n_features_in_, np.nan, dtype=query.dtype
        )
        full_query[self.valid_features_] = query
        distances = np.empty(len(self.donors_), dtype=np.float64)

        for donor_id, donor in enumerate(self.donors_):
            full_donor = np.full(
                self.n_features_in_, np.nan, dtype=donor.dtype
            )
            full_donor[self.valid_features_] = donor

            # A callback cannot mutate stored donors or another call's query.
            value = np.asarray(
                self.metric_callable_(
                    full_query.copy(),
                    full_donor,
                    missing_values=np.nan,
                )
            )

            if value.ndim != 0 or value.dtype.kind not in "iuf":
                raise ValueError(
                    "metric callable must return a real numeric scalar"
                )

            if np.isnan(value):
                distances[donor_id] = np.nan
                continue

            if not np.isfinite(value) or value < 0:
                raise ValueError(
                    "metric callable must return a nonnegative finite "
                    "distance or np.nan"
                )

            distance = float(value)
            if (
                not np.isfinite(distance)
                or (distance == 0.0 and value != 0.0)
            ):
                raise ValueError(
                    "metric distance must be representable as float64"
                )

            distances[donor_id] = distance

        return distances

    def _transform_callable(self, X):
        """Select donors by callable distance, with stable ties."""
        result = self._copy_or_reuse(X)
        missing = np.isnan(X)
        k = int(self.n_neighbors)

        for row in np.flatnonzero(missing.any(axis=1)):
            # Capture the original query before any in-place writes.
            query = X[row].copy()
            missing_columns = np.flatnonzero(missing[row])
            filled = query.copy()
            filled[missing_columns] = self.statistics_[missing_columns]

            # Preserve the existing fallback for entirely missing queries.
            if not missing[row].all():
                distances = self._callable_distances(query)
                finite_ids = np.flatnonzero(np.isfinite(distances))
                ordered_ids = finite_ids[
                    np.argsort(distances[finite_ids], kind="stable")
                ]

                for column in missing_columns:
                    eligible = ordered_ids[
                        ~np.isnan(self.donors_[ordered_ids, column])
                    ]
                    selected = eligible[:k]
                    if selected.size == 0:
                        continue

                    values = self.donors_[selected, column][None, :]
                    if self._uses_uniform_weights():
                        fill = self._aggregate(
                            values, axis=1, ignore_nan=False
                        )[0]
                    else:
                        fill = self._weighted_mean_from_distances(
                            values, distances[selected][None, :]
                        )[0]

                    filled[column] = fill

            result[row] = filled

        return result

    def transform(self, X):
        """
        Impute missing values using fitted donors and the configured metric.

        Parameters:
        - X (array-like): The input data with missing values to be imputed.

        Returns:
        - X_tmp (array-like): Imputed data. May reuse input storage when copy=False.
        """
        
        check_is_fitted(self)

        X = self._validate_input(X, reset=False)

        original = X
        # Capture missingness before any in-place imputation.
        indicators = (
            None
            if self.indicator_ is None
            else self.indicator_.transform(original)
        )

        if not self.valid_features_.all():
            X = X[:, self.valid_features_]
        if X.shape[1] == 0:
            return self._format_output(
                self._copy_or_reuse(X), original, indicators
            )

        if self.metric_callable_ is not None:
            imputed = self._transform_callable(X)
            return self._format_output(imputed, original, indicators)

        if self.donor_policy_ == "available":
            imputed = self._transform_available(X)
            return self._format_output(imputed, original, indicators)

        X_tmp = self._copy_or_reuse(X)
        missing_mask = np.isnan(X)

        # Group rows that have the same observed columns.
        pattern_groups = {}
        missing_row_indices = np.flatnonzero(
            missing_mask.any(axis=1)
        )

        for sample_idx in missing_row_indices:
            observed_mask = ~missing_mask[sample_idx]
            pattern = tuple(observed_mask.tolist())
            pattern_groups.setdefault(pattern, []).append(sample_idx)

        for pattern, sample_indices in pattern_groups.items():
            observed_mask = np.asarray(pattern, dtype=bool)
            observed_cols = np.flatnonzero(observed_mask)
            missing_cols = np.flatnonzero(~observed_mask)

            # If the whole row is missing, use fitted statistics.
            if observed_cols.size == 0:
                X_tmp[np.ix_(sample_indices, missing_cols)] = (
                    self.statistics_[missing_cols]
                )
                continue

            donor_vectors = self._as_faiss_vectors(
                self.donors_[:, observed_cols]
            )
            query_vectors = self._as_faiss_vectors(
                X[np.ix_(sample_indices, observed_cols)]
            )

            index = faiss.index_factory(
                donor_vectors.shape[1],
                self.index_factory,
                self.metric_type_,
            )
            index.train(donor_vectors)
            index.add(donor_vectors)

            _, neighbor_indices = index.search(
                query_vectors,
                min(int(self.n_neighbors), self.donors_.shape[0]),
            )

            if not self._uses_uniform_weights():
                for query_position, sample_idx in enumerate(sample_indices):
                    row_neighbors = neighbor_indices[query_position]
                    valid_neighbors = row_neighbors[
                        (row_neighbors >= 0)
                        & (row_neighbors < self.donors_.shape[0])
                    ]
                    if valid_neighbors.size == 0:
                        raise ValueError(
                            "FAISS did not return any valid neighbors"
                        )

                    selected_donors = self.donors_[valid_neighbors]

                    # Keep FAISS neighbor selection. Compute weighting
                    # distances directly for those selected donors only.
                    delta = selected_donors[:, observed_cols].astype(
                        np.float64
                    )
                    delta -= X[sample_idx, observed_cols].astype(np.float64)
                    squared = np.einsum("ij,ij->i", delta, delta)

                    # Match the missing-feature scaling used by
                    # nan_euclidean distances, including for callables.
                    squared *= self.n_features_in_ / observed_cols.size

                    selected_values = selected_donors[:, missing_cols].T
                    selected_distances = np.broadcast_to(
                        squared, selected_values.shape
                    )
                    X_tmp[sample_idx, missing_cols] = self._weighted_mean(
                        selected_values, selected_distances
                    )
                continue

            # Limit gathered donor values to roughly 8 MiB per chunk,
            # with a one-query minimum. Median and repair use extra storage.
            # Keep the search batch unchanged: only aggregation is split.
            neighbor_count = neighbor_indices.shape[1]
            values_per_query = neighbor_count * missing_cols.size
            bytes_per_query = self.donors_.dtype.itemsize * values_per_query
            aggregation_rows = max(
                1, min(256, (8 * 1024 * 1024) // bytes_per_query),
            )
            for start in range(0, len(sample_indices), aggregation_rows):
                rows = sample_indices[start:start + aggregation_rows]
                neighbors = neighbor_indices[start:start + aggregation_rows]

                if (neighbors < 0).any():
                    # Approximate indexes can return different valid counts.
                    # Retain filtering, duplicate ids and rowwise reduction.
                    for sample_idx, row_neighbors in zip(rows, neighbors):
                        valid_neighbors = row_neighbors[row_neighbors >= 0]
                        if valid_neighbors.size == 0:
                            raise ValueError(
                                "FAISS did not return any valid neighbors"
                            )
                        selected_values = self.donors_[
                            valid_neighbors
                        ][:, missing_cols]
                        X_tmp[sample_idx, missing_cols] = self._aggregate(
                            selected_values, axis=0, ignore_nan=False,
                        )
                    continue

                # A contiguous last axis preserves the neighbor order and
                # reduction layout of the previous columnwise aggregation.
                selected_values = np.ascontiguousarray(self.donors_[
                    neighbors[:, None, :], missing_cols[None, :, None]
                ])
                aggregates = self._aggregate(
                    selected_values.reshape(-1, neighbor_count),
                    axis=1, ignore_nan=False,
                ).reshape(len(rows), missing_cols.size)
                X_tmp[np.ix_(rows, missing_cols)] = aggregates

        return self._format_output(X_tmp, original, indicators)

    def _transform_available(self, X):
        try:
            return self._transform_available_batched(X)
        finally:
            self.available_index_.clear_cache()

    def _transform_available_batched(self, X):
        result = self._copy_or_reuse(X)
        missing = np.isnan(X)
        all_missing = missing.all(axis=1)
        result[all_missing] = self.statistics_
        rows = np.flatnonzero(missing.any(axis=1) & ~all_missing)
        n_donors = self.donors_.shape[0]
        k = min(int(self.n_neighbors), n_donors)
        required = np.minimum(k, self.available_index_.donor_counts)
        batch_size = max(
            1, min(256, (128 * 1024 * 1024) // (12 * n_donors))
        )

        for start in range(0, rows.size, batch_size):
            batch_rows = rows[start:start + batch_size]
            batch_missing = missing[batch_rows]
            columns = np.flatnonzero(batch_missing.any(axis=0))
            queries = np.ascontiguousarray(X[batch_rows])
            result[batch_rows] = np.where(
                batch_missing, self.statistics_, queries
            )
            search_k = min(n_donors, max(16, 2 * k))

            while True:
                distances, ids = self.available_index_.search(
                    queries, int(search_k)
                )
                valid = (
                    (ids >= 0)
                    & (ids < n_donors)
                    & np.isfinite(distances)
                )
                safe_ids = np.where(valid, ids, 0)
                enough = np.ones(batch_rows.size, dtype=bool)
                for col in columns:
                    values = self.donors_[safe_ids, col]
                    usable = valid & ~np.isnan(values)
                    enough &= (
                        ~batch_missing[:, col]
                        | (usable.sum(axis=1) >= required[col])
                    )

                # Use corrected search results: the float32 cache can contain
                # overflowed values that were repaired by precise refinement.
                finished = enough | (~valid).any(axis=1)
                if search_k == n_donors:
                    finished[:] = True

                for col in columns:
                    if not (finished & batch_missing[:, col]).any():
                        continue
                    values = self.donors_[safe_ids, col]
                    usable = valid & ~np.isnan(values)
                    chosen = usable & (np.cumsum(usable, axis=1) <= k)
                    fill_rows = finished & batch_missing[:, col] & chosen.any(axis=1)
                    if not fill_rows.any():
                        continue
                    if self._uses_uniform_weights():
                        selected = np.where(
                            chosen[fill_rows], values[fill_rows], np.nan
                        )
                        fill = self._aggregate(
                            selected, axis=1, ignore_nan=True
                        )
                    else:
                        # Pack only the selected neighbors for this feature.
                        # Missing slots represent unavailable distances.
                        selected_mask = chosen[fill_rows]
                        receiver_rows, candidate_cols = np.nonzero(
                            selected_mask
                        )
                        slots = (
                            np.cumsum(selected_mask, axis=1)[
                                receiver_rows, candidate_cols
                            ] - 1
                        )
                        source_rows = np.flatnonzero(fill_rows)[receiver_rows]
                        shape = (
                            int(fill_rows.sum()),
                            int(required[col]),
                        )
                        selected_values = np.full(
                            shape, np.nan, dtype=self.donors_.dtype
                        )
                        selected_distances = np.full(
                            shape, np.nan, dtype=np.float64
                        )
                        selected_values[receiver_rows, slots] = values[
                            source_rows, candidate_cols
                        ]
                        selected_distances[receiver_rows, slots] = distances[
                            source_rows, candidate_cols
                        ]
                        fill = self._weighted_mean(
                            selected_values, selected_distances
                        )

                    result[batch_rows[fill_rows], col] = fill

                if finished.all():
                    break
                if finished.any():
                    remaining = np.flatnonzero(~finished)
                    batch_rows = batch_rows[remaining]
                    batch_missing = batch_missing[remaining]
                    columns = np.flatnonzero(batch_missing.any(axis=0))
                    queries = self.available_index_.retain_queries(remaining)
                search_k = min(n_donors, 2 * search_k)

        return result
