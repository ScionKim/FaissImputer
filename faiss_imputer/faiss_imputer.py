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

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.transformer_tags.preserves_dtype = ["float32"]
        return tags

    def _aggregate(self, values, *, axis, ignore_nan):
        """Reduce a 2-D array, repairing only nonfinite aggregation results."""
        if self.strategy == "mean":
            aggregate = np.nanmean if ignore_nan else np.mean
        else:
            aggregate = np.nanmedian if ignore_nan else np.median

        # Keep the existing float32 reduction for ordinary inputs. Finite
        # donor values can still overflow its intermediate sum or midpoint.
        with np.errstate(over="ignore", invalid="ignore"):
            result = aggregate(values, axis=axis)

        for position in np.flatnonzero(~np.isfinite(result)):
            selected = values[:, position] if axis == 0 else values[position, :]
            if ignore_nan and np.isnan(selected).all():
                # Preserve the undefined result and original warning for an
                # all-missing slice instead of reducing it a second time.
                continue
            # Recompute one affected slice at a time to bound temporary
            # float64 storage. Assignment retains the original result dtype.
            result[position] = aggregate(selected.astype(np.float64))

        return result

    def _uses_uniform_weights(self):
        return self.weights is None or (
            isinstance(self.weights, str) and self.weights == "uniform"
        )

    def _weighted_mean(self, values, squared_distances):
        """Average selected donors using actual, unsquared distances."""
        valid = ~np.isnan(values) & np.isfinite(squared_distances)
        distances = np.sqrt(
            np.maximum(
                np.asarray(squared_distances, dtype=np.float64), 0.0
            )
        )
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
            np.divide(
                1.0,
                distances,
                out=weights,
                where=valid & (distances > 0),
            )

            # Exact matches exclude all nonzero-distance neighbors.
            zero_distance = valid & (distances == 0)
            zero_rows = zero_distance.any(axis=1)
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

        if (
            not np.isfinite(result).all()
            or (np.abs(result) > np.finfo(np.float32).max).any()
        ):
            raise ValueError(
                "weighted mean must be finite and representable as float32"
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
        X = validate_data(
            self,
            X,
            dtype=np.float32,
            ensure_all_finite='allow-nan',
            reset=True,
        )

        # Check parameters
        if (
            isinstance(self.n_neighbors, (bool, np.bool_))
            or not isinstance(self.n_neighbors, Integral)
            or self.n_neighbors <= 0
        ):
            raise ValueError("n_neighbors must be a positive integer")

        if self.metric not in ('l2', 'ip'):
            raise ValueError("metric must be either 'l2' or 'ip'")

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
            if self.metric != "l2":
                raise ValueError(
                    "non-uniform weights require metric='l2'"
                )

        if self.donor_policy not in ("complete", "available"):
            raise ValueError(
                "donor_policy must be either 'complete' or 'available'"
            )

        if not isinstance(self.add_indicator, (bool, np.bool_)):
            raise ValueError("add_indicator must be a boolean")

        if not isinstance(self.keep_empty_features, (bool, np.bool_)):
            raise ValueError("keep_empty_features must be a boolean")

        if self.donor_policy == "available" and (
            self.metric != "l2" or self.index_factory != "Flat"
        ):
            raise ValueError(
                "donor_policy='available' requires "
                "metric='l2' and index_factory='Flat'"
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
                    faiss.METRIC_L2 if self.metric == "l2"
                    else faiss.METRIC_INNER_PRODUCT
                )
                faiss.index_factory(
                    self.n_features_in_, self.index_factory, metric_type
                )
            # There is no value to estimate and no distance index to build.
            self.statistics_ = np.empty(0, dtype=np.float32)
            self.donors_ = np.empty((0, 0), dtype=np.float32)
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

        # Build faiss index
        self.metric_type_ = (
            faiss.METRIC_L2
            if self.metric == 'l2'
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
            index.train(self.donors_)
            index.add(self.donors_)

        # Store the index as an attribute
        self.index_ = index

        return self

    def _fit_available(self, X):
        observed = ~np.isnan(X)

        self.statistics_ = self._aggregate(X, axis=0, ignore_nan=True)

        nonempty_rows = observed.any(axis=1)
        self.donors_ = X[nonempty_rows].copy()
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

    def _format_output(self, imputed, original):
        """Restore retained empty columns before appending original indicators."""
        if self.keep_empty_features and not self.valid_features_.all():
            restored = np.zeros(original.shape, dtype=np.float32)
            restored[:, self.valid_features_] = imputed
            imputed = restored
        return self._append_indicator(imputed, original)

    def _append_indicator(self, imputed, original):
        """Append indicators computed from the original query values."""
        if self.indicator_ is None:
            return imputed

        indicators = self.indicator_.transform(original)
        if indicators.shape[1] == 0:
            return imputed

        # Boolean indicators become 0/1 in the float32 output array.
        return np.concatenate((imputed, indicators), axis=1)

    def transform(self, X):
        """
        Impute missing values in the provided data using the fitted Faiss index.

        Parameters:
        - X (array-like): The input data with missing values to be imputed.

        Returns:
        - X_tmp (array-like): A copy of the input data with imputed missing values.
        """
        
        # Check if fit is called
        check_is_fitted(self)

        X = validate_data(
            self,
            X,
            dtype=np.float32,
            ensure_all_finite='allow-nan',
            reset=False,
        )

        original = X
        if not self.valid_features_.all():
            X = X[:, self.valid_features_]
        if X.shape[1] == 0:
            return self._format_output(X.copy(), original)

        if self.donor_policy_ == "available":
            imputed = self._transform_available(X)
            return self._format_output(imputed, original)

        # Copy X to avoid modifying the original data
        X_tmp = X.copy()

        # Find the missing values
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

            donor_vectors = np.ascontiguousarray(
                self.donors_[:, observed_cols],
                dtype=np.float32,
            )
            query_vectors = np.ascontiguousarray(
                X[np.ix_(sample_indices, observed_cols)],
                dtype=np.float32,
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
                    delta -= query_vectors[query_position]
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

            # Limit the gathered float32 values to roughly 8 MiB per chunk,
            # with a one-query minimum. Median and repair use extra storage.
            # Keep the FAISS search batch unchanged: only aggregation is split.
            neighbor_count = neighbor_indices.shape[1]
            values_per_query = neighbor_count * missing_cols.size
            aggregation_rows = max(
                1, min(256, (8 * 1024 * 1024) // (4 * values_per_query)),
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

        return self._format_output(X_tmp, original)

    def _transform_available(self, X):
        try:
            return self._transform_available_batched(X)
        finally:
            self.available_index_.clear_cache()

    def _transform_available_batched(self, X):
        result = X.copy()
        missing = np.isnan(X)
        result[missing] = np.broadcast_to(self.statistics_, X.shape)[missing]
        rows = np.flatnonzero(missing.any(axis=1) & ~missing.all(axis=1))
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
            queries = np.ascontiguousarray(X[batch_rows], dtype=np.float32)
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
                            shape, np.nan, dtype=np.float32
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
