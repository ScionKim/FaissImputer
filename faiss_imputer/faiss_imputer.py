import numpy as np
import faiss
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data

class FaissImputer(TransformerMixin, BaseEstimator):
    """Impute missing values using faiss."""

    def __init__(
        self,
        n_neighbors=3,
        metric="l2",
        strategy="mean",
        index_factory="Flat",
        donor_policy="complete",
    ):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.strategy = strategy
        self.index_factory = index_factory
        self.donor_policy = donor_policy

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.transformer_tags.preserves_dtype = ["float32"]
        return tags

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
            "donor_policy_",
            "donor_groups_",
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
        if not isinstance(self.n_neighbors, int) or self.n_neighbors <= 0:
            raise ValueError("n_neighbors must be a positive integer")

        if self.metric not in ('l2', 'ip'):
            raise ValueError("metric must be either 'l2' or 'ip'")

        if self.strategy not in ('mean', 'median'):
            raise ValueError("strategy must be either 'mean' or 'median'")

        if self.donor_policy not in ("complete", "available"):
            raise ValueError(
                "donor_policy must be either 'complete' or 'available'"
            )

        self.donor_policy_ = self.donor_policy
        if self.donor_policy_ == "available":
            return self._fit_available(X)

        if self.strategy == 'mean':
            self.statistics_ = np.nanmean(X, axis=0)
        else:
            self.statistics_ = np.nanmedian(X, axis=0)

        # Extract non-missing data
        mask = ~np.isnan(X).any(axis=1)
        self.donors_ = X[mask].copy()
        
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
        index.train(self.donors_)
        index.add(self.donors_)

        # Store the index as an attribute
        self.index_ = index

        return self

    def _fit_available(self, X):
        if self.metric != "l2" or self.index_factory != "Flat":
            raise ValueError(
                "donor_policy='available' requires "
                "metric='l2' and index_factory='Flat'"
            )

        observed = ~np.isnan(X)
        if not observed.any(axis=0).all():
            raise ValueError("X must not contain all-missing columns")

        if self.strategy == "mean":
            self.statistics_ = np.nanmean(X, axis=0)
        else:
            self.statistics_ = np.nanmedian(X, axis=0)

        nonempty_rows = observed.any(axis=1)
        self.donors_ = X[nonempty_rows].copy()
        observed = observed[nonempty_rows]
        self.metric_type_ = faiss.METRIC_L2

        groups = {}
        for row_idx, row_mask in enumerate(observed):
            pattern = tuple(row_mask.tolist())
            groups.setdefault(pattern, []).append(row_idx)

        self.donor_groups_ = [
            (
                np.asarray(pattern, dtype=bool),
                np.asarray(indices, dtype=np.intp),
            )
            for pattern, indices in groups.items()
        ]

        return self

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

        if self.donor_policy_ == "available":
            all_complete = (
                len(self.donor_groups_) == 1
                and self.donor_groups_[0][0].all()
            )
            if not all_complete:
                return self._transform_available(X)

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
                min(self.n_neighbors, self.donors_.shape[0]),
            )

            for sample_idx, neighbors in zip(
                sample_indices,
                neighbor_indices,
            ):
                valid_neighbors = neighbors[neighbors >= 0]

                if valid_neighbors.size == 0:
                    raise ValueError(
                        "FAISS did not return any valid neighbors"
                    )

                selected_values = self.donors_[
                    valid_neighbors
                ][:, missing_cols]

                if self.strategy == 'mean':
                    column_agg = np.mean(
                        selected_values,
                        axis=0,
                    )
                else:
                    column_agg = np.median(
                        selected_values,
                        axis=0,
                    )

                X_tmp[sample_idx, missing_cols] = column_agg

        return X_tmp

    def _transform_available(self, X):
        result = X.copy()
        missing = np.isnan(X)
        donor_observed = ~np.isnan(self.donors_)

        for row in np.flatnonzero(missing.any(axis=1)):
            missing_cols = np.flatnonzero(missing[row])
            observed_cols = np.flatnonzero(~missing[row])
            result[row, missing_cols] = self.statistics_[missing_cols]
            if observed_cols.size == 0:
                continue

            common_counts = donor_observed[:, observed_cols].sum(axis=1)
            has_target = donor_observed[:, missing_cols].any(axis=1)
            donor_rows = np.flatnonzero((common_counts > 0) & has_target)
            if donor_rows.size == 0:
                continue

            target_present = donor_observed[np.ix_(donor_rows, missing_cols)]
            k = min(self.n_neighbors, donor_rows.size)
            required = np.minimum(k, target_present.sum(axis=0))

            index = faiss.IndexFlatL2(int(observed_cols.size))
            block_size = max(
                1, (16 * 1024 * 1024) // (8 * observed_cols.size)
            )
            for start in range(0, donor_rows.size, block_size):
                block_rows = donor_rows[start:start + block_size]
                offsets = self.donors_[
                    np.ix_(block_rows, observed_cols)
                ].astype(np.float64)
                offsets -= X[row, observed_cols]
                offsets[np.isnan(offsets)] = 0.0
                # Omitting the common factor X.shape[1] preserves ranking.
                offsets /= np.sqrt(common_counts[block_rows])[:, None]
                index.add(np.ascontiguousarray(offsets, dtype=np.float32))

            query = np.zeros((1, observed_cols.size), dtype=np.float32)
            search_k = min(donor_rows.size, max(16, 2 * k))
            while True:
                distances, indices = index.search(query, int(search_k))
                local_ids = indices[0]
                valid = (
                    (local_ids >= 0)
                    & (local_ids < donor_rows.size)
                    & np.isfinite(distances[0])
                )
                local_ids = local_ids[valid]
                _, first = np.unique(local_ids, return_index=True)
                local_ids = local_ids[np.sort(first)]
                hits = target_present[local_ids]

                if (
                    np.all(hits.sum(axis=0) >= required)
                    or search_k == donor_rows.size
                ):
                    break
                search_k = min(donor_rows.size, 2 * search_k)

            for target_idx, col in enumerate(missing_cols):
                selected = local_ids[hits[:, target_idx]][:k]
                if selected.size == 0:
                    continue
                values = self.donors_[donor_rows[selected], col]
                if self.strategy == "mean":
                    result[row, col] = np.mean(values)
                else:
                    result[row, col] = np.median(values)

        return result
