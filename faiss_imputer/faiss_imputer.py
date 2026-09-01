import numpy as np
import faiss
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data

class FaissImputer(BaseEstimator, TransformerMixin):
    """Impute missing values using faiss."""

    def __init__(self, n_neighbors=3, metric='l2', strategy='mean', index_factory='Flat'):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.strategy = strategy
        self.index_factory = index_factory

    def fit(self, X, y=None):
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
                self.n_neighbors,
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
