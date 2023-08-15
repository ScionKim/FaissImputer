import numpy as np
import faiss
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

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
        X = check_array(X, dtype=np.float32, force_all_finite='allow-nan')

        # Check parameters
        if not isinstance(self.n_neighbors, int) or self.n_neighbors <= 0:
            raise ValueError("n_neighbors must be a positive integer")

        if self.metric not in ('l2', 'ip'):
            raise ValueError("metric must be either 'l2' or 'ip'")

        if self.strategy not in ('mean', 'median'):
            raise ValueError("strategy must be either 'mean' or 'median'")

        # Extract non-missing data
        mask = ~np.isnan(X).any(axis=1)
        X_non_missing = X[mask]

        # Build faiss index
        index = faiss.index_factory(X_non_missing.shape[1], self.index_factory)
        index.metric_type = faiss.METRIC_L2 if self.metric == 'l2' else faiss.METRIC_INNER_PRODUCT
        index.train(X_non_missing)
        index.add(X_non_missing)

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
        # Check input data
        X = check_array(X, dtype=np.float32, force_all_finite='allow-nan')

        # Check if fit is called
        check_is_fitted(self)

        # Copy X to avoid modifying the original data
        X_tmp = X.copy()

        # Find the missing values
        missing_mask = np.isnan(X)
        
        # Generate placeholder values for imputation (mean or median)
        if self.strategy == 'mean':
            placeholder_values = np.nanmean(X, axis=0)
        elif self.strategy == 'median':
            placeholder_values = np.nanmedian(X, axis=0)
        
        # Loop over each sample with missing values
        for sample_idx in np.where(missing_mask.any(axis=1))[0]:
            # Extract row and the missing mask for that sample
            sample_row = X[sample_idx, :]
            sample_missing_mask = missing_mask[sample_idx, :]

            # Find the missing values and their corresponding columns
            sample_missing_cols = np.where(sample_missing_mask)[0]
            sample_row[sample_missing_cols] = placeholder_values[sample_missing_cols]

            # Impute missing values using k nearest neighbors
            _, neighbor_indices = self.index_.search(sample_row.reshape(1, -1), self.n_neighbors)
            selected_vectors = X[neighbor_indices[0]]
            selected_values = selected_vectors[:, sample_missing_cols]

            if self.strategy == 'mean':
                column_agg = np.nanmean(selected_values, axis=0)
            elif self.strategy == 'median':
                column_agg = np.nanmedian(selected_values, axis=0)

            sample_row[sample_missing_cols] = column_agg

            # Update the imputed row in the temporary copy
            X_tmp[sample_idx] = sample_row

        return X_tmp