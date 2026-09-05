"""Internal distance backend for partially observed donors."""

import faiss
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.extmath import row_norms


class MatrixNaNIndex:
    def __init__(self, donors):
        # Own this buffer: caller data and public donors_ retain their NaNs.
        self.donors64 = np.array(donors, dtype=np.float64, copy=True)
        self.present = ~np.isnan(self.donors64)
        # Preserve the reduction used by the existing numerical-risk guard.
        self.norms = np.nansum(self.donors64 * self.donors64, axis=1)
        self.missing_donors = ~self.present
        # Internal donors64 is zero-filled; masks preserve missingness.
        self.donors64[self.missing_donors] = 0.0
        self.squared_donors = self.donors64 * self.donors64
        self.zero_norms = row_norms(self.donors64, squared=True)
        self.clear_cache()

    def _prepared_distances(self, queries):
        X = queries.copy()
        missing_X = np.isnan(X)
        X[missing_X] = 0.0
        distances = euclidean_distances(
            X,
            self.donors64,
            squared=True,
            Y_norm_squared=self.zero_norms,
        )
        XX = X * X
        distances -= np.dot(XX, self.missing_donors.T)
        distances -= np.dot(missing_X, self.squared_donors.T)
        np.clip(distances, 0, None, out=distances)
        present_count = np.dot(1 - missing_X, self.present.T)
        distances[present_count == 0] = np.nan
        np.maximum(1, present_count, out=present_count)
        distances /= present_count
        distances *= X.shape[1]
        return distances

    def clear_cache(self):
        self.query_ref = None
        self.query64 = None
        self.matrix = None
        self.precise_rows = {}

    def _direct_distances(self, query):
        shared = self.present & ~np.isnan(query)
        counts = shared.sum(axis=1)
        delta = np.where(shared, self.donors64 - query, 0.0)
        squared = np.sum(delta * delta, axis=1)
        distances = np.full(len(self.donors64), np.inf, dtype=np.float64)
        usable = counts > 0
        distances[usable] = (
            squared[usable] * self.donors64.shape[1] / counts[usable]
        )
        return distances

    def search(self, queries, k):
        if queries is not self.query_ref:
            self.clear_cache()
            query64 = np.asarray(queries, dtype=np.float64)
            distances = self._prepared_distances(query64)
            finite = np.isfinite(distances)
            query_norms = np.nansum(query64 * query64, axis=1)
            p = self.donors64.shape[1]

            # Conservative suspicion test, not a proven error bound.
            tolerance = (
                64 * np.finfo(np.float64).eps * p * p
                * (query_norms[:, None] + self.norms[None, :])
            )
            suspect_pairs = finite & (distances <= tolerance)
            distances[~finite] = np.inf

            with np.errstate(over="ignore", under="ignore"):
                matrix32 = distances.astype(np.float32)
            suspect_pairs |= finite & (
                (matrix32 >= np.finfo(np.float32).max)
                | ((distances > 0) & (matrix32 == 0))
            )

            query_missing = np.isnan(query64)
            suspect = np.zeros(query64.shape[0], dtype=bool)
            chunk_size = max(1, min(4096, (1024 * 1024) // max(p, 1)))
            for row in np.flatnonzero(suspect_pairs.any(axis=1)):
                candidates = np.flatnonzero(suspect_pairs[row])
                for start in range(0, candidates.size, chunk_size):
                    donor_rows = candidates[start:start + chunk_size]
                    can_fill = self.present[donor_rows] & query_missing[row]
                    if can_fill.any():
                        suspect[row] = True
                        break

            self.query64 = query64
            self.matrix = matrix32
            
            for row in np.flatnonzero(suspect):
                self.precise_rows[int(row)] = self._direct_distances(query64[row])
            
            self.query_ref = queries

        k = min(int(k), self.matrix.shape[1])
        probe_k = min(k + 1, self.matrix.shape[1])
        
        probe_values, probe_ids = faiss.kmin(self.matrix, probe_k)
        values = probe_values[:, :k]
        ids = probe_ids[:, :k]
        
        ambiguous = np.zeros(self.matrix.shape[0], dtype=bool)
        
        ambiguous = np.zeros(self.matrix.shape[0], dtype=bool)
        query_missing = np.isnan(self.query64)
        
        for row in range(self.matrix.shape[0]):
            tie_values = []
        
            if k > 1:
                tied = (
                    np.isfinite(values[row, 1:])
                    & (values[row, 1:] == values[row, :-1])
                )
                if tied.any():
                    tie_values.extend(values[row, 1:][tied])
        
            if (
                probe_k > k
                and np.isfinite(probe_values[row, k - 1])
                and probe_values[row, k - 1] == probe_values[row, k]
            ):
                tie_values.append(probe_values[row, k - 1])
        
            for tie_value in np.unique(tie_values):
                tied_donors = np.flatnonzero(
                    np.isfinite(self.matrix[row])
                    & (self.matrix[row] == tie_value)
                )
        
                can_fill = (
                    self.present[tied_donors]
                    & query_missing[row]
                )
        
                # Exact ordering matters only when at least two tied donors
                # compete to fill the same missing feature.
                if np.any(can_fill.sum(axis=0) >= 2):
                    ambiguous[row] = True
                    break
        
        for row in np.flatnonzero(ambiguous):
            row = int(row)
            if row not in self.precise_rows:
                self.precise_rows[row] = self._direct_distances(self.query64[row])
        
        if self.precise_rows:
            values = values.astype(np.float64)
        
            for row, exact in self.precise_rows.items():
                order = np.argsort(exact, kind="stable")[:k]
                row_values = exact[order]
        
                values[row] = row_values
                ids[row] = np.where(np.isfinite(row_values), order, -1)
        
        return values, ids
