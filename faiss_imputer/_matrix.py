"""Internal distance backend for partially observed donors."""

import faiss
import numpy as np


class MatrixNaNIndex:
    def __init__(self, donors):
        self.donors64 = np.asarray(donors, dtype=np.float64)
        self.present = ~np.isnan(self.donors64)
        self.norms = np.nansum(self.donors64 * self.donors64, axis=1)
        self.clear_cache()

    def clear_cache(self):
        self.query_ref = None
        self.matrix = None
        self.use_float64 = False

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
            from sklearn.metrics.pairwise import nan_euclidean_distances

            self.clear_cache()
            query64 = np.asarray(queries, dtype=np.float64)
            distances = nan_euclidean_distances(
                query64, self.donors64, squared=True, copy=True
            )
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

            if suspect.any():
                for row in np.flatnonzero(suspect):
                    distances[row] = self._direct_distances(query64[row])
                self.matrix = distances
                self.use_float64 = True
            else:
                self.matrix = matrix32
            self.query_ref = queries

        k = min(int(k), self.matrix.shape[1])
        if not self.use_float64:
            return faiss.kmin(self.matrix, k)

        ids = np.argpartition(self.matrix, k - 1, axis=1)[:, :k]
        values = np.take_along_axis(self.matrix, ids, axis=1)
        order = np.argsort(values, axis=1, kind="stable")
        values = np.take_along_axis(values, order, axis=1)
        ids = np.take_along_axis(ids, order, axis=1)
        ids = np.where(np.isfinite(values), ids, -1)
        return values, ids
