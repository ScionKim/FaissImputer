"""Internal distance backend for partially observed donors."""

import faiss
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.extmath import row_norms


class MatrixNaNIndex:
    def __init__(self, donors, *, n_features=None):
        source_dtype = np.asarray(donors).dtype
        self._preserve_float64 = (
            source_dtype.kind == "f" and source_dtype.itemsize == 8
        )
        self.donors64 = np.array(donors, dtype=np.float64, copy=True)
        self.n_features = (
            self.donors64.shape[1] if n_features is None else n_features
        )
        self.present = ~np.isnan(self.donors64)
        self.donor_counts = self.present.sum(axis=0)

        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            self.norms = np.nansum(self.donors64 * self.donors64, axis=1)

        self.missing_donors = ~self.present
        self.donors64[self.missing_donors] = 0.0

        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            self.squared_donors = self.donors64 * self.donors64
            self.zero_norms = row_norms(self.donors64, squared=True)

        self.clear_cache()

    def _prepared_distances(self, queries):
        X = np.array(queries, dtype=np.float64, order="C", copy=True)
        missing_X = np.isnan(X)
        X[missing_X] = 0.0
        norms = self.zero_norms if np.isfinite(self.zero_norms).all() else None

        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            distances = euclidean_distances(
                X,
                self.donors64,
                squared=True,
                Y_norm_squared=norms,
            )

            # Reuse one query-by-donor buffer for both corrections
            # and the shared-feature counts.
            workspace = np.empty(distances.shape, dtype=np.float64)

            np.dot(X * X, self.missing_donors.T, out=workspace)
            distances -= workspace

            np.dot(missing_X, self.squared_donors.T, out=workspace)
            distances -= workspace
            np.clip(distances, 0, None, out=distances)

            # Floating-point matrix multiplication avoids the integer
            # dot-product path when counting shared observed features.
            present_X = (~missing_X).astype(np.float64)
            np.dot(present_X, self.present.T, out=workspace)

            distances[workspace == 0] = np.nan
            np.maximum(1.0, workspace, out=workspace)
            distances /= workspace
            distances *= self.n_features

        return distances

    def clear_cache(self):
        self.query_ref = None
        self.matrix = None
        self.precise_rows = {}

    def retain_queries(self, rows):
        """Keep selected cached queries in the supplied row order."""
        queries = np.ascontiguousarray(self.query_ref[rows])
        matrix = self.matrix[rows]
        precise_rows = {
            new_row: self.precise_rows[int(old_row)]
            for new_row, old_row in enumerate(rows)
            if int(old_row) in self.precise_rows
        }
        self.query_ref = queries
        self.matrix = matrix
        self.precise_rows = precise_rows
        return queries

    def _distances_to(self, query, donors, present):
        shared = present & ~np.isnan(query)
        counts = shared.sum(axis=1)
        usable = counts > 0
        delta = np.zeros_like(donors, dtype=np.float64)
        distances = np.full(len(donors), np.inf, dtype=np.float64)

        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            np.subtract(donors, query, out=delta, where=shared)
            squared = np.sum(delta * delta, axis=1)
            distances[usable] = (
                squared[usable] * self.n_features / counts[usable]
            )

            positive = np.any(delta != 0, axis=1)
            repair = usable & (
                ~np.isfinite(distances) | ((distances == 0) & positive)
            )
            if repair.any():
                selected = delta[repair]
                scales = np.max(np.abs(selected), axis=1)
                scaled = selected / scales[:, None]
                normalized = np.sum(scaled * scaled, axis=1)
                normalized *= self.n_features / counts[repair]
                mantissas, exponents = np.frexp(scales)
                distances[repair] = np.ldexp(
                    normalized * mantissas * mantissas, 2 * exponents
                )

        if (
            (~np.isfinite(distances[usable])).any()
            or ((distances == 0) & positive & usable).any()
        ):
            raise ValueError(
                "Squared distances must be finite and representable as float64"
            )

        return distances

    def _direct_distances(self, query):
        return self._distances_to(query, self.donors64, self.present)

    @staticmethod
    def _precise_topk(distances, k):
        if k < distances.size:
            cutoff = np.partition(distances, k - 1)[k - 1]
            closer = np.flatnonzero(distances < cutoff)
            tied = np.flatnonzero(distances == cutoff)[:k - closer.size]
            ids = np.concatenate((closer, tied))
        else:
            ids = np.arange(distances.size)
        # Each equal-distance group starts in training-row order. Sort only
        # the selected candidates, retaining that order for genuine ties.
        order = np.argsort(distances[ids], kind="stable")
        ids = ids[order]
        values = distances[ids]
        return values, np.where(np.isfinite(values), ids, -1)

    def _prepare_search_matrix(self, query64):
        distances = self._prepared_distances(query64)
        matrix32 = np.empty(distances.shape, dtype=np.float32)
        query_missing = np.isnan(query64)
        suspect = np.zeros(query64.shape[0], dtype=bool)
        p = self.n_features

        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            query_norms = np.nansum(query64 * query64, axis=1)
            tolerance_scale = 64 * np.finfo(np.float64).eps * p * p

        # Limit temporary pairwise arrays to a group of query rows.
        pair_budget = 256 * 1024
        rows_per_chunk = max(
            1, pair_budget // max(distances.shape[1], 1)
        )
        donor_chunk_size = max(
            1, min(4096, (1024 * 1024) // max(p, 1))
        )
        maximum32 = np.finfo(np.float32).max

        for row_start in range(0, len(query64), rows_per_chunk):
            row_stop = min(row_start + rows_per_chunk, len(query64))
            block = distances[row_start:row_stop]
            finite = np.isfinite(block)

            # Conservative suspicion test, not a proven error bound.
            with np.errstate(over="ignore", invalid="ignore", under="ignore"):
                tolerance = (
                    query_norms[row_start:row_stop, None]
                    + self.norms[None, :]
                )
                tolerance *= tolerance_scale

            suspect_pairs = ~finite
            suspect_pairs |= block <= tolerance
            block[~finite] = np.inf

            matrix_block = matrix32[row_start:row_stop]
            with np.errstate(over="ignore", under="ignore"):
                matrix_block[:] = block

            suspect_pairs |= finite & (
                (matrix_block >= maximum32)
                | ((block > 0) & (matrix_block == 0))
            )

            for local_row in np.flatnonzero(suspect_pairs.any(axis=1)):
                row = row_start + int(local_row)
                candidates = np.flatnonzero(suspect_pairs[local_row])

                for start in range(0, candidates.size, donor_chunk_size):
                    donor_rows = candidates[start:start + donor_chunk_size]
                    can_fill = self.present[donor_rows] & query_missing[row]
                    has_shared = (
                        self.present[donor_rows] & ~query_missing[row]
                    ).any(axis=1)

                    if np.any(can_fill.any(axis=1) & has_shared):
                        suspect[row] = True
                        break

        # The float64 distance matrix and temporary masks are released
        # when this method returns, before precise rows are recomputed.
        return matrix32, suspect

    def search(self, queries, k):
        if queries is not self.query_ref:
            self.clear_cache()
            query64 = np.asarray(queries, dtype=np.float64)
            self.matrix, suspect = self._prepare_search_matrix(query64)

            for row in np.flatnonzero(suspect):
                self.precise_rows[int(row)] = self._direct_distances(
                    query64[row]
                )
            self.query_ref = queries

        k = min(int(k), self.matrix.shape[1])
        probe_k = min(k + 1, self.matrix.shape[1])
        probe_values, probe_ids = faiss.kmin(self.matrix, probe_k)

        # Includes ties crossing the selection boundary. Ordinary rows require
        # no Python-level loop; rows already refined need no further tie check.
        tied = (
            np.isfinite(probe_values[:, 1:])
            & (probe_values[:, 1:] == probe_values[:, :-1])
        )
        for row in np.flatnonzero(tied.any(axis=1)):
            row = int(row)
            if row in self.precise_rows:
                continue
            missing = np.isnan(queries[row])
            for value in np.unique(probe_values[row, 1:][tied[row]]):
                # Include the whole tie group, even donors outside the probe.
                donor_ids = np.flatnonzero(self.matrix[row] == value)
                can_fill = self.present[donor_ids] & missing
                if np.any(can_fill.sum(axis=0) >= 2):
                    query64 = np.asarray(queries[row], dtype=np.float64)
                    self.precise_rows[row] = self._direct_distances(query64)
                    break

        values = probe_values[:, :k]
        ids = probe_ids[:, :k]
        preserve_float64 = (
            self._preserve_float64 or queries.dtype.itemsize > 4
        )
        if self.precise_rows or preserve_float64:
            values = values.astype(np.float64)

        for row, exact in self.precise_rows.items():
            values[row], ids[row] = self._precise_topk(exact, k)

        if preserve_float64:
            for row in range(len(queries)):
                if row in self.precise_rows:
                    continue
                valid = (ids[row] >= 0) & (ids[row] < len(self.donors64))
                values[row, ~valid] = np.inf
                if valid.any():
                    selected = ids[row, valid]
                    values[row, valid] = self._distances_to(
                        np.asarray(queries[row], dtype=np.float64),
                        self.donors64[selected],
                        self.present[selected],
                    )

        return values, ids
