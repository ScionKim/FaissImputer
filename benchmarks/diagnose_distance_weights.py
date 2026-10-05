"""Reference arithmetic and observation helpers for distance-weight diagnostics."""

from decimal import Decimal, localcontext
from fractions import Fraction
import sys

import numpy as np

from benchmarks.diagnose_real_data_float32 import trace_faiss_searches


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rational(value):
    return {
        "numerator": str(value.numerator),
        "denominator": str(value.denominator),
        "float64": float(value),
    }


def captured_weight_reference(values, weights, returned_value, observed_output):
    """Exact rational aggregation of the captured floating-point operands."""
    values = np.asarray(values)
    weights = np.asarray(weights)
    require(values.ndim == weights.ndim == 1 and values.shape == weights.shape
            and values.size > 0, "Expected aligned one-dimensional values and weights")
    require(np.isfinite(values).all() and np.isfinite(weights).all()
            and (weights >= 0).all(), "Captured operands must be finite with nonnegative weights")
    require(np.isfinite(returned_value) and np.isfinite(observed_output),
            "Observed aggregation outputs must be finite")
    exact_values = [Fraction.from_float(float(value)) for value in values]
    exact_weights = [Fraction.from_float(float(weight)) for weight in weights]
    total = sum(exact_weights, Fraction(0))
    require(total > 0, "Captured weights must have a positive sum")
    mean = sum((v * w for v, w in zip(exact_values, exact_weights)), Fraction(0)) / total
    return {
        "scope": "Exact arithmetic on captured floating-point target values and weights; not exact inverse-distance weights.",
        "exact_weight_sum": rational(total),
        "exact_normalized_weights": [rational(weight / total) for weight in exact_weights],
        "exact_weighted_mean": rational(mean),
        "returned_value_minus_exact_weighted_mean": rational(
            Fraction.from_float(float(returned_value)) - mean
        ),
        "assigned_output_minus_exact_weighted_mean": rational(
            Fraction.from_float(float(observed_output)) - mean
        ),
        "assignment_minus_returned_value": rational(
            Fraction.from_float(float(observed_output))
            - Fraction.from_float(float(returned_value))
        ),
    }


def distance_weighted_reference(values, squared_distances):
    """Rounded decimal inverse-distance reference for exact squared distances."""
    values = np.asarray(values)
    squared = list(squared_distances)
    require(values.ndim == 1 and len(values) == len(squared) and len(squared) > 0,
            "Expected aligned values and exact squared distances")
    require(np.isfinite(values).all(), "Reference target values must be finite")
    require(all(isinstance(d, Fraction) and d >= 0 for d in squared),
            "Reference squared distances must be nonnegative Fractions")
    zero = [d == 0 for d in squared]
    evaluations = []
    for precision in (80, 120):
        with localcontext() as context:
            context.prec = precision
            context.rounding = "ROUND_HALF_EVEN"
            exact_values = [Decimal.from_float(float(value)) for value in values]
            distances = [(Decimal(d.numerator) / Decimal(d.denominator)).sqrt()
                         for d in squared]
            if any(zero):
                weights = [Decimal(int(is_zero)) for is_zero in zero]
            else:
                minimum = min(distances)
                weights = [minimum / d for d in distances]
            total = sum(weights, Decimal(0))
            normalized = [w / total for w in weights]
            mean = sum((v * w for v, w in zip(exact_values, weights)), Decimal(0)) / total
            evaluations.append({
                "decimal_precision": precision,
                "distances": [str(d) for d in distances],
                "normalized_weights": [str(w) for w in normalized],
                "weighted_mean": str(mean),
                "weighted_mean_float64": float(mean),
            })
    return {
        "scope": "Exact rational squared distances from represented inputs, followed by rounded Decimal square roots and weighting. Not a certified exact weighted mean.",
        "zero_distance_rule": "If any selected distance is zero, only selected zero-distance donors contribute equally.",
        "zero_distance_positions": [i for i, is_zero in enumerate(zero) if is_zero],
        "evaluations": evaluations,
        "float64_approximations_agree": (
            evaluations[0]["weighted_mean_float64"] == evaluations[1]["weighted_mean_float64"]
        ),
    }


def trace_distance_faiss_selections(model, train, query, wanted_rows):
    """Observe actual aggregation operands and map them to original query rows."""
    require(model.weights == "distance", "Expected a distance-weighted model")
    previous_profile = sys.getprofile()
    require(previous_profile is None, "Distance tracing requires no active Python profiler")
    weighted_code = type(model)._weighted_mean_from_distances.__code__
    transform_code = type(model)._transform_available_batched.__code__
    wanted = set(int(row) for row in wanted_rows)
    selections = {}

    def observe(frame, event, returned):
        if event != "return" or frame.f_code is not weighted_code:
            return
        local = frame.f_locals
        if local.get("self") is not model:
            return
        parent = frame.f_back
        while parent is not None and parent.f_code is not transform_code:
            parent = parent.f_back
        require(parent is not None and parent.f_locals.get("self") is model,
                "Weighted aggregation is outside the expected available-donor transform")
        context = parent.f_locals
        positions = np.flatnonzero(context["fill_rows"])
        rows = context["batch_rows"][positions]
        column = int(context["col"])
        values, distances = local["values"], local["distances"]
        weights, totals = local["weights"], local["totals"]
        squared = context["selected_distances"]
        require(returned is not None and len(returned) == len(rows),
                "Unexpected weighted aggregation return")
        for position, (row, source_row) in enumerate(zip(rows, positions)):
            row = int(row)
            if row not in wanted:
                continue
            cell = (row, column)
            require(cell not in selections, "Faiss weighted cell captured twice")
            ids = context["safe_ids"][source_row, context["chosen"][source_row]].copy()
            require(len(ids) == int(model.n_neighbors), "Unexpected selected donor count")
            np.testing.assert_array_equal(values[position], train[ids, column])
            np.testing.assert_array_equal(
                distances[position], np.sqrt(np.maximum(squared[position], 0.0)),
            )
            require(np.isfinite(distances[position]).all()
                    and np.isfinite(weights[position]).all()
                    and totals[position] > 0, "Invalid captured Faiss operands")
            selections[cell] = {
                "training_row_indices": ids,
                "target_values": values[position].copy(),
                "search_squared_distances": squared[position].copy(),
                "weight_input_distances": distances[position].copy(),
                "captured_weights": weights[position].copy(),
                "captured_weight_total": float(totals[position]),
                "normalized_weights_float64_from_captured": (
                    weights[position].astype(np.float64) / float(totals[position])
                ),
                "returned_value": float(returned[position]),
                "returned_dtype": str(returned.dtype),
                "target_dtype": str(values.dtype),
                "distance_dtype": str(distances.dtype),
                "weight_dtype": str(weights.dtype),
                "mapping": "Actual batch_rows/fill_rows/column at the aggregation call; not query-value matching.",
                "capture": "Locals at return of _weighted_mean_from_distances; weights include its scaling.",
            }

    try:
        sys.setprofile(observe)
        output, searches, matching_rows = trace_faiss_searches(
            model, train, query, wanted_rows,
        )
    finally:
        sys.setprofile(previous_profile)
    expected = {(row, int(column)) for row in wanted
                for column in np.flatnonzero(np.isnan(query[row]))}
    require(set(selections) == expected, "Some Faiss weighted cells were not captured")
    for (row, column), selection in selections.items():
        assigned = np.asarray(selection["returned_value"], dtype=output.dtype).item()
        require(assigned == output[row, column], "Captured Faiss aggregate differs from assigned output")
        selection["assigned_output"] = float(output[row, column])
    return output, searches, matching_rows, selections
