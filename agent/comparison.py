from __future__ import annotations

from itertools import combinations

from .metrics import METRIC_DEFINITIONS


PERCENTAGE_METRICS = {
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "roc_auc",
    "pr_auc",
    "specificity",
    "mape",
    "bleu",
    "rouge",
}

METRIC_FAMILIES = {
    "accuracy": "classification",
    "precision": "classification",
    "recall": "classification",
    "f1_score": "classification",
    "roc_auc": "classification",
    "pr_auc": "classification",
    "specificity": "classification",
    "log_loss": "classification",
    "mse": "regression",
    "rmse": "regression",
    "mae": "regression",
    "mape": "regression",
    "r_squared": "regression",
    "bleu": "generation",
    "rouge": "generation",
    "reasoning": "benchmark_quality",
    "coding": "benchmark_quality",
    "factuality": "benchmark_quality",
    "math": "benchmark_quality",
    "dataset_size": "dataset_context",
    "parameter_count": "model_context",
    "latency": "efficiency",
    "throughput": "efficiency",
}


def _claim_value(observation, metric):
    value = observation.get("value")
    if not isinstance(value, (int, float)):
        return None

    unit = observation.get("unit")
    if metric in PERCENTAGE_METRICS:
        if unit == "percentage":
            return float(value), float(value) * 100, True
        if unit == "value" and 0 <= value <= 1:
            return float(value), float(value) * 100, True
    if unit == "percentage":
        return float(value), float(value) * 100, True
    return float(value), float(value), False


def compare_claim_metrics(claim_observations):
    models = list(dict.fromkeys(row["model"] for row in claim_observations if row.get("model") != "unspecified"))
    metrics = list(dict.fromkeys(row["metric"] for row in claim_observations))
    metric_results = []

    for metric in metrics:
        observations_by_model = {}
        for observation in claim_observations:
            if observation.get("metric") == metric:
                observations_by_model.setdefault(observation["model"], []).append(observation)

        for model_a, model_b in combinations(models, 2):
            values_a = observations_by_model.get(model_a, [])
            values_b = observations_by_model.get(model_b, [])
            if len(values_a) != 1 or len(values_b) != 1:
                continue

            observation_a = values_a[0]
            observation_b = values_b[0]
            direction = METRIC_DEFINITIONS.get(metric, {}).get("direction", "context_dependent")
            normalized_a = _claim_value(observation_a, metric)
            normalized_b = _claim_value(observation_b, metric)
            measurement_unit_a = observation_a.get("measurement_unit", "")
            measurement_unit_b = observation_b.get("measurement_unit", "")
            units_match = observation_a.get("unit") == observation_b.get("unit")
            ratio_match = metric in PERCENTAGE_METRICS and normalized_a is not None and normalized_b is not None
            measurement_units_match = measurement_unit_a == measurement_unit_b
            comparable = (
                direction != "context_dependent"
                and normalized_a is not None
                and normalized_b is not None
                and measurement_units_match
                and (units_match or ratio_match)
            )

            if not comparable:
                continue
            if normalized_a is None or normalized_b is None:
                continue

            value_a, display_a, percentage_scale_a = normalized_a
            value_b, display_b, percentage_scale_b = normalized_b
            if percentage_scale_a != percentage_scale_b:
                continue

            if value_a == value_b:
                preferred_model = "equal"
            elif direction == "lower_is_better":
                preferred_model = model_a if value_a < value_b else model_b
            else:
                preferred_model = model_a if value_a > value_b else model_b

            display_difference = abs(display_a - display_b)
            difference_unit = (
                "percentage points"
                if percentage_scale_a
                else measurement_unit_a or "reported units"
            )
            relative_difference = abs(value_a - value_b) / abs(value_b) * 100 if value_b != 0 else None
            metric_results.append(
                {
                    "metric": metric,
                    "metric_family": METRIC_FAMILIES.get(metric, "other"),
                    "model_a": model_a,
                    "model_a_display": display_a,
                    "model_b": model_b,
                    "model_b_display": display_b,
                    "absolute_difference": abs(value_a - value_b),
                    "display_difference": display_difference,
                    "difference_unit": difference_unit,
                    "measurement_unit": measurement_unit_a,
                    "relative_difference_percent": relative_difference,
                    "relative_difference_reference": model_b,
                    "direction": direction,
                    "preferred_model": preferred_model,
                    "comparable": True,
                }
            )

    preferred_models = {row["preferred_model"] for row in metric_results if row["preferred_model"] != "equal"}
    if len(preferred_models) > 1:
        comparison_type = "MULTI_METRIC_TRADEOFF"
    elif metric_results and preferred_models:
        comparison_type = "CONSISTENT_REPORTED_ADVANTAGE"
    elif metric_results:
        comparison_type = "EQUAL_REPORTED_VALUES"
    else:
        comparison_type = "COMPARISON_UNAVAILABLE"

    return {
        "models": models,
        "metrics": metrics,
        "comparison_available": bool(metric_results),
        "comparison_type": comparison_type,
        "metric_results": metric_results,
    }
