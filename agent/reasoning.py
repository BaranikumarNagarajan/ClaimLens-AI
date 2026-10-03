from __future__ import annotations

import json
import re
from typing import Any, cast

from .config import settings


METRIC_LABELS = {
    "f1_score": "F1 score",
    "roc_auc": "ROC AUC",
    "pr_auc": "PR AUC",
    "log_loss": "Log loss",
    "r_squared": "R-squared",
    "mse": "MSE",
    "rmse": "RMSE",
    "mae": "MAE",
    "mape": "MAPE",
    "dataset_size": "Dataset size",
    "parameter_count": "Parameter count",
    "reasoning": "Reasoning",
    "coding": "Coding",
    "factuality": "Factuality",
    "math": "Math",
    "benchmark_score": "Benchmark score",
}


def _format_value(value, difference_unit):
    rendered = f"{value:g}"
    return f"{rendered}%" if difference_unit == "percentage points" else rendered


def build_reasoning(
    comparison,
    evidence_count,
    relevant_evidence_count,
    retrieval=None,
    evidence_comparability=None,
):
    metric_results = comparison["metric_results"]
    tradeoff = comparison["comparison_type"] == "MULTI_METRIC_TRADEOFF"

    metric_insights = []
    for result in metric_results:
        label = METRIC_LABELS.get(result["metric"], result["metric"].replace("_", " ").title())
        difference = f"{result['display_difference']:g} {result['difference_unit']}"
        if result["preferred_model"] == "equal":
            conclusion = "The reported values are equal."
        else:
            conclusion = f"The more favorable reported value is {result['preferred_model']} by {difference}."
        metric_insights.append(
            f"{label}: {result['model_a']} = {_format_value(result['model_a_display'], result['difference_unit'])}; "
            f"{result['model_b']} = {_format_value(result['model_b_display'], result['difference_unit'])}. {conclusion}"
        )

    if tradeoff:
        standings = comparison.get("metric_standings") or []
        if standings:
            # Multi-model: describe each metric's ranked leader from standings.
            findings = []
            for standing in standings:
                rows = standing.get("rows") or []
                if not rows:
                    continue
                label = METRIC_LABELS.get(standing["metric"], standing["metric"].replace("_", " ").title())
                best = rows[0]
                runner_up = rows[1] if len(rows) > 1 else None
                if runner_up is not None:
                    gap = abs(best["display"] - runner_up["display"])
                    findings.append(
                        f"{label}: {best['model'].title()} leads at {best['display']:g}"
                        f" (ahead of {runner_up['model'].title()} at {runner_up['display']:g} by {gap:g})"
                    )
                else:
                    findings.append(f"{label}: {best['model'].title()} leads at {best['display']:g}")
            leaders = {standing["rows"][0]["model"] for standing in standings if standing.get("rows")}
            if len(leaders) == 1:
                leader = next(iter(leaders))
                summary = (
                    f"{leader.title()} has the most favorable reported value across all compared metrics: "
                    + "; ".join(findings)
                    + ". This describes the supplied metrics only and does not establish overall model quality."
                )
            else:
                summary = (
                    "The reported metrics are mixed: "
                    + "; ".join(findings)
                    + ". This is a metric-level trade-off, so the available values do not establish one overall winner."
                )
        else:
            findings = []
            for result in metric_results:
                if result["preferred_model"] == "equal":
                    continue
                label = METRIC_LABELS.get(result["metric"], result["metric"].replace("_", " ").title())
                preferred = result["preferred_model"].title()
                difference = f"{result['display_difference']:g} {result['difference_unit']}"
                findings.append(f"{label} favors {preferred} by {difference}")
            summary = (
                "The reported metrics are mixed: "
                + "; ".join(findings)
                + ". This is a metric-level trade-off, so the available values do not establish one overall winner."
            )
        interpretation = (
            "Metric-level trade-off detected. The reported metrics do not establish a single overall conclusion; "
            "no aggregate score or overall winner has been inferred."
        )
    elif metric_results and comparison["comparison_type"] == "CONSISTENT_REPORTED_ADVANTAGE":
        # For 3+ models, the pairwise "preferred model" is ambiguous; the ranked
        # standings leader is the correct answer.
        standings = comparison.get("metric_standings") or []
        leaders = [
            standing["rows"][0]["model"]
            for standing in standings
            if standing.get("rows") and standing["rows"][0].get("is_best")
        ]
        if leaders:
            favored = leaders[0]
        else:
            favored = next(row["preferred_model"] for row in metric_results if row["preferred_model"] != "equal")
        favored_metrics = [
            METRIC_LABELS.get(standing["metric"]) or standing["metric"].replace("_", " ").title()
            for standing in standings
            if standing.get("rows") and standing["rows"][0].get("is_best") and standing["rows"][0]["model"] == favored
        ] or [
            METRIC_LABELS.get(row["metric"]) or row["metric"].replace("_", " ").title()
            for row in metric_results
            if row["preferred_model"] == favored
        ]
        labels = ", ".join(dict.fromkeys(favored_metrics))
        summary = (
            f"{favored.title()} has the more favorable reported values for {labels}. "
            "This describes the supplied comparable metrics only and does not establish overall model quality."
        )
        interpretation = (
            f"{favored} has the more favorable reported value across the comparable metrics provided. "
            "This does not by itself establish overall model superiority."
        )
    elif metric_results:
        summary = "The compared models have equal reported values for the available comparable metrics."
        interpretation = "No metric-level difference was detected in the comparable reported values."
    else:
        summary = "The claim did not provide enough unambiguous, comparable model values for a metric-level comparison."
        interpretation = "A comparison is unavailable. Provide model names and values for the same metric to compare them."

    limitations = (
        "Reported values alone do not establish overall model superiority. A stronger evaluation requires comparable "
        "datasets and test splits, consistent preprocessing and evaluation protocols, relevant class distributions, "
        "and appropriate uncertainty or statistical analysis where applicable."
    )
    retrieval = retrieval or {}
    if relevant_evidence_count:
        evidence_status = (
            f"{relevant_evidence_count} relevant quantitative evidence observation(s) were found across "
            f"{evidence_count} evidence record(s). External evidence should be interpreted in the context of its "
            "dataset, protocol, and comparability to the claim."
        )
    elif retrieval.get("search_status") == "missing_api_key":
        evidence_status = (
            "No external search was performed because TAVILY_API_KEY is not configured. "
            "The comparison uses values supplied in the claim only."
        )
    elif retrieval.get("search_status") == "search_failed":
        evidence_status = (
            "External evidence retrieval failed; no claim is made that the web contains no relevant evidence. "
            "The comparison uses values supplied in the claim only."
        )
    elif retrieval.get("database_status") in {"unavailable", "save_failed"}:
        evidence_status = (
            "No relevant external quantitative evidence was returned, and the database cache was unavailable. "
            "The comparison uses values supplied in the claim only."
        )
    else:
        evidence_status = (
            "No relevant external quantitative evidence was returned. The comparison therefore uses the values "
            "supplied in the claim."
        )

    evidence_comparability = evidence_comparability or {}
    status_counts = evidence_comparability.get("status_counts", {})
    if status_counts:
        mismatch_count = status_counts.get("BENCHMARK_MISMATCH", 0)
        unverified_count = sum(
            count
            for status, count in status_counts.items()
            if "UNCONFIRMED" in status or "UNSPECIFIED" in status or "UNVERIFIED" in status
        )
        if mismatch_count:
            summary += f" Retrieved evidence includes {mismatch_count} source(s) on a different benchmark."
        elif unverified_count:
            summary += f" Benchmark comparability remains unverified for {unverified_count} source(s)."
        summary += " Retrieved evidence does not by itself verify the claim."

    return {
        "summary": summary,
        "metric_insights": metric_insights,
        "overall_interpretation": interpretation,
        "limitations": limitations,
        "evidence_status": evidence_status,
    }


def generate_llm_summary(
    question,
    comparison,
    evidence_metrics,
    deterministic_summary,
    evidence_comparability=None,
    external_model_comparisons=None,
    user_needs_comparison=None,
    external_model_observations=None,
    user_reported_distribution=None,
):
    if not settings.hf_token:
        return {
            "text": deterministic_summary,
            "provider": "Deterministic fallback",
            "status": "not_configured",
        }

    system_message = (
        "You are ClaimLens, an AI/ML claim-analysis assistant. Rewrite the supplied deterministic_interpretation "
        "into a clear, concise summary of the structured findings. Do not add, remove, or recompute any numbers, "
        "models, sources, or conclusions. Do not contradict yourself: if the structured findings contain multiple "
        "source comparisons, present them as separate within-source results and never merge them into one claim. "
        "Keep source-reported comparisons separate from values stated by the user, and state the comparability "
        "caveat exactly as given. Missing evidence means unknown, not a negative result. "
        "If comparison_type is MULTI_METRIC_TRADEOFF, state plainly that different metrics favor different models "
        "and no overall winner is established. If comparison_type is SOURCE_REPORTED_COMPARISON, lead with the "
        "direct answer to the question using the within-source result, then the protocol caveat. "
        "For user_reported_distribution, state each category's count and share, the top-two share, and that it is "
        "an outcome distribution, not a quality score. "
        "Never combine metrics into a single score. Use cautious professional language in two to three sentences."
    )

    user_content = (
        "Condense the following ClaimLens findings into a clear, accurate summary of at most three sentences. "
        "Keep every number, model name, and caveat exactly as stated. Do not add new facts or combine results.\n\n"
        f"Question: {question}\n"
        f"Comparison type: {comparison['comparison_type']}\n"
        f"Findings: {deterministic_summary}"
    )

    try:
        from huggingface_hub import InferenceClient

        client = InferenceClient(
            model=settings.hf_model,
            provider=cast(Any, settings.hf_provider),
            token=settings.hf_token,
            timeout=settings.hf_timeout_seconds,
        )
        response = client.chat_completion(
            messages=[
                {"role": "system", "content": system_message},
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
            max_tokens=240,
            temperature=0.1,
        )
        message = response.choices[0].message.content
        if isinstance(message, list):
            message = " ".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in message
            )
        summary = str(message or "").strip()
        if not summary:
            raise ValueError("The model returned an empty summary.")

        summary = re.sub(
            r"^\s*(?:here\s+is|here'?s|below\s+is|summary\s*:|the\s+following\s+is)[^:.\n]*[:.]?\s*",
            "",
            summary,
            flags=re.IGNORECASE,
        ).strip()

        external_preferences = {
            item["preferred_model"]
            for item in external_model_comparisons or []
            if item.get("preferred_model") != "equal"
        }
        has_tradeoff = (
            comparison["comparison_type"] == "MULTI_METRIC_TRADEOFF"
            or len(external_preferences) > 1
        )
        if not has_tradeoff and re.search(
            r"\b(?:different metrics favor different models|metric[- ]level trade[- ]off)\b",
            summary,
            flags=re.IGNORECASE,
        ):
            return {
                "text": deterministic_summary,
                "provider": "Deterministic fallback",
                "status": "fallback_unsupported_tradeoff",
            }

        if has_tradeoff:
            unsafe_claims = (
                r"\bmodel\s+[a-z0-9]+\s+(?:is|was)\s+(?:the\s+)?(?:overall\s+)?(?:winner|best|superior)\b",
                r"\bmodel\s+[a-z0-9]+\s+(?:is|was)\s+(?:clearly\s+)?better\s+overall\b",
                r"(?<!no )\b(?:overall|single|clear)\s+(?:winner|best model)\s*(?:is|:)\s*model\s+[a-z0-9]+\b",
            )
            if any(re.search(pattern, summary, flags=re.IGNORECASE) for pattern in unsafe_claims):
                return {
                    "text": deterministic_summary,
                    "provider": "Deterministic fallback",
                    "status": "fallback_unsafe_summary",
                }

        return {
            "text": summary,
            "provider": settings.hf_model,
            "status": "success",
        }
    except Exception as error:
        return {
            "text": deterministic_summary,
            "provider": "Deterministic fallback",
            "status": f"fallback_{type(error).__name__}",
        }
