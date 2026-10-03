from __future__ import annotations

import re

import pandas as pd

from .comparison import compare_claim_metrics
from .comparability import assess_evidence_comparability
from .metrics import (
    PERCENTAGE_METRICS,
    build_user_claim_rows,
    detect_benchmarks,
    detect_evaluation_contexts,
    detect_metrics,
    detect_models,
    detect_sample_sizes,
    extract_model_distribution,
    extract_metric_values,
    has_confidence_interval,
    has_statistical_significance,
    normalize_model_name,
)
from .needs import build_needs_evidence, models_match
from .reasoning import build_reasoning, generate_llm_summary
from .retrieval import normalize_text, retrieve_evidence, safe_text


def evidence_relevance_score(row, query_metrics, query_normalized):
    score = 0.0
    metric = safe_text(row["metric"])
    text = normalize_text(row["context"])

    if metric in query_metrics:
        score += 5.0
    if row["models_detected"]:
        score += 1.5
    if row["evaluation_contexts"]:
        score += 1.5

    if any(context in row["evaluation_contexts"] for context in ["test_set", "validation_set", "cross_validation", "holdout", "external_evaluation"]):
        score += 2.0

    if row["confidence_interval_present"]:
        score += 1.0
    if row["statistical_significance_present"]:
        score += 1.0

    query_tokens = [token for token in re.findall(r"[a-z0-9]+", query_normalized) if len(token) >= 4]
    if query_tokens:
        matched_tokens = sum(1 for token in set(query_tokens) if token in text)
        score += min(matched_tokens * 0.15, 1.5)

    return score


def _build_source_metric_comparisons(requested_models, evidence_dataframe, comparability_sources):
    if len(requested_models) < 2 or evidence_dataframe.empty:
        return []

    comparability_by_id = {
        str(item["evidence_id"]): item for item in comparability_sources
    }
    comparisons = []
    for _, evidence_row in evidence_dataframe.iterrows():
        source_text = safe_text(evidence_row["analysis_text"])
        source_observations = build_user_claim_rows(source_text)
        source_comparison = compare_claim_metrics(source_observations)
        source_models = source_comparison["models"]

        for metric_result in source_comparison["metric_results"]:
            compared_models = [metric_result["model_a"], metric_result["model_b"]]
            if not all(
                any(models_match(requested, detected) for detected in source_models)
                for requested in requested_models
            ):
                continue
            if not all(
                any(models_match(requested, compared) for compared in compared_models)
                for requested in requested_models
            ):
                continue

            source_comparability = comparability_by_id.get(str(evidence_row["evidence_id"]), {})
            comparisons.append(
                {
                    **metric_result,
                    "comparison_basis": "retrieved_source",
                    "source": safe_text(evidence_row["source"]),
                    "title": safe_text(evidence_row["title"]),
                    "url": safe_text(evidence_row["url"]),
                    "evidence_id": safe_text(evidence_row["evidence_id"]),
                    "benchmarks": source_comparability.get("source_benchmarks", []),
                    "task_context": source_comparability.get("source_tasks", []),
                    "comparability_status": "WITHIN_SOURCE_PROTOCOL_UNVERIFIED",
                    "comparability_explanation": (
                        "Both reported values were extracted from this source, but dataset version and "
                        "evaluation protocol still require verification."
                    ),
                    "direct_comparison_supported": False,
                }
            )

    return comparisons


def _extract_requested_model_observations(requested_models, evidence_dataframe):
    if not requested_models or evidence_dataframe.empty:
        return []

    observations = []
    seen = set()
    for _, evidence_row in evidence_dataframe.iterrows():
        evidence_text = safe_text(evidence_row["analysis_text"])
        benchmarks = detect_benchmarks(evidence_text)
        contexts = detect_evaluation_contexts(evidence_text)
        for claim_row in build_user_claim_rows(evidence_text):
            matched_model = next(
                (
                    requested
                    for requested in requested_models
                    if models_match(requested, claim_row["model"])
                ),
                None,
            )
            if matched_model is None:
                continue
            key = (
                str(evidence_row["evidence_id"]),
                matched_model,
                claim_row["metric"],
                claim_row["value"],
                claim_row["unit"],
            )
            if key in seen:
                continue
            seen.add(key)
            is_percentage = claim_row["unit"] == "percentage" or (
                claim_row["measurement_unit"] == "ratio"
                and claim_row["metric"] in PERCENTAGE_METRICS
            )
            observations.append(
                {
                    "model": claim_row["model"],
                    "requested_model": matched_model,
                    "metric": claim_row["metric"],
                    "value": float(claim_row["value"]),
                    "display_value": float(claim_row["value"]) * 100 if is_percentage else float(claim_row["value"]),
                    "unit": "percentage" if is_percentage else claim_row["measurement_unit"],
                    "direction": claim_row["direction"],
                    "source": safe_text(evidence_row["source"]),
                    "title": safe_text(evidence_row["title"]),
                    "url": safe_text(evidence_row["url"]),
                    "evidence_id": safe_text(evidence_row["evidence_id"]),
                    "benchmarks": benchmarks,
                    "evaluation_contexts": contexts,
                    "raw_text": claim_row["raw_text"],
                    "comparison_status": "SOURCE_REPORTED_PROTOCOL_UNVERIFIED",
                }
            )
    return observations


def run_claimlens_pipeline(
    question: str,
    investigation_id: str | None = None,
    country: str | None = None,
    investigation_domain: str = "general",
    claim_types: list | None = None,
    df_evidence_input: pd.DataFrame | None = None,
    *,
    market: str | None = None,
    domain: str | None = None,
):
    if not question or not question.strip():
        raise ValueError("A question is required.")

    user_query = question.strip()
    selected_market = market if market is not None else country
    selected_domain = domain if domain is not None else investigation_domain
    query_normalized = normalize_text(" ".join(filter(None, [user_query, selected_market, selected_domain])))
    model_distribution = extract_model_distribution(user_query)
    query_metrics = detect_metrics(user_query)
    if "accuracy" in query_normalized and "accuracy" not in query_metrics:
        query_metrics.append("accuracy")
    query_metrics = sorted(set(query_metrics))

    df_evidence, retrieval = retrieve_evidence(
        user_query,
        market=selected_market,
        domain=selected_domain,
        investigation_id=investigation_id,
        candidate_df=df_evidence_input,
    )
    df_user_claim = pd.DataFrame(build_user_claim_rows(user_query))
    if not df_user_claim.empty:
        df_user_claim = df_user_claim.drop_duplicates().reset_index(drop=True)
    claim_model_names = set(df_user_claim["model"]) if not df_user_claim.empty else set()
    detected_models = detect_models(user_query)
    requested_models = [] if model_distribution["available"] else [
        model
        for model in detected_models
        if normalize_model_name(model) not in claim_model_names
    ]
    distribution_models = [item["model"] for item in model_distribution["results"]]
    comparability_models = requested_models or distribution_models or sorted(claim_model_names) or None
    evidence_comparability = assess_evidence_comparability(
        user_query,
        df_evidence,
        model_names=comparability_models,
    )
    external_model_comparisons = _build_source_metric_comparisons(
        requested_models,
        df_evidence,
        evidence_comparability["sources"],
    )
    external_model_observations = _extract_requested_model_observations(
        requested_models,
        df_evidence,
    )
    user_needs_comparison = build_needs_evidence(
        user_query,
        requested_models,
        df_evidence,
    )
    comparability_by_evidence_id = {
        str(item["evidence_id"]): item for item in evidence_comparability["sources"]
    }

    quantitative_rows = []
    for _, row in df_evidence.iterrows():
        evidence_id = row["evidence_id"]
        text = safe_text(row["analysis_text"])
        source = safe_text(row["source"])
        title = safe_text(row["title"])
        url = safe_text(row["url"])
        source_comparability = comparability_by_evidence_id.get(str(evidence_id), {})

        metrics = detect_metrics(text)
        models = detect_models(text)
        evaluation_contexts = detect_evaluation_contexts(text)
        sample_sizes = detect_sample_sizes(text)
        ci_present = has_confidence_interval(text)
        significance_present = has_statistical_significance(text)

        for metric in metrics:
            for observation in extract_metric_values(text, metric):
                quantitative_rows.append(
                    {
                        "evidence_id": evidence_id,
                        "comparability_status": source_comparability.get("status", "UNASSESSED"),
                        "comparability_explanation": source_comparability.get("explanation", "No comparison context available."),
                        "source": source,
                        "title": title,
                        "url": url,
                        "metric": observation["metric"],
                        "value": observation["value"],
                        "unit": observation["unit"],
                        "measurement_unit": observation["measurement_unit"],
                        "direction": observation["direction"],
                        "models_detected": models,
                        "evaluation_contexts": evaluation_contexts,
                        "sample_sizes_detected": sample_sizes,
                        "confidence_interval_present": ci_present,
                        "statistical_significance_present": significance_present,
                        "raw_text": observation["raw_text"],
                        "context": observation["context"],
                    }
                )

    df_quantitative_all = pd.DataFrame(quantitative_rows)
    if df_quantitative_all.empty:
        df_quantitative_all = pd.DataFrame(
            columns=[
                "evidence_id",
                "comparability_status",
                "comparability_explanation",
                "source",
                "title",
                "url",
                "metric",
                "value",
                "unit",
                "measurement_unit",
                "direction",
                "models_detected",
                "evaluation_contexts",
                "sample_sizes_detected",
                "confidence_interval_present",
                "statistical_significance_present",
                "raw_text",
                "context",
            ]
        )

    if not df_quantitative_all.empty:
        df_quantitative_all = df_quantitative_all.drop_duplicates(
            subset=["evidence_id", "metric", "value", "raw_text"]
        ).reset_index(drop=True)

    if not df_quantitative_all.empty:
        df_quantitative_all["relevance_score"] = df_quantitative_all.apply(
            lambda row: evidence_relevance_score(row, query_metrics, query_normalized), axis=1
        )
        df_quantitative_all["evidence_priority"] = df_quantitative_all["metric"].isin(query_metrics).map(
            {True: "primary", False: "secondary"}
        )
    else:
        df_quantitative_all["relevance_score"] = pd.Series(dtype=float)
        df_quantitative_all["evidence_priority"] = pd.Series(dtype=str)

    if not df_quantitative_all.empty:
        requested_model_matches = pd.Series(False, index=df_quantitative_all.index)
        if len(requested_models) >= 2:
            requested_model_matches = df_quantitative_all["models_detected"].apply(
                lambda source_models: all(
                    any(models_match(requested, detected) for detected in source_models)
                    for requested in requested_models
                )
            )
        df_quantitative_relevant = (
            df_quantitative_all[
                (df_quantitative_all["metric"].isin(query_metrics))
                | (df_quantitative_all["relevance_score"] >= 3.0)
                | requested_model_matches
            ]
            .sort_values(by="relevance_score", ascending=False, kind="mergesort")
            .sort_values(by="evidence_priority", ascending=True, kind="mergesort")
            .reset_index(drop=True)
        )
    else:
        df_quantitative_relevant = pd.DataFrame()

    metric_counts = df_quantitative_all["metric"].value_counts().to_dict() if not df_quantitative_all.empty else {}
    metrics_detected = sorted({str(metric) for metric in metric_counts} | set(query_metrics))
    quantitative_status = "AI_ML_QUANTITATIVE_EVIDENCE_FOUND" if not df_quantitative_all.empty else "NO_AI_ML_QUANTITATIVE_EVIDENCE_FOUND"
    claim_observations = df_user_claim.to_dict(orient="records") if not df_user_claim.empty else []
    comparison = compare_claim_metrics(claim_observations)
    reasoning = build_reasoning(
        comparison,
        evidence_count=len(df_evidence),
        relevant_evidence_count=len(df_quantitative_relevant),
        retrieval=retrieval,
        evidence_comparability=evidence_comparability,
    )
    comparable_metrics = list(
        dict.fromkeys(
            [row["metric"] for row in comparison["metric_results"]]
            + [row["metric"] for row in external_model_comparisons]
        )
    )
    evidence_metrics = []
    for metric in metrics_detected:
        if df_quantitative_relevant.empty:
            metric_rows = pd.DataFrame()
        else:
            metric_rows = df_quantitative_relevant[df_quantitative_relevant["metric"] == metric]
        observations = []
        for _, evidence_row in metric_rows.head(10).iterrows():
            observations.append(
                {
                    "evidence_id": safe_text(evidence_row["evidence_id"]),
                    "value": float(evidence_row["value"]),
                    "unit": safe_text(evidence_row["unit"]),
                    "measurement_unit": safe_text(evidence_row["measurement_unit"]),
                    "comparability_status": safe_text(evidence_row["comparability_status"]),
                    "comparability_explanation": safe_text(evidence_row["comparability_explanation"]),
                    "source": safe_text(evidence_row["source"]),
                    "title": safe_text(evidence_row["title"]),
                    "url": safe_text(evidence_row["url"]),
                    "models_detected": evidence_row["models_detected"],
                    "evaluation_contexts": evidence_row["evaluation_contexts"],
                    "sample_sizes_detected": evidence_row["sample_sizes_detected"],
                    "confidence_interval_present": bool(evidence_row["confidence_interval_present"]),
                    "statistical_significance_present": bool(evidence_row["statistical_significance_present"]),
                    "raw_text": safe_text(evidence_row["raw_text"]),
                }
            )
        evidence_metrics.append(
            {
                "metric": metric,
                "observation_count": int(len(metric_rows)),
                "unique_sources": int(metric_rows["source"].nunique()) if not metric_rows.empty else 0,
                "observations": observations,
                "comparability_statuses": sorted(
                    {observation["comparability_status"] for observation in observations}
                ),
                "interpretation": (
                    "Retrieved observations are contextual evidence, not direct verification of the claim; "
                    "dataset and protocol comparability must be checked."
                    if observations
                    else "No relevant external observation was extracted for this metric."
                ),
            }
        )

    deterministic_summary = reasoning["summary"]
    if model_distribution["available"]:
        ranked_distribution = sorted(
            model_distribution["results"],
            key=lambda item: item["count"],
            reverse=True,
        )
        denominator = model_distribution["reported_total"] or model_distribution["observed_total"]
        distribution_findings = "; ".join(
            f"{item['model']} {item['count']}/{denominator} {item['unit']} ({item['share_percent']:g}%)"
            for item in ranked_distribution
        )
        if model_distribution["total_consistent"] is True:
            total_note = "The counts reconcile with the stated total."
        elif model_distribution["total_consistent"] is False:
            total_note = "The extracted counts do not reconcile with the stated total."
        else:
            total_note = "No total was stated; shares use the sum of extracted counts."
        top_two_note = (
            f"The top two categories account for {model_distribution['top_two_share_percent']:g}% of the total, "
            f"while the remaining {len(ranked_distribution) - 2} categories share "
            f"{100 - model_distribution['top_two_share_percent']:g}%. "
            "Focusing only on the top two therefore omits a material portion of the benchmark's outcomes. "
            if len(ranked_distribution) > 2
            else "The reported total is split between the two categories shown. "
        )
        deterministic_summary = (
            f"User-reported outcome distribution: {distribution_findings}. "
            f"{top_two_note}"
            f"{total_note} This distribution describes task outcomes, not overall model quality."
        )
    mixed_pair_request = bool(comparison["metric_results"] and requested_models)
    claim_pair = " vs. ".join(model.title() for model in comparison["models"])
    requested_pair = " vs. ".join(model.title() for model in requested_models)
    if mixed_pair_request:
        claim_findings = "; ".join(reasoning["metric_insights"]).rstrip(" .")
        deterministic_summary = (
            f"Two separate comparisons were requested. User-supplied results for {claim_pair}: "
            f"{claim_findings}. These claim results are not assigned to {requested_pair}. "
        )
    if external_model_comparisons:
        source_findings = []
        for item in external_model_comparisons[:6]:
            source_name = item["title"] or item["source"] or "a retrieved source"
            benchmark_text = ", ".join(item["benchmarks"]) or "benchmark not identified"
            value_suffix = "%" if item["difference_unit"] == "percentage points" else f" {item['measurement_unit']}" if item.get("measurement_unit") else ""
            source_findings.append(
                f"{item['metric'].replace('_', ' ')} in {source_name} ({benchmark_text}): "
                f"{item['model_a'].title()} reported {item['model_a_display']:g}{value_suffix}; "
                f"{item['model_b'].title()} reported {item['model_b_display']:g}{value_suffix}. "
                f"{item['preferred_model'].title()} has the more favorable reported value by "
                f"{item['display_difference']:g} {item['difference_unit']}; dataset/protocol comparability is unverified"
            )
        deterministic_summary += " Retrieved-source comparisons: " + "; ".join(source_findings) + "."
    elif external_model_observations:
        deterministic_summary += (
            f" Retrieved sources contain {len(external_model_observations)} model-specific metric observation(s), "
            "but the requested models were not compared together in one source. "
            "Cross-source differences are not treated as directly comparable."
        )
    elif mixed_pair_request:
        deterministic_summary += (
            f" No same-source metric comparison was found for {requested_pair}; "
            "no results from the user-claim pair are attributed to those products."
        )

    requested_needs = user_needs_comparison["requested_needs"]
    if requested_needs:
        found_need_rows = user_needs_comparison["evidence_found_count"]
        deterministic_summary += (
            f" User priorities assessed: {found_need_rows} of "
            f"{len(user_needs_comparison['rows'])} model-priority combinations had linked source passages. "
            "Missing evidence is unknown, not a negative product result."
        )

    summary_comparison = comparison
    if external_model_comparisons and not comparison["comparison_available"]:
        summary_comparison = {
            **comparison,
            "comparison_type": "SOURCE_REPORTED_COMPARISON",
            "comparison_available": True,
            "metric_results": external_model_comparisons,
            "metrics": list(dict.fromkeys(item["metric"] for item in external_model_comparisons)),
        }

    if mixed_pair_request:
        ai_summary = {
            "text": deterministic_summary,
            "provider": "Deterministic mixed-pair safeguard",
            "status": "deterministic_mixed_comparison",
        }
    else:
        ai_summary = generate_llm_summary(
            user_query,
            summary_comparison,
            evidence_metrics,
            deterministic_summary,
            evidence_comparability,
            external_model_comparisons,
            user_needs_comparison,
            external_model_observations,
            model_distribution,
        )

    claim_metric_results = comparison["metric_results"]
    display_metric_results = claim_metric_results or external_model_comparisons
    comparison_available = bool(
        comparison["comparison_available"]
        or external_model_comparisons
        or user_needs_comparison["evidence_found_count"]
        or model_distribution["available"]
    )
    if comparison["comparison_available"]:
        display_comparison_type = comparison["comparison_type"]
    elif external_model_comparisons:
        display_comparison_type = "SOURCE_REPORTED_COMPARISON"
    elif user_needs_comparison["evidence_found_count"]:
        display_comparison_type = "USER_NEEDS_EVIDENCE_FOUND"
    elif model_distribution["available"]:
        display_comparison_type = "USER_REPORTED_DISTRIBUTION"
    elif external_model_observations:
        display_comparison_type = "SOURCE_METRIC_EVIDENCE_ONLY"
    else:
        display_comparison_type = comparison["comparison_type"]
    output_models = list(
        dict.fromkeys(
            comparison["models"]
            + requested_models
            + (distribution_models if model_distribution["available"] else [])
        )
    )
    output_metrics = list(
        dict.fromkeys(
            comparison["metrics"]
            + [item["metric"] for item in external_model_comparisons]
            + [item["metric"] for item in external_model_observations]
        )
    )

    claimlens_quantitative = {
        "investigation_id": investigation_id,
        "question": user_query,
        "market_context": selected_market,
        "domain": selected_domain,
        "claim_types": claim_types or [],
        "evidence_records": int(len(df_evidence)),
        "quantitative_observations": int(len(df_quantitative_all)),
        "relevant_quantitative_observations": int(len(df_quantitative_relevant)),
        "query_metrics": query_metrics,
        "metrics_detected": metrics_detected,
        "metric_counts": metric_counts,
        "quantitative_status": quantitative_status,
        "user_claim_observations": claim_observations,
        "comparison": comparison["comparison_type"],
        "comparison_available": comparison_available,
        "comparison_type": display_comparison_type,
        "metric_results": comparison["metric_results"],
        "external_model_comparisons": external_model_comparisons,
        "external_model_observations": external_model_observations,
        "user_needs_comparison": user_needs_comparison,
        "model_distribution": model_distribution,
        "evidence_metrics": evidence_metrics,
        "evidence_comparability": evidence_comparability,
        "retrieval": retrieval,
        "summary": ai_summary["text"],
        "summary_provider": ai_summary["provider"],
        "summary_status": ai_summary["status"],
        "currency": "NOT_REQUIRED_FOR_AI_ML_EVALUATION",
    }

    return {
        "question": user_query,
        "market": selected_market,
        "domain": selected_domain,
        "comparison_available": comparison_available,
        "comparison_type": display_comparison_type,
        "models": output_models,
        "claim_models": comparison["models"],
        "requested_models": requested_models,
        "metrics": output_metrics,
        "comparable_metrics": comparable_metrics,
        "metric_results": comparison["metric_results"],
        "claim_metric_results": claim_metric_results,
        "metric_standings": comparison.get("metric_standings", []),
        "external_model_comparisons": external_model_comparisons,
        "external_model_observations": external_model_observations,
        "display_metric_results": display_metric_results,
        "comparison_basis": "claim" if claim_metric_results else "retrieved_source" if external_model_comparisons else "source_evidence_only" if external_model_observations else "user_needs" if user_needs_comparison["evidence_found_count"] else "user_reported_distribution" if model_distribution["available"] else "unavailable",
        "user_needs_comparison": user_needs_comparison,
        "model_distribution": model_distribution,
        "evidence_metrics": evidence_metrics,
        "evidence_comparability": evidence_comparability,
        "retrieval": retrieval,
        "graph_data": {
            "claim_comparisons": claim_metric_results,
            "source_comparisons": external_model_comparisons,
            "source_model_observations": external_model_observations,
            "user_reported_distribution": model_distribution,
            "retrieved_observations": [
                observation
                for metric_summary in evidence_metrics
                for observation in metric_summary["observations"]
            ],
        },
        "summary": ai_summary["text"],
        "deterministic_summary": deterministic_summary,
        "summary_provider": ai_summary["provider"],
        "summary_status": ai_summary["status"],
        "metric_insights": reasoning["metric_insights"],
        "overall_interpretation": reasoning["overall_interpretation"],
        "limitations": reasoning["limitations"],
        "evidence_status": reasoning["evidence_status"],
        "evidence_count": int(len(df_evidence)),
        "relevant_evidence": int(len(df_quantitative_relevant)),
        "evidence_available": not df_quantitative_relevant.empty,
        "status": "COMPLETE",
        "df_evidence": df_evidence,
        "df_user_claim": df_user_claim,
        "df_quantitative_all": df_quantitative_all,
        "df_quantitative_relevant": df_quantitative_relevant,
        "claimlens_quantitative": claimlens_quantitative,
        "query_metrics": query_metrics,
        "metrics_detected": metrics_detected,
    }
