import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from agent.config import settings
from agent.pipeline import run_claimlens_pipeline


def _comparison_table_rows(items, result, retrieved_source=False):
    evidence_by_metric = {item["metric"]: item for item in result["evidence_metrics"]}
    rows = []
    for item in items:
        suffix = "%" if item["difference_unit"] == "percentage points" else ""
        evidence_summary = evidence_by_metric.get(item["metric"], {})
        observations = evidence_summary.get("observations", [])
        row = {
            "Metric": item["metric"].replace("_", " ").title(),
            "Model A": f"{item['model_a_display']:g}{suffix} ({item['model_a']})",
            "Model B": f"{item['model_b_display']:g}{suffix} ({item['model_b']})",
            "Difference": f"{item['display_difference']:g} {item['difference_unit']}",
            "Direction": item["direction"].replace("_", " "),
            "More favorable reported value": item["preferred_model"],
        }
        if retrieved_source:
            row["Source"] = item.get("title") or item.get("source") or "Retrieved source"
            row["URL"] = item.get("url", "")
            benchmarks = item.get("benchmarks", [])
            if benchmarks:
                row["Benchmark"] = ", ".join(benchmarks)
            row["Protocol status"] = item.get("comparability_status", "Not assessed")
        else:
            row["External evidence"] = (
                f"{evidence_summary.get('observation_count', 0)} observation(s) / "
                f"{evidence_summary.get('unique_sources', 0)} source(s)"
                if observations
                else "None found"
            )
        rows.append(row)
    return rows


def _plot_claim_and_evidence(result):
    claim_comparisons = result["claim_metric_results"]
    source_comparisons = result["external_model_comparisons"]
    if not claim_comparisons and not source_comparisons and not result["external_model_observations"]:
        st.info("No numeric claim or evidence observations are available to graph.")
        return

    st.subheader("Metric Graphs")
    st.caption("Claim values and source-reported comparisons are plotted separately. Metrics are never added into a shared score.")
    comparison_groups = [
        ("Claim values", claim_comparisons),
        ("Same-source model comparisons", source_comparisons),
    ]
    rendered_metrics = set()
    for group_label, comparisons in comparison_groups:
        for comparison in comparisons:
            metric_key = (group_label, comparison["metric"], comparison["model_a"], comparison["model_b"], comparison.get("url", ""))
            if metric_key in rendered_metrics:
                continue
            rendered_metrics.add(metric_key)
            labels = [comparison["model_a"], comparison["model_b"]]
            values = [comparison["model_a_display"], comparison["model_b_display"]]
            figure, axis = plt.subplots(figsize=(8, 2.7))
            axis.barh(labels, values, color=["#167c80", "#df7545"])
            for index, value in enumerate(values):
                axis.text(value, index, f"  {value:g}", va="center", fontsize=9)
            source_suffix = ""
            if group_label != "Claim values":
                source_suffix = f" · {comparison.get('title') or comparison.get('source') or 'Retrieved source'}"
            axis.set_title(f"{comparison['metric'].replace('_', ' ').title()} · {group_label}{source_suffix}")
            axis.set_xlabel(comparison["difference_unit"])
            axis.grid(axis="x", alpha=0.2)
            axis.set_axisbelow(True)
            figure.tight_layout()
            st.pyplot(figure)
            plt.close(figure)

    if not claim_comparisons and not source_comparisons and result["external_model_observations"]:
        st.caption("These dots are per-source observations only; they are not cross-source model comparisons.")
        by_metric = {}
        for observation in result["external_model_observations"]:
            by_metric.setdefault(observation["metric"], []).append(observation)
        for metric, observations in by_metric.items():
            visible = observations[:10]
            labels = [
                f"{item['requested_model']} · {item['title'] or item['source'] or 'Source'}"
                for item in visible
            ]
            values = [item["display_value"] for item in visible]
            figure, axis = plt.subplots(figsize=(9, max(2.4, 0.45 * len(labels) + 1)))
            axis.scatter(values, range(len(values)), color="#344b87", s=48)
            for index, value in enumerate(values):
                axis.text(value, index, f"  {value:g}", va="center", fontsize=9)
            axis.set_yticks(range(len(labels)), labels)
            axis.set_title(f"{metric.replace('_', ' ').title()} · Per-source observations")
            axis.set_xlabel(visible[0]["unit"] or "Reported value")
            axis.grid(axis="x", alpha=0.2)
            axis.set_axisbelow(True)
            figure.tight_layout()
            st.pyplot(figure)
            plt.close(figure)


def _render_user_reported_distribution(distribution):
    if not distribution["available"]:
        return

    st.subheader("User-Reported Outcome Distribution")
    st.caption("Reported task outcomes only; this distribution is not a measure of overall model quality.")
    reported_total = distribution["reported_total"]
    total_consistent = distribution["total_consistent"]
    reconciliation = (
        "Matches"
        if total_consistent is True
        else "Does not match"
        if total_consistent is False
        else "Not stated"
    )
    total_columns = st.columns(4)
    total_columns[0].metric("Stated total", reported_total if reported_total is not None else "Not stated")
    total_columns[1].metric("Observed total", distribution["observed_total"])
    total_columns[2].metric("Total reconciliation", reconciliation)
    top_two_share = distribution["top_two_share_percent"]
    total_columns[3].metric("Top-two share", f"{top_two_share:g}%" if top_two_share is not None else "Not available")

    ranked = sorted(distribution["results"], key=lambda item: item["count"], reverse=True)
    rows = [
        {
            "Rank": index,
            "Category / model": item["model"],
            f"Reported {item['unit']}": item["count"],
            "Share of total": f"{item['share_percent']:g}%" if item["share_percent"] is not None else "Not available",
            "Outcome context": item["outcome_context"] or "Not stated",
        }
        for index, item in enumerate(ranked, start=1)
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    chart_items = sorted(distribution["results"], key=lambda item: item["count"])
    labels = [item["model"] for item in chart_items]
    counts = [item["count"] for item in chart_items]

    st.markdown("**Chart 1 · Reported counts by category**")
    figure, axis = plt.subplots(figsize=(9, max(2.7, 0.48 * len(labels) + 1)))
    bars = axis.barh(labels, counts, color="#167c80")
    axis.bar_label(bars, padding=4, fmt="%g")
    axis.set_xlabel(f"Reported {distribution.get('unit', 'count')}")
    axis.grid(axis="x", alpha=0.2)
    axis.set_axisbelow(True)
    figure.tight_layout()
    st.pyplot(figure)
    plt.close(figure)

    shares = [
        (item["model"], item["share_percent"])
        for item in ranked
        if item["share_percent"] is not None
    ]
    if shares:
        st.markdown("**Chart 2 · Share of the reported total**")
        figure, axis = plt.subplots(figsize=(7.5, 4.2))
        axis.pie(
            [share for _, share in shares],
            labels=[model for model, _ in shares],
            autopct="%1.1f%%",
            startangle=90,
            counterclock=False,
            colors=["#167c80", "#df7545", "#344b87", "#5f9ea0", "#c97c5d", "#8c7851"],
            wedgeprops={"width": 0.42, "edgecolor": "white"},
        )
        figure.tight_layout()
        st.pyplot(figure)
        plt.close(figure)

    if top_two_share is not None and len(ranked) >= 3:
        remaining_share = round(100 - top_two_share, 1)
        remaining_count = sum(item["count"] for item in ranked[2:])
        st.markdown("**Chart 3 · Top two vs. all remaining categories**")
        figure, axis = plt.subplots(figsize=(7, 2.8))
        labels = ["Top two models", f"All other {len(ranked) - 2} categories"]
        values = [top_two_share, remaining_share]
        bars = axis.barh(labels, values, color=["#167c80", "#df7545"])
        axis.bar_label(bars, padding=4, fmt="%g%%")
        axis.set_xlabel("Share of reported total (%)")
        axis.set_xlim(0, 100)
        axis.grid(axis="x", alpha=0.2)
        axis.set_axisbelow(True)
        figure.tight_layout()
        st.pyplot(figure)
        plt.close(figure)
        st.caption(
            f"The top two categories account for {top_two_share:g}% "
            f"({ranked[0]['count'] + ranked[1]['count']} {distribution.get('unit', 'count')}); "
            f"the remaining {len(ranked) - 2} categories hold {remaining_share:g}% "
            f"({remaining_count} {distribution.get('unit', 'count')})."
        )


st.set_page_config(page_title=settings.streamlit_title, layout="wide")
st.title("CLAIMLENS")
st.subheader("AI / ML Claim Investigation Engine")
st.caption("Analyze quantitative claims, compare reported metrics, and identify limitations in the available evidence.")

with st.form("claimlens_form"):
    question = st.text_area(
        "Question",
        placeholder="Example: Model A has 92% accuracy and 88% F1 score, while Model B has 90% accuracy and 91% F1 score.",
        height=140,
    )
    market_choice = st.selectbox(
        "Market / Country",
        ["Not specified", "Singapore", "India", "United States", "United Kingdom", "Canada", "Australia", "Other"],
    )
    custom_market = st.text_input("Custom country or market (optional)", placeholder="Overrides the selection above")
    market = custom_market.strip() or (None if market_choice in {"Not specified", "Other"} else market_choice)
    domain = st.text_input("Domain", value="AI / ML evaluation")
    submitted = st.form_submit_button("Run investigation", type="primary")

if submitted:
    if not question.strip():
        st.warning("Enter a claim or question before running the investigation.")
        st.stop()

    with st.spinner("Running the ClaimLens investigation..."):
        result = run_claimlens_pipeline(
            question=question,
            market=(market.strip() or None) if market else None,
            domain=domain.strip() or "AI / ML evaluation",
        )

    st.divider()
    st.header("Investigation Result")
    if result["metric_results"]:
        status_label = "CLAIM COMPARISON READY"
    elif result["external_model_comparisons"]:
        status_label = "SOURCE-REPORTED COMPARISON"
    elif result["external_model_observations"]:
        status_label = "MODEL EVIDENCE FOUND; PAIRING UNVERIFIED"
    elif result["model_distribution"]["available"]:
        status_label = "USER-REPORTED DISTRIBUTION"
    elif result["user_needs_comparison"]["evidence_found_count"]:
        status_label = "USER-PRIORITY EVIDENCE FOUND"
    else:
        status_label = "COMPARISON UNAVAILABLE"
    st.write(f"**Status:** {status_label} · {result['status']}")

    st.subheader("Claim Summary")
    st.write(result["question"])

    if result["claim_metric_results"]:
        st.subheader("Claim-Reported Comparison")
        st.caption("This table compares values entered in your question; metric preferences do not establish overall model quality.")
        st.dataframe(
            pd.DataFrame(_comparison_table_rows(result["claim_metric_results"], result)),
            hide_index=True,
            width="stretch",
        )
    elif not result["requested_models"] and not result["external_model_comparisons"] and not result["model_distribution"]["available"]:
        st.subheader("Claim-Reported Comparison")
        st.info("No numeric model values were supplied in the question.")

    if result["requested_models"]:
        st.subheader(f"Requested Model Comparison: {' vs. '.join(result['requested_models'])}")
        if result["external_model_comparisons"]:
            source_rows = _comparison_table_rows(
                result["external_model_comparisons"],
                result,
                retrieved_source=True,
            )
            st.dataframe(
                pd.DataFrame(source_rows),
                hide_index=True,
                width="stretch",
                column_config={"URL": st.column_config.LinkColumn("Source URL")},
            )
            st.caption("Values are reported by the cited source. Dataset/protocol alignment and overall product quality are not inferred.")
        else:
            st.info("No single retrieved source reported matching numeric values for both requested models.")

    if result["external_model_observations"]:
        st.subheader("Retrieved Model Metrics")
        st.caption("Values below are reported by individual sources. Values from separate sources are not ranked against each other unless benchmark and protocol are aligned.")
        model_metric_rows = [
            {
                "Requested model": item["requested_model"],
                "Model/version in source": item["model"],
                "Metric": item["metric"].replace("_", " ").title(),
                "Reported value": (
                    f"{item['display_value']:g}%"
                    if item["unit"] == "percentage"
                    else f"{item['display_value']:g} {item['unit']}".strip()
                ),
                "Direction": item["direction"].replace("_", " "),
                "Benchmark": ", ".join(item["benchmarks"]) or "Not stated",
                "Evaluation context": ", ".join(item["evaluation_contexts"]) or "Not stated",
                "Source": item["title"] or item["source"],
                "Protocol status": item["comparison_status"],
                "URL": item["url"],
            }
            for item in result["external_model_observations"]
        ]
        st.dataframe(
            pd.DataFrame(model_metric_rows),
            hide_index=True,
            width="stretch",
            column_config={"URL": st.column_config.LinkColumn("Source URL")},
        )

    needs_result = result["user_needs_comparison"]
    if needs_result["requested_needs"]:
        st.subheader("Your Priorities")
        st.caption(needs_result["summary"])
        needs_rows = []
        for item in needs_result["rows"]:
            if item["findings"]:
                for finding in item["findings"]:
                    needs_rows.append(
                        {
                            "Priority": item["criterion"],
                            "Model": item["model"],
                            "Evidence status": item["status"],
                            "Source finding": finding["finding"],
                            "Extracted value": (
                                f"{finding['value']} {finding['unit']}"
                                if finding["value"] is not None
                                else "Qualitative evidence"
                            ),
                            "Source": finding["title"] or finding["source"],
                            "URL": finding["url"],
                        }
                    )
            else:
                needs_rows.append(
                    {
                        "Priority": item["criterion"],
                        "Model": item["model"],
                        "Evidence status": item["status"],
                        "Source finding": item["explanation"],
                        "Extracted value": "Not found",
                        "Source": "",
                        "URL": "",
                    }
                )
        st.dataframe(
            pd.DataFrame(needs_rows),
            hide_index=True,
            width="stretch",
            column_config={"URL": st.column_config.LinkColumn("Source URL")},
        )

    st.subheader("AI Summary")
    if result["summary_status"] == "success":
        st.caption(f"Generated by {result['summary_provider']}")
    elif result["summary_status"] == "not_configured":
        st.caption("Hugging Face is not configured; showing the deterministic reasoning fallback.")
    elif result["summary_status"] == "deterministic_mixed_comparison":
        st.caption("Deterministic summary keeps the user-claim pair separate from the requested-product pair.")
    else:
        st.caption(f"Hugging Face summary unavailable ({result['summary_status']}); showing the deterministic fallback.")
    st.write(result["summary"])

    if result["metric_insights"]:
        st.subheader("Metric Insights")
        for insight in result["metric_insights"]:
            st.markdown(f"- {insight}")

    if result["model_distribution"]["available"]:
        st.subheader("Interpretation")
        st.write(
            "The full distribution shows how outcomes spread across all reported categories. "
            "A top-two-only view concentrates on a subset of the benchmark and omits the remainder; "
            "the charts below quantify that remainder."
        )
    else:
        st.subheader("Overall Interpretation")
        st.write(result["overall_interpretation"])

    st.subheader("Limitations")
    st.write(result["limitations"])

    st.subheader("Evidence Status")
    st.write(result["evidence_status"])
    retrieval = result["retrieval"]
    st.caption(
        f"Evidence available: {'Yes' if result['evidence_available'] else 'No'} · "
        f"Relevant quantitative observations: {result['relevant_evidence']} · "
        f"Retrieval: {retrieval['search_status']} · Database cache: {retrieval['database_status']}"
    )

    comparability = result["evidence_comparability"]
    show_comparability = bool(
        comparability["claim_benchmarks"]
        or any(source["source_benchmarks"] for source in comparability["sources"])
    )
    if show_comparability:
        st.subheader("Evidence Comparability")
        st.caption(comparability["summary"])
        for source in comparability["sources"]:
            label = f"{source['status']}: {source['title'] or source['source'] or 'Retrieved source'}"
            with st.expander(label):
                st.write(source["explanation"])
                st.write(f"**Metrics shared with claim:** {', '.join(source['shared_metrics']) or 'None'}")
                st.write(f"**Claim benchmarks:** {', '.join(source['claim_benchmarks']) or 'Not stated'}")
                st.write(f"**Source benchmarks:** {', '.join(source['source_benchmarks']) or 'Not stated'}")
                st.write(f"**Claim models:** {', '.join(source['claim_models']) or 'Not detected'}")
                st.write(f"**Models in source:** {', '.join(source['source_models']) or 'Not detected'}")
                st.write(f"**Claim task(s):** {', '.join(source['claim_tasks']) or 'Not detected'}")
                st.write(f"**Source task(s):** {', '.join(source['source_tasks']) or 'Not detected'}")
                st.write(f"**Shared task(s):** {', '.join(source['shared_tasks']) or 'None established'}")
                st.write(f"**Evaluation context overlap:** {', '.join(set(source['claim_contexts']) & set(source['source_contexts'])) or 'Not established'}")
                st.write("**Direct claim verification:** Not established")
                if source["url"]:
                    st.link_button("Open source", source["url"])

    _plot_claim_and_evidence(result)
    _render_user_reported_distribution(result["model_distribution"])

    if any(item["observations"] for item in result["evidence_metrics"]):
        st.subheader("Retrieved Metric Evidence")
        for item in result["evidence_metrics"]:
            if not item["observations"]:
                continue
            with st.expander(f"{item['metric'].replace('_', ' ').title()} · {item['observation_count']} observation(s)"):
                st.caption(item["interpretation"])
                evidence_rows = [
                    {
                        "Value": (
                            f"{observation['value'] * 100:g}%"
                            if observation["unit"] == "percentage" or observation["measurement_unit"] == "ratio"
                            else f"{observation['value']:g} {observation['measurement_unit']}".strip()
                        ),
                        "Source": observation["source"],
                        "Title": observation["title"],
                        "Models mentioned": ", ".join(observation["models_detected"]),
                        "Evaluation context": ", ".join(observation["evaluation_contexts"]),
                        "Sample size": ", ".join(map(str, observation["sample_sizes_detected"])),
                        "Confidence interval": observation["confidence_interval_present"],
                        "Statistical significance": observation["statistical_significance_present"],
                        "URL": observation["url"],
                    }
                    for observation in item["observations"]
                ]
                st.dataframe(
                    pd.DataFrame(evidence_rows),
                    hide_index=True,
                    width="stretch",
                    column_config={"URL": st.column_config.LinkColumn("Source URL")},
                )
    with st.expander("Investigation Details"):
        st.write(f"**Detected metrics:** {', '.join(result['metrics']) or 'None'}")
        st.write(f"**Comparable metrics:** {', '.join(result['comparable_metrics']) or 'None'}")
        st.write(f"**Comparison type:** {result['comparison_type']}")
        st.write(f"**Claim benchmarks:** {', '.join(comparability['claim_benchmarks']) or 'Not detected'}")
        st.write(f"**Claim tasks:** {', '.join(comparability['claim_tasks']) or 'Not detected'}")
        st.write(f"**Evidence comparability statuses:** {comparability['status_counts'] or 'None'}")
        st.write(f"**Evidence records:** {result['evidence_count']}")
        st.write(f"**Search query:** {retrieval['search_query']}")
        st.write(f"**Retrieved from:** {retrieval['source']}")
        st.write(f"**Cache hit:** {'Yes' if retrieval['cache_hit'] else 'No'}")
        st.write(f"**Results fetched / deduplicated:** {retrieval['fetched_count']} / {retrieval['deduplicated_count']}")
        st.write(f"**Persisted to database:** {'Yes' if retrieval['persisted'] else 'No'}")
        st.write(f"**Pipeline status:** {result['status']}")
        st.write(f"**Market:** {result['market'] or 'Not specified'}")
        st.write(f"**Domain:** {result['domain'] or 'Not specified'}")
