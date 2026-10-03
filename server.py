"""ClaimLens AI · FastAPI report server.

Serves the professional HTML/CSS/JS frontend and exposes the
ClaimLens investigation pipeline as a JSON API. Charts are rendered
server-side with Matplotlib and returned as base64 PNG images.
"""

from __future__ import annotations

import base64
import io
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent.pipeline import run_claimlens_pipeline

FRONTEND_DIR = "frontend"

BRAND_COLORS = ["#167c80", "#df7545", "#344b87", "#5f9ea0", "#c97c5d", "#8c7851"]

app = FastAPI(title="ClaimLens AI", version="1.0.0")


class InvestigationRequest(BaseModel):
    question: str = Field(..., min_length=3)
    market: str | None = None
    domain: str = "AI / ML evaluation"


def _figure_to_base64(figure) -> str:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
    plt.close(figure)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("ascii")


def _comparison_charts(result: dict) -> list[dict]:
    charts: list[dict] = []
    comparison_groups = [
        ("Claim values", result.get("claim_metric_results") or []),
        ("Same-source model comparisons", result.get("external_model_comparisons") or []),
    ]
    rendered: set[tuple] = set()
    for group_label, comparisons in comparison_groups:
        for comparison in comparisons:
            metric_key = (
                group_label,
                comparison["metric"],
                comparison["model_a"],
                comparison["model_b"],
                comparison.get("url", ""),
            )
            if metric_key in rendered:
                continue
            rendered.add(metric_key)
            labels = [comparison["model_a"], comparison["model_b"]]
            values = [comparison["model_a_display"], comparison["model_b_display"]]
            figure, axis = plt.subplots(figsize=(8, 2.7))
            axis.barh(labels, values, color=[BRAND_COLORS[0], BRAND_COLORS[1]])
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
            charts.append(
                {
                    "kind": "metric_comparison",
                    "title": f"{comparison['metric'].replace('_', ' ').title()} · {group_label}",
                    "image": _figure_to_base64(figure),
                }
            )

    claim_comparisons = result.get("claim_metric_results") or []
    source_comparisons = result.get("external_model_comparisons") or []
    observations = result.get("external_model_observations") or []
    if not claim_comparisons and not source_comparisons and observations:
        by_metric: dict[str, list] = {}
        for observation in observations:
            by_metric.setdefault(observation["metric"], []).append(observation)
        for metric, metric_observations in by_metric.items():
            visible = metric_observations[:10]
            labels = [
                f"{item['requested_model']} · {item['title'] or item['source'] or 'Source'}"
                for item in visible
            ]
            values = [item["display_value"] for item in visible]
            figure, axis = plt.subplots(figsize=(9, max(2.4, 0.45 * len(labels) + 1)))
            axis.scatter(values, range(len(values)), color=BRAND_COLORS[2], s=48)
            for index, value in enumerate(values):
                axis.text(value, index, f"  {value:g}", va="center", fontsize=9)
            axis.set_yticks(range(len(labels)), labels)
            axis.set_title(f"{metric.replace('_', ' ').title()} · Per-source observations")
            axis.set_xlabel(visible[0]["unit"] or "Reported value")
            axis.grid(axis="x", alpha=0.2)
            axis.set_axisbelow(True)
            figure.tight_layout()
            charts.append(
                {
                    "kind": "source_observations",
                    "title": f"{metric.replace('_', ' ').title()} · Per-source observations",
                    "image": _figure_to_base64(figure),
                }
            )
    return charts


def _distribution_charts(distribution: dict) -> list[dict]:
    if not distribution.get("available"):
        return []

    charts: list[dict] = []
    ranked = sorted(distribution["results"], key=lambda item: item["count"], reverse=True)
    unit = distribution.get("unit", "count")

    chart_items = sorted(distribution["results"], key=lambda item: item["count"])
    labels = [item["model"] for item in chart_items]
    counts = [item["count"] for item in chart_items]
    figure, axis = plt.subplots(figsize=(9, max(2.7, 0.48 * len(labels) + 1)))
    bars = axis.barh(labels, counts, color=BRAND_COLORS[0])
    axis.bar_label(bars, padding=4, fmt="%g")
    axis.set_xlabel(f"Reported {unit}")
    axis.grid(axis="x", alpha=0.2)
    axis.set_axisbelow(True)
    figure.tight_layout()
    charts.append(
        {
            "kind": "distribution_counts",
            "title": "Reported counts by category",
            "image": _figure_to_base64(figure),
        }
    )

    shares = [
        (item["model"], item["share_percent"])
        for item in ranked
        if item["share_percent"] is not None
    ]
    if shares:
        figure, axis = plt.subplots(figsize=(7.5, 4.2))
        axis.pie(
            [share for _, share in shares],
            labels=[model for model, _ in shares],
            autopct="%1.1f%%",
            startangle=90,
            counterclock=False,
            colors=BRAND_COLORS,
            wedgeprops={"width": 0.42, "edgecolor": "white"},
        )
        figure.tight_layout()
        charts.append(
            {
                "kind": "distribution_share",
                "title": "Share of the reported total",
                "image": _figure_to_base64(figure),
            }
        )

    top_two_share = distribution.get("top_two_share_percent")
    if top_two_share is not None and len(ranked) >= 3:
        remaining_share = round(100 - top_two_share, 1)
        figure, axis = plt.subplots(figsize=(7, 2.8))
        bar_labels = ["Top two models", f"All other {len(ranked) - 2} categories"]
        values = [top_two_share, remaining_share]
        bars = axis.barh(bar_labels, values, color=[BRAND_COLORS[0], BRAND_COLORS[1]])
        axis.bar_label(bars, padding=4, fmt="%g%%")
        axis.set_xlabel("Share of reported total (%)")
        axis.set_xlim(0, 100)
        axis.grid(axis="x", alpha=0.2)
        axis.set_axisbelow(True)
        figure.tight_layout()
        charts.append(
            {
                "kind": "distribution_top_two",
                "title": "Top two vs. all remaining categories",
                "image": _figure_to_base64(figure),
            }
        )
    return charts


def _sanitize(value: Any) -> Any:
    """Convert the pipeline result into JSON-serializable data."""
    import numpy as np
    import pandas as pd

    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, dict):
        return {key: _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and (value != value):
        return None
    return value


@app.post("/api/investigate")
def investigate(request: InvestigationRequest):
    try:
        result = run_claimlens_pipeline(
            question=request.question,
            market=request.market or None,
            domain=request.domain or "AI / ML evaluation",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Investigation failed: {exc}") from exc

    charts = _comparison_charts(result) + _distribution_charts(result.get("model_distribution") or {})
    payload = _sanitize(result)
    for heavy_key in [
        "df_evidence",
        "df_user_claim",
        "df_quantitative_all",
        "df_quantitative_relevant",
        "claimlens_quantitative",
    ]:
        payload.pop(heavy_key, None)
    payload["charts"] = charts
    return payload


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(f"{FRONTEND_DIR}/index.html")


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="127.0.0.1", port=8600, reload=False)
