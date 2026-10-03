from __future__ import annotations

import re

from .metrics import (
    detect_benchmarks,
    detect_evaluation_contexts,
    detect_metrics,
    detect_models,
)
from .retrieval import safe_text


STATUS_EXPLANATIONS = {
    "NO_SHARED_METRICS": "The source does not report a metric requested by the claim.",
    "BENCHMARK_MISMATCH": "The source reports a different named benchmark from the claim.",
    "EVALUATION_CONTEXT_MISMATCH": "The named evaluation split or protocol context differs from the claim.",
    "CLAIM_BENCHMARK_UNCONFIRMED": "The claim names a benchmark, but the source does not identify that benchmark.",
    "CLAIM_BENCHMARK_UNSPECIFIED": "The source names a benchmark, but the claim does not; benchmark comparability cannot be confirmed.",
    "BENCHMARK_MATCH_PROTOCOL_UNVERIFIED": "The named benchmark and models match, but dataset version and evaluation protocol are not fully verified.",
    "BENCHMARK_MATCH_MODELS_UNSPECIFIED": "The named benchmark matches, but the source does not identify the evaluated models.",
    "BENCHMARK_MATCH_SOME_MODELS": "The named benchmark matches, but the source mentions only some of the models in the claim.",
    "BENCHMARK_MATCH_DIFFERENT_MODELS": "The named benchmark matches, but the source does not report the models in the claim.",
    "METRIC_MATCH_BENCHMARK_UNSPECIFIED": "A metric overlaps, but neither claim nor source identifies a benchmark.",
    "NO_CLAIM_METRICS": "No quantitative metric was detected in the claim, so metric comparability cannot be assessed.",
    "BENCHMARK_MATCH_NO_CLAIM_METRICS": "The claim identifies this benchmark and the source reports the requested models, but the claim supplied no metric values; this is a source-reported comparison, not verification of a user-reported value.",
    "TASK_MISMATCH": "The source describes a different task from the one named in the claim.",
    "TASK_MATCH_BENCHMARK_UNSPECIFIED": "The task matches, but benchmark identity and evaluation protocol are not established.",
}

BENCHMARK_TASKS = {
    "mmlu": "general_knowledge_reasoning",
    "mmlu_pro": "general_knowledge_reasoning",
    "gsm8k": "math_reasoning",
    "math": "math_reasoning",
    "gpqa": "science_reasoning",
    "hellaswag": "commonsense_reasoning",
    "truthfulqa": "truthfulness",
    "winogrande": "commonsense_reasoning",
    "arc_challenge": "science_reasoning",
    "humaneval": "code_generation",
    "mbpp": "code_generation",
    "bigbench_hard": "multi_task_reasoning",
    "mt_bench": "chat_preference",
    "arena_hard": "chat_preference",
    "imagenet": "image_classification",
    "coco": "vision_and_captioning",
    "squad": "reading_comprehension",
    "glue": "language_understanding",
    "superglue": "language_understanding",
}

TASK_PATTERNS = {
    "math_reasoning": r"\b(?:math(?:ematical)? reasoning|math word problems?)\b",
    "code_generation": r"\b(?:code generation|code completion|program synthesis|coding task)\b",
    "question_answering": r"\b(?:question answering|open[- ]domain qa)\b",
    "text_summarization": r"\b(?:text summarization|summarisation)\b",
    "machine_translation": r"\b(?:machine translation|translation task)\b",
    "image_classification": r"\b(?:image classification|visual recognition)\b",
    "object_detection": r"\b(?:object detection|instance segmentation)\b",
}


def _detect_tasks(text, benchmarks):
    tasks = {BENCHMARK_TASKS[benchmark] for benchmark in benchmarks if benchmark in BENCHMARK_TASKS}
    normalized = safe_text(text).lower()
    tasks.update(
        task
        for task, pattern in TASK_PATTERNS.items()
        if re.search(pattern, normalized, flags=re.IGNORECASE)
    )
    return sorted(tasks)


def assess_evidence_comparability(question, evidence_dataframe, model_names=None):
    claim_metrics = set(detect_metrics(question))
    claim_models = set(model_names) if model_names is not None else set(detect_models(question))
    claim_benchmarks = set(detect_benchmarks(question))
    claim_contexts = set(detect_evaluation_contexts(question))
    claim_tasks = set(_detect_tasks(question, claim_benchmarks))
    results = []

    for _, row in evidence_dataframe.iterrows():
        text = safe_text(row.get("analysis_text", ""))
        evidence_metrics = set(detect_metrics(text))
        evidence_models = set(detect_models(text))
        evidence_benchmarks = set(detect_benchmarks(text))
        evidence_contexts = set(detect_evaluation_contexts(text))
        evidence_tasks = set(_detect_tasks(text, evidence_benchmarks))
        shared_metrics = sorted(claim_metrics & evidence_metrics)
        shared_benchmarks = sorted(claim_benchmarks & evidence_benchmarks)

        if claim_benchmarks and evidence_benchmarks and not shared_benchmarks:
            status = "BENCHMARK_MISMATCH"
        elif claim_tasks and evidence_tasks and not claim_tasks.intersection(evidence_tasks):
            status = "TASK_MISMATCH"
        elif claim_contexts and evidence_contexts and not claim_contexts.intersection(evidence_contexts):
            status = "EVALUATION_CONTEXT_MISMATCH"
        elif claim_benchmarks and not evidence_benchmarks:
            status = "CLAIM_BENCHMARK_UNCONFIRMED"
        elif evidence_benchmarks and not claim_benchmarks:
            status = "CLAIM_BENCHMARK_UNSPECIFIED"
        elif shared_benchmarks and not claim_metrics:
            if not evidence_models:
                status = "BENCHMARK_MATCH_MODELS_UNSPECIFIED"
            elif not claim_models.intersection(evidence_models):
                status = "BENCHMARK_MATCH_DIFFERENT_MODELS"
            elif not claim_models.issubset(evidence_models):
                status = "BENCHMARK_MATCH_SOME_MODELS"
            else:
                status = "BENCHMARK_MATCH_NO_CLAIM_METRICS"
        elif not claim_metrics:
            status = "NO_CLAIM_METRICS"
        elif not shared_metrics:
            status = "NO_SHARED_METRICS"
        elif shared_benchmarks:
            if not evidence_models:
                status = "BENCHMARK_MATCH_MODELS_UNSPECIFIED"
            elif not claim_models.intersection(evidence_models):
                status = "BENCHMARK_MATCH_DIFFERENT_MODELS"
            elif not claim_models.issubset(evidence_models):
                status = "BENCHMARK_MATCH_SOME_MODELS"
            else:
                status = "BENCHMARK_MATCH_PROTOCOL_UNVERIFIED"
        elif claim_tasks and evidence_tasks:
            status = "TASK_MATCH_BENCHMARK_UNSPECIFIED"
        else:
            status = "METRIC_MATCH_BENCHMARK_UNSPECIFIED"

        results.append(
            {
                "evidence_id": safe_text(row.get("evidence_id", "")),
                "title": safe_text(row.get("title", "")),
                "url": safe_text(row.get("url", "")),
                "source": safe_text(row.get("source", "")),
                "status": status,
                "explanation": STATUS_EXPLANATIONS[status],
                "shared_metrics": shared_metrics,
                "claim_models": sorted(claim_models),
                "source_models": sorted(evidence_models),
                "claim_benchmarks": sorted(claim_benchmarks),
                "source_benchmarks": sorted(evidence_benchmarks),
                "shared_benchmarks": shared_benchmarks,
                "claim_contexts": sorted(claim_contexts),
                "source_contexts": sorted(evidence_contexts),
                "claim_tasks": sorted(claim_tasks),
                "source_tasks": sorted(evidence_tasks),
                "shared_tasks": sorted(claim_tasks & evidence_tasks),
                "direct_comparison_supported": False,
            }
        )

    status_counts = {}
    for result in results:
        status_counts[result["status"]] = status_counts.get(result["status"], 0) + 1

    return {
        "claim_metrics": sorted(claim_metrics),
        "claim_models": sorted(claim_models),
        "claim_benchmarks": sorted(claim_benchmarks),
        "claim_contexts": sorted(claim_contexts),
        "claim_tasks": sorted(claim_tasks),
        "source_count": len(results),
        "status_counts": status_counts,
        "sources": results,
        "direct_comparison_supported": False,
        "summary": (
            "No direct benchmark comparison is claimed. Each source is classified by reported metric, benchmark, "
            "model, and evaluation context; dataset version and full protocol still require source verification."
            if results
            else "No retrieved sources were available to assess for benchmark comparability."
        ),
    }
