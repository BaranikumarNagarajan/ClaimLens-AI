import re

from .retrieval import normalize_text, safe_float, safe_text


METRIC_DEFINITIONS = {
    "accuracy": {"patterns": [r"\baccuracy\b", r"\baccurate\b"], "direction": "higher_is_better"},
    "precision": {"patterns": [r"\bprecision\b"], "direction": "higher_is_better"},
    "recall": {"patterns": [r"\brecall\b", r"\bsensitivity\b", r"\btrue positive rate\b"], "direction": "higher_is_better"},
    "f1_score": {"patterns": [r"\bf1\b", r"\bf1[- ]score\b", r"\bf1 score\b"], "direction": "higher_is_better"},
    "roc_auc": {"patterns": [r"\broc[- ]auc\b", r"\broc auc\b", r"\bauc\b"], "direction": "higher_is_better"},
    "pr_auc": {"patterns": [r"\bpr[- ]auc\b", r"\bprecision[- ]recall auc\b"], "direction": "higher_is_better"},
    "specificity": {"patterns": [r"\bspecificity\b", r"\btrue negative rate\b"], "direction": "higher_is_better"},
    "log_loss": {"patterns": [r"\blog[- ]loss\b", r"\blog loss\b"], "direction": "lower_is_better"},
    "mse": {"patterns": [r"\bmse\b", r"\bmean squared error\b"], "direction": "lower_is_better"},
    "rmse": {"patterns": [r"\brmse\b", r"\broot mean squared error\b"], "direction": "lower_is_better"},
    "mae": {"patterns": [r"\bmae\b", r"\bmean absolute error\b"], "direction": "lower_is_better"},
    "mape": {"patterns": [r"\bmape\b", r"\bmean absolute percentage error\b"], "direction": "lower_is_better"},
    "r_squared": {"patterns": [r"\br[²2]\b", r"\br-squared\b", r"\br squared\b", r"\br_squared\b"], "direction": "higher_is_better"},
    "bleu": {"patterns": [r"\bbleu\b"], "direction": "higher_is_better"},
    "rouge": {"patterns": [r"\brouge\b"], "direction": "higher_is_better"},
    "dataset_size": {"patterns": [r"\bdataset size\b", r"\bsample size\b", r"\btraining samples?\b", r"\btest samples?\b", r"\bdata points?\b"], "direction": "context_dependent"},
    "parameter_count": {"patterns": [r"\bparameters?\b", r"\bparameter count\b", r"\bmodel size\b"], "direction": "context_dependent"},
    "latency": {"patterns": [r"\blatency\b", r"\binference time\b", r"\bresponse time\b"], "direction": "lower_is_better"},
    "throughput": {"patterns": [r"\bthroughput\b", r"\brequests per second\b", r"\bqueries per second\b"], "direction": "higher_is_better"},
    "reasoning": {"patterns": [r"\breasoning\b"], "direction": "higher_is_better"},
    "coding": {"patterns": [r"\bcoding\b", r"\bcode generation\b"], "direction": "higher_is_better"},
    "factuality": {"patterns": [r"\bfactuality\b", r"\bfactual accuracy\b"], "direction": "higher_is_better"},
    "math": {"patterns": [r"\bmath\b", r"\bmathematics\b", r"\bmaths\b"], "direction": "higher_is_better"},
    "benchmark_score": {"patterns": [r"\bbenchmark scores?\b(?!\s+is\b)", r"\bbenchmark results?\b"], "direction": "higher_is_better"},
}


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
    "reasoning",
    "coding",
    "factuality",
    "math",
}


EVALUATION_CONTEXT_PATTERNS = {
    "test_set": ["test set", "test dataset", "testing set"],
    "validation_set": ["validation set", "validation dataset"],
    "training_set": ["training set", "training dataset", "train set"],
    "cross_validation": ["cross-validation", "cross validation", "k-fold", "k fold"],
    "benchmark": ["benchmark", "benchmarking"],
    "holdout": ["holdout", "hold-out"],
    "external_evaluation": ["external validation", "external test"],
}


BENCHMARK_PATTERNS = {
    "mmlu_pro": [r"\bmmlu[\s-]*pro\b"],
    "mmlu": [r"\bmmlu\b"],
    "gsm8k": [r"\bgsm[\s-]?8k\b"],
    "math": [r"\bmath[- ]?benchmark\b"],
    "gpqa": [r"\bgpqa(?:[- ](?:diamond|extended))?\b"],
    "hellaswag": [r"\bhellaswag\b"],
    "truthfulqa": [r"\btruthfulqa\b"],
    "winogrande": [r"\bwinogrande\b"],
    "arc_challenge": [r"\barc[- ]challenge\b", r"\bai2 reasoning challenge\b"],
    "humaneval": [r"\bhuman[- ]?eval\b"],
    "mbpp": [r"\bmbpp\b", r"\bmostly basic python problems\b"],
    "bigbench_hard": [r"\bbig[- ]?bench hard\b", r"\bbbh\b"],
    "mt_bench": [r"\bmt[- ]bench\b"],
    "arena_hard": [r"\barena[- ]hard\b"],
    "imagenet": [r"\bimagenet(?:[- ]1k)?\b"],
    "coco": [r"\bmscoco\b", r"\bcoco(?:[- ]\d+)?\b"],
    "squad": [r"\bsquad(?:\s*2(?:\.0)?)?\b"],
    "glue": [r"\bglue benchmark\b"],
    "superglue": [r"\bsuperglue\b"],
}


MODEL_NAME_PATTERNS = [
    r"\bgpt[\s-]?\d+(?:\.\d+)?(?:[\s-]?(?:turbo|instruct|mini|preview))?\b",
    r"\bgemma(?:[\s-]?\d+(?:\.\d+)?)?(?:[\s-]?(?:it|flash|pro))?\b",
    r"\bgemini(?:[\s-]?(?:\d+(?:\.\d+)?|pro|flash|ultra))?(?:[\s-]?(?:pro|flash|ultra))?\b",
    r"\bclaude(?:[\s-]?\d+(?:\.\d+)?(?:[\s-]?(?:opus|sonnet|haiku))?)?\b",
    r"\bproduct\s+[a-z0-9]+\b",
    r"\b(?:bert|llama|resnet|mistral|mixtral|qwen|deepseek|phi|grok)(?:\s?-?\d+(?:\.\d+)?[a-z]?)?\b",
    r"\b(?:xgboost|random forest|logistic regression|svm|cnn|rnn|transformer)\b",
]


NUMBER_PATTERN = r"(?<![A-Za-z0-9])\d+(?:,\d{3})*(?:\.\d+)?(?![A-Za-z0-9])"
MEASUREMENT_UNITS = {
    "milliseconds": "ms",
    "millisecond": "ms",
    "ms": "ms",
    "seconds": "s",
    "second": "s",
    "secs": "s",
    "sec": "s",
    "s": "s",
    "microseconds": "us",
    "microsecond": "us",
    "us": "us",
    "µs": "us",
    "μs": "us",
    "requests per second": "requests/s",
    "request per second": "requests/s",
    "queries per second": "queries/s",
    "query per second": "queries/s",
    "req/s": "requests/s",
    "rps": "requests/s",
    "tokens per second": "tokens/s",
    "token per second": "tokens/s",
    "samples per second": "samples/s",
    "sample per second": "samples/s",
}


def _measurement_unit(text, number_end, unit, metric):
    if unit == "percentage":
        return "%"
    tail = text[number_end:number_end + 40].strip().lower()
    for label in sorted(MEASUREMENT_UNITS, key=len, reverse=True):
        if re.match(r"^" + re.escape(label) + r"\b", tail):
            return MEASUREMENT_UNITS[label]
    if metric in PERCENTAGE_METRICS:
        return "ratio"
    return ""


def detect_metrics(text):
    normalized = normalize_text(text)
    detected = []
    for metric, definition in METRIC_DEFINITIONS.items():
        for pattern in definition["patterns"]:
            try:
                if re.search(pattern, normalized, flags=re.IGNORECASE):
                    detected.append(metric)
                    break
            except Exception:
                pass
    result = sorted(set(detected))
    # benchmark_score is a generic fallback; drop it when a concrete metric is present.
    if "benchmark_score" in result and len(result) > 1:
        result = [metric for metric in result if metric != "benchmark_score"]
    return result


def detect_models(text):
    model_text = safe_text(text).lower()
    mentions = []

    for match in re.finditer(r"\bmodel[\s-]+([a-z]|[0-9]+)\b", model_text, flags=re.IGNORECASE):
        mentions.append((match.start(), "model " + match.group(1)))

    for pattern in MODEL_NAME_PATTERNS:
        for match in re.finditer(pattern, model_text, flags=re.IGNORECASE):
            value = match.group(0).strip()
            if value:
                mentions.append((match.start(), value))

    # Drop vendor mentions that are part of a composite "Vendor-Model-X" label;
    # the "Model X" tail is the usable model identifier.
    mentions = [
        (start, model)
        for start, model in mentions
        if not re.match(r"^[\s-]*model[\s-]+[a-z0-9]\b", model_text[start + len(model):])
    ]

    models = []
    for _, model in sorted(mentions, key=lambda mention: mention[0]):
        if model not in models:
            models.append(model)
    return models


def normalize_model_name(model):
    normalized = safe_text(model).lower()
    normalized = re.sub(r"[^a-z0-9.+\-\s]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def detect_benchmarks(text):
    normalized = normalize_text(text)
    detected = {
        benchmark
        for benchmark, patterns in BENCHMARK_PATTERNS.items()
        if any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in patterns)
    }
    if "mmlu_pro" in detected:
        detected.discard("mmlu")
    return sorted(detected)


MODEL_DISTRIBUTION_PATTERN = re.compile(
    r"(?P<model>(?:(?:GPT|Claude|Gemini|Gemma|Llama|DeepSeek|Mistral)\s+Model\s+[A-Z0-9]+|"
    r"another\s+open[- ]source\s+model|Model\s+[A-Z0-9]+))"
    r"\s*[—–-]\s*(?P<count>\d+)(?:\s+(?P<unit>tasks?|votes?|wins?))?\b"
    r"(?P<context>[^,.;]*)",
    flags=re.IGNORECASE,
)

COMPOSITE_MODEL_NAME = r"[A-Za-z][A-Za-z0-9]*-Model-[A-Z]"

TASK_COUNT_PATTERN = re.compile(
    r"\b(?P<model>" + COMPOSITE_MODEL_NAME + r"|Model\s+[A-Z0-9]+)\b"
    r"(?:(?!\bacross\b)[^.;]){0,60}?\bon\s+(?P<count>\d+)\s*(?P<unit>tasks?|votes?|wins?)?\b",
    flags=re.IGNORECASE,
)

RESPECTIVELY_COUNT_PATTERN = re.compile(
    r"\bon\s+(?P<counts>\d+(?:\s*(?:,|\band\b)\s*\d+)*)\s+(?P<unit>tasks?|votes?|wins?)\s+respectively\b",
    flags=re.IGNORECASE,
)


def _extract_task_count_distribution(text, default_unit):
    results = []
    seen = set()
    for match in TASK_COUNT_PATTERN.finditer(text):
        model = re.sub(r"\s+", " ", match.group("model")).strip()
        key = model.casefold()
        if key in seen:
            continue
        seen.add(key)
        results.append(
            {
                "model": model,
                "count": int(match.group("count")),
                "unit": (match.group("unit") or default_unit).lower(),
                "outcome_context": "",
            }
        )
    return results


def _extract_respectively_distribution(text, default_unit):
    list_match = RESPECTIVELY_COUNT_PATTERN.search(text)
    if not list_match:
        return []
    counts = [int(value) for value in re.findall(r"\d+", list_match.group("counts"))]
    unit = (list_match.group("unit") or default_unit).lower()
    models = list(dict.fromkeys(re.findall(r"\b" + COMPOSITE_MODEL_NAME + r"\b", text)))
    if len(counts) < 2 or len(models) < len(counts):
        return []
    return [
        {"model": model, "count": count, "unit": unit, "outcome_context": ""}
        for model, count in zip(models, counts)
    ]


def extract_model_distribution(text):
    text = safe_text(text)
    results = []
    seen = set()
    unit_match = re.search(r"\b(tasks?|votes?|wins?)\b", text, flags=re.IGNORECASE)
    default_unit = unit_match.group(1).lower() if unit_match else "count"
    for match in MODEL_DISTRIBUTION_PATTERN.finditer(text):
        model = re.sub(r"\s+", " ", match.group("model")).strip()
        key = model.casefold()
        if key in seen:
            continue
        seen.add(key)
        context = match.group("context").strip()
        results.append(
            {
                "model": model,
                "count": int(match.group("count")),
                "unit": (match.group("unit") or default_unit).lower(),
                "outcome_context": context,
            }
        )

    if not results:
        results = _extract_task_count_distribution(text, default_unit)
    if not results:
        results = _extract_respectively_distribution(text, default_unit)

    if not results:
        return {
            "available": False,
            "results": [],
            "reported_total": None,
            "observed_total": 0,
            "total_consistent": None,
            "top_two_share_percent": None,
        }

    total_match = re.search(
        r"\bacross\s+(\d+)\s+(?:(?:evaluation|benchmark)\s+)?tasks?\b",
        text,
        flags=re.IGNORECASE,
    )
    reported_total = int(total_match.group(1)) if total_match else None
    observed_total = sum(item["count"] for item in results)
    denominator = reported_total or observed_total
    for item in results:
        item["share_percent"] = round(item["count"] / denominator * 100, 1) if denominator else None

    top_two = sorted(results, key=lambda item: item["count"], reverse=True)[:2]
    top_two_share = (
        round(sum(item["count"] for item in top_two) / denominator * 100, 1)
        if denominator
        else None
    )
    return {
        "available": True,
        "results": results,
        "reported_total": reported_total,
        "observed_total": observed_total,
        "total_consistent": observed_total == reported_total if reported_total is not None else None,
        "top_two_share_percent": top_two_share,
        "unit": results[0]["unit"],
    }


def detect_evaluation_contexts(text):
    normalized = normalize_text(text)
    contexts = []
    for context, terms in EVALUATION_CONTEXT_PATTERNS.items():
        for term in terms:
            if term in normalized:
                contexts.append(context)
                break
    return sorted(set(contexts))


def detect_sample_sizes(text):
    values = []
    text = safe_text(text)
    patterns = [
        r"\bn\s*=\s*(\d[\d,]*)\b",
        r"\b(\d[\d,]*)\s+samples?\b",
        r"\b(\d[\d,]*)\s+observations?\b",
        r"\b(\d[\d,]*)\s+examples?\b",
        r"\b(\d[\d,]*)\s+records?\b",
        r"\b(\d[\d,]*)\s+data points?\b",
    ]

    for pattern in patterns:
        try:
            matches = re.finditer(pattern, text, flags=re.IGNORECASE)
        except Exception:
            continue

        for match in matches:
            number_groups = [group for group in match.groups() if group is not None]
            if not number_groups:
                continue
            value = safe_float(number_groups[-1].replace(",", ""))
            if value is not None:
                values.append(value)

    return sorted(set(values))


def has_confidence_interval(text):
    normalized = normalize_text(text)
    patterns = ["confidence interval", "confidence intervals", "95% ci", "95% confidence", "ci =", "ci:"]
    return any(pattern in normalized for pattern in patterns)


def has_statistical_significance(text):
    normalized = normalize_text(text)
    patterns = ["statistically significant", "statistical significance", "p-value", "p value", "p <", "p=", "p ="]
    return any(pattern in normalized for pattern in patterns)


def extract_metric_values(text, metric):
    text = safe_text(text)
    observations = []
    definition = METRIC_DEFINITIONS.get(metric, {})
    patterns = definition.get("patterns", [])

    metric_matches = []
    for metric_pattern in sorted(patterns, key=len, reverse=True):
        try:
            matches = re.finditer(metric_pattern, text, flags=re.IGNORECASE)
        except Exception:
            continue
        for match in matches:
            metric_matches.append((match.start(), match.end()))

    selected_matches = []
    for metric_start, metric_end in sorted(
        set(metric_matches),
        key=lambda span: (span[0], -(span[1] - span[0])),
    ):
        if any(
            metric_start >= selected_start and metric_end <= selected_end
            for selected_start, selected_end in selected_matches
        ):
            continue
        selected_matches.append((metric_start, metric_end))

    for metric_start, metric_end in selected_matches:
            start = max(0, metric_start - 80)
            end = min(len(text), metric_end + 120)
            local_text = text[start:end]

            candidates = []
            local_metric_position = metric_start - start
            local_metric_end = metric_end - start

            for number_match in re.finditer(NUMBER_PATTERN + r"\s*%", local_text):
                value = safe_float(number_match.group(0).replace("%", "").replace(",", ""))
                if value is not None:
                    candidates.append((number_match.start(), number_match.end(), value / 100.0, "percentage"))

            if not candidates:
                for number_match in re.finditer(NUMBER_PATTERN, local_text):
                    if (
                        number_match.start() < local_metric_end
                        and number_match.end() > local_metric_position
                    ):
                        continue
                    raw_number = number_match.group(0)
                    value = safe_float(raw_number)
                    if value is None:
                        continue
                    if 1900 <= value <= 2100:
                        continue
                    if 0 <= value <= 1:
                        candidates.append((number_match.start(), number_match.end(), value, "value"))
                    elif 1 < value <= 100 and metric in PERCENTAGE_METRICS:
                        candidates.append((number_match.start(), number_match.end(), value / 100.0, "percentage"))
                    else:
                        candidates.append((number_match.start(), number_match.end(), value, "value"))

            if not candidates:
                continue

            following = [
                candidate
                for candidate in candidates
                if candidate[0] >= local_metric_end
                and re.fullmatch(
                    r"\s*(?:(?:is|was|of|at|equals?)\s*)?[:=]?\s*",
                    local_text[local_metric_end:candidate[0]],
                    flags=re.IGNORECASE,
                )
            ]
            preceding = [
                candidate
                for candidate in candidates
                if candidate[1] <= local_metric_position
                and not local_text[candidate[1]:local_metric_position].strip(" \t,;:")
            ]
            associated = following or preceding or candidates
            _, candidate_end, value, unit = min(
                associated,
                key=lambda candidate: abs(candidate[0] - local_metric_position),
            )
            measurement_unit = _measurement_unit(local_text, candidate_end, unit, metric)
            context_start = max(0, metric_start - 300)
            context_end = min(len(text), metric_end + 300)
            context = text[context_start:context_end]
            raw_text = local_text.strip()

            observations.append(
                {
                    "metric": metric,
                    "value": value,
                    "unit": unit,
                    "measurement_unit": measurement_unit,
                    "direction": definition.get("direction", "context_dependent"),
                    "raw_text": raw_text,
                    "context": context,
                }
            )

    unique_observations = {}
    for observation in observations:
        key = (observation["value"], observation["unit"], observation["raw_text"])
        unique_observations[key] = observation
    return list(unique_observations.values())


ANONYMOUS_MODEL_PATTERN = re.compile(
    r"\b(?:"
    r"another\s+model"
    r"|the\s+other\s+model"
    r"|(?:the\s+)?(?:first|second|third|fourth)\s+model"
    r"|(?:its\s+|the\s+|an?\s+)?(?:previous|prior|older?|earlier)\s+(?:model|version)"
    r"|(?:its\s+|the\s+|an?\s+)?new\s+(?:[a-z0-9-]+[\s-]){0,3}(?:model|version)"
    r"|an?\s+(?:[a-z][a-z0-9-]*[\s-]){0,4}?model"
    r")\b",
    flags=re.IGNORECASE,
)

ANONYMOUS_MODEL_LABELS = ["first model", "second model", "third model", "fourth model"]


def _anonymous_role(matched_text):
    """Map an anonymous mention to a stable role so re-mentions deduplicate."""
    text = matched_text.lower()
    if re.search(r"\b(?:previous|prior|older?|earlier)\b", text):
        return "second model"
    if re.search(r"\bnew\b", text):
        return "first model"
    if re.search(r"\b(?:another|other|second)\b", text):
        return "second model"
    if re.search(r"\bfirst\b", text):
        return "first model"
    if re.search(r"\bthird\b", text):
        return "third model"
    if re.search(r"\bfourth\b", text):
        return "fourth model"
    return None


def _anonymous_model_mentions(text):
    raw = []
    for match in ANONYMOUS_MODEL_PATTERN.finditer(safe_text(text)):
        if raw and match.start() < raw[-1][1]:
            continue
        raw.append((match.start(), match.end(), match.group(0)))
    if len(raw) < 2:
        return []

    # Assign a stable role to each mention; re-mentions of the same role reuse it.
    roles = []
    unnamed_index = 0
    for start, end, matched in raw:
        role = _anonymous_role(matched)
        if role is None:
            role = ANONYMOUS_MODEL_LABELS[min(unnamed_index, len(ANONYMOUS_MODEL_LABELS) - 1)]
            unnamed_index += 1
        roles.append((start, end, role))

    # Keep only the first occurrence of each role, in text order.
    seen_roles = {}
    for start, end, role in roles:
        if role not in seen_roles:
            seen_roles[role] = (start, end, role)
    mentions = sorted(seen_roles.values(), key=lambda item: item[0])
    if len(mentions) < 2:
        return []
    return mentions


def _anonymous_value(number_text, has_percent, metric):
    value = safe_float(number_text.replace(",", ""))
    if value is None:
        return None
    if has_percent or (metric in PERCENTAGE_METRICS and 1 < value <= 100):
        return value / 100.0, "percentage", "%"
    if 0 <= value <= 1 and metric in PERCENTAGE_METRICS:
        return value, "percentage", "%"
    return value, "value", ""


def _supplement_anonymous_pair_values(question, metrics, rows):
    """Assign paired values for anonymous first/second-model questions.

    Handles two common phrasings the segment splitter cannot pair:
    - "reasoning improving from 80 to 92" -> old value = second model, new value = first model
    - "98% accuracy compared with 93%" -> first value = first model, second value = second model

    Transition matches override segment-extracted rows because the from/to wording
    states both values explicitly, while the segment window can pick up a number
    belonging to a neighboring metric.
    """
    covered = {(row["model"], row["metric"]) for row in rows}
    additions = []
    override_metrics = set()
    for metric in metrics:
        definition = METRIC_DEFINITIONS.get(metric, {})
        direction = definition.get("direction", "context_dependent")
        for pattern in definition.get("patterns", []):
            transition = re.search(
                pattern
                + r"[^0-9.]{0,40}?\bfrom\s+(?P<old>"
                + NUMBER_PATTERN
                + r")\s*(?P<old_pct>%)?\s+to\s+(?P<new>"
                + NUMBER_PATTERN
                + r")\s*(?P<new_pct>%)?",
                question,
                flags=re.IGNORECASE,
            )
            if transition:
                override_metrics.add(metric)
                pairs = (
                    ("second model", transition.group("old"), bool(transition.group("old_pct"))),
                    ("first model", transition.group("new"), bool(transition.group("new_pct"))),
                )
                for label, raw, has_percent in pairs:
                    normalized = _anonymous_value(raw, has_percent, metric)
                    if normalized:
                        value, unit, measurement_unit = normalized
                        additions.append(
                            {
                                "model": label,
                                "metric": metric,
                                "value": value,
                                "unit": unit,
                                "measurement_unit": measurement_unit,
                                "direction": direction,
                                "raw_text": transition.group(0),
                            }
                        )
                break
            compared = re.search(
                r"(?P<first>"
                + NUMBER_PATTERN
                + r")\s*(?P<first_pct>%)?\s+"
                + pattern
                + r"[^0-9.]{0,40}?\bcompared\s+(?:with|to)\s+(?:an?\s+|about\s+|around\s+)?(?P<second>"
                + NUMBER_PATTERN
                + r")\s*(?P<second_pct>%)?",
                question,
                flags=re.IGNORECASE,
            )
            if compared:
                # "compared with" states both values explicitly, so it overrides
                # any segment-extracted value for the same (model, metric).
                override_metrics.add(metric)
                pairs = (
                    ("first model", compared.group("first"), bool(compared.group("first_pct"))),
                    ("second model", compared.group("second"), bool(compared.group("second_pct"))),
                )
                for label, raw, has_percent in pairs:
                    normalized = _anonymous_value(raw, has_percent, metric)
                    if normalized:
                        value, unit, measurement_unit = normalized
                        additions.append(
                            {
                                "model": label,
                                "metric": metric,
                                "value": value,
                                "unit": unit,
                                "measurement_unit": measurement_unit,
                                "direction": direction,
                                "raw_text": compared.group(0),
                            }
                        )
                break
            # "the new model has an F1 of 81%, while the previous model has 89%"
            while_match = re.search(
                r"new\s+(?:[a-z0-9-]+[\s-]){0,3}(?:model|version)[^0-9.]{0,30}?"
                + pattern
                + r"[^0-9.]{0,20}?(?P<first>" + NUMBER_PATTERN + r")\s*(?P<first_pct>%)?"
                + r"[^0-9.]{0,40}?\bwhile\b[^0-9.]{0,50}?(?:previous|prior|old)\s+(?:model|version)"
                + r"[^0-9.]{0,30}?(?:" + pattern + r")?[^0-9.]{0,20}?(?P<second>" + NUMBER_PATTERN + r")\s*(?P<second_pct>%)?",
                question,
                flags=re.IGNORECASE,
            )
            if while_match:
                override_metrics.add(metric)
                pairs = (
                    ("first model", while_match.group("first"), bool(while_match.group("first_pct"))),
                    ("second model", while_match.group("second"), bool(while_match.group("second_pct"))),
                )
                for label, raw, has_percent in pairs:
                    normalized = _anonymous_value(raw, has_percent, metric)
                    if normalized:
                        value, unit, measurement_unit = normalized
                        additions.append(
                            {
                                "model": label,
                                "metric": metric,
                                "value": value,
                                "unit": unit,
                                "measurement_unit": measurement_unit,
                                "direction": direction,
                                "raw_text": while_match.group(0),
                            }
                        )
                break
    return additions, override_metrics


def build_user_claim_rows(question):
    metrics = detect_metrics(question)
    models = detect_models(question)
    model_mentions = []

    for model in models:
        tolerant_pattern = r"[\s-]+".join(
            re.escape(part) for part in re.split(r"[\s-]+", model) if part
        )
        for match in re.finditer(tolerant_pattern, question, flags=re.IGNORECASE):
            model_mentions.append((match.start(), match.end(), model))

    anonymous_used = False
    if not model_mentions:
        model_mentions = _anonymous_model_mentions(question)
        anonymous_used = bool(model_mentions)

    model_mentions.sort(key=lambda item: item[0])
    distinct_mentions = []
    for mention in model_mentions:
        if not distinct_mentions or mention[0] >= distinct_mentions[-1][1]:
            distinct_mentions.append(mention)

    segments = []
    for index, (start, end, model) in enumerate(distinct_mentions):
        segment_end = distinct_mentions[index + 1][0] if index + 1 < len(distinct_mentions) else len(question)
        segments.append((model, question[end:segment_end]))

    if not segments:
        segments = [("unspecified", question)]

    rows = []
    for model, segment in segments:
        normalized_model = normalize_model_name(model)
        segment_row_count = 0
        for metric in metrics:
            observations = extract_metric_values(segment, metric)
            if observations:
                for observation in observations:
                    rows.append(
                        {
                            "model": normalized_model,
                            "metric": metric,
                            "value": observation["value"],
                            "unit": observation["unit"],
                            "measurement_unit": observation["measurement_unit"],
                            "direction": observation["direction"],
                            "raw_text": observation["raw_text"],
                        }
                    )
                    segment_row_count += 1
                continue
            # Fallback: assign a bare score only when this segment produced no
            # metric values at all (e.g. "Gemma achieves 84%" or "Model A = 72").
            # If the segment already yielded a value for another metric, its
            # numbers belong to that metric — never steal them.
            if segment_row_count > 0:
                continue
            definition = METRIC_DEFINITIONS.get(metric, {})
            direction = definition.get("direction", "context_dependent")
            value_match = re.search(
                r"(?P<value>" + NUMBER_PATTERN + r")\s*(?P<pct>%)?",
                segment,
            )
            if value_match:
                normalized = _anonymous_value(
                    value_match.group("value"),
                    bool(value_match.group("pct")),
                    metric,
                )
                if normalized:
                    value, unit, measurement_unit = normalized
                    rows.append(
                        {
                            "model": normalized_model,
                            "metric": metric,
                            "value": value,
                            "unit": unit,
                            "measurement_unit": measurement_unit,
                            "direction": direction,
                            "raw_text": segment.strip(),
                        }
                    )

    # If the claim names a single metric but models carry bare scores with no
    # metric keyword (e.g. "Model A = 72"), treat the metric as that score.
    if metrics:
        covered_pairs = {(row["model"], row["metric"]) for row in rows}
        single_metric = metrics[0]
        definition = METRIC_DEFINITIONS.get(single_metric, {})
        direction = definition.get("direction", "context_dependent")
        for model, segment in segments:
            normalized_model = normalize_model_name(model)
            if (normalized_model, single_metric) in covered_pairs:
                continue
            value_match = re.search(
                r"(?P<value>" + NUMBER_PATTERN + r")\s*(?P<pct>%)?",
                segment,
            )
            if not value_match:
                continue
            normalized = _anonymous_value(
                value_match.group("value"),
                bool(value_match.group("pct")),
                single_metric,
            )
            if normalized:
                value, unit, measurement_unit = normalized
                rows.append(
                    {
                        "model": normalized_model,
                        "metric": single_metric,
                        "value": value,
                        "unit": unit,
                        "measurement_unit": measurement_unit,
                        "direction": direction,
                        "raw_text": segment.strip(),
                    }
                )

    if anonymous_used and len({label for _, _, label in model_mentions}) >= 2:
        additions, override_metrics = _supplement_anonymous_pair_values(question, metrics, rows)
        if override_metrics:
            rows = [
                row
                for row in rows
                if not (
                    row["metric"] in override_metrics
                    and row["model"] in {"first model", "second model"}
                )
            ]
        rows.extend(additions)

    deduplicated = {}
    for row in rows:
        key = (row["model"], row["metric"], row["value"], row["unit"])
        deduplicated[key] = row
    return list(deduplicated.values())
