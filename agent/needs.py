from __future__ import annotations

import re

import pandas as pd

from .metrics import detect_models, normalize_model_name
from .retrieval import safe_text


NEED_ALIASES = {
    "languages_supported": {
        "label": "Languages supported",
        "patterns": [
            r"\blanguages?\b",
            r"\bmultilingual\b",
            r"\blanguage support\b",
        ],
    },
    "readability": {
        "label": "Readability",
        "patterns": [
            r"\breadability\b",
            r"\breadable\b",
            r"\bread(?:s|ing)? easily\b",
            r"\bclear (?:writing|responses|prose)\b",
            r"\beasy to (?:read|understand)\b",
        ],
    },
    "cost": {
        "label": "Cost and pricing",
        "patterns": [r"\bcost\b", r"\bpricing\b", r"\bprice\b", r"\bper token\b"],
    },
    "speed": {
        "label": "Speed and latency",
        "patterns": [r"\bspeed\b", r"\blatency\b", r"\btokens? per second\b", r"\binference time\b"],
    },
    "context_window": {
        "label": "Context window",
        "patterns": [r"\bcontext window\b", r"\bcontext length\b", r"\btoken window\b"],
    },
    "coding": {
        "label": "Coding capability",
        "patterns": [r"\bcoding\b", r"\bcode generation\b", r"\bprogramming\b"],
    },
    "reasoning": {
        "label": "Reasoning capability",
        "patterns": [r"\breasoning\b", r"\blogical reasoning\b", r"\bproblem solving\b"],
    },
    "multimodal": {
        "label": "Multimodal support",
        "patterns": [r"\bmultimodal\b", r"\bimage input\b", r"\baudio input\b", r"\bvideo input\b"],
    },
    "privacy": {
        "label": "Privacy and data handling",
        "patterns": [r"\bprivacy\b", r"\bdata retention\b", r"\btraining on (?:user )?data\b"],
    },
}

NEEDS_MARKER = re.compile(
    r"(?:i\s+care\s+(?:most\s+)?about|my\s+(?:needs|priorities|criteria)\s+(?:are|include)|priorities|criteria)\s*:\s*(.+?)\s*$",
    flags=re.IGNORECASE | re.DOTALL,
)


def _criterion_identity(criterion):
    normalized = re.sub(r"[^a-z0-9\s]", " ", criterion.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    for key, definition in NEED_ALIASES.items():
        if any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in definition["patterns"]):
            return key, definition["label"]
    return normalized.replace(" ", "_"), criterion.strip().capitalize()


def extract_requested_needs(question):
    match = NEEDS_MARKER.search(safe_text(question))
    if not match:
        return []

    raw_needs = re.split(r"[,;\n]|\band\b", match.group(1), flags=re.IGNORECASE)
    needs = []
    seen = set()
    for raw_need in raw_needs:
        raw_need = re.sub(r"^\s*(?:and|also)\s+", "", raw_need, flags=re.IGNORECASE)
        raw_need = raw_need.strip(" \t\r\n.,:;-")
        if not raw_need:
            continue
        key, label = _criterion_identity(raw_need)
        if key in seen:
            continue
        seen.add(key)
        needs.append({"key": key, "criterion": label, "user_wording": raw_need})
    return needs


def _model_family(model):
    normalized = normalize_model_name(model)
    for family in ("claude", "gemini", "gemma", "gpt", "llama", "mistral", "mixtral", "qwen", "deepseek", "grok", "product a", "product b"):
        if normalized == family or normalized.startswith(family + " ") or normalized.startswith(family + "-"):
            return family
    return normalized


def _matches_requested_model(requested, detected):
    requested_normalized = normalize_model_name(requested)
    detected_normalized = normalize_model_name(detected)
    has_requested_version = bool(re.search(r"\d", requested_normalized))
    if has_requested_version or requested_normalized.startswith("product "):
        compact_requested = re.sub(r"[\s-]+", "", requested_normalized)
        compact_detected = re.sub(r"[\s-]+", "", detected_normalized)
        return compact_requested == compact_detected
    return _model_family(requested_normalized) == _model_family(detected_normalized)


def models_match(requested, detected):
    return _matches_requested_model(requested, detected)


def _sentences(text):
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|[\n;]+", safe_text(text))
        if sentence.strip()
    ]


def _language_count(text):
    patterns = [
        r"\b(?:supports?|available in|understands?|speaks|covers)\s+(\d{1,4})\+?\s+languages?\b",
        r"\b(\d{1,4})\+?\s+languages?\s+(?:supported|available)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def _criterion_matches(criterion, text):
    definition = NEED_ALIASES.get(criterion["key"])
    if definition:
        return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in definition["patterns"])
    words = [word for word in re.findall(r"[a-z0-9]+", criterion["user_wording"].lower()) if len(word) > 2]
    normalized = text.lower()
    return bool(words) and all(word in normalized for word in words)


def build_needs_evidence(question, requested_models, evidence_dataframe):
    needs = extract_requested_needs(question)
    rows = []
    for need in needs:
        for requested_model in requested_models:
            findings = []
            criterion_seen_for_model = False
            for _, evidence_row in evidence_dataframe.iterrows():
                text = safe_text(evidence_row.get("raw_evidence", "")) or safe_text(
                    evidence_row.get("analysis_text", "")
                )
                sentences = _sentences(text)
                matched_model_sentences = []
                criterion_sentences = []
                source_models = detect_models(text)
                model_seen_in_source = any(
                    _matches_requested_model(requested_model, source_model)
                    for source_model in source_models
                )
                for sentence in sentences:
                    sentence_models = detect_models(sentence)
                    model_matches = any(
                        _matches_requested_model(requested_model, sentence_model)
                        for sentence_model in sentence_models
                    )
                    matches_criterion = _criterion_matches(need, sentence)
                    if model_matches and matches_criterion:
                        matched_model_sentences.append(sentence)
                    elif matches_criterion:
                        criterion_sentences.append(sentence)

                if model_seen_in_source and criterion_sentences:
                    criterion_seen_for_model = True

                for sentence in matched_model_sentences[:2]:
                    findings.append(
                        {
                            "value": _language_count(sentence) if need["key"] == "languages_supported" else None,
                            "unit": "languages" if need["key"] == "languages_supported" else None,
                            "finding": sentence[:600],
                            "source": safe_text(evidence_row.get("source", "")),
                            "title": safe_text(evidence_row.get("title", "")),
                            "url": safe_text(evidence_row.get("url", "")),
                            "evidence_id": safe_text(evidence_row.get("evidence_id", "")),
                        }
                    )

            if findings:
                status = "evidence_found"
                explanation = "Retrieved source text explicitly links this model to the requested criterion; review the cited source for context."
            elif criterion_seen_for_model:
                status = "criterion_found_model_not_linked"
                explanation = "The source mentions this criterion but does not clearly link it to this model."
            else:
                status = "not_found_in_retrieved_evidence"
                explanation = "No retrieved passage was found that supports this criterion for this model."

            rows.append(
                {
                    "criterion": need["criterion"],
                    "criterion_key": need["key"],
                    "user_wording": need["user_wording"],
                    "model": requested_model,
                    "status": status,
                    "explanation": explanation,
                    "findings": findings,
                }
            )

    return {
        "requested_needs": needs,
        "requested_models": requested_models,
        "rows": rows,
        "evidence_found_count": sum(row["status"] == "evidence_found" for row in rows),
        "summary": (
            "Requested preferences are assessed from retrieved passages per model; missing evidence is not interpreted as a negative result."
            if needs
            else "No explicit user priorities were supplied."
        ),
    }
