"""JSON schema for teacher output, plus label-consistency validation.

The teacher (Fable) returns a batch of samples matching SAMPLE_BATCH_SCHEMA.
`validate_sample` enforces the invariants the schema cannot express
(excerpts must appear in the text, compliant <=> no violations, the
communication_type must match the recipient metadata, ...).
"""

from __future__ import annotations

import re

from .taxonomy import (
    COMMUNICATION_TYPES,
    DIFFICULTIES,
    RULE_CITES,
    SEVERITIES,
    VIOLATION_CATEGORIES,
)

VIOLATION_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": sorted(VIOLATION_CATEGORIES)},
        "rule": {"type": "string", "enum": RULE_CITES},
        "excerpt": {
            "type": "string",
            "description": "Verbatim substring of the communication (<= 200 chars) that triggers the finding.",
        },
        "explanation": {"type": "string", "description": "1-2 sentences: why this excerpt violates the cited rule."},
    },
    "required": ["category", "rule", "excerpt", "explanation"],
    "additionalProperties": False,
}

LABEL_SCHEMA = {
    "type": "object",
    "properties": {
        "communication_type": {"type": "string", "enum": COMMUNICATION_TYPES},
        "compliant": {"type": "boolean"},
        "violations": {"type": "array", "items": VIOLATION_SCHEMA},
        "severity": {"type": "string", "enum": SEVERITIES},
        "required_actions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "What the reviewer should do: e.g. 'Return to rep for revision', 'Route for principal approval', 'File with FINRA within 10 business days', 'No action'.",
        },
        "rationale": {"type": "string", "description": "2-4 sentences a compliance reviewer would write, citing rule paragraphs."},
    },
    "required": ["communication_type", "compliant", "violations", "severity", "required_actions", "rationale"],
    "additionalProperties": False,
}

METADATA_SCHEMA = {
    "type": "object",
    "properties": {
        "sender_role": {"type": "string"},
        "firm": {"type": "string", "description": "Fictional member firm name. Never a real firm."},
        "recipient_description": {"type": "string"},
        "recipient_count_30d": {"type": "integer", "description": "Number of distinct recipients this communication (or template) reached in the trailing 30 days."},
        "recipient_type": {"type": "string", "enum": ["retail", "institutional", "mixed"]},
        "subject": {"type": "string", "description": "Subject line or empty string."},
    },
    "required": ["sender_role", "firm", "recipient_description", "recipient_count_30d", "recipient_type", "subject"],
    "additionalProperties": False,
}

SAMPLE_SCHEMA = {
    "type": "object",
    "properties": {
        "scenario_id": {"type": "string", "description": "Echo the scenario_id you were given."},
        "difficulty": {"type": "string", "enum": DIFFICULTIES},
        "channel": {"type": "string"},
        "metadata": METADATA_SCHEMA,
        "communication": {"type": "string", "description": "The full text of the communication exactly as sent, including greeting, body, sign-off, and any footer/disclaimer."},
        "label": LABEL_SCHEMA,
        "generator_notes": {"type": "string", "description": "Why this sample is at its difficulty level and what makes it non-trivial. Not used for training."},
    },
    "required": ["scenario_id", "difficulty", "channel", "metadata", "communication", "label", "generator_notes"],
    "additionalProperties": False,
}

SAMPLE_BATCH_SCHEMA = {
    "type": "object",
    "properties": {"samples": {"type": "array", "items": SAMPLE_SCHEMA}},
    "required": ["samples"],
    "additionalProperties": False,
}


_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS.sub(" ", s).strip().lower()


def validate_sample(sample: dict, scenario: dict | None = None) -> list[str]:
    """Return a list of problems; empty list means the sample is acceptable."""
    problems: list[str] = []
    label = sample.get("label", {})
    meta = sample.get("metadata", {})
    text = sample.get("communication", "")
    norm_text = _norm(text)

    if len(text) < 40:
        problems.append("communication too short")
    if len(text) > 6000:
        problems.append("communication too long")

    violations = label.get("violations", [])
    compliant = label.get("compliant")
    severity = label.get("severity")

    if compliant and violations:
        problems.append("compliant=True but violations listed")
    if not compliant and not violations:
        problems.append("compliant=False but no violations listed")
    if compliant and severity != "none":
        problems.append("compliant=True but severity != none")
    if not compliant and severity == "none":
        problems.append("compliant=False but severity == none")

    ctype = label.get("communication_type")
    rtype = meta.get("recipient_type")
    count = meta.get("recipient_count_30d", 0)
    if rtype == "institutional" and ctype != "institutional_communication":
        # allowed only if the label calls out institutional_forwarding (then it is really retail/correspondence)
        if not any(v.get("category") == "institutional_forwarding" for v in violations):
            problems.append("institutional recipients but communication_type is not institutional_communication")
    if rtype == "retail":
        expected = "correspondence" if count <= 25 else "retail_communication"
        if ctype != expected:
            problems.append(f"retail count {count} implies {expected}, got {ctype}")
    if rtype == "mixed" and ctype == "institutional_communication":
        problems.append("mixed audience cannot be an institutional_communication")

    seen_cats = set()
    for v in violations:
        cat = v.get("category")
        spec = VIOLATION_CATEGORIES.get(cat)
        if spec is None:
            problems.append(f"unknown category {cat}")
            continue
        if v.get("rule") != spec["rule"]:
            problems.append(f"category {cat} must cite {spec['rule']}, got {v.get('rule')}")
        if ctype not in spec["applies_to"]:
            problems.append(f"category {cat} does not apply to {ctype}")
        ex = _norm(v.get("excerpt", ""))
        if not ex or ex not in norm_text:
            problems.append(f"excerpt not found verbatim in communication: {v.get('excerpt', '')[:60]!r}")
        if cat in seen_cats:
            problems.append(f"duplicate category {cat}")
        seen_cats.add(cat)

    if scenario is not None:
        if sample.get("difficulty") != scenario["difficulty"]:
            problems.append("difficulty does not match scenario")
        if bool(compliant) != bool(scenario["compliant"]):
            problems.append("compliant flag does not match scenario")
        if not compliant:
            wanted = set(scenario["violation_categories"])
            if wanted and not (wanted & seen_cats):
                problems.append(f"none of the requested categories {sorted(wanted)} present")

    return problems
