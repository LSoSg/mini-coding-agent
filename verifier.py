"""Independent, read-only verifier protocol for the dual-model pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any

MAX_VERIFIER_ATTEMPTS = 2
MAX_REVIEW_ITEMS = 20
MAX_REVIEW_TEXT_CHARS = 1_000

VERIFIER_SYSTEM_PROMPT = """You are an independent code verifier, not the builder.
Your job is to challenge whether the final implementation satisfies the original user
requirement, which is authoritative. Builder-authored success criteria may be incomplete.
Look for counterexamples, missing edge cases, and assumptions introduced by the builder.
Do not reward the implementation merely because its self-tests passed.

You have no tools and cannot modify files. Review only the supplied requirement contract
and non-test implementation files. Return strict JSON only, with exactly this shape:
{
  "verdict": "PASS or FAIL",
  "summary": "concise independent conclusion",
  "requirement_checks": [
    {"requirement": "requirement text", "satisfied": true, "reason": "evidence"}
  ],
  "counterexamples": ["concrete input or scenario that may break the implementation"],
  "unresolved_assumptions": ["assumption not justified by the user requirement"]
}

Use FAIL when a required behavior is missing, a concrete counterexample remains, or an
unresolved assumption can affect correctness. Use PASS only when no correctness-blocking
issue is found. Do not include Markdown fences or text outside the JSON object."""


class VerifierVerdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"


@dataclass(frozen=True)
class RequirementCheck:
    requirement: str
    satisfied: bool
    reason: str


@dataclass(frozen=True)
class VerifierReview:
    verdict: VerifierVerdict
    summary: str
    requirement_checks: list[RequirementCheck]
    counterexamples: list[str]
    unresolved_assumptions: list[str]


class VerifierProtocolError(ValueError):
    """Raised when the verifier does not follow the structured response protocol."""


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise VerifierProtocolError(f"{field} must be a non-empty string.")
    value = value.strip()
    if len(value) > MAX_REVIEW_TEXT_CHARS:
        raise VerifierProtocolError(
            f"{field} cannot exceed {MAX_REVIEW_TEXT_CHARS} characters."
        )
    return value


def _text_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or len(value) > MAX_REVIEW_ITEMS:
        raise VerifierProtocolError(
            f"{field} must be a list with at most {MAX_REVIEW_ITEMS} items."
        )
    return [_text(item, f"{field} item") for item in value]


def parse_verifier_review(raw_review: str) -> VerifierReview:
    if not isinstance(raw_review, str):
        raise VerifierProtocolError("Verifier response must be a JSON string.")
    try:
        data = json.loads(raw_review)
    except json.JSONDecodeError as exc:
        raise VerifierProtocolError(f"Verifier response is not valid JSON: {exc}") from exc
    expected = {
        "verdict",
        "summary",
        "requirement_checks",
        "counterexamples",
        "unresolved_assumptions",
    }
    if not isinstance(data, dict) or set(data) != expected:
        raise VerifierProtocolError(
            f"Verifier fields must be exactly {sorted(expected)}."
        )
    try:
        verdict = VerifierVerdict(data["verdict"])
    except (TypeError, ValueError) as exc:
        raise VerifierProtocolError("verdict must be PASS or FAIL.") from exc

    raw_checks = data["requirement_checks"]
    if (
        not isinstance(raw_checks, list)
        or not raw_checks
        or len(raw_checks) > MAX_REVIEW_ITEMS
    ):
        raise VerifierProtocolError(
            "requirement_checks must contain between 1 and "
            f"{MAX_REVIEW_ITEMS} items."
        )
    checks: list[RequirementCheck] = []
    for index, item in enumerate(raw_checks, start=1):
        if not isinstance(item, dict) or set(item) != {
            "requirement", "satisfied", "reason"
        }:
            raise VerifierProtocolError(
                f"requirement_checks item {index} has invalid fields."
            )
        if not isinstance(item["satisfied"], bool):
            raise VerifierProtocolError(
                f"requirement_checks item {index} satisfied must be boolean."
            )
        checks.append(RequirementCheck(
            requirement=_text(item["requirement"], "requirement"),
            satisfied=item["satisfied"],
            reason=_text(item["reason"], "reason"),
        ))

    counterexamples = _text_list(data["counterexamples"], "counterexamples")
    assumptions = _text_list(
        data["unresolved_assumptions"], "unresolved_assumptions"
    )
    if verdict is VerifierVerdict.PASS and (
        any(not check.satisfied for check in checks)
        or counterexamples
        or assumptions
    ):
        raise VerifierProtocolError(
            "PASS cannot contain failed checks, counterexamples, or unresolved assumptions."
        )
    if verdict is VerifierVerdict.FAIL and (
        all(check.satisfied for check in checks)
        and not counterexamples
        and not assumptions
    ):
        raise VerifierProtocolError(
            "FAIL must identify a failed check, counterexample, or unresolved assumption."
        )
    return VerifierReview(
        verdict=verdict,
        summary=_text(data["summary"], "summary"),
        requirement_checks=checks,
        counterexamples=counterexamples,
        unresolved_assumptions=assumptions,
    )


def verifier_messages(context: dict[str, Any]) -> list[dict[str, str]]:
    """Build a fresh verifier conversation with no Builder message history."""

    return [
        {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Independently review this completed coding task:\n"
                + json.dumps(context, ensure_ascii=False, indent=2)
            ),
        },
    ]
