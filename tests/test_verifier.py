"""Tests for the independent verifier JSON protocol."""

import json

import pytest

from verifier import (
    VerifierProtocolError,
    VerifierVerdict,
    parse_verifier_review,
    verifier_messages,
)


def review_json(verdict: str = "PASS") -> str:
    return json.dumps({
        "verdict": verdict,
        "summary": "The implementation satisfies the supplied contract.",
        "requirement_checks": [{
            "requirement": "Return the correct result",
            "satisfied": verdict == "PASS",
            "reason": "The implementation handles the required behavior",
        }],
        "counterexamples": [] if verdict == "PASS" else ["Empty input fails"],
        "unresolved_assumptions": [],
    })


def test_parse_valid_pass_review() -> None:
    review = parse_verifier_review(review_json())

    assert review.verdict is VerifierVerdict.PASS
    assert review.requirement_checks[0].satisfied is True


def test_pass_cannot_hide_counterexamples() -> None:
    data = json.loads(review_json())
    data["counterexamples"] = ["A known failing input"]

    with pytest.raises(VerifierProtocolError, match="PASS cannot contain"):
        parse_verifier_review(json.dumps(data))


def test_review_cannot_use_a_vacuous_requirement_check_list() -> None:
    data = json.loads(review_json())
    data["requirement_checks"] = []

    with pytest.raises(VerifierProtocolError, match="between 1 and"):
        parse_verifier_review(json.dumps(data))


def test_unknown_fields_and_invalid_json_are_rejected() -> None:
    data = json.loads(review_json())
    data["extra"] = True

    with pytest.raises(VerifierProtocolError, match="fields must be exactly"):
        parse_verifier_review(json.dumps(data))
    with pytest.raises(VerifierProtocolError, match="not valid JSON"):
        parse_verifier_review("not-json")


def test_verifier_messages_are_a_fresh_two_message_context() -> None:
    messages = verifier_messages({"original_requirement": "Implement add"})

    assert [message["role"] for message in messages] == ["system", "user"]
    assert "independent code verifier" in messages[0]["content"]
    assert "Implement add" in messages[1]["content"]
