"""Eval harness: checks the model, guardrails and advisory behaviour. Run: pytest -v"""
import sys
from pathlib import Path

import pytest
from langchain_core.runnables import RunnableLambda

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.chain import LANGS, SCRIPT_RANGES, check_guardrails, generate_advisory  # noqa: E402
from app.predictor import predict  # noqa: E402
from app.rag import answer, build_retriever  # noqa: E402

# (soil, temp, age, rain, expect_irrigate)
CASES = [
    (12, 36, 7, 10, True), (15, 38, 8, 5, True), (18, 34, 7, 20, True), (10, 30, 5, 30, True),
    (14, 40, 9, 0, True), (35, 30, 7, 10, False), (40, 28, 6, 10, False), (42, 33, 8, 5, False),
    (12, 36, 7, 90, False), (15, 38, 8, 85, False), (18, 34, 7, 75, False), (38, 25, 4, 60, False),
]


@pytest.mark.parametrize("soil,temp,age,rain,expect", CASES)
def test_model_decision(soil, temp, age, rain, expect):
    assert predict(soil, temp, age, rain)["irrigate"] == expect


@pytest.mark.parametrize("lang", list(LANGS))
@pytest.mark.parametrize("soil,temp,age,rain,expect", CASES[:3] + CASES[8:10])
def test_template_advisory_matches_prediction(lang, soil, temp, age, rain, expect):
    pred = predict(soil, temp, age, rain)
    adv = generate_advisory(pred, lang, use_llm=False)
    assert adv.irrigate == expect
    assert adv.source == "template"
    ok, notes = check_guardrails(adv.advisory_text, pred, lang)
    assert ok, notes
    if expect:
        assert str(pred["duration_hours"]) in adv.advisory_text.translate(
            str.maketrans("०१२३४५६७८९೦೧೨೩೪೫೬೭೮೯", "01234567890123456789"))
    if lang in SCRIPT_RANGES:
        lo, hi = SCRIPT_RANGES[lang]
        assert any(lo <= c <= hi for c in adv.advisory_text)


def test_no_rain_advice_when_irrigation_not_needed():
    pred = predict(12, 36, 7, 90)
    adv = generate_advisory(pred, "English", use_llm=False)
    assert "not required" in adv.advisory_text.lower()


def test_guardrail_rejects_wrong_duration_and_falls_back():
    pred = predict(12, 36, 7, 10)
    bad = RunnableLambda(lambda _: "Irrigate tomorrow for 9.9 hours.")
    adv = generate_advisory(pred, "English", llm=bad)
    assert adv.source == "template" and not adv.guardrail_ok


def test_guardrail_accepts_good_llm_output():
    pred = predict(12, 36, 7, 10)
    good = RunnableLambda(lambda _: f"Soil is dry. Irrigate tomorrow morning for {pred['duration_hours']} hours.")
    adv = generate_advisory(pred, "English", llm=good)
    assert adv.source == "llm" and adv.guardrail_ok


def test_guardrail_rejects_wrong_language_script():
    pred = predict(12, 36, 7, 10)
    wrong = RunnableLambda(lambda _: f"Irrigate for {pred['duration_hours']} hours.")  # English, asked Hindi
    adv = generate_advisory(pred, "Hindi", llm=wrong)
    assert adv.source == "template"


def test_llm_failure_falls_back():
    def boom(_):
        raise RuntimeError("rate limited")
    adv = generate_advisory(predict(12, 36, 7, 10), "English", llm=RunnableLambda(boom))
    assert adv.source == "template" and "llm error" in adv.guardrail_notes


def test_rag_retrieves_fertigation_context():
    r = answer("When should I apply fertigation?", retriever=build_retriever())
    assert r["sources"] and "fertigation" in r["answer"].lower()


def test_rag_unknown_question_is_safe():
    r = answer("zzzz qqqq xxxx", retriever=build_retriever())
    assert "not sure" in r["answer"].lower()
