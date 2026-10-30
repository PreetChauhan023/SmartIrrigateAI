"""Compare prompt v1 vs v2 on the guardrail pass-rate using the REAL LLM (needs HF_TOKEN).
Run: HF_TOKEN=xxx python tests/compare_prompts.py   -> prints a results table for your report.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.chain import LANGS, generate_advisory, get_llm  # noqa: E402
from app.predictor import predict  # noqa: E402

CASES = [(12, 36, 7, 10), (15, 38, 8, 5), (35, 30, 7, 10), (12, 36, 7, 90), (18, 34, 7, 75), (14, 40, 9, 0)]

llm = get_llm()
if llm is None:
    sys.exit("Set HF_TOKEN to run the real-LLM comparison.")

print("| prompt | language | cases | guardrail pass | used LLM |")
print("|---|---|---|---|---|")
for v in ["v1", "v2"]:
    for lang in LANGS:
        ok = llm_used = 0
        for c in CASES:
            adv = generate_advisory(predict(*c), lang, llm=llm, prompt_version=v)
            ok += adv.guardrail_ok
            llm_used += adv.source == "llm"
        print(f"| {v} | {lang} | {len(CASES)} | {ok}/{len(CASES)} | {llm_used}/{len(CASES)} |")
