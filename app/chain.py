"""LangChain advisory chain: prediction dict -> farmer-friendly multilingual advisory.

prompt (versioned file) | LLM (HF Inference endpoint) | parser  -> guardrail -> result
If no HF_TOKEN / LLM failure / guardrail failure, a deterministic template is used,
so the demo never breaks. The LLM only PHRASES the decision; it never computes it.
"""
import os
import re
from pathlib import Path
from typing import Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
LANGS = {"English": "en", "Hindi": "hi", "Marathi": "mr", "Kannada": "kn"}
DEFAULT_REPO_LLM = os.getenv("HF_LLM_REPO", "Qwen/Qwen2.5-7B-Instruct")

_DIGITS = str.maketrans("०१२३४५६७८९೦೧೨೩೪೫೬೭೮೯", "01234567890123456789")

TEMPLATES = {
    "English": {
        "yes": "Soil is dry. Irrigate tomorrow morning for {h} hours.",
        "no": "Irrigation is not required today.",
        "no_rain": "Rainfall is expected ({r}% chance); irrigation is not required today.",
    },
    "Hindi": {
        "yes": "मिट्टी सूखी है। कल सुबह {h} घंटे सिंचाई करें।",
        "no": "आज सिंचाई की आवश्यकता नहीं है।",
        "no_rain": "बारिश की संभावना है ({r}%); आज सिंचाई की आवश्यकता नहीं है।",
    },
    "Marathi": {
        "yes": "माती कोरडी आहे. उद्या सकाळी {h} तास पाणी द्या.",
        "no": "आज पाणी देण्याची गरज नाही.",
        "no_rain": "पावसाची शक्यता आहे ({r}%); आज पाणी देण्याची गरज नाही.",
    },
    "Kannada": {
        "yes": "ಮಣ್ಣು ಒಣಗಿದೆ. ನಾಳೆ ಬೆಳಿಗ್ಗೆ {h} ಗಂಟೆ ನೀರು ಹಾಯಿಸಿ.",
        "no": "ಇಂದು ನೀರು ಹಾಯಿಸುವ ಅಗತ್ಯವಿಲ್ಲ.",
        "no_rain": "ಮಳೆಯ ಸಾಧ್ಯತೆ ಇದೆ ({r}%); ಇಂದು ನೀರು ಹಾಯಿಸುವ ಅಗತ್ಯವಿಲ್ಲ.",
    },
}

SCRIPT_RANGES = {"Hindi": ("ऀ", "ॿ"), "Marathi": ("ऀ", "ॿ"), "Kannada": ("ಀ", "೿")}


class Advisory(BaseModel):
    advisory_text: str
    irrigate: bool
    duration_hours: float
    language: str
    source: str  # "llm" | "template"
    prompt_version: str
    guardrail_ok: bool
    guardrail_notes: str = ""


def template_advisory(pred: dict, language: str) -> str:
    t = TEMPLATES[language]
    if pred["irrigate"]:
        return t["yes"].format(h=pred["duration_hours"])
    if pred["rain_prob"] >= 50:
        return t["no_rain"].format(r=int(pred["rain_prob"]))
    return t["no"]


def load_prompt(version: str = "v1") -> PromptTemplate:
    text = (ROOT / "prompts" / f"advisory_{version}.txt").read_text(encoding="utf-8")
    return PromptTemplate.from_template(text)


def get_llm() -> Optional[Runnable]:
    """HF-hosted LLM through langchain-huggingface. Returns None if no token is set."""
    if not os.getenv("HF_TOKEN"):
        return None
    from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

    ep = HuggingFaceEndpoint(
        repo_id=DEFAULT_REPO_LLM, task="text-generation", max_new_tokens=120,
        temperature=0.2, huggingfacehub_api_token=os.getenv("HF_TOKEN"),
    )
    return ChatHuggingFace(llm=ep)


def check_guardrails(text: str, pred: dict, language: str) -> tuple[bool, str]:
    """Deterministic checks on LLM output vs. the model's decision."""
    notes = []
    if not text.strip():
        return False, "empty output"
    nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", text.translate(_DIGITS))]
    allowed = {float(pred["duration_hours"]), float(pred["rain_prob"]), float(int(pred["rain_prob"])),
               float(pred["soil_moisture"]), float(round(pred["soil_moisture"]))}
    if pred["irrigate"]:
        if float(pred["duration_hours"]) not in nums:
            notes.append("duration missing/mismatch")
    else:
        # no irrigation: must not tell farmer to irrigate for some hours
        if any(n not in allowed for n in nums):
            notes.append("unexpected number in no-irrigation advisory")
    if any(n not in allowed for n in nums):
        notes.append("number not from prediction")
    if language in SCRIPT_RANGES:
        lo, hi = SCRIPT_RANGES[language]
        if not any(lo <= c <= hi for c in text):
            notes.append(f"wrong script for {language}")
    return (not notes), "; ".join(sorted(set(notes)))


def build_chain(llm: Optional[Runnable], prompt_version: str = "v1") -> Runnable:
    prompt = load_prompt(prompt_version)
    if llm is None:
        return RunnableLambda(lambda _: "")
    return prompt | llm | StrOutputParser()


def generate_advisory(pred: dict, language: str = "English", llm: Optional[Runnable] = None,
                      prompt_version: str = "v1", use_llm: bool = True) -> Advisory:
    if llm is None and use_llm:
        llm = get_llm()
    source, text, ok, notes = "template", "", True, ""
    if llm is not None and use_llm:
        try:
            chain = build_chain(llm, prompt_version)
            text = chain.invoke({
                "language": language, "irrigate": pred["irrigate"],
                "duration_hours": pred["duration_hours"], "rain_prob": pred["rain_prob"],
                "soil_moisture": pred["soil_moisture"],
            }).strip()
            ok, notes = check_guardrails(text, pred, language)
            source = "llm"
        except Exception as e:  # network / rate limit
            ok, notes, text = False, f"llm error: {type(e).__name__}", ""
    if source != "llm" or not ok:
        text = template_advisory(pred, language)
        source = "template"
    return Advisory(
        advisory_text=text, irrigate=pred["irrigate"], duration_hours=pred["duration_hours"],
        language=language, source=source, prompt_version=prompt_version,
        guardrail_ok=ok, guardrail_notes=notes,
    )
