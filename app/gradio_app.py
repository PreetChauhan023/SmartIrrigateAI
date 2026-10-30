"""Gradio UI (deploy on HF Spaces: set app_file to app/gradio_app.py or copy to root as app.py).

Tabs: Advisory (with Accept / Override human-in-the-loop) and Ask the Agronomy Bot (RAG).
Feedback is appended to feedback/feedback.csv = retraining/evaluation signal.
"""
import csv
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gradio as gr  # noqa: E402

from app.chain import LANGS, generate_advisory  # noqa: E402
from app.predictor import predict  # noqa: E402
from app.rag import answer  # noqa: E402

FB = ROOT / "feedback" / "feedback.csv"


def log_feedback(decision, lang, text, soil, temp, age, rain):
    FB.parent.mkdir(exist_ok=True)
    new = not FB.exists()
    with open(FB, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "decision", "language", "advisory", "soil", "temp", "age", "rain"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), decision, lang, text, soil, temp, age, rain])
    return f"Recorded: {decision}"


def run_advisory(soil, temp, age, rain, lang, prompt_version):
    pred = predict(soil, temp, age, rain)
    adv = generate_advisory(pred, lang, prompt_version=prompt_version)
    info = (f"Model {pred['model_version']} | irrigate={pred['irrigate']} | "
            f"hours={pred['duration_hours']} | confidence={pred['confidence']} | "
            f"source={adv.source} | guardrail_ok={adv.guardrail_ok} {adv.guardrail_notes}")
    return adv.advisory_text, info


def run_chat(q, lang):
    r = answer(q, lang)
    return f"{r['answer']}\n\n(sources: {', '.join(set(r['sources']))}; mode: {r['mode']})"


with gr.Blocks(title="Sugarcane Irrigation Advisor") as demo:
    gr.Markdown("# Sugarcane Irrigation Advisor\nHF model + LangChain advisory. Decision support only; farmer/agronomist confirms.")
    with gr.Tab("Advisory"):
        with gr.Row():
            soil = gr.Slider(5, 50, 22, label="Soil moisture (%)")
            temp = gr.Slider(15, 45, 32, label="Temperature (°C)")
            age = gr.Slider(0.5, 12, 7, label="Crop age (months)")
            rain = gr.Slider(0, 100, 20, label="Rain probability next 24h (%)")
        with gr.Row():
            lang = gr.Dropdown(list(LANGS), value="English", label="Language")
            pv = gr.Dropdown(["v1", "v2"], value="v1", label="Prompt version")
        btn = gr.Button("Get advisory", variant="primary")
        out = gr.Textbox(label="Advisory", lines=3)
        meta = gr.Textbox(label="Details (for MLOps demo)")
        btn.click(run_advisory, [soil, temp, age, rain, lang, pv], [out, meta])
        with gr.Row():
            acc = gr.Button("Accept")
            ovr = gr.Button("Override")
        status = gr.Textbox(label="Feedback status")
        acc.click(lambda *a: log_feedback("accept", *a), [lang, out, soil, temp, age, rain], status)
        ovr.click(lambda *a: log_feedback("override", *a), [lang, out, soil, temp, age, rain], status)
    with gr.Tab("Ask the Agronomy Bot"):
        q = gr.Textbox(label="Your question", placeholder="When should I apply fertigation?")
        lang2 = gr.Dropdown(list(LANGS), value="English", label="Language")
        a = gr.Textbox(label="Answer", lines=6)
        gr.Button("Ask").click(run_chat, [q, lang2], a)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", 7860)))
