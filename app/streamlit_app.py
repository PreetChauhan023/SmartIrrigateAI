"""
SmartIrrigate AI
Multilingual RAG-Based Irrigation Advisory Assistant

Technologies:
- Streamlit
- Hugging Face Multilingual Embeddings
- FAISS Vector Database
- LangChain
- Ollama (Qwen 2.5 1.5B)

Run:
python -m streamlit run app/streamlit_app.py
"""

import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

# --------------------------------------------------
# PROJECT PATH
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.rag_demo import DemoRetriever, ask, DEFAULT_LLM


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="SmartIrrigateAI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# THEME / STYLING
# --------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --green-900: #0F3D24;
        --green-800: #14532D;
        --green-700: #166534;
        --green-600: #16A34A;
        --green-100: #DCFCE7;
        --green-50:  #F0FDF4;
        --ink:       #0F172A;
        --muted:     #64748B;
        --line:      #E2E8F0;
        --bg:        #F4F7F5;
        --card:      #FFFFFF;
        --amber-bg:  #FFFBEB;
        --amber-bd:  #FDE68A;
        --amber-tx:  #92400E;
    }

    html, body, [class*="css"], .stApp {
        font-family: 'Inter', sans-serif;
    }

    .stApp { background-color: var(--bg); }

    /* Hide default Streamlit chrome */
    #MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }

    .block-container {
        max-width: 100%;
        padding-top: 1.2rem;
        padding-left: 3rem;
        padding-right: 3rem;
        padding-bottom: 6rem;
    }

    /* Keep headings inside answers proportionate */
    [data-testid="stChatMessage"] h1,
    [data-testid="stChatMessage"] h2,
    [data-testid="stChatMessage"] h3 {
        font-size: 1.1rem;
        font-weight: 600;
        margin: 1rem 0 0.4rem 0;
        padding: 0;
    }

    /* ---------- Hero ---------- */
    .hero {
        background: linear-gradient(135deg, var(--green-900) 0%, var(--green-700) 60%, #22A053 100%);
        border-radius: 18px;
        padding: 28px 32px;
        color: #fff;
        box-shadow: 0 10px 30px rgba(20, 83, 45, 0.18);
        margin-bottom: 18px;
    }
    .hero-top { display: flex; align-items: center; gap: 14px; }
    .hero-logo {
        width: 52px; height: 52px; border-radius: 14px;
        background: rgba(255,255,255,0.16);
        display: flex; align-items: center; justify-content: center;
        font-size: 28px;
    }
    .hero h1 {
        color: #fff !important;
        font-size: 1.7rem; font-weight: 700; margin: 0; padding: 0;
        letter-spacing: -0.02em;
    }
    .hero .tagline { color: #D1FAE5; font-size: 0.95rem; margin-top: 2px; }
    .hero p.desc {
        color: #ECFDF5; font-size: 0.95rem; line-height: 1.55;
        margin: 16px 0 14px 0; max-width: 720px;
    }
    .chip {
        display: inline-block;
        background: rgba(255,255,255,0.16);
        border: 1px solid rgba(255,255,255,0.28);
        color: #fff; font-size: 0.78rem; font-weight: 500;
        padding: 4px 12px; border-radius: 999px; margin: 0 6px 6px 0;
    }

    /* ---------- Stat cards ---------- */
    .stat {
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 14px 18px;
    }
    .stat .label {
        color: var(--muted); font-size: 0.74rem; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.06em;
    }
    .stat .value { color: var(--green-800); font-size: 1.6rem; font-weight: 700; margin-top: 2px; }

    /* ---------- Section titles ---------- */
    .section-title {
        color: var(--ink); font-size: 1.05rem; font-weight: 600;
        margin: 22px 0 10px 0;
    }
    .section-sub { color: var(--muted); font-size: 0.88rem; margin-top: -6px; margin-bottom: 12px; }

    /* ---------- Feature cards ---------- */
    .feature {
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 18px;
        height: 100%;
    }
    .feature .icon { font-size: 1.5rem; }
    .feature .title { font-weight: 600; color: var(--ink); margin-top: 6px; font-size: 0.95rem; }
    .feature .text { color: var(--muted); font-size: 0.85rem; margin-top: 4px; line-height: 1.5; }

    /* ---------- Chat ---------- */
    [data-testid="stChatMessage"] {
        background-color: var(--card);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 14px 16px;
        margin-bottom: 12px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
    }
    [data-testid="stChatMessage"] p { line-height: 1.65; }

    [data-testid="stChatInput"] {
        border-radius: 14px;
        border: 1px solid var(--line);
    }

    .advisory-tag {
        display: inline-block;
        background: var(--green-50);
        color: var(--green-700);
        border: 1px solid var(--green-100);
        font-size: 0.7rem; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.06em;
        padding: 3px 10px; border-radius: 999px; margin-bottom: 8px;
    }
    .meta { color: var(--muted); font-size: 0.75rem; margin-top: 6px; }

    .source-card-title { color: var(--green-800); font-weight: 600; font-size: 0.9rem; }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid var(--line);
    }
    .brand { display: flex; align-items: center; gap: 10px; margin-bottom: 2px; }
    .brand .logo {
        width: 38px; height: 38px; border-radius: 10px;
        background: var(--green-50); border: 1px solid var(--green-100);
        display: flex; align-items: center; justify-content: center; font-size: 20px;
    }
    .brand .name { font-weight: 700; color: var(--green-800); font-size: 1.05rem; line-height: 1.1; }
    .brand .sub { color: var(--muted); font-size: 0.75rem; }

    .sys-row {
        display: flex; justify-content: space-between; align-items: baseline;
        padding: 8px 0; border-bottom: 1px dashed var(--line); font-size: 0.82rem;
    }
    .sys-row .k { color: var(--muted); }
    .sys-row .v { color: var(--ink); font-weight: 500; text-align: right; }

    .status-ok {
        display: inline-flex; align-items: center; gap: 6px;
        background: var(--green-50); color: var(--green-700);
        border: 1px solid var(--green-100);
        border-radius: 999px; padding: 4px 12px; font-size: 0.78rem; font-weight: 500;
    }
    .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--green-600); display: inline-block; }

    .notice {
        background: var(--amber-bg); border: 1px solid var(--amber-bd);
        color: var(--amber-tx); border-radius: 12px;
        padding: 12px 14px; font-size: 0.8rem; line-height: 1.5;
    }

    /* ---------- Buttons ---------- */
    div.stButton > button {
        border-radius: 10px;
        border: 1px solid var(--line);
        background: #fff;
        color: var(--ink);
        font-weight: 500;
        text-align: left;
        transition: all 0.15s ease;
    }
    div.stButton > button:hover {
        border-color: var(--green-600);
        color: var(--green-700);
        background: var(--green-50);
    }

    [data-testid="stDownloadButton"] button {
        border-radius: 10px; width: 100%;
    }

    h1, h2, h3 { color: var(--green-800); }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "rag_messages" not in st.session_state:
    st.session_state.rag_messages = []

if "rag_results" not in st.session_state:
    st.session_state.rag_results = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def render_evidence(result: dict) -> None:
    """Show retrieved knowledge as numbered evidence cards."""
    sources = result.get("retrieved", [])

    if not sources:
        st.caption("No supporting records were retrieved for this answer.")
        return

    with st.expander(f"📚 Supporting evidence ({len(sources)} records)"):
        for i, source in enumerate(sources, start=1):
            with st.container(border=True):
                st.markdown(
                    f"<div class='source-card-title'>"
                    f"[{i}] {source.get('topic', 'Untitled')}</div>",
                    unsafe_allow_html=True,
                )
                st.write(source.get("text", ""))


def render_assistant_message(message: dict, show_evidence: bool) -> None:
    st.markdown(
        "<span class='advisory-tag'>🌱 Irrigation Advisory</span>",
        unsafe_allow_html=True,
    )
    st.markdown(message["content"])

    if message.get("time"):
        st.markdown(
            f"<div class='meta'>Generated at {message['time']}</div>",
            unsafe_allow_html=True,
        )

    if show_evidence:
        idx = message.get("result_index")
        if idx is not None and idx < len(st.session_state.rag_results):
            render_evidence(st.session_state.rag_results[idx])


def build_transcript() -> str:
    lines = ["SmartIrrigate AI - Conversation Transcript", "=" * 44, ""]
    for m in st.session_state.rag_messages:
        who = "You" if m["role"] == "user" else "SmartIrrigate AI"
        lines.append(f"{who}:\n{m['content']}\n")
    lines.append("-" * 44)
    lines.append(
        "Educational prototype. Recommendations are not field-validated."
    )
    return "\n".join(lines)


# --------------------------------------------------
# CACHED RETRIEVER
# --------------------------------------------------

@st.cache_resource(show_spinner="Loading irrigation knowledge base...")
def load_retriever():
    return DemoRetriever(
        mode="huggingface",
        k=4,
    )


# --------------------------------------------------
# LOAD KNOWLEDGE BASE
# --------------------------------------------------

kb_error = None
retriever = None

try:
    retriever = load_retriever()
except Exception as exc:
    kb_error = exc


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.markdown(
        """
        <div class="brand">
            <div class="logo">🌱</div>
            <div>
                <div class="name">SmartIrrigate AI</div>
                <div class="sub">Irrigation Advisory Assistant</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    if kb_error is None:
        st.markdown(
            "<span class='status-ok'><span class='dot'></span>Knowledge base online</span>",
            unsafe_allow_html=True,
        )

    st.markdown("<div class='section-title'>System</div>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="sys-row"><span class="k">Language model</span><span class="v">{DEFAULT_LLM}</span></div>
        <div class="sys-row"><span class="k">Runtime</span><span class="v">Ollama (local)</span></div>
        <div class="sys-row"><span class="k">Embeddings</span><span class="v">Multilingual MiniLM</span></div>
        <div class="sys-row"><span class="k">Vector store</span><span class="v">FAISS</span></div>
        <div class="sys-row"><span class="k">Framework</span><span class="v">LangChain</span></div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='section-title'>Settings</div>", unsafe_allow_html=True)

    show_sources = st.toggle(
        "Show supporting evidence",
        value=False,
        help="Display the knowledge records used to generate each answer.",
    )

    if st.button("🗑️  Clear conversation", use_container_width=True):
        st.session_state.rag_messages = []
        st.session_state.rag_results = []
        st.session_state.pending_question = None
        st.rerun()
    st.write("")

# --------------------------------------------------
# HERO HEADER
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <div class="hero-top">
            <div class="hero-logo">🌱</div>
            <div>
                <h1>SmartIrrigate AI</h1>
                <div class="tagline">Multilingual irrigation advisory powered by retrieval-augmented AI</div>
            </div>
        </div>
        <p class="desc">
            Get guidance on irrigation timing, soil moisture, crop water requirements,
            and rainfall planning, grounded in a curated agricultural knowledge base.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
# --------------------------------------------------
# KNOWLEDGE BASE ERROR
# --------------------------------------------------

if kb_error is not None:
    st.error(f"Unable to load the irrigation knowledge base: {kb_error}")
    st.info(
        "Check that knowledge/synthetic_guidance.json exists "
        "and the required embedding dependencies are installed."
    )
    st.stop()


# --------------------------------------------------
# WELCOME / SUGGESTED QUESTIONS
# --------------------------------------------------

if not st.session_state.rag_messages:

    st.markdown(
        "<div class='section-title'>What I can help with</div>",
        unsafe_allow_html=True,
    )

    f1, f2, f3 = st.columns(3)

    features = [
        (f1, "⏱️", "Irrigation timing", "When and how often to irrigate different crops across growth stages."),
        (f2, "🌡️", "Soil & weather", "How soil moisture and rainfall should shape your watering decisions."),
        (f3, "🗣️", "Your language", "Ask in English, Hindi, Gujarati, or Marathi and get answers back in kind."),
    ]

    for col, icon, title, text in features:
        with col:
            st.markdown(
                f"<div class='feature'><div class='icon'>{icon}</div>"
                f"<div class='title'>{title}</div>"
                f"<div class='text'>{text}</div></div>",
                unsafe_allow_html=True,
            )

    st.markdown(
        "<div class='section-title'>Try asking</div>"
        "<div class='section-sub'>Select a question to get started.</div>",
        unsafe_allow_html=True,
    )

    suggestions = [
        "When should I irrigate sugarcane?",
        "गन्ने की फसल में सिंचाई कब करनी चाहिए?",
        "શેરડીના પાકમાં પાણી ક્યારે આપવું જોઈએ?",
        "How does soil moisture affect irrigation?",
    ]

    left, right = st.columns(2)

    for index, question in enumerate(suggestions):
        with left if index % 2 == 0 else right:
            if st.button(
                question,
                key=f"suggestion_{index}",
                use_container_width=True,
            ):
                st.session_state.pending_question = question
                st.rerun()

else:
    st.markdown(
        "<div class='section-title'>Conversation</div>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# DISPLAY CHAT HISTORY
# --------------------------------------------------

for message in st.session_state.rag_messages:

    avatar = "🧑‍🌾" if message["role"] == "user" else "🌱"

    with st.chat_message(message["role"], avatar=avatar):
        if message["role"] == "assistant":
            render_assistant_message(message, show_sources)
        else:
            st.markdown(message["content"])


# --------------------------------------------------
# QUESTION INPUT
# --------------------------------------------------

typed_question = st.chat_input("Ask your irrigation question in any supported language...")

suggested_question = st.session_state.pending_question

if suggested_question:
    st.session_state.pending_question = None

question = suggested_question or typed_question


# --------------------------------------------------
# GENERATE RESPONSE
# --------------------------------------------------

if question:

    previous_history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.rag_messages
    ]

    st.session_state.rag_messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user", avatar="🧑‍🌾"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🌱"):

        with st.spinner("Analysing your question and retrieving guidance..."):

            try:

                result = ask(
                    question=question,
                    language="Auto",
                    retriever=retriever,
                    use_llm=True,
                    history=previous_history,
                )

                answer = result.get(
                    "answer",
                    "Sorry, I couldn't generate a response.",
                )

                result_index = len(st.session_state.rag_results)
                st.session_state.rag_results.append(result)

                assistant_message = {
                    "role": "assistant",
                    "content": answer,
                    "result_index": result_index,
                    "time": datetime.now().strftime("%H:%M"),
                }

                st.session_state.rag_messages.append(assistant_message)

                render_assistant_message(assistant_message, show_sources)

                if result.get("error"):
                    st.error("The advisory engine encountered an error.")

                    with st.expander("View error details", expanded=True):
                        st.code(str(result["error"]))

                    print(
                        "\n========== STREAMLIT RAG ERROR ==========\n",
                        result["error"],
                        "\n=========================================\n",
                        flush=True,
                    )

            except Exception as exc:

                st.error(f"Unable to generate an advisory: {exc}")

