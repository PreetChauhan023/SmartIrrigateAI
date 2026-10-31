"""
SmartIrrigate AI
Multilingual RAG-Based Agricultural Advisory Chatbot
with Integrated ML Irrigation Prediction

Run:
python -m streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path
from datetime import datetime

import streamlit as st


# ============================================================
# PROJECT PATH
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================
# IMPORT EXISTING PROJECT MODULES
# ============================================================

from app.rag_demo import (
    DemoRetriever,
    ask,
    DEFAULT_LLM,
)

from app.prediction_chat import (
    is_prediction_request,
    extract_inputs,
    answer_prediction,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SmartIrrigate AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM WHITE AND GREEN THEME
# ============================================================

st.markdown(
    """
    <style>

    @import url(
        'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap'
    );

    :root {
        --green-dark: #0F3D24;
        --green-primary: #166534;
        --green-light: #DCFCE7;
        --green-bg: #F0FDF4;

        --background: #F4F7F5;
        --white: #FFFFFF;

        --text: #0F172A;
        --muted: #64748B;
        --border: #E2E8F0;
    }

    html, body, .stApp {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: var(--background);
        color: var(--text);
    }

    .block-container {
        max-width: 1250px;
        padding-top: 1.5rem;
        padding-bottom: 5rem;
    }

    /* Sidebar */

    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid var(--border);
    }

    /* Hero */

    .hero {
        background: linear-gradient(
            135deg,
            #0F3D24,
            #166534,
            #22A053
        );

        padding: 30px;
        border-radius: 18px;
        margin-bottom: 24px;
        color: white;
    }

    .hero h1 {
        color: white !important;
        font-size: 30px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .hero p {
        color: #ECFDF5;
        font-size: 15px;
        line-height: 1.7;
        margin-bottom: 0;
    }

    /* Feature Cards */

    .feature-card {
        background: white;
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 20px;
        min-height: 130px;
        margin-bottom: 12px;
    }

    .feature-icon {
        font-size: 26px;
        margin-bottom: 8px;
    }

    .feature-title {
        color: #0F3D24;
        font-weight: 600;
        font-size: 15px;
    }

    .feature-description {
        color: #64748B;
        font-size: 13px;
        margin-top: 6px;
        line-height: 1.6;
    }

    /* Chat */

    [data-testid="stChatMessage"] {
        background-color: white;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 14px 16px;
        margin-bottom: 12px;
    }

    [data-testid="stChatMessage"] p {
        line-height: 1.7;
    }

    [data-testid="stChatInput"] {
        border-radius: 14px;
    }

    /* Labels */

    .advisory-tag {
        display: inline-block;
        background: #F0FDF4;
        color: #166534;
        border: 1px solid #DCFCE7;
        border-radius: 20px;
        padding: 5px 12px;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 10px;
    }

    .prediction-tag {
        display: inline-block;
        background: #EFF6FF;
        color: #1D4ED8;
        border: 1px solid #BFDBFE;
        border-radius: 20px;
        padding: 5px 12px;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 10px;
    }

    .message-time {
        color: #64748B;
        font-size: 11px;
        margin-top: 12px;
    }

    /* Buttons */

    div.stButton > button {
        border-radius: 10px;
        background-color: white;
        color: #0F172A;
        border: 1px solid #E2E8F0;
        font-weight: 500;
        text-align: left;
    }

    div.stButton > button:hover {
        border-color: #16A34A;
        background-color: #F0FDF4;
        color: #166534;
    }

    h1, h2, h3 {
        color: #14532D;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "rag_messages" not in st.session_state:
    st.session_state.rag_messages = []

if "rag_results" not in st.session_state:
    st.session_state.rag_results = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

if "pending_prediction" not in st.session_state:
    st.session_state.pending_prediction = None


# ============================================================
# LOAD RAG RETRIEVER
# ============================================================

@st.cache_resource(
    show_spinner="Loading agricultural knowledge base..."
)
def load_retriever():

    return DemoRetriever(
        mode="huggingface",
        k=4,
    )


retriever = None
kb_error = None

try:
    retriever = load_retriever()

except Exception as exc:
    kb_error = str(exc)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def current_time():

    return datetime.now().strftime("%H:%M")


def is_cancel_request(text):
    """
    Detect an explicit request to cancel an unfinished prediction.
    """

    return text.strip().lower() in {
        "cancel",
        "cancel prediction",
        "stop prediction",
        "exit prediction",
        "reset prediction",
    }


def is_prediction_followup(text):
    """
    Detect whether a message contains named prediction inputs.

    A normal agricultural question must not be treated
    as a prediction follow-up.
    """

    return bool(extract_inputs(text))


def add_assistant_message(
    content,
    kind="rag",
    result_index=None,
):

    message = {
        "role": "assistant",
        "content": content,
        "kind": kind,
        "result_index": result_index,
        "time": current_time(),
    }

    st.session_state.rag_messages.append(message)

    return message


def render_evidence(result):

    sources = result.get("retrieved", [])

    if not sources:
        return

    with st.expander(
        f"📚 Supporting evidence ({len(sources)} records)"
    ):

        for index, source in enumerate(
            sources,
            start=1,
        ):

            with st.container(border=True):

                st.markdown(
                    f"**[{index}] "
                    f"{source.get('topic', 'Agricultural Knowledge')}**"
                )

                st.write(
                    source.get("text", "")
                )


def render_assistant_message(message, show_sources):

    kind = message.get("kind", "rag")

    if kind == "prediction":

        st.markdown(
            '<span class="prediction-tag">'
            '🧮 ML Irrigation Prediction'
            '</span>',
            unsafe_allow_html=True,
        )

    elif kind == "system":

        st.markdown(
            '<span class="advisory-tag">'
            '🌱 SmartIrrigate AI'
            '</span>',
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            '<span class="advisory-tag">'
            '🌱 Agricultural Advisory'
            '</span>',
            unsafe_allow_html=True,
        )

    st.markdown(
        message["content"]
    )

    if message.get("time"):

        st.markdown(
            f'<div class="message-time">'
            f'{message["time"]}'
            f'</div>',
            unsafe_allow_html=True,
        )

    if show_sources and kind == "rag":

        result_index = message.get("result_index")

        if (
            result_index is not None
            and result_index < len(st.session_state.rag_results)
        ):

            result = st.session_state.rag_results[
                result_index
            ]

            render_evidence(result)


def build_transcript():

    lines = [
        "SmartIrrigate AI",
        "Conversation Transcript",
        "=" * 45,
        "",
    ]

    for message in st.session_state.rag_messages:

        role = (
            "You"
            if message["role"] == "user"
            else "SmartIrrigate AI"
        )

        lines.append(
            f"{role}:\n{message['content']}\n"
        )

    lines.append(
        "\nEducational prototype. "
        "Predictions are not field-validated."
    )

    return "\n".join(lines)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🌱 SmartIrrigate AI")

    st.caption(
        "Multilingual Agricultural Advisory Assistant"
    )

    st.divider()

    st.markdown("### System Information")

    st.markdown(
        f"**Language Model:** `{DEFAULT_LLM}`"
    )

    st.markdown(
        "**RAG Framework:** LangChain"
    )

    st.markdown(
        "**Embeddings:** Hugging Face MiniLM"
    )

    st.markdown(
        "**Vector Database:** FAISS"
    )

    st.markdown(
        "**ML Prediction:** Scikit-learn"
    )

    st.divider()

    st.markdown("### Settings")

    show_sources = st.toggle(
        "Show supporting evidence",
        value=False,
    )

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True,
    ):

        st.session_state.rag_messages = []
        st.session_state.rag_results = []
        st.session_state.pending_question = None
        st.session_state.pending_prediction = None

        st.rerun()

    if st.session_state.pending_prediction is not None:

        st.info(
            "An irrigation prediction is waiting "
            "for additional input values."
        )

        if st.button(
            "Cancel Pending Prediction",
            use_container_width=True,
        ):

            st.session_state.pending_prediction = None

            st.rerun()

    st.divider()

    if st.session_state.rag_messages:

        transcript = build_transcript()

        st.download_button(
            "📥 Download Conversation",
            data=transcript,
            file_name="SmartIrrigate_Conversation.txt",
            mime="text/plain",
            use_container_width=True,
        )

    st.divider()

    st.caption(
        "Educational AI prototype. "
        "Agricultural recommendations should be "
        "verified against local field conditions."
    )


# ============================================================
# HERO SECTION
# ============================================================

st.markdown(
    """
    <div class="hero">

        <h1>🌱 SmartIrrigate AI</h1>

        <p>
            Your intelligent multilingual agricultural assistant.
            Ask questions about irrigation, soil moisture,
            rainfall, crop water requirements, or generate
            ML-based irrigation predictions.
        </p>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# KNOWLEDGE BASE STATUS
# ============================================================

if kb_error:

    st.warning(
        "The agricultural knowledge base could not be loaded. "
        "RAG responses may be unavailable."
    )

    with st.expander("Knowledge base error"):
        st.code(kb_error)


# ============================================================
# WELCOME SCREEN
# ============================================================

if not st.session_state.rag_messages:

    st.subheader("What I Can Help With")

    col1, col2, col3 = st.columns(3)

    features = [
        (
            col1,
            "🌱",
            "Agricultural Advisory",
            "Ask questions about irrigation timing, "
            "water conservation, and crop management.",
        ),
        (
            col2,
            "🧮",
            "ML Irrigation Prediction",
            "Predict irrigation requirements using "
            "soil moisture, temperature, crop age, "
            "and rainfall probability.",
        ),
        (
            col3,
            "🌍",
            "Multilingual Chatbot",
            "Ask agricultural questions in English, "
            "Hindi, or Gujarati.",
        ),
    ]

    for column, icon, title, description in features:

        with column:

            st.markdown(
                f"""
                <div class="feature-card">

                    <div class="feature-icon">
                        {icon}
                    </div>

                    <div class="feature-title">
                        {title}
                    </div>

                    <div class="feature-description">
                        {description}
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

    st.subheader("Try Asking")

    suggestions = [
        "What are the benefits of drip irrigation?",
        "When should I irrigate sugarcane?",
        "ગન્નાના પાકમાં સિંચાઈ ક્યારે કરવી જોઈએ?",
        (
            "Predict irrigation: soil moisture 25%, "
            "temperature 34 C, crop age 5 months, "
            "rain probability 20%"
        ),
    ]

    left, right = st.columns(2)

    for index, suggestion in enumerate(suggestions):

        column = left if index % 2 == 0 else right

        with column:

            if st.button(
                suggestion,
                key=f"suggestion_{index}",
                use_container_width=True,
            ):

                st.session_state.pending_question = suggestion

                st.rerun()


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.rag_messages:

    avatar = (
        "🧑‍🌾"
        if message["role"] == "user"
        else "🌱"
    )

    with st.chat_message(
        message["role"],
        avatar=avatar,
    ):

        if message["role"] == "assistant":

            render_assistant_message(
                message,
                show_sources,
            )

        else:

            st.markdown(
                message["content"]
            )


# ============================================================
# CHAT INPUT
# ============================================================

typed_question = st.chat_input(
    "Ask an agricultural question or request an irrigation prediction..."
)

suggested_question = st.session_state.pending_question

if suggested_question:

    st.session_state.pending_question = None

question = suggested_question or typed_question


# ============================================================
# CHAT RESPONSE ROUTING
# ============================================================

if question and question.strip():

    question = question.strip()

    previous_history = [
        {
            "role": message["role"],
            "content": message["content"],
        }
        for message in st.session_state.rag_messages
    ]

    st.session_state.rag_messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message(
        "user",
        avatar="🧑‍🌾",
    ):

        st.markdown(question)

    with st.chat_message(
        "assistant",
        avatar="🌱",
    ):

        try:

            # ==========================================
            # 1. CANCEL AN UNFINISHED PREDICTION
            # ==========================================

            if is_cancel_request(question):

                st.session_state.pending_prediction = None

                answer = (
                    "The pending irrigation prediction "
                    "has been cancelled. You can ask "
                    "another agricultural question or "
                    "start a new prediction."
                )

                message = add_assistant_message(
                    answer,
                    kind="system",
                )

                render_assistant_message(
                    message,
                    show_sources,
                )

            else:

                # ======================================
                # 2. DETECT REQUEST TYPE
                # ======================================

                prediction_requested = (
                    is_prediction_request(question)
                )

                prediction_pending = (
                    st.session_state.pending_prediction
                    is not None
                )

                prediction_followup = False

                if prediction_pending:

                    prediction_followup = (
                        is_prediction_followup(question)
                    )

                # IMPORTANT:
                #
                # An unfinished prediction alone does
                # not force the next message into ML.
                #
                # Normal agricultural questions must
                # continue using RAG.

                prediction_mode = (
                    prediction_requested
                    or (
                        prediction_pending
                        and prediction_followup
                    )
                )

                # ======================================
                # 3. ML PREDICTION
                # ======================================

                if prediction_mode:

                    with st.spinner(
                        "Running ML irrigation prediction..."
                    ):

                        # Explicit new prediction requests
                        # start with fresh input values.

                        if prediction_requested:

                            pending_values = None

                        else:

                            pending_values = (
                                st.session_state.pending_prediction
                            )

                        answer, updated_pending = (
                            answer_prediction(
                                text=question,
                                pending=pending_values,
                            )
                        )

                        st.session_state.pending_prediction = (
                            updated_pending
                        )

                        message = add_assistant_message(
                            answer,
                            kind="prediction",
                        )

                        render_assistant_message(
                            message,
                            show_sources,
                        )

                # ======================================
                # 4. NORMAL RAG QUESTION
                # ======================================

                else:

                    with st.spinner(
                        "Searching agricultural knowledge..."
                    ):

                        if retriever is None:

                            answer = (
                                "The agricultural knowledge "
                                "base is currently unavailable. "
                                "Please check the retrieval "
                                "configuration."
                            )

                            message = add_assistant_message(
                                answer,
                                kind="system",
                            )

                            render_assistant_message(
                                message,
                                show_sources,
                            )

                        else:

                            result = ask(
                                question=question,
                                language="Auto",
                                retriever=retriever,
                                use_llm=True,
                                history=previous_history,
                            )

                            answer = result.get(
                                "answer",
                                (
                                    "Sorry, I couldn't "
                                    "generate a response."
                                ),
                            )

                            result_index = len(
                                st.session_state.rag_results
                            )

                            st.session_state.rag_results.append(
                                result
                            )

                            message = add_assistant_message(
                                answer,
                                kind="rag",
                                result_index=result_index,
                            )

                            render_assistant_message(
                                message,
                                show_sources,
                            )

                            if result.get("error"):

                                st.warning(
                                    "The language model "
                                    "encountered an error."
                                )

                                with st.expander(
                                    "View error details"
                                ):

                                    st.code(
                                        str(result["error"])
                                    )

        except Exception as exc:

            st.error(
                f"Unable to process your request: {exc}"
            )

            print(
                "SMARTIRRIGATE ERROR:",
                repr(exc),
                flush=True,
            )