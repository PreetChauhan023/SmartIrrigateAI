"""
SmartIrrigate RAG Assistant: Hugging Face embeddings + FAISS + local Ollama.

Features:
- 300 synthetic agricultural guidance records
- Hugging Face sentence embeddings
- FAISS vector similarity search
- LangChain prompt orchestration
- Local Ollama Qwen 2.5 answer generation
- Conversational history
- Source citations
- Retrieval fallback
- Persistent FAISS index
"""

import json
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "knowledge" / "synthetic_guidance.json"
INDEX_PATH = ROOT / "knowledge" / "faiss_index"

EMBEDDING_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

DEFAULT_LLM = "qwen2.5:1.5b"


# --------------------------------------------------
# LOAD SYNTHETIC KNOWLEDGE
# --------------------------------------------------

def load_documents():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Synthetic knowledge base missing. "
            "Run: python data/generate_guidance.py"
        )

    records = json.loads(
        DATA_PATH.read_text(encoding="utf-8")
    )

    documents = []

    for record in records:

        content = (
            f"Category: {record['category']}\n"
            f"Topic: {record['topic']}\n"
            f"Context: {record['context']}\n"
            f"Question: {record['question']}\n"
            f"Guidance: {record['answer']}"
        )

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "id": record["id"],
                    "category": record["category"],
                    "topic": record["topic"],
                    "source": record["source"],
                    "synthetic": True,
                },
            )
        )

    return documents


# --------------------------------------------------
# HUGGING FACE EMBEDDINGS
# --------------------------------------------------

def get_embeddings():

    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 32,
        },
    )


# --------------------------------------------------
# FAISS VECTOR DATABASE
# --------------------------------------------------

class DemoRetriever:

    def __init__(self, mode="huggingface", k=4):

        self.mode = mode
        self.k = k

        self.documents = load_documents()

        if mode == "huggingface":

            from langchain_community.vectorstores import FAISS

            embeddings = get_embeddings()

            if INDEX_PATH.exists():

                self.vectorstore = FAISS.load_local(
                    str(INDEX_PATH),
                    embeddings,
                    allow_dangerous_deserialization=True,
                )

            else:

                self.vectorstore = FAISS.from_documents(
                    self.documents,
                    embeddings,
                )

                INDEX_PATH.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                self.vectorstore.save_local(
                    str(INDEX_PATH)
                )

            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k},
            )

        else:

            from sklearn.feature_extraction.text import TfidfVectorizer

            self.vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
            )

            self.matrix = self.vectorizer.fit_transform(
                [d.page_content for d in self.documents]
            )

    def invoke(self, question):

        if self.mode == "huggingface":

            return self.retriever.invoke(question)

        from sklearn.metrics.pairwise import cosine_similarity

        query_vector = self.vectorizer.transform(
            [question]
        )

        similarities = cosine_similarity(
            query_vector,
            self.matrix,
        )[0]

        indices = similarities.argsort()[::-1][:self.k]

        return [
            self.documents[i]
            for i in indices
            if similarities[i] > 0.025
        ]


# --------------------------------------------------
# LOCAL OLLAMA LANGUAGE MODEL
# --------------------------------------------------

def get_llm():
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=DEFAULT_LLM,
        temperature=0.2,
        num_predict=450,
    )


# --------------------------------------------------
# CONVERSATIONAL RAG PROMPT
# --------------------------------------------------

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are SmartIrrigate AI, an educational agricultural
assistant specializing in sugarcane irrigation and MLOps.

IMPORTANT RULES:

1. Answer using only the provided synthetic knowledge.
2. Explain concepts clearly and conversationally.
3. Do not copy entire retrieved documents.
4. Combine relevant information into a useful answer.
5. Do not invent agricultural measurements or prescriptions.
6. If the context is insufficient, acknowledge that limitation.
7. Cite supporting source IDs such as [DEMO-001].
8. Do not cite sources that were not provided.
9. Respond in the same language as the latest user question.
   Detect the language from the question itself (including its script).
   If the question mixes languages, follow its dominant language.
   If the language is unclear, use English.
   Keep technical names such as FAISS and MLOps unchanged.
10. Remember the conversation history when answering follow-ups.
11. Explain that recommendations are educational, not field-validated,
    when discussing real irrigation decisions.

Language preference:
{language} (Auto means infer from the latest user question; never infer from retrieved English documents.)

Conversation history:
{history}

Retrieved knowledge:
{context}
""",
        ),
        (
            "human",
            "{question}",
        ),
    ]
)


# --------------------------------------------------
# FORMAT CONVERSATION HISTORY
# --------------------------------------------------

def format_history(history):

    if not history:
        return "No previous conversation."

    lines = []

    for message in history[-8:]:

        role = message.get("role", "user")
        content = message.get("content", "")

        lines.append(
            f"{role}: {content}"
        )

    return "\n".join(lines)


# --------------------------------------------------
# FOLLOW-UP QUESTION REWRITING
# --------------------------------------------------

REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
Rewrite the latest question as a standalone search query
using the conversation history.

Do not answer the question.
Do not add new facts.
Return only the rewritten query.

History:
{history}
""",
        ),
        (
            "human",
            "{question}",
        ),
    ]
)


def rewrite_question(question, history, llm):

    if not history or llm is None:
        return question

    try:

        chain = REWRITE_PROMPT | llm | StrOutputParser()

        rewritten = chain.invoke({
            "question": question,
            "history": format_history(history),
        })

        return rewritten.strip() or question

    except Exception:
        return question


# --------------------------------------------------
# FORMAT RETRIEVED CONTEXT
# --------------------------------------------------

def format_context(documents):

    sections = []

    for doc in documents:

        doc_id = doc.metadata["id"]

        sections.append(
            f"[{doc_id}]\n{doc.page_content}"
        )

    return "\n\n".join(sections)


# --------------------------------------------------
# RETRIEVAL FALLBACK
# --------------------------------------------------

def fallback_answer(documents):

    if not documents:

        return (
            "I couldn't find relevant information in the "
            "synthetic knowledge base. Please ask another question."
        )

    answer = (
        "I couldn't generate an LLM response, but here is "
        "the most relevant information from the synthetic "
        "knowledge base:\n\n"
    )

    for doc in documents[:2]:

        content = doc.page_content

        guidance = content.split("Guidance:")[-1].strip()

        answer += (
            f"**[{doc.metadata['id']}] "
            f"{doc.metadata['topic']}**\n\n"
            f"{guidance}\n\n"
        )

    return answer


# --------------------------------------------------
# MAIN RAG PIPELINE
# --------------------------------------------------

def ask(
    question,
    language="Auto",
    retriever=None,
    use_llm=True,
    history=None,
):

    history = history or []

    if retriever is None:
        retriever = DemoRetriever()

    llm = None
    llm_error = None

    if use_llm:

        try:
            llm = get_llm()
        except Exception as exc:
            llm_error = str(exc)

    search_query = rewrite_question(
        question,
        history,
        llm,
    )

    documents = retriever.invoke(search_query)

    sources = [
        doc.metadata["id"]
        for doc in documents
    ]

    retrieved = [
        {
            "id": doc.metadata["id"],
            "topic": doc.metadata["topic"],
            "category": doc.metadata["category"],
            "text": doc.page_content,
        }
        for doc in documents
    ]

    if not documents:

        return {
            "answer": (
                "I couldn't find sufficient information "
                "in the synthetic knowledge base."
            ),
            "sources": [],
            "retrieved": [],
            "mode": "no-match",
            "search_query": search_query,
            "error": None,
        }

    context = format_context(documents)

    answer = fallback_answer(documents)

    mode = "retrieval-fallback"

    if llm is not None:

        try:

            chain = (
                RAG_PROMPT
                | llm
                | StrOutputParser()
            )

            generated = chain.invoke({
                "question": question,
                "context": context,
                "history": format_history(history),
                "language": language,
            })

            generated = generated.strip()

            valid_sources = set(sources)

            cited_sources = set(
                re.findall(
                    r"DEMO-\d+",
                    generated,
                )
            )

            if generated and cited_sources.issubset(valid_sources):
                # If Qwen omitted citations, attach actual retrieved source IDs.
                if not cited_sources:
                    generated += "\n\nSources: " + ", ".join(
                        f"[{source}]" for source in sources[:2]
                    )
                answer = generated
                mode = "langchain-ollama-rag"
            else:
                mode = "retrieval-fallback-invalid-citations"


        except Exception as exc:

            import traceback

            llm_error = traceback.format_exc()

            print("\n========== OLLAMA LLM ERROR ==========")

            print(llm_error)

            print("============================================\n")

            mode = "retrieval-fallback-llm-error"

    elif not use_llm:

        mode = "retrieval-only"

    elif llm is None:

        mode = "retrieval-fallback-ollama-unavailable"

    return {
        "answer": answer,
        "sources": sources,
        "retrieved": retrieved,
        "mode": mode,
        "search_query": search_query,
        "error": llm_error,
    }
