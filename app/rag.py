"""RAG over sugarcane agronomy notes.

Retriever: FAISS + multilingual HF embeddings when USE_HF_EMBEDDINGS=1,
otherwise a light TF-IDF retriever so the demo works offline and starts fast.
Answer: HF LLM via LangChain if HF_TOKEN set, else returns the retrieved passages.
"""
import os
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
KB_DIR = ROOT / "knowledge"

RAG_PROMPT = ChatPromptTemplate.from_template(
    "Answer the sugarcane farmer's question using ONLY the context. "
    "Reply in {language}, in simple words, max 3 sentences. "
    "If the context does not contain the answer, say you are not sure and to ask an agronomist.\n\n"
    "Context:\n{context}\n\nQuestion: {question}"
)


def load_chunks() -> list[Document]:
    docs = []
    for f in sorted(KB_DIR.glob("*.md")):
        section = ""
        for block in f.read_text(encoding="utf-8").split("\n## "):
            block = block.strip()
            if len(block) > 40:
                docs.append(Document(page_content=block, metadata={"source": f.name, "section": section}))
    return docs


class TfidfRetriever(BaseRetriever):
    docs: list
    vec: object
    mat: object
    k: int = 3

    def _get_relevant_documents(self, query, *, run_manager=None):
        sims = cosine_similarity(self.vec.transform([query]), self.mat)[0]
        idx = sims.argsort()[::-1][: self.k]
        return [self.docs[i] for i in idx if sims[i] > 0]


def build_retriever(k: int = 3) -> BaseRetriever:
    docs = load_chunks()
    if os.getenv("USE_HF_EMBEDDINGS") == "1":
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings

        emb = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")
        return FAISS.from_documents(docs, emb).as_retriever(search_kwargs={"k": k})
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    mat = vec.fit_transform([d.page_content for d in docs])
    return TfidfRetriever(docs=docs, vec=vec, mat=mat, k=k)


def answer(question: str, language: str = "English", retriever: BaseRetriever = None, llm=None) -> dict:
    retriever = retriever or build_retriever()
    hits = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in hits)
    if llm is None and os.getenv("HF_TOKEN"):
        from app.chain import get_llm

        llm = get_llm()
    if llm is None or not hits:
        text = context[:600] if hits else "I am not sure. Please ask your agronomist."
        return {"answer": text, "sources": [d.metadata["source"] for d in hits], "mode": "retrieval-only"}
    out = (RAG_PROMPT | llm | StrOutputParser()).invoke(
        {"context": context, "question": question, "language": language}
    )
    return {"answer": out.strip(), "sources": [d.metadata["source"] for d in hits], "mode": "rag-llm"}
