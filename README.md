# 🌱 SmartIrrigate AI

**Multilingual RAG-Based Agricultural Advisory Chatbot**  
*M.Tech Artificial Intelligence & Data Science — MLOps Mini-Project*

SmartIrrigate AI is a conversational agricultural advisory application that uses **Retrieval-Augmented Generation (RAG)** to answer questions about sugarcane irrigation, soil moisture, crop water requirements, drip irrigation, and water conservation.

Built with **Streamlit, LangChain, Hugging Face sentence embeddings, FAISS, and Ollama (Qwen 2.5)**, it supports questions in **English, Hindi, and Gujarati**. It is designed as an educational demonstration, not a field-validated irrigation decision system.

## ✨ Features

- **Multilingual chat:** Ask questions in English, Hindi, or Gujarati.
- **RAG-based answers:** Retrieve relevant passages before generating an answer.
- **Semantic retrieval:** Use multilingual Hugging Face embeddings and FAISS.
- **Local text generation:** Run Qwen 2.5 1.5B through Ollama without a hosted LLM API.
- **Conversation history:** Continue conversations with follow-up questions.
- **Streamlit interface:** Suggested questions, chat history, and optional source display.
- **Agricultural references:** Collected FAO and Tamil Nadu Agricultural University (TNAU) material, alongside existing synthetic demonstration data.

## 🛠️ Technology Stack

| Component | Technology | Purpose |
| --- | --- | --- |
| Frontend | Streamlit | Interactive chatbot interface |
| RAG orchestration | LangChain | Retrieval and generation pipeline |
| Embeddings | Hugging Face / Sentence Transformers | Multilingual text vectors |
| Vector database | FAISS | Similarity search |
| Local LLM runtime | Ollama | Local model inference |
| Language model | Qwen 2.5 1.5B | Response generation |
| Language | Python | Application code |

**Embedding model:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`  
**Generation model:** `qwen2.5:1.5b`

## 🏗️ Architecture

```text
User question (English / Hindi / Gujarati)
                    |
                    v
             Streamlit Chat UI
                    |
                    v
            LangChain RAG Logic
                    |
                    v
       Hugging Face Text Embeddings
                    |
                    v
              FAISS Retrieval
                    |
                    v
        Retrieved Knowledge Passages
                    |
                    v
       LangChain Prompt + Context
                    |
                    v
            Ollama / Qwen 2.5
                    |
                    v
           Chatbot Response
```

## 📁 Project Structure

```text
irrigation-advisor/
├── app/
│   ├── streamlit_app.py       # Streamlit interface
│   └── rag_demo.py            # RAG retrieval and generation
├── knowledge/
│   ├── faiss_index/           # Original vector index
│   ├── agricultural_faiss_cleaned/  # Prepared agricultural index
│   ├── synthetic_guidance.json
│   └── sugarcane_guide.md
├── requirements.txt
└── README.md
```

> **Note:** The repository may include additional training, testing, and monitoring files from earlier experiments. The active knowledge base is determined by the configuration in `app/rag_demo.py`. The presence of an index does not mean the running chatbot automatically uses it.

## 🚀 Run Locally

### Prerequisites

- Python 3.10 (or a compatible version)
- [Ollama](https://ollama.com/)
- Git and an internet connection for the initial setup

### 1. Clone the repository

```bash
git clone https://github.com/PreetChauhan023/SmartIrrigateAI.git
cd SmartIrrigateAI
```

### 2. Create and activate a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Python dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Download the Ollama model

```powershell
ollama pull qwen2.5:1.5b
ollama list
```

Ensure the Ollama service is running.

### 5. Start the Streamlit app

```powershell
python -m streamlit run app\streamlit_app.py
```

Visit **http://localhost:8501**.

## 💬 Example Questions

| Language | Example |
| --- | --- |
| English | When should sugarcane be irrigated? |
| English | What are the benefits of drip irrigation? |
| Hindi | गन्ने की फसल में सिंचाई कब करनी चाहिए? |
| Hindi | ड्रिप सिंचाई के क्या फायदे हैं? |
| Gujarati | શેરડીના પાકને પાણી ક્યારે આપવું જોઈએ? |
| Gujarati | ટપક સિંચાઈના ફાયદા શું છે? |

Response quality and language consistency depend on the retrieved content and the local language model.

## 📚 Knowledge Base

During development, reference material was collected from institutional sources, including the **Food and Agriculture Organization (FAO)** and **Tamil Nadu Agricultural University (TNAU)**.

| Processing stage | Result |
| --- | ---: |
| Sources considered | 13 |
| Sources successfully collected | 10 |
| Unique chunks extracted | 418 |
| Chunks retained after initial cleaning | 416 |

A separate FAISS index was generated from the cleaned agricultural passages. The project also retains synthetic educational guidance records. These collection statistics describe the development process; **they do not prove that every chunk is used by the current application**.

## 🔎 How the RAG Pipeline Works

1. A user submits a question in the Streamlit chat.
2. LangChain coordinates retrieval and response generation.
3. The Hugging Face embedding model converts the question into a vector.
4. FAISS retrieves similar knowledge passages from the configured index.
5. The passages and question are supplied to the generation prompt.
6. Ollama runs Qwen 2.5 and returns a natural-language answer.
7. The chatbot displays the response and maintains conversation history.

RAG provides supporting context, but it does not guarantee factual correctness or eliminate hallucinations.

## ☁️ Deployment Notes

The application runs locally with Ollama. **Streamlit Community Cloud cannot automatically access Ollama on a developer's laptop.** An online deployment needs a reachable hosted LLM service or securely configured remote inference server. Additional memory constraints may apply when loading the embedding model in cloud hosting.

## 🎓 Educational Scope and Limitations

This mini-project demonstrates practical integration of **Hugging Face, LangChain, vector retrieval, local LLM inference, and Streamlit**.

- Some guidance is synthetic or intended only for educational use.
- Institutional documents may not reflect current or local field conditions.
- The chatbot does not directly measure soil moisture or retrieve live weather data.
- Predictions and sensor-based irrigation automation are **not part of the current chatbot implementation**.
- Small local LLMs may produce incomplete, inaccurate, or inconsistent multilingual answers.
- The application is **not a field-validated agricultural advisory system**. Verify decisions with local conditions and qualified agricultural guidance.

## 🔮 Potential Enhancements

- ML-based irrigation prediction integrated into the conversation
- Weather and soil-moisture sensor data
- Additional crop-specific reference documents
- Improved multilingual retrieval
- Feedback collection and evaluation
- Cloud-accessible LLM deployment

---

**Project:** SmartIrrigate AI  
**Academic context:** M.Tech AI & DS — MLOps Mini-Project  
**Primary focus:** Hugging Face + LangChain + RAG
