# 🌱 SmartIrrigate AI
### Multilingual RAG-Based Agricultural Advisory Chatbot

SmartIrrigate AI is an intelligent agricultural advisory chatbot developed as an **M.Tech Artificial Intelligence and Data Science MLOps mini-project**. The system uses **Retrieval-Augmented Generation (RAG)** to provide contextual agricultural information about irrigation management, soil moisture, crop water requirements, drip irrigation, and water conservation.

The chatbot integrates **Hugging Face, LangChain, FAISS, Ollama, and Streamlit** to deliver an interactive conversational experience in multiple languages.

## 📌 Project Overview

SmartIrrigate AI aims to make agricultural information accessible through a conversational interface. Users can ask questions in English, Hindi, or Gujarati and receive AI-generated explanations supported by retrieved agricultural knowledge.

The application uses a locally running Qwen 2.5 language model through Ollama, eliminating the need for a paid LLM inference API during local operation.

The project demonstrates the integration of semantic retrieval, vector databases, local language models, and conversational AI.

## ✨ Key Features

- **Multilingual Chatbot:** Supports agricultural questions in English, Hindi, and Gujarati.
- **Retrieval-Augmented Generation:** Retrieves relevant information before generating responses.
- **Agricultural Knowledge Base:** Includes irrigation-related guidance and collected reference material from agricultural institutions.
- **Semantic Search:** Uses multilingual Hugging Face sentence embeddings.
- **FAISS Vector Database:** Stores and retrieves agricultural knowledge efficiently.
- **Local LLM Integration:** Uses Ollama with Qwen 2.5 1.5B.
- **LangChain Integration:** Orchestrates document retrieval, prompting, and response generation.
- **Conversation History:** Supports contextual follow-up questions.
- **Interactive Streamlit Interface:** Includes suggested questions, chatbot conversations, and optional source information.
- **Offline LLM Generation:** After downloading the required models, generation can run locally without a hosted inference API.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Backend development |
| Streamlit | Chatbot interface |
| LangChain | RAG orchestration |
| Hugging Face | Multilingual text embeddings |
| Sentence Transformers | Embedding model execution |
| FAISS | Vector storage and semantic retrieval |
| Ollama | Local LLM runtime |
| Qwen 2.5 1.5B | Response generation |
| JSON | Knowledge-base records and metadata |

### Models

**Embedding model:**

`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

**Language model:**

`qwen2.5:1.5b`

## 🏗️ System Architecture

```text
                USER QUESTION
                      |
                      v
              STREAMLIT CHATBOT
                      |
                      v
                LANGCHAIN RAG
                      |
                      v
          HUGGING FACE EMBEDDINGS
                      |
                      v
                FAISS DATABASE
                      |
                      v
           RETRIEVED AGRICULTURAL
                  KNOWLEDGE
                      |
                      v
               LANGCHAIN PROMPT
                      |
                      v
             OLLAMA QWEN 2.5
                      |
                      v
             AI-GENERATED ANSWER
                      |
                      v
              STREAMLIT CHATBOT
```

## 📂 Project Structure

The essential chatbot components are organized as follows:

```text
irrigation-advisor/
|
├── app/
│   ├── streamlit_app.py
│   └── rag_demo.py
│
├── knowledge/
│   ├── synthetic_guidance.json
│   ├── faiss_index/
│   ├── agricultural_faiss_cleaned/
│   └── sugarcane_guide.md
│
├── requirements.txt
└── README.md
```

The project may also contain agricultural source documents, processed knowledge files, and supporting scripts. The active knowledge base depends on the configuration in `app/rag_demo.py`.

## ⚙️ Installation and Setup

### 1. Prerequisites

- Python 3.10 or another compatible version
- Git
- Ollama
- Internet connection for initial installation and model downloads

### 2. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/SmartIrrigate-AI.git
cd SmartIrrigate-AI
```

Replace `YOUR_USERNAME` with your GitHub username.

### 3. Create a Virtual Environment

On Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

The requirements file must contain the packages used by the application's current implementation.

### 5. Install Ollama

Download Ollama from:

https://ollama.com/

After installation, download the language model:

```powershell
ollama pull qwen2.5:1.5b
```

Verify that the model is installed:

```powershell
ollama list
```

Ensure the Ollama service is running before starting the chatbot.

### 6. Run SmartIrrigate AI

```powershell
python -m streamlit run app\streamlit_app.py
```

Open the local application at:

http://localhost:8501

## 💬 Example Questions

**English**

- When should sugarcane be irrigated?
- What are the advantages of drip irrigation?
- How does soil moisture affect irrigation?
- What are the effects of waterlogging?

**Hindi**

- गन्ने की फसल में सिंचाई कब करनी चाहिए?
- ड्रिप सिंचाई के क्या फायदे हैं?

**Gujarati**

- શેરડીના પાકને પાણી ક્યારે આપવું જોઈએ?
- ટપક સિંચાઈના ફાયદા શું છે?

The chatbot attempts to generate answers in the requested language. Actual response quality depends on the language model and retrieved knowledge.

## 📚 Agricultural Knowledge Base

The project contains agricultural guidance relevant to sugarcane irrigation and general water management.

During development, information was collected from institutional agricultural resources, including:

- Food and Agriculture Organization (FAO)
- Tamil Nadu Agricultural University (TNAU)

A document-processing pipeline was used to extract, clean, and split agricultural reference material into searchable chunks.

**Knowledge-base development summary:**

- 13 reference sources initially considered
- 10 sources successfully collected
- 418 unique text chunks initially extracted
- 416 chunks retained after initial cleaning
- A separate agricultural FAISS index generated using the cleaned chunks

These figures describe the knowledge-base preparation process. They do not imply that every record is currently used by the running chatbot.

Some original synthetic guidance records are also retained for demonstration and fallback purposes.

## 🧠 How the RAG Chatbot Works

1. The user enters an agricultural question through Streamlit.
2. LangChain processes the question and coordinates retrieval.
3. Hugging Face embeddings represent the question as a numerical vector.
4. FAISS searches for semantically relevant agricultural information.
5. Retrieved passages are included in the generation prompt.
6. Ollama runs the Qwen 2.5 model to generate a contextual response.
7. The response is displayed in the chatbot interface.
8. Conversation history allows the application to handle follow-up questions.

RAG helps ground responses in retrieved information, but it does not guarantee that every generated statement is correct.

## 🚀 Deployment

The application can be run locally using Streamlit and Ollama.

For online deployment, platforms such as Streamlit Community Cloud can host the Streamlit frontend.

However, **the current Ollama-based implementation requires access to a running Ollama server**. A deployed application cannot automatically connect to Ollama running on the developer's laptop.

Cloud deployment therefore requires an accessible remote LLM service or another supported generation backend.

The application has not been assumed to be production-deployed merely because it runs locally.

## 🎓 Academic Relevance

This project demonstrates the integration of several technologies relevant to modern AI application development:

- Retrieval-Augmented Generation
- Large Language Models
- Semantic search and vector embeddings
- Vector databases
- LangChain-based AI orchestration
- Multilingual conversational AI
- Local model execution
- Interactive AI application development

The project is intended as a practical implementation and demonstration of Hugging Face and LangChain integration.

## ⚠️ Limitations

- Some agricultural guidance records are synthetic or educational.
- Collected reference material may not reflect current local agricultural conditions.
- The language model may produce incomplete or inaccurate answers.
- The application does not directly measure soil moisture or retrieve live weather conditions.
- Multilingual response quality may vary.
- The chatbot is not a field-validated irrigation decision system.
- Local Ollama execution requires adequate CPU and memory resources.
- Agricultural recommendations should be verified against local field conditions and qualified agricultural guidance.

## 🔮 Future Enhancements

Potential improvements include:

- Integration of ML-based irrigation prediction within the chatbot
- Real-time weather information
- Soil moisture sensor integration
- Additional crop-specific knowledge
- Improved multilingual retrieval
- Context-aware agricultural recommendations
- User feedback collection
- Cloud-hosted LLM deployment

These are proposed improvements and are not claimed as completed features.

## 📄 Conclusion

SmartIrrigate AI demonstrates how Retrieval-Augmented Generation can support an interactive, multilingual agricultural advisory application. By integrating Hugging Face embeddings, FAISS retrieval, LangChain orchestration, and local Ollama inference, the project provides a practical example of an AI-powered chatbot designed to make agricultural information easier to access.

**Project Type:** M.Tech AI & DS — MLOps Mini-Project

**Primary Focus:** Hugging Face + LangChain + RAG Integration#   S m a r t I r r i g a t e A I  
 