
# 🤖 Coding Ninja Multi-Agent Chatbot

## 🧩 Problem Statement

Students and learners often struggle to find accurate and prompt answers related to:

* Customer support queries (refunds, enrollment, etc.)
* Content or documentation on the Coding Ninjas website
* Coding help and debugging
* Course recommendations based on their interests
* Interview preparation materials
* Project ideas aligned with their skillsets

**Manual support is time-consuming, often generic, and lacks personalization.**

### 💡 Why Multi-Agent AI?

AI Agents are ideal here because:

* They can specialize in distinct domains, improving accuracy and relevance.
* Multi-agent collaboration ensures the system can delegate user queries to the most suitable agent.
* This results in faster, modular, and context-aware responses, improving user satisfaction drastically.

Instead of building a monolithic chatbot, using **specialized agents** for each query type enables the system to scale, adapt, and deliver **expert-level interaction per domain**.

---

## 🛠️ Project Description

The **Coding Ninja Multi-Agent Chatbot** is an intelligent assistant that intelligently routes and answers queries using a network of task-specific AI agents:

### 🧠 Supported Agents

1. **Customer Support Agent** – Handles account, payment, and support-related queries.
2. **Content Fetcher Agent** – Scrapes and summarizes relevant content from the Coding Ninjas website.
3. **Code Helper Agent** – Helps debug or write code with explanations and examples.
4. **Course Advisor Agent** – Recommends Coding Ninjas courses based on goals.
5. **Interview Prep Agent** – Offers mock questions, answers, and prep tips.
6. **Project Ideas Agent** – Suggests project ideas based on user interests and skills.

### 🔄 How Agents Collaborate

* A **router node** powered by keyword-based logic routes incoming user queries to the most appropriate agent.
* Each agent works independently but conforms to a unified response structure.
* The **state machine graph** (via LangGraph) manages agent transitions and execution paths.

---

## 🧰 Tools, Libraries, and Frameworks Used

| Type                          | Tool                                                      |
| ----------------------------- | --------------------------------------------------------- |
| **Backend**                   | [FastAPI](https://fastapi.tiangolo.com/)                  |
| **Frontend**                  | [Streamlit](https://streamlit.io/)                        |
| **Multi-Agent Orchestration** | [LangGraph](https://python.langchain.com/docs/langgraph/) |
| **LLM Integration**           | [LangChain](https://www.langchain.com/)                   |
| **Web Scraping**              | `WebBaseLoader`, `BeautifulSoup`                          |
| **Search Tool**               | DuckDuckGoSearchRun (LangChain Community Tool)            |
| **Env Management**            | `python-dotenv`                                           |
| **CORS Middleware**           | For cross-origin API calls                                |
| **Async Support**             | `asyncio` for agent concurrency                           |

---

## 🧠 LLM Selection

### ✅ Ideal LLM for Production:

* ** Groq api for faster inference **
* *** model name :-  "deepseek-r1-distill-llama-70b" ***
  

### 🆓 Free-Tier / Open-Source LLMs Considered:

* **[Gemma (via Hugging Face)](https://huggingface.co/google/gemma-7b)**
* **[Mistral-7B-Instruct (Hugging Face)](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)**
* **[GPT-3.5-Turbo](https://platform.openai.com/docs/guides/gpt)** (free credits on OpenAI)
* **[Google Gemini 1.5 Flash](https://cloud.google.com/vertex-ai/docs/generative-ai/overview)** (via Google Cloud)

### 🎯 Justification for LLM Choices:

* **Performance-Driven**: Groq-hosted DeepSeek LLaMA 70B offers high speed and cost-efficiency for real-time apps.
* **Domain Versatility**: DeepSeek  provide good performance for summarization and Q\&A with prompt-tuning.
* **Cost Consideration**: Used Groq + DeepSeek for zero-cost high-speed inference during development and demo stages.

---

## 📦 How to Run

### Backend (FastAPI)

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

### Frontend (Streamlit)

```bash
streamlit run app.py
```

Ensure `.env` contains your Groq API key or substitute with any supported LLM.

---

## 📌 Future Improvements

* Integrate memory & long-term context tracking


## Links

[Try Here](https://example.com)

## Agent Architecture

![Chatbot Architecture](I:\codingninja\architecture.png)


