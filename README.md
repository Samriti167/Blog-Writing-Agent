# 🤖 Autonomous Multi-Agent Blog Writer

An intelligent, stateful AI blog-generation pipeline built with **LangGraph**, **Groq (Llama-3.1)**, and **Streamlit**.

This application utilizes an advanced **Map-Reduce (Fan-out/Fan-in)** architecture to parallelize the generation of complex, multi-section documents.

It dynamically routes topics to perform web research, strictly enforces JSON schema structures using **Pydantic**, and outputs fully formatted **Markdown blogs**.

---

# ✨ Features

## 🧠 Autonomous Orchestration
A planner agent breaks down a topic into a highly structured outline containing:

- 5-9 sections
- Specific word counts
- Educational goals
- Actionable bullet points

## 🔎 Dynamic Routing & RAG
The system intelligently decides between:

- **Closed Book Mode** → Evergreen concepts without external research
- **Hybrid/Open Book Mode** → Uses Tavily web search for updated information

Research is triggered only when required.

## 📚 Optimized Context Handling
- Filters relevant research results
- Removes duplicate URLs
- Controls context size before sending data to LLMs

## ⚡ Concurrent Multi-Agent Fanout
Multiple worker agents generate different blog sections independently using LangGraph fan-out architecture.

Includes:

- Controlled concurrency
- API rate-limit handling
- Structured generation

## 🖥️ Interactive Streamlit UI

The dashboard provides:

- Real-time graph progress
- Research evidence display
- Blog plan visualization
- Markdown preview
- Downloadable output


---

# 🧠 System Architecture

```
                User Topic
                    |
                    ↓
              Router Node
                    |
        ┌───────────┴───────────┐
        ↓                       ↓
 Research Node            Direct Planning
        |
        ↓
 Orchestrator Node
        |
        ↓
   Fan-out Workers
        |
        ↓
   Section Generation
        |
        ↓
  Reducer Node (Merge)
        |
        ↓
 Final Markdown Blog
```

---

# 🔹 Components

## Router Node

Analyzes the user topic and decides:

- Whether research is required
- Which generation mode to use

---

## Research Node

Uses Tavily Search API to:

- Fetch relevant sources
- Extract evidence
- Remove duplicate URLs

---

## Orchestrator Node

Creates the blog blueprint using:

- LLM structured output
- Pydantic validation
- Strict JSON schema enforcement

---

## Worker Nodes (Fan-out)

Multiple agents write individual sections based on:

- Blog plan
- Section goals
- Research evidence
- Word requirements

---

## Reducer Node (Fan-in)

Combines all generated sections into:

- A complete Markdown blog
- A downloadable `.md` file


---

# 🛠️ Installation & Setup

## Prerequisites

- Python 3.9+
- Groq API Key
- Tavily API Key


---

## Clone Repository

```bash
git clone https://github.com/Samriti167/AI-Blog-Writing-Agent.git

cd AI-Blog-Writing-Agent
```

---

## Create Virtual Environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Mac/Linux

```bash
python -m venv myvenv

source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here

TAVILY_API_KEY=your_tavily_api_key_here
```

---

# 🚀 Run Application

Start Streamlit:

```bash
streamlit run frontend.py
```

The application will open at:

```
http://localhost:8501
```

---

# 💻 Core Tech Stack

- Python
- LangGraph
- LangChain
- Groq Llama-3.1
- Streamlit
- Tavily Search API
- Pydantic
- Markdown


---

# 🚀 Usage

1. Open the Streamlit dashboard.

2. Enter a blog topic.

Example:

```
The State of Multimodal LLMs in 2026
```

3. Click:

```
Generate Blog
```

4. The agent pipeline will execute:

```
Routing
   ↓
Research
   ↓
Planning
   ↓
Section Generation
   ↓
Final Blog
```

5. View:

- Generated plan
- Research evidence
- Markdown preview

6. Download the final `.md` blog file.


---

# ⭐ Highlights

✅ Multi-Agent Architecture  
✅ LangGraph State Management  
✅ Map-Reduce Blog Generation  
✅ Dynamic Research Routing  
✅ Structured LLM Outputs  
✅ RAG-based Evidence Handling  
✅ Streamlit Interactive Dashboard  


---

# Built With ❤️ Using

**LangGraph + Groq + Streamlit**
