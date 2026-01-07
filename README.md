# 🧠 Intelligent Medical Assistant Agent

<div align="center">

![Medical AI](https://img.shields.io/badge/AI-Medical%20Assistant-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10-green?style=for-the-badge&logo=python)
![LangGraph](https://img.shields.io/badge/LangGraph-Agentic%20AI-purple?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

*An advanced conversational AI medical assistant powered by stateful multi-tool agentic framework*

[Features](#-key-features) • [Installation](#-installation--setup) • [Architecture](#%EF%B8%8F-tech-stack--architecture) • [Usage](#-usage) • [Contributing](#-contributing)

</div>

---

## 📌 Project Overview

This project is an **advanced, conversational AI medical assistant** built on a stateful, multi-tool agentic framework. It goes beyond simple Q&A by dynamically selecting the best information source for any query, proactively offering supplementary advice, and anticipating user needs by finding nearby medical specialists.

The agent uses a **Retrieval-Augmented Generation (RAG)** architecture with a **Pinecone vector database** as its foundational knowledge, but enhances it with live data from the **PubMed API** and a **web scraper**. All decisions are orchestrated by a central routing brain built with **LangGraph**.

### 🌟 What Makes This Special?

- 🔹 **Stateful Multi-Tool Agent**: Uses LangGraph to manage conversation state and dynamically route tasks
- 🔹 **Dynamic Knowledge Sources**: Chooses between its internal Pinecone DB, live PubMed research, or web scraping for the best context
- 🔹 **Proactive & Conversational**: Not only answers questions but offers unsolicited advice and anticipates user needs
- 🔹 **Location-Aware**: Integrates a free OpenStreetMap tool to find and suggest local specialists with addresses and distances
- 🔹 **Modular & Scalable Architecture**: Built with a clean, multi-file Python structure for easy maintenance and expansion

---

## ⚙️ Tech Stack & Architecture

### 🛠️ Core Technologies Used

| Component | Technology |
|-----------|-----------|
| **Agentic Framework** | LangGraph for building the stateful, dynamic agent |
| **LLM** | Google's Gemini family for reasoning, routing, and response generation |
| **Vector Search & Embeddings** | Pinecone for high-speed semantic search on a custom medical knowledge base |
| **Live Data Tools** | PubMed API for real-time research, BeautifulSoup for web scraping |
| **Geospatial Tools** | OpenStreetMap (Overpass API) and Haversine for key-free location finding and distance calculation |
| **Core Libraries** | LangChain, Python, Dotenv |

### 🏗️ System Architecture

The agent's workflow is cyclical and intelligent, managed by a central router that directs tasks to the appropriate tool before synthesizing a final answer.

```
┌─────────────────────────────────────────────────────────────┐
│                    User Query Input                         │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              LangGraph State Manager                        │
│         (Maintains Conversation Context)                    │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│            Central Router (Gemini LLM)                      │
│      Analyzes query & selects best tool                     │
└──┬──────────┬──────────┬──────────┬──────────┬─────────────┘
   │          │          │          │          │
   ▼          ▼          ▼          ▼          ▼
┌─────┐  ┌────────┐  ┌──────┐  ┌─────┐  ┌──────────┐
│ RAG │  │PubMed  │  │ Web  │  │ OSM │  │Proactive │
│Tool │  │  API   │  │Scrape│  │Geo  │  │ Advisor  │
└──┬──┘  └───┬────┘  └───┬──┘  └──┬──┘  └────┬─────┘
   │         │           │        │          │
   └─────────┴───────────┴────────┴──────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│           Response Synthesis & Generation                   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Conversational Response                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.10 or higher
- Conda (recommended) or virtualenv
- API keys for Google Gemini and Pinecone

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Arpitkushwahaa/Medical-Chatbot.git
cd Medical-Chatbot
```

### 2️⃣ Create an Isolated Environment

Using Conda (recommended):
```bash
conda create -n medibot python=3.10 -y
conda activate medibot
```

Or using venv:
```bash
python -m venv medibot
# On Windows:
medibot\Scripts\activate
# On macOS/Linux:
source medibot/bin/activate
```

### 3️⃣ Install Required Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Set Up API Keys & Environment Variables

Create a `.env` file in the root directory with your credentials:

```env
GOOGLE_API_KEY="your_google_api_key_here"
PINECONE_API_KEY="your_pinecone_api_key_here"
```

> **Note**: Never commit your `.env` file to version control. It's already included in `.gitignore`.

### 5️⃣ Ingest Data into Pinecone

Before running the agent, populate its knowledge base using your local medical documents:

```bash
python ingest.py
```

This will process all documents in the `medical_documents/` folder and upload them to your Pinecone index.

### 6️⃣ Run the Agent Application

```bash
python main.py
```

The agent will now be running in your terminal, ready to answer questions! 🎉

---

## 🔬 Core Functionalities & Innovations

### 1️⃣ Dynamic Tool-Using Agent

- A **central router**, powered by the LLM, analyzes the conversation's state at every turn to decide the single best next action
- The agent can seamlessly transition from answering a factual question with RAG to conducting live research on PubMed, to finding a local doctor on OpenStreetMap
- Intelligent decision-making ensures the most relevant and up-to-date information is always provided

### 2️⃣ Proactive Assistance

- The agent is prompted to be **more than just a reactive bot**
- After answering a question, it analyzes the context to provide supplementary advice, such as lifestyle tips or related symptoms to watch for
- It anticipates user needs by proactively offering to find relevant local help (e.g., *"Would you like me to find a dermatologist near you?"*)

### 3️⃣ Free & Scalable Geolocation

- Uses the **OpenStreetMap Overpass API**, a completely free alternative to paid services like Google Maps
- The agent gets the user's location once, caches it in the conversation state, and uses it to find and rank nearby specialists by distance
- Provides complete addresses and calculated distances for easy access to medical care

---

## 📂 Project Structure

```
📦 medical_agent/
├── 📁 medical_documents/     # PDFs and documents for the knowledge base
├── 📄 config.py              # Handles API keys and constants
├── 📄 tools.py               # Defines all agent tools (RAG, PubMed, OSM, etc.)
├── 📄 agent.py               # Builds and compiles the LangGraph agent
├── 📄 ingest.py              # Script to load data into Pinecone
├── 📄 main.py                # Main entry point to run the application
├── 📄 app.py                 # Application logic and utilities
├── 📄 requirements.txt       # Python dependency list
├── 📄 Dockerfile             # Docker configuration for containerization
├── 📄 index.html             # Web interface (if applicable)
├── 📄 LICENSE                # MIT License file
├── 📄 README.md              # This file
└── 🔑 .env                   # API keys (ignored in version control)
```

---

## 💡 Usage

### Basic Conversation

```
You: What are the symptoms of diabetes?
Agent: [Provides detailed answer from RAG/PubMed]
       Additionally, I notice you're asking about diabetes symptoms. 
       Would you like tips on prevention or finding an endocrinologist near you?
```

### Finding Local Specialists

```
You: I need to find a cardiologist
Agent: I can help you find cardiologists nearby. Could you share your location 
       (city or coordinates)?
You: New York, NY
Agent: [Lists top 5 cardiologists with addresses and distances]
```

### Research Queries

```
You: What's the latest research on immunotherapy for cancer?
Agent: [Searches PubMed for recent studies and summarizes findings]
```

---

## 🚀 Future Enhancements

- [ ] 🐳 **Containerize with Docker** for consistent deployment across environments
- [ ] 🌐 **Build a Web Interface** using Streamlit or Flask for a user-friendly experience
- [ ] 🎯 **Integrate a Reranker Model** to improve the quality of retrieved RAG context
- [ ] 🧬 **Fine-Tune an LLM** with medical-specific datasets for even higher accuracy
- [ ] 📱 **Mobile App Development** for on-the-go medical assistance
- [ ] 🔊 **Voice Interface** for hands-free interaction
- [ ] 📊 **Analytics Dashboard** to track common queries and improve knowledge base

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Please ensure your code follows the project's coding standards and includes appropriate tests.

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Arpit Kushwaha**

- GitHub: [@Arpitkushwahaa](https://github.com/Arpitkushwahaa)

---

## 🙏 Acknowledgments

- **LangChain** and **LangGraph** for the powerful agentic framework
- **Google Gemini** for advanced language model capabilities
- **Pinecone** for efficient vector search
- **OpenStreetMap** for free geolocation services
- The open-source community for continuous inspiration

---

<div align="center">

**⭐ If you find this project helpful, please consider giving it a star!**

Made with ❤️ and 🤖 by developers who care about accessible healthcare

</div>
