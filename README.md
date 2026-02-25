# 🤖 Chatter AI

A premium, state-of-the-art Multi-PDF RAG (Retrieval-Augmented Generation) assistant. Built with a high-performance FastAPI backend and a stunning React frontend, Chatter AI allows you to search, analyze, and chat across your entire document library with professional-grade precision.

---

## ✨ Key Features

- 📄 **Multi-PDF Intelligence**: Unlike standard RAG apps, Chatter AI can query across your *entire* indexed library simultaneously.
- 🖼️ **Visual Document Hub**: Modern card-based management with automatic first-page PDF previews (thumbnails).
- 🧬 **Smart Routing**: Intelligent conditional logic that automatically switches between "Querying recently uploaded" and "Library-wide search."
- ⚡ **Groq-Powered Speed**: Near-instant responses using Llama 3 (via Groq API) for high-performance reasoning.
- 🔍 **Semantic Search**: FAISS-based vector storage with `all-MiniLM-L6-v2` embeddings and cosine similarity.
- � **Deep Analytics**: Track token usage, similarity scores, and document-specific query frequency.
- 🎨 **Premium Aesthetic**: A sophisticated dark-themed interface featuring glassmorphism effects and smooth micro-animations.

---

## 🛠️ Architecture

### **Frontend**
- **Framework**: [React](https://reactjs.org/) + [Vite](https://vitejs.dev/)
- **Language**: [TypeScript](https://www.typescriptlang.org/)
- **State**: Context API for global chat & document state.
- **API Client**: Axios with centralized routing.
- **Icons**: Lucide React.
- **Styling**: Vanilla CSS Design System with custom utility tokens.

### **Backend**
- **Framework**: [FastAPI](https://fastapi.tiangolo.com/) (Python 3.10+)
- **Database**: SQLAlchemy with SQLite (managed via CRUD patterns).
- **RAG Engine**:
  - **PDF Processing**: `pypdf` & `LangChain` text splitters.
  - **Embeddings**: `sentence-transformers` (Pre-loaded singleton).
  - **Vector DB**: FAISS (local index management).
  - **Thumbnails**: `PyMuPDF` (fitz) for generating page previews.
- **LLM**: Groq Cloud (Llama 3.3 70B Versatile).

---

## 🚀 Getting Started

### **1. Clone & Environment**
```bash
git clone <repo-url>
cd Chatter_AI
```

Create a `.env` file in the root:
```env
GROQ_API_KEY=your_api_key_here
```

### **2. Backend Setup**
```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn backend.main:app --reload
```
*The backend will run on `http://localhost:8000`*

### **3. Frontend Setup**
```bash
cd frontend

# Install dependencies
npm install

# Start the dev server
npm run dev
```
*Visit `http://localhost:5173` to start chatting!*

---

## 📂 Project Structure

```
Chatter_AI/
├── backend/
│   ├── api/             # FastAPI Routers (Query, Docs, Chat, Analytics)
│   ├── db/              # SQLAlchemy Models & CRUD
│   ├── rag_engine/      # Core logic (Ingestion, Retrieval, LLM)
│   └── main.py          # App Entry Point
├── frontend/
│   ├── src/
│   │   ├── api/         # Centralized API calls
│   │   ├── context/     # Global State (ChatContext)
│   │   ├── hooks/       # Custom React Hooks
│   │   └── pages/       # Dashboards & Chat UI
│   └── vite.config.ts   # Reverse Proxy Configuration
├── rag_cache/           # FAISS indices & Thumbnails
└── uploaded_pdfs/       # RAW PDF storage
```

---

## 🛡️ RAG Logic details

1. **Ingestion**: PDFs are split into 1000-character chunks with overlap.
2. **Indexing**: Chunks are embedded using `sentence-transformers` and stored in a local FAISS index.
3. **Retrieval**: User queries are embedded and compared against all *active* document indices using cosine similarity.
4. **Filtering**: The engine respects logical "soft-deletes" from the DB, ensuring deleted PDFs stay in cache but never appear in answers.
5. **Generation**: Top chunks are fed to Groq with professional system prompts for high-accuracy response generation.

---

## 👤 Author

**Onkar Hadgaonkar**
- 📧 [onkarhadgaonkar22@gmail.com](mailto:onkarhadgaonkar22@gmail.com)

---
*Developed with ❤️ as a next-gen document assistant.*
