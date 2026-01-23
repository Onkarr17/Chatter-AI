# 🤖 Chatter AI

A powerful RAG (Retrieval-Augmented Generation) chatbot that can answer questions from PDF documents or provide general knowledge answers using LLM.

The project is deployed and here is Live link : https://chatter-ai-rag.streamlit.app/

## Features

- 📄 **PDF RAG Mode**: Upload PDFs and ask questions about them
- 🌍 **LLM Chat Mode**: General knowledge conversations
- ⚡ **Fast Responses**: Powered by Groq API (Llama 3.3 70B)
- 🔍 **Smart Retrieval**: FAISS-based semantic search with cosine similarity
- 💾 **Intelligent Caching**: Automatic cache management for faster responses
- 🎨 **Modern UI**: Beautiful dark theme with ChatGPT-style chat interface
- ✨ **Streaming Answers**: Real-time typing effect for answers

## Tech Stack

- **Frontend**: Streamlit
- **LLM**: Groq API (Llama 3.3 70B Versatile)
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Store**: FAISS (cosine similarity)
- **PDF Processing**: LangChain PyPDFLoader

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd Chatter_AI
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
```

4. Run the app:
```bash
streamlit run streamlit_app.py
```

## Usage

1. **LLM Chat Mode**: Select "🌍 LLM Chat" mode for all the questions in the universe
2. **PDF RAG Mode**: 
   - Select "📄 PDF RAG" mode
   - Upload a PDF file (max 10MB)
   - Wait for processing
   - Ask questions about the PDF

## Project Structure

```
Chatter_AI/
├── streamlit_app.py    # Main Streamlit UI
├── rag.py              # RAG backend logic
├── requirements.txt     # Python dependencies
├── .env                # API keys (not in repo)
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

## Configuration

- **Max PDF Size**: 10MB (configurable in `.streamlit/config.toml`)
- **Answer Quality**: Small, Medium, Large presets
- **Cache**: Automatically managed in `rag_cache/` directory

## Author

Onkar Hadgaonkar
Email : onkarhadgaonkar22@gmail.com
