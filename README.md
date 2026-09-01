# FACE Prep RAG Mini-Project

A lightweight Retrieval-Augmented Generation (RAG) pipeline built in Python. This project demonstrates how to retrieve relevant context from a knowledge base and use an LLM to answer questions while strictly preventing hallucinations.

## Architecture
1. **Chunking**: Line-based chunking of `faceprep.txt` to preserve individual semantic facts.
2. **Retrieval**: Uses `gemini-embedding-001` to vectorize text and calculates Cosine Similarity in pure Python to rank the most relevant chunks.
3. **Generation**: Uses `gemini-3.6-flash` to synthesize an answer based *only* on the retrieved context, with explicit grounding instructions to handle retrieval misses gracefully.

## How to Run

1. Clone the repository.
2. Create and activate a virtual environment (`python -m venv venv`).
3. Install dependencies: `pip install -r requirements.txt`
4. Add your Gemini API key to a `.env` file: `GEMINI_API_KEY=your_key_here`
5. Run the pipeline: `python main.py`