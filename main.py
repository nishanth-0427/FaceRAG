import os
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()
client = genai.Client() # Automatically picks up GEMINI_API_KEY from .env

def chunk_text(filepath):
    """Reads a text file and returns a list of non-empty lines."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def retrieve(question, chunks, top_k=3, threshold=0.4):
    """Scores chunks against the question using Gemini embeddings and cosine similarity."""
    
    # 1. Embed the question and all chunks in one fast API call
    # 1. Embed the question and all chunks in one fast API call
    texts_to_embed = [question] + chunks
    response = client.models.embed_content(
        model="gemini-embedding-001",  # <--- Update this line
        contents=texts_to_embed
    )
    
    # Extract vector values. Question is index 0, chunks are the rest.
    embeddings = [e.values for e in response.embeddings]
    q_emb, chunk_embs = embeddings[0], embeddings[1:]
    
    # 2. Calculate Cosine Similarity (pure Python, no extra libraries)
    def cosine_sim(v1, v2):
        dot = sum(a * b for a, b in zip(v1, v2))
        norm = (sum(a * a for a in v1) ** 0.5) * (sum(b * b for b in v2) ** 0.5)
        return dot / norm if norm else 0.0

    # 3. Score, sort, and return the best chunks that beat our threshold
    scored_chunks = [(cosine_sim(q_emb, emb), chunk) for emb, chunk in zip(chunk_embs, chunks)]
    scored_chunks.sort(reverse=True, key=lambda x: x[0])
    
    # Only keep chunks that score high enough (this handles the "Retrieval Miss")
    return [item for item in scored_chunks[:top_k] if item[0] >= threshold]

# Let's test the retrieval engine!
if __name__ == "__main__":
    chunks = chunk_text("faceprep.txt")
    
    # A test query that SHOULD match something
    question = "Where is the head office?"
    print(f"\nQuestion: {question}")
    
    results = retrieve(question, chunks)
    for score, chunk in results:
        print(f"Score: {score:.3f} | Chunk: {chunk}")