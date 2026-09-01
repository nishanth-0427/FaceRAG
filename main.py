import os
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()
client = genai.Client()

def chunk_text(filepath):
    """Reads a text file and returns a list of non-empty lines."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def retrieve(question, chunks, top_k=3, threshold=0.4):
    """Scores chunks against the question using Gemini embeddings and cosine similarity."""
    texts_to_embed = [question] + chunks
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=texts_to_embed
    )
    
    embeddings = [e.values for e in response.embeddings]
    q_emb, chunk_embs = embeddings[0], embeddings[1:]
    
    def cosine_sim(v1, v2):
        dot = sum(a * b for a, b in zip(v1, v2))
        norm = (sum(a * a for a in v1) ** 0.5) * (sum(b * b for b in v2) ** 0.5)
        return dot / norm if norm else 0.0

    scored_chunks = [(cosine_sim(q_emb, emb), chunk) for emb, chunk in zip(chunk_embs, chunks)]
    scored_chunks.sort(reverse=True, key=lambda x: x[0])
    
    return [item for item in scored_chunks[:top_k] if item[0] >= threshold]


# --- STEP 3: Generation ---
def generate_answer(question, retrieved_chunks):
    """Builds a prompt from the retrieved chunks and calls the Gemini LLM to answer."""
    # 1. Handle the retrieval miss (if no chunks met the threshold)
    if not retrieved_chunks:
        return "⚠️ Retrieval miss: no chunk scored above threshold (0.4).\nAnswer: I don't have enough information to answer this."

    # 2. Build the context string
    context_text = "\n".join([chunk for score, chunk in retrieved_chunks])
    
    # 3. Construct the prompt with strict grounding instructions
    prompt = f"""Context:
{context_text}

Question: {question}

Instructions: Answer using only the context above.
If the context doesn't contain the answer, say "I don't have enough information to answer this."
"""

    # 4. Call the LLM to generate the answer
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )
    return response.text


# --- STEP 4: Wire it together and run the demo! ---
if __name__ == "__main__":
    chunks = chunk_text("faceprep.txt")
    
    # Two test cases to show the system works perfectly (hit) and fails gracefully (miss)
    test_questions = [
        "What programs does FACE Prep offer for colleges?",  # Should be a HIT
        "What is FACE Prep's total revenue last year?"       # Should be a MISS
    ]
    
    for q in test_questions:
        print(f"\n{'='*50}\nQuestion: {q}")
        
        results = retrieve(q, chunks)
        print("\n--- Retrieved Context ---")
        if not results:
             print("None.")
        for score, chunk in results:
            print(f"[{score:.2f}] {chunk}")
            
        print("\n--- LLM Answer ---")
        answer = generate_answer(q, results)
        print(answer)
        print("\n")