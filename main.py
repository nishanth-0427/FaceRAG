import os
import json
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()
client = genai.Client()


def load_knowledge_base(filepath):
    """Loads the JSON knowledge base and returns the list of chunk objects."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def retrieve(question, kb, top_k=3, threshold=0.4):
    """Scores chunks against the question using Gemini embeddings and cosine similarity."""
    texts_to_embed = [question] + [item["text"] for item in kb]
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

    scored = [(cosine_sim(q_emb, emb), item) for emb, item in zip(chunk_embs, kb)]
    scored.sort(reverse=True, key=lambda x: x[0])

    return [pair for pair in scored[:top_k] if pair[0] >= threshold]


# --- STEP 3: Generation ---
def generate_answer(question, retrieved):
    """Builds a prompt from the retrieved chunks and calls the Gemini LLM to answer."""
    if not retrieved:
        return "⚠️ Retrieval miss: no chunk scored above threshold (0.4).\nAnswer: I don't have enough information to answer this."

    context_text = "\n".join([item["text"] for score, item in retrieved])

    prompt = f"""Context:
{context_text}

Question: {question}

Instructions: Answer using only the context above.
If the context doesn't contain the answer, say "I don't have enough information to answer this."
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )
    return response.text


# --- STEP 4: Wire it together and run the demo! ---
if __name__ == "__main__":
    kb = load_knowledge_base("knowledge_base.json")

    test_questions = [
        "What is your Cash on Delivery fee?",              # should be a HIT
        "Do you ship internationally?",                     # HIT or MISS depending on what you wrote
        "What is Kaira Home's total revenue last year?"     # should be a MISS
    ]

    for q in test_questions:
        print(f"\n{'='*50}\nQuestion: {q}")

        results = retrieve(q, kb)
        print("\n--- Retrieved Context ---")
        if not results:
            print("None.")
        for score, item in results:
            print(f"[{score:.2f}] ({item['id']} / {item['category']}) {item['text']}")

        print("\n--- LLM Answer ---")
        answer = generate_answer(q, results)
        print(answer)
        print("\n")