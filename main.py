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

def load_index(cache_path="embeddings_cache.json"):
    with open(cache_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def retrieve(question, kb, index, top_k=3, threshold=0.4):
    """Embeds only the question; compares against cached chunk vectors."""
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=[question]
    )
    q_emb = response.embeddings[0].values
    chunk_embs = index["vectors"]

    def cosine_sim(v1, v2):
        dot = sum(a * b for a, b in zip(v1, v2))
        norm = (sum(a * a for a in v1) ** 0.5) * (sum(b * b for b in v2) ** 0.5)
        return dot / norm if norm else 0.0

    scored = [(cosine_sim(q_emb, emb), item) for emb, item in zip(chunk_embs, kb)]
    scored.sort(reverse=True, key=lambda x: x[0])

    return [pair for pair in scored[:top_k] if pair[0] >= threshold]

def rewrite_query(question, history):
    """Rewrites a follow-up question into a standalone one using recent chat history."""
    if not history:
        return question

    recent = history[-3:]  # last 3 turns is enough context
    history_text = "\n".join(
        f"Q: {turn['question']}\nA: {turn['answer']}" for turn in recent
    )

    prompt = f"""Given this conversation history:
{history_text}

And this follow-up question: "{question}"

Rewrite the follow-up question to be a standalone question that includes any 
context it depends on from the history. If it's already standalone, return it unchanged.
Return ONLY the rewritten question, nothing else.
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )
    return response.text.strip()

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
    index = load_index("embeddings_cache.json")

    # --- safety check: make sure kb and index are in sync ---
    if len(kb) != index["count"]:
        raise ValueError(
            f"knowledge_base.json has {len(kb)} entries but embeddings_cache.json "
            f"has {index['count']}. Run build_index.py to rebuild the cache."
        )

    test_questions = [
        "What is your Cash on Delivery fee?",
        "Do you ship internationally?",
        "What is Kaira Home's total revenue last year?"
    ]

    for q in test_questions:
        print(f"\n{'='*50}\nQuestion: {q}")

        results = retrieve(q, kb, index)
        print("\n--- Retrieved Context ---")
        if not results:
            print("None.")
        for score, item in results:
            print(f"[{score:.2f}] ({item['id']} / {item['category']}) {item['text']}")

        print("\n--- LLM Answer ---")
        answer = generate_answer(q, results)
        print(answer)
        print("\n")