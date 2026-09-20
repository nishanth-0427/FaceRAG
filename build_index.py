import json
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client()

def build_embeddings(kb_path="knowledge_base.json", out_path="embeddings_cache.json"):
    with open(kb_path, 'r', encoding='utf-8') as f:
        kb = json.load(f)

    texts = [item["text"] for item in kb]
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=texts
    )
    vectors = [e.values for e in response.embeddings]

    cache = {
        "kb_path": kb_path,
        "count": len(kb),
        "ids": [item["id"] for item in kb],
        "vectors": vectors
    }

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(cache, f)

    print(f"Cached {len(kb)} embeddings to {out_path}")

if __name__ == "__main__":
    build_embeddings()