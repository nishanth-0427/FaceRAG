from fastapi import FastAPI
from pydantic import BaseModel
from main import load_knowledge_base, load_index, retrieve, generate_answer, rewrite_query

app = FastAPI()

# Load once at startup, not per-request
kb = load_knowledge_base("knowledge_base.json")
index = load_index("embeddings_cache.json")

if len(kb) != index["count"]:
    raise ValueError(
        f"knowledge_base.json has {len(kb)} entries but embeddings_cache.json "
        f"has {index['count']}. Run build_index.py to rebuild the cache."
    )

# In-memory session store: {session_id: [ {question, answer}, ... ]}
sessions = {}


class ChatRequest(BaseModel):
    session_id: str
    question: str


@app.post("/chat")
def chat(req: ChatRequest):
    history = sessions.get(req.session_id, [])

    standalone_question = rewrite_query(req.question, history)
    results = retrieve(standalone_question, kb, index)
    answer = generate_answer(req.question, results)  # answer the ORIGINAL question

    history.append({"question": req.question, "answer": answer})
    sessions[req.session_id] = history

    return {
        "answer": answer,
        "rewritten_question": standalone_question,  # add this for debugging visibility
        "retrieved": [
            {"id": item["id"], "category": item["category"], "score": round(score, 2)}
            for score, item in results
        ],
        "turn_count": len(history)
    }


@app.get("/history/{session_id}")
def get_history(session_id: str):
    return {"session_id": session_id, "history": sessions.get(session_id, [])}