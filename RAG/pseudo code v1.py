from fastapi import FastAPI, Request
import openai
from vectordb import search_vector_db, rerank_results
from cache import response_cache, embedding_cache

app = FastAPI()

# === RAG Handler ===
async def handle_study_query(query, user_id):
    # Step 1: Check response cache
    if response_cache.exists(query):
        return {"answer": response_cache.get(query)}

    # Step 2: Check embedding cache
    if embedding_cache.exists(query):
        embedding = embedding_cache.get(query)
    else:
        embedding = openai.Embedding.create(
            model="text-embedding-3-small",
            input=query
        )["data"][0]["embedding"]
        embedding_cache.set(query, embedding)

    # Step 3: Vector search
    candidates = search_vector_db(embedding, top_k=10)

    # Step 4: Rerank results
    ranked_contexts = rerank_results(query, candidates, top_k=5)

    # Step 5: Prepare context
    context_text = "\n\n".join([doc["text"] for doc in ranked_contexts])

    prompt = f"""
    You are a study assistant. 
    Answer the following question using the provided context only.
    
    Context:
    {context_text}

    Question: {query}
    """

    # Step 6: LLM Call
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )

    # Step 7: Cache + stream
    full_answer = ""
    for chunk in response:
        if "content" in chunk["choices"][0]["delta"]:
            piece = chunk["choices"][0]["delta"]["content"]
            full_answer += piece
            yield piece  # stream to frontend

    response_cache.set(query, full_answer)

    return {"answer": full_answer}
