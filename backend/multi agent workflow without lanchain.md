

* 🔑 **Auth** → Supabase (JWT-based).
* 📦 **Storage** → Cloudflare R2 for docs.
* 🧠 **RAG** → embeddings + vector DB (Milvus/Qdrant).
* 💼 **Career Agent** → function-calling with LinkedIn/GitHub/Search MCP tools.
* ⚡ **Caching** → Redis (response + embedding cache).
* 🚦 **Rate limiting** → token bucket per user.
* 🔀 **Routing** → Decide if query is study (→ RAG) or career (→ agent).

---

## 🔹 Backend Architecture

```
FastAPI (or Node/Express)
│
├── Auth middleware (Supabase JWT)
├── Rate limiting (Redis token bucket)
│
├── /query   → router (study vs career)
│    ├── study → RAG pipeline
│    └── career → Career agent
│
├── /upload  → store notes in Cloudflare R2 + embed → vector DB
│
├── Cache layer (Redis for responses + embeddings)
│
└── Vector DB (Milvus/Qdrant)
```

---

## 🔹 Pseudo-Code (Python / FastAPI style)

```python
from fastapi import FastAPI, Request, Depends, HTTPException
import openai
from auth import verify_jwt
from cache import response_cache, embedding_cache, rate_limiter
from storage import upload_to_r2
from vectordb import search_vector_db, rerank_results, insert_document
from tools import linkedin_api, github_api, web_search

app = FastAPI()

# Middleware: Auth + Rate limiting
@app.middleware("http")
async def auth_and_rate_limit(request: Request, call_next):
    token = request.headers.get("Authorization")
    user = verify_jwt(token)  # Supabase check
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    if not rate_limiter.allow(user["id"]):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    request.state.user = user
    return await call_next(request)

# === Query Router ===
@app.post("/query")
async def handle_query(req: Request):
    data = await req.json()
    query = data["query"]
    user_id = req.state.user["id"]

    if is_study_query(query):
        return await handle_study_query(query, user_id)
    else:
        return await handle_career_query(query, user_id)

# === RAG Handler ===
async def handle_study_query(query, user_id):
    if response_cache.exists(query):
        return {"answer": response_cache.get(query)}

    if embedding_cache.exists(query):
        embedding = embedding_cache.get(query)
    else:
        embedding = openai.Embedding.create(
            model="text-embedding-3-small",
            input=query
        )["data"][0]["embedding"]
        embedding_cache.set(query, embedding)

    candidates = search_vector_db(embedding, top_k=10)
    ranked = rerank_results(query, candidates, top_k=5)
    context_text = "\n\n".join([doc["text"] for doc in ranked])

    prompt = f"""
    You are a study assistant. Use only the context to answer.

    Context:
    {context_text}

    Question: {query}
    """

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    final_answer = response["choices"][0]["message"]["content"]

    response_cache.set(query, final_answer)
    return {"answer": final_answer}

# === Career Agent ===
async def handle_career_query(query, user_id):
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are a career advisor agent."},
                  {"role": "user", "content": query}],
        functions=[
            {"name": "search_jobs", "parameters": {"keyword": "string"}},
            {"name": "fetch_github_repos", "parameters": {"topic": "string"}},
            {"name": "web_search", "parameters": {"query": "string"}}
        ],
        function_call="auto"
    )

    msg = response["choices"][0]["message"]
    if msg.get("function_call"):
        tool = msg["function_call"]["name"]
        args = msg["function_call"]["arguments"]

        if tool == "search_jobs":
            tool_result = linkedin_api.search_jobs(args["keyword"])
        elif tool == "fetch_github_repos":
            tool_result = github_api.fetch_repos(args["topic"])
        else:
            tool_result = web_search(args["query"])

        final = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a career advisor agent."},
                {"role": "user", "content": query},
                {"role": "function", "name": tool, "content": str(tool_result)}
            ]
        )
        return {"answer": final["choices"][0]["message"]["content"]}
    else:
        return {"answer": msg["content"]}

# === Upload Notes ===
@app.post("/upload")
async def upload_notes(req: Request):
    file = await req.form()
    content = file["file"].read()

    # Store in Cloudflare R2
    url = upload_to_r2(content)

    # Embed + Insert into vector DB
    embedding = openai.Embedding.create(
        model="text-embedding-3-small",
        input=content
    )["data"][0]["embedding"]

    insert_document(user_id=req.state.user["id"], text=content, embedding=embedding, url=url)

    return {"status": "uploaded", "url": url}
```

---

## 🔹 Key Features

* **Auth**: Supabase JWT checked on every request.
* **Rate limiting**: Redis token bucket prevents abuse (e.g., 100 queries/day).
* **Routing**: Simple function `is_study_query()` decides → RAG or Career Agent.
* **Caching**: Redis used for query-response + embeddings.
* **Storage**: R2 stores files; embeddings stored in vector DB.
* **Career Agent**: Function-calling system, no LangChain needed.

---

⚡ This backend is lean, modular, and scalable. You can host on Render/Modal/RunPod and scale each component separately (RAG workers, Career agent workers, API auth workers).


