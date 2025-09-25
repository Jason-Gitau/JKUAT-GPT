Alright 🔥 let’s extend the backend pseudo-code so that **students are grouped by course** (e.g. Math/CS, Medicine), and notes/handouts can be:

* **Private** → only the uploader sees them.
* **Course-shared** → everyone in the same course sees them.
* **Public** → visible across the entire university (if allowed).

We’ll also allow **class reps/university staff** to upload official handouts.

---

## 🔹 Database Schema (Supabase / Neon)

```sql
-- Users
users (
  id UUID PRIMARY KEY,
  email TEXT,
  name TEXT,
  course_id UUID REFERENCES courses(id),
  role TEXT CHECK (role IN ('student', 'classrep', 'staff'))
)

-- Courses
courses (
  id UUID PRIMARY KEY,
  name TEXT,
  faculty TEXT
)

-- Notes
notes (
  id UUID PRIMARY KEY,
  uploader_id UUID REFERENCES users(id),
  course_id UUID REFERENCES courses(id),
  visibility TEXT CHECK (visibility IN ('private','course','public')),
  url TEXT, -- Cloudflare R2 link
  created_at TIMESTAMP DEFAULT now()
)

-- Embeddings
embeddings (
  id UUID PRIMARY KEY,
  note_id UUID REFERENCES notes(id),
  embedding VECTOR(1536), -- Milvus/Qdrant stores actual vector
  text TEXT
)
```

---

## 🔹 Upload Endpoint (with visibility)

```python
@app.post("/upload")
async def upload_notes(req: Request):
    form = await req.form()
    file = form["file"].read()
    visibility = form.get("visibility", "private")  # default private
    user = req.state.user

    # 1. Upload file to R2
    url = upload_to_r2(file)

    # 2. Store note metadata
    note_id = db.insert("notes", {
        "uploader_id": user["id"],
        "course_id": user["course_id"],
        "visibility": visibility,
        "url": url
    })

    # 3. Embed + Insert into vector DB
    embedding = openai.Embedding.create(
        model="text-embedding-3-small",
        input=file
    )["data"][0]["embedding"]

    insert_document(
        note_id=note_id,
        text=file.decode("utf-8"),
        embedding=embedding,
        course_id=user["course_id"],
        visibility=visibility
    )

    return {"status": "uploaded", "note_id": note_id, "visibility": visibility}
```

---

## 🔹 Query with Course-Aware Retrieval

```python
async def handle_study_query(query, user_id):
    user = db.get_user(user_id)

    # Step 1: Cache check
    if response_cache.exists(query):
        return {"answer": response_cache.get(query)}

    # Step 2: Embedding check
    if embedding_cache.exists(query):
        embedding = embedding_cache.get(query)
    else:
        embedding = openai.Embedding.create(
            model="text-embedding-3-small",
            input=query
        )["data"][0]["embedding"]
        embedding_cache.set(query, embedding)

    # Step 3: Vector search (filter by course/public)
    candidates = search_vector_db(
        embedding,
        top_k=10,
        filters={
            "course_id": [user["course_id"]],
            "visibility": ["course", "public"]
        }
    )

    # Step 4: Rerank + prompt prep
    ranked = rerank_results(query, candidates, top_k=5)
    context_text = "\n\n".join([doc["text"] for doc in ranked])

    prompt = f"""
    You are a study assistant. 
    Use the following class notes and handouts:

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
```

---

## 🔹 User Journey Example

👤 John (Math/CS student at JKUAT):

* Logs in → system knows his `course_id = math_cs`.
* Class rep uploads lecturer’s **handouts** with `visibility="course"`.
* Another student uploads **personal notes** but sets `visibility="private"`.
* The university staff uploads **past papers** with `visibility="public"`.

When John asks *“Explain Fourier Transforms”*:

1. System filters vector DB → pulls **Math/CS course notes** + **public resources**, ignores private notes.
2. Reranker ranks best documents.
3. LLM builds answer from context.

---

## 🔹 Advantages

* Keeps notes organized by course.
* Encourages collaboration (shared handouts, upvotes possible later).
* Private notes stay private unless user chooses otherwise.
* University staff can upload **official materials**.

---


