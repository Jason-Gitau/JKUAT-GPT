
---

## 🔹 1. Handling Duplicate Lecture Notes

Since lecture handouts are **shared resources**, we want to store each unique file **once** only.

### Approach:

* When a file is uploaded:

  1. **Hash the file** (e.g., SHA256).
  2. Check if the hash already exists in DB.
  3. If yes → don’t store the file again, just **link user → existing file**.
  4. If no → store the file in storage + save the hash + metadata in DB.

This way, even if 200 students upload the same PDF, you only pay storage for **one copy**.

📌 Example Schema:

```sql
table: lecture_files
- id (uuid)
- filename
- file_url
- hash (unique)
- uploader_id
- created_at
```

📌 Pseudocode:

```python
def upload_lecture(file, user_id):
    file_bytes = file.read()
    file_hash = sha256(file_bytes)

    existing = db.find_one("lecture_files", {"hash": file_hash})
    if existing:
        db.insert("user_files", {"user_id": user_id, "file_id": existing["id"]})
        return {"status": "duplicate", "file_url": existing["file_url"]}
    else:
        file_url = storage.upload(file, path=f"lectures/{file_hash}.pdf")
        file_id = db.insert("lecture_files", {
            "filename": file.name,
            "file_url": file_url,
            "hash": file_hash,
            "uploader_id": user_id
        })
        db.insert("user_files", {"user_id": user_id, "file_id": file_id})
        return {"status": "new", "file_url": file_url}
```

---

## 🔹 2. Handling Personal Notes

Here’s the nuance:

* **Lecture handouts → deduplicated** (global resources).
* **Personal notes → not deduplicated**, because:

  * Even if content is similar, they belong to **different students**.
  * Students may want their notes private.
  * Notes could contain annotations, highlights, or personal interpretations → not exact duplicates.

### Optimization:

* For personal notes, we can still **hash** them. If identical, you could:

  * Option 1: Still store separately (because they’re private).
  * Option 2: Store once, but encrypt metadata so only the uploading student can see their “copy.”

If storage is **very expensive** at scale, you might apply deduplication even for notes, but usually personal notes are **smaller** compared to lecture PDFs (mostly text or small docs). The biggest cost sink is **duplicate lecture files**, not notes.

---

## 🔹 3. Cost Implications

* Deduplicating lecture files = **huge savings** (think: 500 students uploading the same 20MB handout = 10GB saved instantly).
* Personal notes are much lighter → less urgent to deduplicate.
* You can combine this with **tiered storage** (move older/unused files to cheaper storage like Cloudflare R2 deep storage).

---

✅ **My recommendation**:

* Implement **hash-based deduplication** for lecture handouts.
* Allow **personal notes to always be uploaded**, since they’re small and unique to each student.
* Later, you can run background jobs that **detect identical notes** and optimize if storage bills rise.

---
design a unified upload flow that handles both **lecture notes** (shared, deduplicated) and **personal notes** (private, usually unique).

---

# 🔹 Pseudocode: File Upload System

```python
def upload_file(file, user_id, file_type):
    """
    file_type: "lecture" or "personal"
    """

    # Step 1: Compute file hash
    file_bytes = file.read()
    file_hash = sha256(file_bytes)

    if file_type == "lecture":
        # --- Deduplicate lecture files ---
        existing = db.find_one("lecture_files", {"hash": file_hash})

        if existing:
            # Just link user to existing file
            db.insert("user_files", {"user_id": user_id, "file_id": existing["id"], "type": "lecture"})
            return {
                "status": "duplicate",
                "file_url": existing["file_url"],
                "message": "Lecture file already exists, linked to your account."
            }

        else:
            # Upload new file
            file_url = storage.upload(file, path=f"lectures/{file_hash}.pdf")
            file_id = db.insert("lecture_files", {
                "filename": file.name,
                "file_url": file_url,
                "hash": file_hash,
                "uploader_id": user_id,
                "created_at": now()
            })
            db.insert("user_files", {"user_id": user_id, "file_id": file_id, "type": "lecture"})
            return {
                "status": "new",
                "file_url": file_url,
                "message": "Lecture file uploaded successfully."
            }

    elif file_type == "personal":
        # --- Personal notes (no deduplication by default) ---
        file_url = storage.upload(file, path=f"personal/{user_id}/{file_hash}.pdf")
        file_id = db.insert("personal_notes", {
            "filename": file.name,
            "file_url": file_url,
            "hash": file_hash,
            "owner_id": user_id,
            "created_at": now()
        })
        return {
            "status": "personal_uploaded",
            "file_url": file_url,
            "message": "Personal note uploaded successfully."
        }

    else:
        return {"status": "error", "message": "Invalid file type"}
```

---

# 🔹 Database Design

### Lecture Files

```sql
table: lecture_files
- id (uuid, pk)
- filename (text)
- file_url (text)
- hash (text, unique)
- uploader_id (uuid fk users)
- created_at (timestamp)
```

### Personal Notes

```sql
table: personal_notes
- id (uuid, pk)
- filename (text)
- file_url (text)
- hash (text)   -- not unique (same notes can exist across users)
- owner_id (uuid fk users)
- created_at (timestamp)
```

### User ↔ File Links (for lecture materials)

```sql
table: user_files
- id (uuid, pk)
- user_id (uuid fk users)
- file_id (uuid fk lecture_files)
- type (enum: "lecture" | "personal")
- created_at (timestamp)
```

---

# 🔹 Workflow

* **Lecture notes**

  * If duplicate → don’t upload again, just link.
  * Saves storage & bandwidth.

* **Personal notes**

  * Always uploaded.
  * Hash still computed → useful for future analytics (e.g., detecting clusters of students writing same notes).
  * Stored under `personal/{user_id}/` namespace → fully private to that student.

---

✅ This system gives you **deduplication where it matters (lecture notes)** while keeping **flexibility and privacy for personal notes**.

---

 **deduplication + personal/lecture notes system** into the **RAG ingestion pipeline**.

---

# 🔹 Core Idea

* **Lecture notes** → Deduplicated → Ingest once → Stored globally in **VectorDB (course-level namespace)**.
* **Personal notes** → Not deduplicated → Ingest separately → Stored privately in **VectorDB (user-level namespace)**.
* At **query time**, the system combines:

  * **Global course knowledge (lecture notes + top upvoted student notes)**
  * **Private personal notes (if user chooses to include them)**

---

# 🔹 Ingestion Flow

```python
def ingest_file(file_id, file_type, user_id):
    """
    Takes a file (already uploaded) and prepares embeddings + VectorDB storage
    """

    if file_type == "lecture":
        # Get file from DB
        file = db.get("lecture_files", file_id)

        # Avoid duplicate ingestion (hash ensures uniqueness)
        if db.exists("ingested_files", {"hash": file["hash"]}):
            return {"status": "skipped", "message": "Already ingested"}

        # Extract text (OCR/PDF parsing)
        text_chunks = chunk_text(extract_text(file["file_url"]))

        # Generate embeddings
        embeddings = openai.embed(text_chunks)

        # Store embeddings into course namespace
        vector_db.insert(
            namespace=f"course:{file['course_id']}",
            embeddings=embeddings,
            metadata={"file_id": file_id, "type": "lecture"}
        )

        # Mark as ingested
        db.insert("ingested_files", {"file_id": file_id, "hash": file["hash"], "type": "lecture"})

        return {"status": "ingested", "message": "Lecture notes added to RAG"}

    elif file_type == "personal":
        file = db.get("personal_notes", file_id)

        # Extract & chunk text
        text_chunks = chunk_text(extract_text(file["file_url"]))

        # Generate embeddings
        embeddings = openai.embed(text_chunks)

        # Store embeddings into private namespace
        vector_db.insert(
            namespace=f"user:{user_id}",
            embeddings=embeddings,
            metadata={"file_id": file_id, "type": "personal"}
        )

        return {"status": "ingested", "message": "Personal notes added to RAG"}

    else:
        return {"status": "error", "message": "Invalid file type"}
```

---

# 🔹 Query Flow

```python
def query_ai(user_id, course_id, query_text):
    """
    Routes query through cache -> embeddings -> retrieval -> rerank -> LLM
    """

    # 1. Get embeddings for query
    query_embedding = cache.get(f"embed:{query_text}")
    if not query_embedding:
        query_embedding = openai.embed([query_text])
        cache.set(f"embed:{query_text}", query_embedding)

    # 2. Retrieve from lecture/course notes
    lecture_results = vector_db.search(
        namespace=f"course:{course_id}",
        embedding=query_embedding,
        top_k=10
    )

    # 3. Retrieve from user’s personal notes
    personal_results = vector_db.search(
        namespace=f"user:{user_id}",
        embedding=query_embedding,
        top_k=5
    )

    # 4. Combine + rerank results
    combined = reranker.rank(lecture_results + personal_results, query_text)

    # 5. Build prompt
    context = build_context(combined)
    llm_input = f"Context:\n{context}\n\nQuestion: {query_text}"

    # 6. Get final LLM response
    response = llm.generate(llm_input)

    return response
```

---

# 🔹 Benefits of This Design

1. **Lecture notes (deduped, global)** → Avoids storage bloat.
2. **Personal notes (private)** → Encourages individual input without risk of data leakage.
3. **Flexible retrieval** → User benefits from both community + private context.
4. **Upvoted notes weighting** → You can apply retrieval bias (e.g., multiply vector score × upvote factor).

---



