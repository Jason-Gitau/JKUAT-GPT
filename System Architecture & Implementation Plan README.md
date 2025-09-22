# AI Study Assistant - System Architecture & Implementation Plan

## 1. High-Level Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │   API Gateway    │    │   Worker Layer  │
│   (React PWA)   │◄──►│   (FastAPI)      │◄──►│   (Celery/RQ)   │
│                 │    │                  │    │                 │
│ • Chat UI       │    │ • Auth           │    │ • Doc Processing│
│ • Upload UI     │    │ • Ingestion      │    │ • Embedding Gen │
│ • Group UI      │    │ • Query          │    │ • OCR           │
│ • Dashboard     │    │ • Groups         │    │ • Summarization │
└─────────────────┘    │ • Feedback       │    └─────────────────┘
                       │ • Admin          │
                       └──────────────────┘
                                │
                 ┌──────────────┼──────────────┐
                 │              │              │
        ┌─────────▼──┐  ┌────────▼───┐  ┌──────▼──────┐
        │ PostgreSQL │  │   Redis    │  │ Vector Store│
        │            │  │            │  │ (pgvector)  │
        │ • Users    │  │ • Cache    │  │             │
        │ • Docs     │  │ • Sessions │  │ • Embeddings│
        │ • Groups   │  │ • Queues   │  │ • Chunks    │
        │ • Logs     │  └────────────┘  └─────────────┘
        └────────────┘
                                │
                       ┌────────▼────────┐
                       │  External APIs  │
                       │                 │
                       │ • DeepSeek LLM  │
                       │ • Qwen LLM      │
                       │ • Embedding API │
                       │ • Research APIs │
                       └─────────────────┘
```

## 2. Database Schema Implementation

### PostgreSQL Tables

```sql
-- Users table
CREATE TABLE users (
    user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    student_id VARCHAR(100),
    display_name VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    role VARCHAR(20) DEFAULT 'student' CHECK (role IN ('student', 'admin', 'lecturer')),
    consent_uploads BOOLEAN DEFAULT FALSE,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Documents table
CREATE TABLE documents (
    doc_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_id UUID REFERENCES users(user_id) ON DELETE CASCADE,
    title VARCHAR(500) NOT NULL,
    filename VARCHAR(255) NOT NULL,
    source_type VARCHAR(50) DEFAULT 'lecture' CHECK (source_type IN ('lecture', 'paper', 'book', 'pastpaper')),
    uploaded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    storage_path TEXT NOT NULL,
    language VARCHAR(10) DEFAULT 'en',
    hash VARCHAR(64) UNIQUE NOT NULL, -- SHA256
    status VARCHAR(20) DEFAULT 'processing' CHECK (status IN ('processing', 'ready', 'error')),
    metadata JSONB DEFAULT '{}',
    token_count INTEGER DEFAULT 0
);

-- Vector chunks table with pgvector
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE chunks (
    chunk_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id UUID REFERENCES documents(doc_id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    tokens INTEGER NOT NULL,
    metadata JSONB DEFAULT '{}',
    embedding vector(768), -- 768-dimensional embeddings
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    checksum VARCHAR(40) NOT NULL, -- SHA1 of text
    chunk_index INTEGER NOT NULL -- Position within document
);

-- Create HNSW index for fast similarity search
CREATE INDEX idx_chunks_embedding ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX idx_chunks_doc_id ON chunks(doc_id);
CREATE INDEX idx_chunks_checksum ON chunks(checksum);

-- Groups table
CREATE TABLE groups (
    group_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    is_anonymous BOOLEAN DEFAULT TRUE,
    created_by UUID REFERENCES users(user_id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    settings JSONB DEFAULT '{}'
);

-- Group membership
CREATE TABLE group_members (
    group_id UUID REFERENCES groups(group_id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(user_id) ON DELETE CASCADE,
    joined_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    role VARCHAR(20) DEFAULT 'member' CHECK (role IN ('member', 'admin')),
    PRIMARY KEY (group_id, user_id)
);

-- Query logs
CREATE TABLE query_logs (
    query_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(user_id),
    group_id UUID REFERENCES groups(group_id) NULL,
    query_text TEXT NOT NULL,
    retrieved_chunk_ids UUID[] DEFAULT '{}',
    llm_response TEXT,
    tokens_input INTEGER DEFAULT 0,
    tokens_output INTEGER DEFAULT 0,
    cost_usd DECIMAL(10,6) DEFAULT 0,
    rating VARCHAR(10) CHECK (rating IN ('up', 'down')),
    correction TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    response_time_ms INTEGER,
    model_used VARCHAR(100)
);

-- Create indexes for performance
CREATE INDEX idx_query_logs_user_id ON query_logs(user_id);
CREATE INDEX idx_query_logs_created_at ON query_logs(created_at);
CREATE INDEX idx_documents_owner_id ON documents(owner_id);
CREATE INDEX idx_documents_status ON documents(status);
```

## 3. Core API Endpoints Implementation

### FastAPI Application Structure

```python
# app/main.py
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import asyncio
import httpx
from typing import List, Optional
import uuid
from datetime import datetime

app = FastAPI(title="AI Study Assistant", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

# Configuration
class Config:
    DEEPSEEK_API_KEY = "your-deepseek-key"
    DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
    EMBEDDING_MODEL = "text-embedding-ada-002"  # Configurable
    LLM_MODEL = "deepseek-chat"
    CHUNK_SIZE = 300
    CHUNK_OVERLAP = 50
    MAX_CONTEXT_TOKENS = 2000
    DEFAULT_TOP_K = 10

config = Config()
```

### Core Endpoints

```python
# app/routes/ingestion.py
from fastapi import APIRouter, UploadFile, BackgroundTasks
from app.services.document_processor import DocumentProcessor
from app.models.schemas import IngestionResponse, IngestionStatus

router = APIRouter(prefix="/api/v1/ingest")

@router.post("/", response_model=IngestionResponse)
async def upload_document(
    file: UploadFile,
    title: str,
    source_type: str = "lecture",
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_user)
):
    """Upload and process document"""
    job_id = str(uuid.uuid4())
    
    # Save file and enqueue processing
    file_path = await save_uploaded_file(file, current_user.user_id)
    
    # Enqueue background processing
    background_tasks.add_task(
        process_document,
        job_id=job_id,
        file_path=file_path,
        title=title,
        source_type=source_type,
        owner_id=current_user.user_id
    )
    
    return IngestionResponse(
        job_id=job_id,
        status="processing",
        message="Document uploaded successfully"
    )

@router.get("/{job_id}", response_model=IngestionStatus)
async def get_ingestion_status(job_id: str):
    """Get processing status"""
    # Implementation to check job status
    pass
```

```python
# app/routes/query.py
from fastapi import APIRouter
from app.services.rag_pipeline import RAGPipeline
from app.models.schemas import QueryRequest, QueryResponse

router = APIRouter(prefix="/api/v1")

@router.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    current_user = Depends(get_current_user)
):
    """Process user query using RAG pipeline"""
    
    # Initialize RAG pipeline
    rag = RAGPipeline(config=config)
    
    # Execute query
    result = await rag.process_query(
        query=request.query,
        user_id=current_user.user_id,
        group_id=request.group_id,
        max_tokens=request.max_tokens_context or config.MAX_CONTEXT_TOKENS,
        response_mode=request.response_mode or "concise"
    )
    
    # Log query for analytics
    await log_query(
        user_id=current_user.user_id,
        query=request.query,
        result=result
    )
    
    return result
```

## 4. RAG Pipeline Implementation

### Document Processing Service

```python
# app/services/document_processor.py
import hashlib
from pathlib import Path
from typing import List, Dict, Any
import asyncio
import httpx

class DocumentProcessor:
    def __init__(self, config):
        self.config = config
        self.embedding_client = httpx.AsyncClient()
    
    async def process_document(
        self,
        file_path: str,
        title: str,
        source_type: str,
        owner_id: str
    ) -> Dict[str, Any]:
        """Main document processing pipeline"""
        
        try:
            # 1. Extract text
            text_content = await self.extract_text(file_path)
            
            # 2. Generate document hash
            doc_hash = hashlib.sha256(text_content.encode()).hexdigest()
            
            # 3. Check for duplicates
            existing_doc = await self.check_duplicate(doc_hash)
            if existing_doc:
                return {"status": "duplicate", "doc_id": existing_doc}
            
            # 4. Save document metadata
            doc_id = await self.save_document_metadata(
                title=title,
                filename=Path(file_path).name,
                source_type=source_type,
                owner_id=owner_id,
                storage_path=file_path,
                doc_hash=doc_hash
            )
            
            # 5. Chunk text
            chunks = await self.chunk_text(text_content, title)
            
            # 6. Generate embeddings
            embeddings = await self.generate_embeddings([chunk["text"] for chunk in chunks])
            
            # 7. Save chunks with embeddings
            await self.save_chunks(doc_id, chunks, embeddings)
            
            # 8. Update document status
            await self.update_document_status(doc_id, "ready")
            
            return {
                "status": "success",
                "doc_id": doc_id,
                "chunks_created": len(chunks)
            }
            
        except Exception as e:
            await self.update_document_status(doc_id, "error")
            raise e
    
    async def chunk_text(self, text: str, title: str) -> List[Dict[str, Any]]:
        """Chunk text into smaller pieces with overlap"""
        
        chunks = []
        sentences = text.split('. ')
        
        current_chunk = ""
        current_tokens = 0
        chunk_index = 0
        
        for sentence in sentences:
            sentence_tokens = len(sentence.split())  # Rough token count
            
            if current_tokens + sentence_tokens > self.config.CHUNK_SIZE:
                if current_chunk:
                    # Save current chunk
                    chunks.append({
                        "text": current_chunk.strip(),
                        "tokens": current_tokens,
                        "chunk_index": chunk_index,
                        "metadata": {
                            "title": title,
                            "chunk_index": chunk_index
                        }
                    })
                    
                    # Start new chunk with overlap
                    overlap_sentences = current_chunk.split('. ')[-2:]  # Last 2 sentences
                    current_chunk = '. '.join(overlap_sentences) + '. ' + sentence
                    current_tokens = sum(len(s.split()) for s in overlap_sentences) + sentence_tokens
                    chunk_index += 1
            else:
                current_chunk += sentence + '. '
                current_tokens += sentence_tokens
        
        # Add final chunk
        if current_chunk:
            chunks.append({
                "text": current_chunk.strip(),
                "tokens": current_tokens,
                "chunk_index": chunk_index,
                "metadata": {
                    "title": title,
                    "chunk_index": chunk_index
                }
            })
        
        return chunks
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for text chunks"""
        
        # Batch embedding generation
        embeddings = []
        batch_size = 100
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            response = await self.embedding_client.post(
                f"{self.config.DEEPSEEK_BASE_URL}/embeddings",
                json={
                    "model": self.config.EMBEDDING_MODEL,
                    "input": batch
                },
                headers={
                    "Authorization": f"Bearer {self.config.DEEPSEEK_API_KEY}"
                }
            )
            
            batch_embeddings = [item["embedding"] for item in response.json()["data"]]
            embeddings.extend(batch_embeddings)
        
        return embeddings
```

### RAG Pipeline Service

```python
# app/services/rag_pipeline.py
from typing import List, Dict, Any, Optional
import asyncio
import httpx
from app.models.schemas import QueryResponse, Citation
from app.services.vector_search import VectorSearchService
from app.services.reranker import RerankerService

class RAGPipeline:
    def __init__(self, config):
        self.config = config
        self.vector_search = VectorSearchService()
        self.reranker = RerankerService()
        self.llm_client = httpx.AsyncClient()
    
    async def process_query(
        self,
        query: str,
        user_id: str,
        group_id: Optional[str] = None,
        max_tokens: int = 2000,
        response_mode: str = "concise"
    ) -> QueryResponse:
        """Main RAG pipeline execution"""
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # 1. Generate query embedding
            query_embedding = await self.generate_embedding(query)
            
            # 2. Retrieve relevant chunks
            retrieved_chunks = await self.vector_search.search(
                query_embedding=query_embedding,
                user_id=user_id,
                group_id=group_id,
                top_k=self.config.DEFAULT_TOP_K
            )
            
            # 3. Rerank chunks (optional but recommended)
            if len(retrieved_chunks) > 5:
                retrieved_chunks = await self.reranker.rerank(
                    query=query,
                    chunks=retrieved_chunks,
                    top_k=5
                )
            
            # 4. Build context from chunks
            context, citations = await self.build_context(
                chunks=retrieved_chunks,
                max_tokens=max_tokens
            )
            
            # 5. Generate response using LLM
            llm_response, tokens_used = await self.generate_response(
                query=query,
                context=context,
                response_mode=response_mode
            )
            
            end_time = asyncio.get_event_loop().time()
            response_time = int((end_time - start_time) * 1000)
            
            return QueryResponse(
                answer=llm_response,
                citations=citations,
                tokens_input=tokens_used["input"],
                tokens_output=tokens_used["output"],
                cost_usd=self.calculate_cost(tokens_used),
                response_time_ms=response_time,
                query_id=str(uuid.uuid4())
            )
            
        except Exception as e:
            # Error handling and fallback
            return QueryResponse(
                answer="I encountered an error processing your query. Please try again.",
                citations=[],
                tokens_input=0,
                tokens_output=0,
                cost_usd=0.0,
                response_time_ms=0,
                query_id=str(uuid.uuid4()),
                error=str(e)
            )
    
    async def build_context(
        self,
        chunks: List[Dict],
        max_tokens: int
    ) -> tuple[str, List[Citation]]:
        """Build context string from retrieved chunks"""
        
        context_parts = []
        citations = []
        total_tokens = 0
        
        for i, chunk in enumerate(chunks, 1):
            chunk_tokens = chunk["tokens"]
            
            if total_tokens + chunk_tokens > max_tokens:
                break
            
            context_parts.append(f"[{i}] SOURCE: {chunk['metadata']['title']}\n{chunk['text']}\n---\n")
            
            citations.append(Citation(
                chunk_id=chunk["chunk_id"],
                doc_id=chunk["doc_id"],
                title=chunk["metadata"]["title"],
                score=chunk["score"],
                source_number=i
            ))
            
            total_tokens += chunk_tokens
        
        context = "\n".join(context_parts)
        return context, citations
    
    async def generate_response(
        self,
        query: str,
        context: str,
        response_mode: str
    ) -> tuple[str, Dict[str, int]]:
        """Generate LLM response with context"""
        
        # Select appropriate prompt template
        if response_mode == "quiz":
            prompt_template = self.get_quiz_prompt_template()
        elif response_mode == "detailed":
            prompt_template = self.get_detailed_prompt_template()
        else:
            prompt_template = self.get_concise_prompt_template()
        
        # Format prompt
        formatted_prompt = prompt_template.format(
            sources=context,
            user_query=query
        )
        
        # Call LLM API
        response = await self.llm_client.post(
            f"{self.config.DEEPSEEK_BASE_URL}/chat/completions",
            json={
                "model": self.config.LLM_MODEL,
                "messages": [
                    {"role": "user", "content": formatted_prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 1000
            },
            headers={
                "Authorization": f"Bearer {self.config.DEEPSEEK_API_KEY}"
            }
        )
        
        result = response.json()
        
        return (
            result["choices"][0]["message"]["content"],
            {
                "input": result["usage"]["prompt_tokens"],
                "output": result["usage"]["completion_tokens"]
            }
        )
    
    def get_concise_prompt_template(self) -> str:
        return """You are an academic study assistant. Use ONLY the information in the following sources to answer. If the answer is not present, say "I don't know; please provide more notes."

SOURCES:
{sources}

QUESTION:
{user_query}

INSTRUCTIONS:
- Answer concisely in plain English.
- Provide numbered citations like (source #1).
- If you mention facts not found in sources, label them "assumption".
- End with 2 suggested practice questions."""
```

## 5. Monitoring & Analytics

### Metrics Collection

```python
# app/services/metrics.py
from prometheus_client import Counter, Histogram, Gauge
import time
from functools import wraps

# Prometheus metrics
query_counter = Counter('queries_total', 'Total queries processed', ['user_type', 'response_mode'])
query_duration = Histogram('query_duration_seconds', 'Query processing time')
token_usage = Counter('tokens_used_total', 'Total tokens consumed', ['model', 'type'])
retrieval_precision = Gauge('retrieval_precision_at_k', 'Retrieval precision at K')

def track_metrics(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            result = await func(*args, **kwargs)
            
            # Track success metrics
            query_counter.labels(
                user_type='student',
                response_mode=kwargs.get('response_mode', 'concise')
            ).inc()
            
            return result
        
        finally:
            # Track duration
            duration = time.time() - start_time
            query_duration.observe(duration)
    
    return wrapper
```

## 6. Deployment Configuration

### Docker Configuration

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    postgresql-client \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Start application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Kubernetes Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-study-assistant
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ai-study-assistant
  template:
    metadata:
      labels:
        app: ai-study-assistant
    spec:
      containers:
      - name: api
        image: ai-study-assistant:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secrets
              key: connection-string
        - name: DEEPSEEK_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: deepseek-key
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
```

## 7. Implementation Roadmap

### Week 1-2: Foundation
- [ ] Set up PostgreSQL with pgvector
- [ ] Implement basic FastAPI application structure
- [ ] Create document ingestion pipeline
- [ ] Basic text chunking and embedding generation

### Week 3-4: Core RAG
- [ ] Implement vector search functionality
- [ ] Build query processing pipeline
- [ ] Add LLM integration with DeepSeek
- [ ] Create basic React frontend

### Week 5-6: Features & Polish
- [ ] Add group functionality
- [ ] Implement reranking
- [ ] Add monitoring and metrics
- [ ] Create admin dashboard

### Week 7-8: Testing & Deployment
- [ ] Load testing and optimization
- [ ] Deploy to test cluster
- [ ] User acceptance testing
- [ ] Prepare for pilot launch

## 8. Success Metrics for Pilot

### Technical Metrics
- **Latency**: < 5s median response time
- **Accuracy**: > 80% user satisfaction on relevance
- **Availability**: > 99% uptime
- **Cost**: < $0.10 per active user per month

### Business Metrics
- **Adoption**: 500-1000 pilot users
- **Engagement**: > 20% weekly active users
- **Retention**: > 60% 30-day retention
- **Satisfaction**: > 4.0/5.0 average rating

This architecture provides a solid foundation for the AI study assistant with room for scaling and improvement based on pilot feedback.
