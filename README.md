# Enterprise AI Knowledge Assistant

Assistant documentaire basé sur une architecture RAG permettant d'ingérer des documents, de rechercher les passages pertinents, de générer des réponses sourcées et d'exécuter des outils via LangGraph et MCP.

Projet personnel réalisé pour pratiquer les principales briques d'un système d'IA appliquée : backend Python, embeddings, RAG, évaluation, MLOps, conteneurisation et agents.

---

## Demo

The application provides:

- PDF ingestion
- automatic chunking
- multilingual embeddings
- PostgreSQL + pgvector storage
- semantic and lexical retrieval
- hybrid retrieval with Reciprocal Rank Fusion
- local LLM generation with Ollama
- sourced answers
- document classification with PyTorch and Hugging Face
- experiment tracking with MLflow
- LangGraph tool orchestration
- MCP tool exposure

Screenshots and demonstrations are available in `docs/screenshots/`.

---

## Architecture

```mermaid
flowchart TD
    U[User] --> API[FastAPI]

    API --> INGEST[Document ingestion]
    INGEST --> PDF[PDF extraction]
    PDF --> CHUNK[Token-aware chunking]
    CHUNK --> EMB[Sentence Transformers]
    EMB --> DB[(PostgreSQL + pgvector)]

    API --> RAG[RAG engine]

    RAG --> HYBRID[Hybrid retrieval]
    HYBRID --> VECTOR[Vector search]
    HYBRID --> FTS[PostgreSQL Full Text Search]
    VECTOR --> DB
    FTS --> DB

    HYBRID --> LLM[Ollama / Qwen 2.5]
    LLM --> ANSWER[Sourced answer]

    API --> AGENT[LangGraph Agent]
    AGENT --> SEARCH[search_documents]
    AGENT --> SQL[query_database]
    AGENT --> REPORT[generate_report]

    SEARCH --> HYBRID
    SQL --> DB

    MCP[MCP Server] --> SEARCH
    MCP --> SQL

    MLFLOW[MLflow] --> EXP[Experiment tracking]
```

---

## Tech Stack

### Backend

- Python 3.10
- FastAPI
- Pydantic
- SQLAlchemy
- pytest

### RAG and NLP

- Sentence Transformers
- Hugging Face Transformers
- Ollama
- Qwen 2.5 3B
- PostgreSQL Full Text Search
- pgvector

### Machine Learning

- PyTorch
- scikit-learn
- Hugging Face Transformers

### MLOps and quality

- MLflow
- Docker
- Docker Compose
- GitHub Actions
- Ruff

### Agents

- LangGraph
- Model Context Protocol (MCP)

---

## Main Features

### Document ingestion

PDF documents are:

1. uploaded through FastAPI,
2. parsed page by page,
3. split into token-aware chunks,
4. embedded with Sentence Transformers,
5. stored in PostgreSQL with pgvector.

Document and page metadata are preserved so retrieved chunks can be linked back to their source.

Duplicate PDF ingestion is rejected.

---

## Embeddings

The project uses:

`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

Embedding dimension:

`384`

Embeddings are normalized before being stored in PostgreSQL.

---

## Retrieval

Three retrieval strategies were evaluated:

1. semantic vector search,
2. PostgreSQL Full Text Search,
3. hybrid retrieval using Reciprocal Rank Fusion.

A cross-encoder reranker was also evaluated.

### Retrieval results

| Strategy | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Avg latency |
|---|---:|---:|---:|---:|---:|
| Vector search | 52% | 72% | 88% | 96% | 0.040 s |
| PostgreSQL lexical search | 60% | 84% | 88% | 92% | 0.016 s |
| Hybrid RRF | 72% | 88% | 92% | 92% | 0.058 s |
| Hybrid RRF + reranker | 72% | 92% | 92% | 92% | 0.360 s |

The production retrieval strategy is:

`Hybrid retrieval with Reciprocal Rank Fusion`

The reranker was not kept in the main pipeline because the small retrieval improvement did not justify the large latency increase.

---

## RAG Pipeline

The final RAG pipeline follows:

```text
Question
   |
   v
Hybrid retrieval
   |
   v
Relevant chunks
   |
   v
Prompt construction
   |
   v
Local LLM
   |
   v
Answer + sources
```

The LLM is instructed to answer only from the retrieved context.

If the answer cannot be found in the documents, the system can abstain instead of inventing information.

---

## RAG Evaluation

A versioned evaluation dataset containing 30 questions is used.

It contains:

- answerable questions,
- unanswerable questions,
- expected pages,
- reference answers,
- expected terms.

Baseline RAG results:

| Metric | Result |
|---|---:|
| Global accuracy | 63.33% |
| Answerable accuracy | 56% |
| Correct abstention rate | 100% |
| Hallucinations detected | 0 |
| Average generation latency | ~14.6 s |

Generation latency is mainly caused by running the LLM locally on CPU.

---

## Document Classification

A secondary ML experiment classifies documents into:

- Security
- Infrastructure
- Development
- HR

Two approaches were evaluated.

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| PyTorch bag-of-words MLP | 70% | 0.70 |
| Fine-tuned multilingual Transformer | 90% | 0.90 |

The evaluation split contains only 10 examples, so these results are used as an educational comparison rather than a general performance estimate.

The predicted category and confidence can be stored as document metadata.

The category is not used as a mandatory retrieval filter because documents can contain several topics and classification confidence can be low.

---

## MLflow

MLflow is used to track technical experiments.

Tracked information includes:

- retrieval strategy,
- embedding model,
- chunk size,
- overlap,
- top-k,
- reranker usage,
- LLM,
- Recall@k,
- latency,
- classification metrics.

Start MLflow locally:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open:

```text
http://localhost:5000
```

---

## LangGraph Agent

A LangGraph agent chooses between structured tools depending on the user request.

Available tools include:

- `search_documents`
- `query_database`
- `generate_report`

Example execution paths:

```text
Question about documentation
-> router
-> search_documents
-> final_answer
```

```text
Question about stored data
-> router
-> query_database
-> final_answer
```

```text
Report request
-> router
-> search_documents
-> generate_report
-> final_answer
```

The graph uses structured Pydantic outputs and execution limits to avoid uncontrolled loops.

Ten agentic scenarios were used to validate routing and tool chaining.

---

## Model Context Protocol

The project exposes several capabilities through a local MCP server.

Available MCP tools:

- `search_knowledge`
- `get_document_count`
- `get_category_counts`
- `get_recent_documents`

Architecture:

```text
MCP client
    |
    v
MCP server
    |
    v
Application services
    |
    +--> Hybrid retrieval
    |
    +--> PostgreSQL / pgvector
```

LangGraph and MCP have different roles:

- LangGraph orchestrates tools internally.
- MCP exposes tools through a standardized protocol to compatible AI clients.

The implementation currently uses the MCP `stdio` transport.

---

## API

Start FastAPI locally:

```bash
uvicorn app.main:app --reload
```

Swagger:

```text
http://localhost:8000/docs
```

Health check:

```text
GET /health
```

Main endpoints include:

```text
POST /documents
GET  /documents
POST /documents/upload
GET  /documents/{id}/chunks

POST /search
POST /rag
```

---

## Docker

The project contains:

- a FastAPI Docker image,
- PostgreSQL with pgvector,
- persistent database storage,
- health checks,
- environment-based configuration.

Start the stack:

```bash
docker compose up --build
```

Stop it:

```bash
docker compose down
```

Database data is stored in a Docker volume and survives container restarts.

---

## Tests

Start PostgreSQL:

```bash
docker compose up -d postgres
```

Run tests:

```bash
python -m pytest -v
```

Code quality:

```bash
ruff check .
ruff format --check .
```

---

## CI/CD

GitHub Actions automatically runs on pushes and pull requests.

The pipeline performs:

```text
install dependencies
        |
        v
PostgreSQL + pgvector
        |
        v
Ruff lint
        |
        v
Ruff formatting check
        |
        v
pytest
        |
        v
Docker image build
```

A failing test causes the CI pipeline to fail.

---

## Installation

Clone the repository:

```bash
git clone <REPOSITORY_URL>
cd enterprise-ai-assistant
```

Create a virtual environment:

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Copy the environment configuration:

```bash
copy .env.example .env
```

Start PostgreSQL:

```bash
docker compose up -d postgres
```

Start Ollama and download the model:

```bash
ollama pull qwen2.5:3b
```

Start FastAPI:

```bash
uvicorn app.main:app --reload
```

---

## Environment Variables

Example:

```env
DATABASE_URL=postgresql+psycopg://postgres:password@localhost:5433/enterprise_ai_assistant
TEST_DATABASE_URL=postgresql+psycopg://postgres:password@localhost:5433/enterprise_ai_assistant_test
OLLAMA_URL=http://localhost:11434/api/generate
OLLAMA_MODEL=qwen2.5:3b
```

Real credentials must never be committed.

---

## Cloud

The project runs locally at zero cost.

A possible Azure deployment architecture is documented in:

`docs/azure_deployment.md`

No paid Azure infrastructure is currently deployed.

The documented migration path is:

```text
Docker image
-> Container Registry
-> Azure Container Apps
-> managed PostgreSQL / pgvector
```

---

## Known Limitations

Current limitations include:

- the local Qwen model runs on CPU and generation is relatively slow,
- the evaluation corpus is still small,
- the current demo knowledge base contains a limited number of documents,
- document classification was trained on a very small dataset,
- document-level categories can be too coarse for multi-topic documents,
- the classifier model artifact is not committed to Git,
- no production authentication or authorization layer is implemented,
- Azure deployment is documented but not deployed,
- the current MCP implementation is local through stdio.

---

## Cost

The complete development version can run locally without paid APIs.

| Component | Cost |
|---|---|
| FastAPI | Free |
| PostgreSQL | Free |
| pgvector | Free |
| Hugging Face | Free |
| Sentence Transformers | Free |
| Ollama | Free |
| Qwen | Free |
| Docker | Free for local personal use |
| MLflow | Free |
| LangGraph | Free |
| MCP | Free |
| Azure | Not deployed |

---

## Project Structure

```text
enterprise-ai-assistant/
|
+-- app/
|   +-- agents/
|   +-- api/
|   +-- database/
|   +-- mcp_server/
|   +-- ml/
|   +-- models/
|   +-- schemas/
|   +-- services/
|
+-- data/
|
+-- docs/
|
+-- evaluation/
|   +-- results/
|
+-- scripts/
|
+-- tests/
|
+-- .github/
|   +-- workflows/
|
+-- Dockerfile
+-- docker-compose.yml
+-- requirements.txt
+-- pyproject.toml
+-- README.md
```

---

## Main Engineering Decisions

### Why PostgreSQL + pgvector?

It keeps relational data and vector embeddings in the same database and avoids adding an external vector database for a small project.

### Why hybrid retrieval?

Semantic search handles meaning and reformulations while lexical search performs well on precise terminology.

Combining both improved Recall@1 from 52% with vector search to 72%.

### Why no reranker in production?

The reranker slightly improved Recall@3 but increased retrieval latency from approximately 58 ms to 360 ms.

The trade-off was not considered worthwhile for the main pipeline.

### Why a local LLM?

Ollama allows the RAG pipeline to run without a paid external API and keeps the project reproducible locally.

### Why LangGraph?

Some user requests require different execution paths and can combine multiple tools.

A deterministic RAG pipeline remains preferable for simple document questions.

### Why MCP?

MCP provides a standardized way to expose application capabilities to external AI clients without exposing unrestricted application access.

---

## Future Improvements

Possible future improvements include:

- larger and more diverse evaluation datasets,
- additional document formats,
- authentication and authorization,
- better document deduplication using file hashes,
- stronger document classifier training data,
- streaming LLM responses,
- production observability,
- remote MCP transport,
- optional cloud deployment,
- larger or faster LLM inference.

---

## Status

The project currently includes:

- FastAPI backend
- PostgreSQL
- pgvector
- PDF ingestion
- embeddings
- vector retrieval
- lexical retrieval
- hybrid RRF retrieval
- RAG
- reproducible evaluation
- reranking experiments
- Docker Compose
- automated tests
- GitHub Actions CI/CD
- PyTorch
- Hugging Face
- MLflow
- LangGraph
- MCP

Cloud deployment remains intentionally optional.
