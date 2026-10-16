# Enterprise AI Knowledge Assistant

Assistant IA multi-sources basé sur une architecture RAG.

## Status

Project in development.

## Docker

Le projet peut être lancé avec Docker Compose.

Prérequis :
- Docker Desktop
- Ollama lancé localement
- modèle `qwen2.5:3b`
- fichier `.env` contenant `POSTGRES_PASSWORD`

Lancer le projet :

```bash
docker compose up

## Experiment tracking with MLflow

MLflow is used locally to track and compare the main experiments of the project.

### Retrieval comparison

| Strategy | Recall@1 | Recall@3 | Recall@5 | Average latency |
|---|---:|---:|---:|---:|
| Vector search | 52% | 72% | 88% | 0.040 s |
| PostgreSQL lexical search | 60% | 84% | 88% | 0.016 s |
| Hybrid search with RRF | 72% | 88% | 92% | 0.058 s |
| Hybrid RRF + reranker | 72% | 92% | 92% | 0.360 s |

Hybrid retrieval with Reciprocal Rank Fusion was retained for the main
pipeline. The reranker slightly improved Recall@3 but introduced a large
latency increase without improving Recall@1 or Recall@5.

### Document classification

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| PyTorch baseline | 70% | 0.70 |
| Multilingual Transformer | 90% | 0.90 |

The Transformer improved results on the evaluation split, although the
dataset remains intentionally small and the result should not be interpreted
as a general performance estimate.

### MLflow

Start the local tracking interface with:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db