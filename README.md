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