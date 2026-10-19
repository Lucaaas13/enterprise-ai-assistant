import httpx

from app.agents.schemas import AgentDecision
from app.services.llm_service import MODEL_NAME, OLLAMA_URL


def route_question(
    question: str,
) -> AgentDecision:
    prompt = f"""
Tu es le routeur d'un assistant d'entreprise.

Tu dois choisir exactement UN outil.

RÈGLE PRINCIPALE :

Utilise query_database UNIQUEMENT si la question demande
l'une de ces trois opérations structurées :

1. Compter tous les documents
   -> document_count

2. Compter les documents par catégorie
   -> category_counts

3. Afficher les documents les plus récents
   -> recent_documents

Si l'utilisateur demande explicitement un rapport ou une synthèse
structurée, choisis generate_report.

Sinon, pour toute question documentaire classique,
choisis search_documents.

search_documents sert notamment à :
- chercher une information dans un PDF
- expliquer une technologie
- trouver une procédure
- répondre à une question sur le contenu documentaire
- chercher des informations sur l'architecture
- Docker
- Ollama
- FastAPI
- LangGraph
- MLflow
- RAG


Utilise cet outil lorsque l'utilisateur demande explicitement :
- un rapport
- une synthèse structurée
- un compte rendu basé sur les documents

generate_report doit chercher les informations dans les documents
avant de produire le rapport.

Exemples :

Question :
"Quel outil est utilisé pour exécuter le LLM localement ?"

Décision :
tool = search_documents
database_action = null

Question :
"Comment fonctionne le RAG ?"

Décision :
tool = search_documents
database_action = null

Question :
"Combien de documents sont enregistrés ?"

Décision :
tool = query_database
database_action = document_count

Question :
"Combien de documents y a-t-il par catégorie ?"

Décision :
tool = query_database
database_action = category_counts

Question :
"Quels sont les derniers documents ajoutés ?"

Décision :
tool = query_database
database_action = recent_documents

Question :
"Génère-moi un rapport sur les technologies utilisées dans le projet."

Décision :
tool = generate_report
database_action = null

IMPORTANT :
Si la question ne correspond pas clairement à une des
trois opérations de base de données,
choisis search_documents.

Question utilisateur :
{question}


Retourne uniquement la décision structurée.
"""

    response = httpx.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "format": AgentDecision.model_json_schema(),
            "keep_alive": "10m",
            "options": {
                "temperature": 0.0,
                "num_predict": 80,
            },
        },
        timeout=120.0,
    )

    response.raise_for_status()

    data = response.json()

    decision = AgentDecision.model_validate_json(data["response"])

    # Sécurisation de la décision du LLM :
    # query_database nécessite obligatoirement une action.
    if decision.tool == "query_database" and decision.database_action is None:
        return AgentDecision(
            tool="search_documents",
            database_action=None,
        )

    if decision.tool in {
        "search_documents",
        "generate_report",
    }:
        return AgentDecision(
            tool=decision.tool,
            database_action=None,
        )

    return decision
