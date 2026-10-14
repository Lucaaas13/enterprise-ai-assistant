import httpx


import os

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434/api/generate",
)

MODEL_NAME = os.getenv(
    "OLLAMA_MODEL",
    "qwen2.5:3b",
)

def build_prompt(
    question: str,
    context: str,
) -> str:
    return f"""
Tu réponds à une question uniquement à partir des sources fournies.

Instructions :
- Trouve le passage qui répond le plus directement à la question.
- Privilégie les phrases contenant les mêmes concepts ou noms techniques que la question.
- Une déduction simple à partir du texte est autorisée.
  Exemple : si une section sur le LLM local dit "Installer OUTIL_X",
  alors OUTIL_X peut être identifié comme l'outil utilisé localement.
- Si une phrase répond directement à la question, ne choisis pas une information
  moins pertinente située juste après dans une nouvelle section.
- N'invente aucune étape ou recommandation.
- Réponds en 1 ou 2 phrases maximum.
- Termine par la source utilisée, par exemple [Source 1].
- Si aucune source ne permet de répondre, réponds exactement :
"Je ne trouve pas cette information dans les documents fournis."

Exemple 1 :
Source :
Objectif : utiliser un modèle local.
Installer OUTIL_X et choisir un modèle.

Question :
Quel outil est utilisé pour le modèle local ?

Réponse :
L'outil utilisé est OUTIL_X [Source 1].

Exemple 2 :
Source :
N'ajoute pas encore TECHNOLOGIE_X. Commence avec une solution simple.
Phase suivante : installer FRAMEWORK_Y.

Question :
Que faut-il faire avant d'utiliser TECHNOLOGIE_X ?

Réponse :
Il faut commencer avec une solution simple et ne pas encore ajouter TECHNOLOGIE_X [Source 1].

Sources :
{context}

Question :
{question}

Réponse :
"""


def call_ollama(
    question: str,
    context: str,
) -> dict:
    prompt = build_prompt(
        question=question,
        context=context,
    )

    response = httpx.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "10m",
            "options": {
                "temperature": 0.0,
                "num_predict": 120,
            },
        },
        timeout=120.0,
    )

    response.raise_for_status()

    return response.json()


def generate_answer(
    question: str,
    context: str,
) -> str:
    data = call_ollama(
        question=question,
        context=context,
    )

    return data["response"]