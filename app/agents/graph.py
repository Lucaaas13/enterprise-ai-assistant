import operator
from typing import Annotated, TypedDict

from langgraph.graph import (
    END,
    START,
    StateGraph,
)

from app.agents.router_service import (
    route_question,
)
from app.agents.schemas import (
    DatabaseQueryInput,
    GenerateReportInput,
    SearchDocumentsInput,
)
from app.agents.tools import (
    generate_report,
    query_database,
    search_documents,
)
from app.services.llm_service import generate_answer


class AgentState(TypedDict, total=False):
    question: str

    route: str

    database_action: str | None

    tool_result: dict

    answer: str

    steps: int

    trace: Annotated[
        list[str],
        operator.add,
    ]


def router_node(
    state: AgentState,
) -> dict:
    decision = route_question(state["question"])

    route = decision.tool
    database_action = decision.database_action

    # Sécurité :
    # query_database nécessite obligatoirement
    # une action structurée.
    # Sinon on retombe sur la recherche documentaire.
    if route == "query_database" and database_action is None:
        route = "search_documents"

    return {
        "route": route,
        "database_action": database_action,
        "steps": (state.get("steps", 0) + 1),
        "trace": ["router"],
    }


def search_documents_node(
    state: AgentState,
) -> dict:
    result = search_documents(
        SearchDocumentsInput(
            query=state["question"],
            top_k=5,
        )
    )

    return {
        "tool_result": (result.model_dump()),
        "answer": (f"{len(result.results)} passages pertinents trouvés."),
        "steps": (state.get("steps", 0) + 1),
        "trace": ["search_documents"],
    }


def query_database_node(
    state: AgentState,
) -> dict:
    action = state.get("database_action")

    if action is None:
        raise ValueError("database_action is required")

    result = query_database(
        DatabaseQueryInput(
            action=action,
            limit=5,
        )
    )

    return {
        "tool_result": (result.model_dump()),
        "answer": str(result.data),
        "steps": (state.get("steps", 0) + 1),
        "trace": ["query_database"],
    }


def final_answer_node(
    state: AgentState,
) -> dict:
    route = state["route"]
    tool_result = state["tool_result"]

    if route == "generate_report":
        return {
            "answer": tool_result["report"],
            "steps": (state.get("steps", 0) + 1),
            "trace": ["final_answer"],
        }

    if route == "search_documents":
        results = tool_result["results"]

        if not results:
            answer = "Je ne trouve pas cette information dans les documents disponibles."

        else:
            context_parts = []

            for index, result in enumerate(
                results,
                start=1,
            ):
                context_parts.append(
                    (
                        f"SOURCE {index}\n"
                        f"Document : "
                        f"{result['document_title']}\n"
                        f"Page : "
                        f"{result['page_number']}\n"
                        f"Contenu : "
                        f"{result['content']}"
                    )
                )

            context = "\n\n".join(context_parts)

            answer = generate_answer(
                question=state["question"],
                context=context,
            )

    elif route == "query_database":
        action = tool_result["action"]
        data = tool_result["data"]

        if action == "document_count":
            answer = f"La base contient {data['count']} document(s)."

        elif action == "category_counts":
            parts = [(f"{item['category']} : {item['count']}") for item in data]

            answer = "Répartition des documents par catégorie : " + ", ".join(parts) + "."

        elif action == "recent_documents":
            if not data:
                answer = "Aucun document n'est enregistré."
            else:
                titles = [item["title"] for item in data]

                answer = "Documents les plus récents : " + ", ".join(titles) + "."

        else:
            answer = str(data)

    else:
        answer = "Impossible de produire une réponse."

    return {
        "answer": answer,
        "steps": (state.get("steps", 0) + 1),
        "trace": ["final_answer"],
    }


def choose_tool(
    state: AgentState,
) -> str:
    route = state["route"]

    if route in {
        "search_documents",
        "generate_report",
    }:
        return route

    if route == "query_database":
        return "query_database"

    raise ValueError(f"Unknown route: {route}")


def generate_report_node(
    state: AgentState,
) -> dict:
    results = state["tool_result"]["results"]

    if not results:
        report = generate_report(
            GenerateReportInput(
                title="Rapport",
                items=[("Aucune information pertinente n'a été trouvée.")],
            )
        )

        return {
            "tool_result": {
                "report": report.report,
            },
            "answer": report.report,
            "steps": (state.get("steps", 0) + 1),
            "trace": ["generate_report"],
        }

    context_parts = []

    for index, result in enumerate(
        results,
        start=1,
    ):
        context_parts.append(
            (
                f"SOURCE {index}\n"
                f"Document : "
                f"{result['document_title']}\n"
                f"Page : "
                f"{result['page_number']}\n"
                f"Contenu : "
                f"{result['content']}"
            )
        )

    context = "\n\n".join(context_parts)

    report_question = (
        "À partir exclusivement des sources fournies, "
        "rédige une synthèse concise répondant à cette "
        f"demande : {state['question']}. "
        "Liste uniquement les informations réellement "
        "présentes dans les sources. "
        "Ne présente pas comme effectivement implémentée "
        "une technologie qui est seulement prévue, optionnelle "
        "ou mentionnée dans une feuille de route. "
        "Si le statut d'implémentation n'est pas prouvé par "
        "les sources, indique qu'elle est mentionnée ou prévue."
    )

    synthesis = generate_answer(
        question=report_question,
        context=context,
    )

    report = generate_report(
        GenerateReportInput(
            title=("Rapport généré à partir des documents"),
            items=[
                synthesis,
            ],
        )
    )

    return {
        "tool_result": {
            "report": report.report,
        },
        "answer": report.report,
        "steps": (state.get("steps", 0) + 1),
        "trace": ["generate_report"],
    }


def after_search(
    state: AgentState,
) -> str:
    if state["route"] == "generate_report":
        return "generate_report"

    return "final_answer"


builder = StateGraph(AgentState)

builder.add_node(
    "router",
    router_node,
)

builder.add_node(
    "search_documents",
    search_documents_node,
)

builder.add_node(
    "query_database",
    query_database_node,
)

builder.add_edge(
    START,
    "router",
)

builder.add_conditional_edges(
    "router",
    choose_tool,
    {
        "search_documents": "search_documents",
        "query_database": "query_database",
        "generate_report": "search_documents",
    },
)

builder.add_node(
    "final_answer",
    final_answer_node,
)

builder.add_node(
    "generate_report",
    generate_report_node,
)

builder.add_conditional_edges(
    "search_documents",
    after_search,
    {
        "generate_report": "generate_report",
        "final_answer": "final_answer",
    },
)

builder.add_edge(
    "generate_report",
    "final_answer",
)

builder.add_edge(
    "query_database",
    "final_answer",
)

builder.add_edge(
    "final_answer",
    END,
)

agent_graph = builder.compile()


def run_agent(
    question: str,
) -> AgentState:
    result = agent_graph.invoke(
        {
            "question": question,
            "steps": 0,
            "trace": [],
        },
        {
            "recursion_limit": 6,
        },
    )

    return result
