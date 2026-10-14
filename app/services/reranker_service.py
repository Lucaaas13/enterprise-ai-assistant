from functools import lru_cache

from sentence_transformers import CrossEncoder


RERANKER_MODEL = (
    "cross-encoder/"
    "mmarco-mMiniLMv2-L12-H384-v1"
)


@lru_cache
def get_reranker() -> CrossEncoder:
    return CrossEncoder(
        RERANKER_MODEL
    )


def rerank_results(
    query: str,
    results: list[dict],
    top_k: int = 3,
) -> list[dict]:
    if not results:
        return []

    model = get_reranker()

    pairs = [
        (
            query,
            result["content"],
        )
        for result in results
    ]

    scores = model.predict(pairs)

    reranked_results = []

    for result, score in zip(
        results,
        scores,
    ):
        reranked_result = result.copy()

        reranked_result[
            "reranker_score"
        ] = float(score)

        reranked_results.append(
            reranked_result
        )

    reranked_results.sort(
        key=lambda result: (
            result["reranker_score"]
        ),
        reverse=True,
    )

    return reranked_results[:top_k]