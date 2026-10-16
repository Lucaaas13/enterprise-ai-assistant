import json
from pathlib import Path

import mlflow


PROJECT_ROOT = Path(__file__).parent.parent

RESULTS_DIR = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
)

RETRIEVAL_RESULTS = (
    RESULTS_DIR
    / "retrieval_strategies_comparison.json"
)

RAG_RESULTS = (
    RESULTS_DIR
    / "rag_baseline.json"
)

MLFLOW_DB = (
    PROJECT_ROOT
    / "mlflow.db"
)


def configure_mlflow():
    tracking_uri = (
        f"sqlite:///{MLFLOW_DB.as_posix()}"
    )

    mlflow.set_tracking_uri(
        tracking_uri
    )

    mlflow.set_experiment(
        "Enterprise AI Knowledge Assistant"
    )


def log_retrieval_experiments():
    with open(
        RETRIEVAL_RESULTS,
        "r",
        encoding="utf-8",
    ) as file:
        benchmark = json.load(file)

    base_config = benchmark[
        "configuration"
    ]

    for strategy in benchmark[
        "strategies"
    ]:
        metrics = strategy["metrics"]

        with mlflow.start_run(
            run_name=strategy["name"]
        ):
            mlflow.log_param(
                "experiment_type",
                "retrieval",
            )

            mlflow.log_param(
                "retrieval_strategy",
                strategy["name"],
            )

            mlflow.log_param(
                "embedding_model",
                base_config[
                    "embedding_model"
                ],
            )

            mlflow.log_param(
                "embedding_dimension",
                base_config[
                    "embedding_dimension"
                ],
            )

            mlflow.log_param(
                "chunk_size_tokens",
                base_config[
                    "chunk_size_tokens"
                ],
            )

            mlflow.log_param(
                "chunk_overlap_tokens",
                base_config[
                    "chunk_overlap_tokens"
                ],
            )

            mlflow.log_param(
                "max_results",
                base_config[
                    "max_results"
                ],
            )

            mlflow.log_param(
                "reranker",
                (
                    "reranker"
                    in strategy["name"]
                ),
            )

            mlflow.log_metric(
                "recall_at_1",
                metrics["recall_at_1"],
            )

            mlflow.log_metric(
                "recall_at_3",
                metrics["recall_at_3"],
            )

            mlflow.log_metric(
                "recall_at_5",
                metrics["recall_at_5"],
            )

            mlflow.log_metric(
                "recall_at_10",
                metrics["recall_at_10"],
            )

            mlflow.log_metric(
                "average_latency_seconds",
                metrics[
                    "average_latency_seconds"
                ],
            )


def log_rag_baseline():
    with open(
        RAG_RESULTS,
        "r",
        encoding="utf-8",
    ) as file:
        benchmark = json.load(file)

    config = benchmark[
        "configuration"
    ]

    metrics = benchmark[
        "metrics"
    ]

    with mlflow.start_run(
        run_name="rag_baseline"
    ):
        mlflow.log_param(
            "experiment_type",
            "rag",
        )

        mlflow.log_param(
            "retrieval",
            config["retrieval"],
        )

        mlflow.log_param(
            "embedding_model",
            config[
                "embedding_model"
            ],
        )

        mlflow.log_param(
            "chunk_size_tokens",
            config[
                "chunk_size_tokens"
            ],
        )

        mlflow.log_param(
            "chunk_overlap_tokens",
            config[
                "chunk_overlap_tokens"
            ],
        )

        mlflow.log_param(
            "top_k",
            config["top_k"],
        )

        mlflow.log_param(
            "reranker",
            str(
                config["reranker"]
            ),
        )

        mlflow.log_param(
            "llm",
            config["llm"],
        )

        mlflow.log_metric(
            "accuracy",
            metrics["accuracy"],
        )

        mlflow.log_metric(
            "answerable_accuracy",
            metrics[
                "answerable_accuracy"
            ],
        )

        mlflow.log_metric(
            "abstention_accuracy",
            metrics[
                "abstention_accuracy"
            ],
        )

        mlflow.log_metric(
            "hallucinations",
            metrics[
                "hallucinations"
            ],
        )

        mlflow.log_metric(
            "average_latency_seconds",
            metrics[
                "average_latency_seconds"
            ],
        )


def main():
    configure_mlflow()

    log_retrieval_experiments()
    log_rag_baseline()

    print(
        "Expériences enregistrées "
        "dans MLflow."
    )


if __name__ == "__main__":
    main()