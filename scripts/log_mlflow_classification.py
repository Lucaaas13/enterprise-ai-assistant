from pathlib import Path

import mlflow


PROJECT_ROOT = Path(__file__).parent.parent

MLFLOW_DB = PROJECT_ROOT / "mlflow.db"


def configure_mlflow():
    mlflow.set_tracking_uri(
        f"sqlite:///{MLFLOW_DB.as_posix()}"
    )

    mlflow.set_experiment(
        "Enterprise AI Knowledge Assistant"
    )


def log_pytorch_baseline():
    with mlflow.start_run(
        run_name="classification_pytorch_baseline"
    ):
        mlflow.log_param(
            "experiment_type",
            "document_classification",
        )

        mlflow.log_param(
            "model_type",
            "pytorch_mlp",
        )

        mlflow.log_param(
            "dataset_size",
            40,
        )

        mlflow.log_param(
            "test_size",
            10,
        )

        mlflow.log_param(
            "epochs",
            30,
        )

        mlflow.log_param(
            "hidden_size",
            64,
        )

        mlflow.log_metric(
            "accuracy",
            0.70,
        )

        mlflow.log_metric(
            "macro_precision",
            0.75,
        )

        mlflow.log_metric(
            "macro_recall",
            0.71,
        )

        mlflow.log_metric(
            "macro_f1",
            0.70,
        )

        mlflow.log_metric(
            "weighted_f1",
            0.69,
        )


def log_transformer():
    with mlflow.start_run(
        run_name="classification_huggingface_transformer"
    ):
        mlflow.log_param(
            "experiment_type",
            "document_classification",
        )

        mlflow.log_param(
            "model_type",
            "transformer",
        )

        mlflow.log_param(
            "model_name",
            "distilbert-base-multilingual-cased",
        )

        mlflow.log_param(
            "dataset_size",
            40,
        )

        mlflow.log_param(
            "test_size",
            10,
        )

        mlflow.log_param(
            "epochs",
            3,
        )

        mlflow.log_param(
            "batch_size",
            4,
        )

        mlflow.log_param(
            "learning_rate",
            2e-5,
        )

        mlflow.log_param(
            "max_length",
            128,
        )

        mlflow.log_metric(
            "accuracy",
            0.90,
        )

        mlflow.log_metric(
            "macro_precision",
            0.92,
        )

        mlflow.log_metric(
            "macro_recall",
            0.92,
        )

        mlflow.log_metric(
            "macro_f1",
            0.90,
        )

        mlflow.log_metric(
            "weighted_f1",
            0.90,
        )


def main():
    configure_mlflow()

    log_pytorch_baseline()
    log_transformer()

    print(
        "Expériences de classification "
        "enregistrées dans MLflow."
    )


if __name__ == "__main__":
    main()