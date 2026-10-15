from app.ml.classification_service import (
    predict_category,
)

TESTS = [
    "Une vulnérabilité critique a été détectée sur le serveur.",
    "Le pipeline CI exécute les tests Python à chaque push.",
    "Le salarié souhaite poser trois jours de congés.",
    "Le serveur PostgreSQL manque de mémoire disponible.",
]


def main():
    for text in TESTS:
        result = predict_category(text)

        print()
        print("Texte :", text)
        print(
            "Catégorie :",
            result["category"],
        )
        print(
            "Confiance :",
            f"{result['confidence']:.2%}",
        )


if __name__ == "__main__":
    main()
