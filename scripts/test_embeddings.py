from app.services.embedding_service import (
    create_embedding,
    cosine_similarity,
)
from app.services.embedding_service import get_embedding_model


model = get_embedding_model()

print("Longueur maximale :", model.max_seq_length)

def main():
    text_a = "PostgreSQL est utilisé pour stocker les données."
    text_b = "La base de données conserve les informations."
    text_c = "Il fait beau aujourd'hui à Paris."

    embedding_a = create_embedding(text_a)
    embedding_b = create_embedding(text_b)
    embedding_c = create_embedding(text_c)

    similarity_ab = cosine_similarity(
        embedding_a,
        embedding_b,
    )

    similarity_ac = cosine_similarity(
        embedding_a,
        embedding_c,
    )

    print("Texte A :", text_a)
    print("Texte B :", text_b)
    print("Texte C :", text_c)

    print()
    print("Taille de l'embedding :", embedding_a.shape)

    print()
    print(
        "Similarité A / B :",
        round(similarity_ab, 4),
    )

    print(
        "Similarité A / C :",
        round(similarity_ac, 4),
    )


if __name__ == "__main__":
    main()