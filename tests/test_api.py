import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import document_service


client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_documents():
    document_service.documents.clear()
    document_service.next_id = 1


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_create_document():
    response = client.post(
        "/documents",
        json={
            "title": "Guide de déploiement",
            "content": "Le déploiement commence par les tests.",
        },
    )

    assert response.status_code == 201

    data = response.json()

    assert data["id"] == 1
    assert data["title"] == "Guide de déploiement"
    assert data["content"] == "Le déploiement commence par les tests."


def test_get_documents():
    client.post(
        "/documents",
        json={
            "title": "Documentation PostgreSQL",
            "content": "PostgreSQL stocke les données de l'application.",
        },
    )

    response = client.get("/documents")

    assert response.status_code == 200

    data = response.json()

    assert len(data) == 1
    assert data[0]["id"] == 1
    assert data[0]["title"] == "Documentation PostgreSQL"


def test_create_document_without_title():
    response = client.post(
        "/documents",
        json={
            "title": "",
            "content": "Document sans titre.",
        },
    )

    assert response.status_code == 422