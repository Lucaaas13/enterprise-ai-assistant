import os

import pytest
from dotenv import load_dotenv
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.database.database import Base, get_db
from app.main import app

load_dotenv()

TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")

if not TEST_DATABASE_URL:
    raise RuntimeError("TEST_DATABASE_URL is not defined")


test_engine = create_engine(TEST_DATABASE_URL)

TestingSessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=test_engine,
)

with test_engine.begin() as connection:
    connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

Base.metadata.create_all(bind=test_engine)


def override_get_db():
    db = TestingSessionLocal()

    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_database():
    with test_engine.begin() as connection:
        connection.execute(text("TRUNCATE TABLE documents RESTART IDENTITY CASCADE"))


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


def test_duplicate_pdf_upload(
    monkeypatch,
):
    monkeypatch.setattr(
        "app.api.documents.extract_pages_from_pdf",
        lambda _: [
            (
                1,
                "Same PDF content.",
            )
        ],
    )

    monkeypatch.setattr(
        "app.api.documents.predict_category",
        lambda _: {
            "category": "developpement",
            "confidence": 0.90,
        },
    )

    def upload_pdf():
        return client.post(
            "/documents/upload",
            files={
                "file": (
                    "duplicate.pdf",
                    b"fake-pdf-content",
                    "application/pdf",
                )
            },
        )

    first_response = upload_pdf()

    assert first_response.status_code == 201

    second_response = upload_pdf()

    assert second_response.status_code == 409

    assert second_response.json() == {"detail": ("This document has already been uploaded.")}
