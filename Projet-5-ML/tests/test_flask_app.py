# tests/test_flask_app.py
import pytest
from flask_app import app

@pytest.fixture
def client():
    """Fixture pour configurer un client de test Flask."""
    app.config['TESTING'] = True  # Indique à Flask que l'application est en mode test
    with app.test_client() as client:
        yield client

def test_generate_tags_valid_question(client):
    # Simule une requête POST avec une question valide
    response = client.post("/generate_tags", json={"question": "Qu'est-ce que Python ?"})

    # Vérifie que le code de réponse est 200 (succès)
    assert response.status_code == 200

    # Vérifie que la réponse contient bien des tags
    data = response.get_json()
    assert "tags" in data
    assert isinstance(data["tags"], list)  # Vérifie que c'est une liste


def test_generate_tags_missing_question(client):
    # Simule une requête POST sans question
    response = client.post("/generate_tags", json={})

    # Vérifie que le code de réponse est 400 (erreur côté client)
    assert response.status_code == 400

    # Vérifie que l'erreur est correctement signalée
    data = response.get_json()
    assert "error" in data
    assert data["error"] == "Question is required"

def test_generate_tags_invalid_question_type(client):
    # Simule une requête POST avec un type de question incorrect (ex. un entier)
    response = client.post("/generate_tags", json={"question": 12345})

    # Vérifie que le code de réponse est 400 (erreur côté client)
    assert response.status_code == 400

    # Vérifie que l'erreur est correctement signalée
    data = response.get_json()
    assert "error" in data
    assert data["error"] == "Question must be a string"


