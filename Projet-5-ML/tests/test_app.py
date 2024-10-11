# tests/test_app.py
import pytest
import requests

# Test de l'API externe
def test_generate_tags_api():
    url = 'http://127.0.0.1:8000/generate_tags'
    response = requests.post(url, json={"question": "Quel est le langage le plus utilisé en data science ?"})
    assert response.status_code == 200  # Vérifie que l'API répond avec succès
    data = response.json()
    assert 'tags' in data  # Vérifie que la réponse contient les tags


# Test avec simulation de l'API
def test_generate_tags_with_mock(requests_mock):
    url = 'http://127.0.0.1:8000/generate_tags'
    
    # Simuler la réponse de l'API
    mock_response = {"tags": ["R", "Python", "Java"]}
    requests_mock.post(url, json=mock_response, status_code=200)
    
    # Appeler l'API avec la simulation
    response = requests.post(url, json={"question": "Quel est le langage le plus utilisé en data science ?"})
    assert response.status_code == 200
    data = response.json()
    
    # Vérifier que les tags sont retournés
    assert 'tags' in data
    assert data['tags'] == ["R", "Python", "Java"]

