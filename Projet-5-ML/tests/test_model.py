import pytest
import joblib
import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.multiclass import OneVsRestClassifier


@pytest.fixture(scope="module")
def load_model():
    # Chargement des modèles et du TF-IDF vectorizer
    tfidf = joblib.load('tfidf.joblib')
    classifier = joblib.load('classifier.joblib')
    mlb = joblib.load('mlb.joblib')
    return tfidf, classifier, mlb

def test_tfidf_transformer(load_model):
    tfidf, _, _ = load_model
    
    # Tester la transformation de données avec TF-IDF
    test_text = ["Qu'est-ce que Python ?"]
    transformed_data = tfidf.transform(test_text)

    assert transformed_data.shape[0] == 1  # Doit renvoyer 1 ligne
    assert transformed_data.shape[1] > 0  # Le nombre de caractéristiques doit être > 0


def test_invalid_input_format(load_model):
    _, classifier, tfidf = load_model
    
    # Essayer de faire une prédiction avec un mauvais format d'entrée
    with pytest.raises(ValueError):
        invalid_input = ["Invalid input type"]  # Un texte non transformé
        classifier.predict(invalid_input)

