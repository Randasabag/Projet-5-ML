import pytest
import joblib
import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.multiclass import OneVsRestClassifier


@pytest.fixture(scope="module")
def load_model():
    # Chargement des modèles et du TF-IDF vectorizer
    tfidf = joblib.load('https://drive.google.com/file/d/13AYA7oPlaDyxoMuq6vG2Mm27riXoTP4f/view?usp=drive_link/tfidf.joblib')
    classifier = joblib.load('https://drive.google.com/file/d/1vtx6wj3Gdkb1n96dCJcWjaott11wgWX8/view?usp=drive_link/classifier.joblib')
    mlb = joblib.load('https://drive.google.com/file/d/1iMJwgMA4yRhtbAN09732aqZEsmtBBSBe/view?usp=drive_link/mlb.joblib')
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

