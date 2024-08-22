from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow_hub as hub
import numpy as np
import joblib

# Charger votre modèle de machine learning
model = joblib.load("use_model.pkl")  # Chemin vers ton modèle sauvegardé

# Initialiser l'application FastAPI
app = FastAPI()
print("FastAPIutilisé")

class Question(BaseModel):
    question: str

@app.post("/generate_tags")
def generate_tags(question: Question):
    # Prétraitement de la question si nécessaire
    input_data = [question.question]  # Modifie ceci selon le format attendu par ton modèle
    
    # Faire une prédiction avec le modèle
    predicted_tags = model.predict(input_data)
    
    # Renvoyer les tags comme réponse
    return {"tags": predicted_tags}

