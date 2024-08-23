from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import tensorflow_hub as hub


# Charger votre modèle de machine learning et le MultiLabelBinarizer
model = joblib.load("use_model.pkl")  # Chemin vers ton modèle sauvegardé
mlb = joblib.load("mlb.pkl")  # Charger le MultiLabelBinarizer
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")


# Initialiser l'application FastAPI
app = FastAPI()
print("FastAPI utilisé")

class Question(BaseModel):
    question: str

@app.post("/generate_tags")
def generate_tags(question: Question):
    try:
        # Prétraitement de la question si nécessaire
        input_data = embed([question.question])  # Modifie ceci selon le format attendu par ton modèle
        
        # Faire une prédiction avec le modèle
        predicted_tags = model.predict(input_data)
        
        # Convertir les prédictions en labels lisibles
        # Si 'predicted_tags' est une matrice binaire, utilisez mlb.inverse_transform
        if predicted_tags.ndim == 2:  # Si le modèle renvoie une matrice binaire
            predicted_tags = mlb.inverse_transform(predicted_tags)
        else:
            predicted_tags = mlb.inverse_transform(predicted_tags.reshape(1, -1))
        
        # Renvoyer les tags comme réponse
        return {"tags": predicted_tags[0].tolist()}  # Convertir en liste si nécessaire
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
