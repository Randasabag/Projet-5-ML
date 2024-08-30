from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import tensorflow as tf
import tensorflow_hub as hub
import mlflow.pyfunc
import mlflow


# Charger le modèle depuis MLflow
def load_model():
    # URI du modèle MLflow
    model_uri = "runs:/09259b68390948849fd6beffe80015b0/model"
    model = mlflow.pyfunc.load_model(model_uri)
    return model

model = load_model()

# Charger votre modèle de machine learning et le MultiLabelBinarizer
#model = joblib.load("/Users/randaalsabbagh/Desktop/MACHINE_LEARNING/P5/Projet-5-ML/use_model.pkl")  # Chemin vers ton modèle sauvegardé
mlb = joblib.load("mlb.pkl")  # Charger le MultiLabelBinarizer
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")


# Initialiser l'application FastAPI
app = FastAPI()
print("FastAPI utilisé")

class Question(BaseModel):
    question: str

class TagsResponse(BaseModel):
    predicted_tags: list[str]

@app.post("/generate_tags", response_model=TagsResponse)
def generate_tags(question: Question):
    try:
        # Prétraitement de la question
        input_data = embed([question.question])  # Modifie ceci selon le format attendu par ton modèle
        
        # Faire une prédiction avec le modèle
        predicted_tags = model.predict(input_data)
        print("donne moi les tags predits:", predicted_tags)
        
        # Convertir les prédictions en labels lisibles
        if predicted_tags.ndim == 2:  # Si le modèle renvoie une matrice binaire
            predicted_tags = mlb.inverse_transform(predicted_tags)
        else:
            predicted_tags = mlb.inverse_transform(predicted_tags.reshape(1, -1))
        
        # Convertir les prédictions en liste de chaînes
        tags_list = [f"#{tag}" for sublist in predicted_tags for tag in sublist]
        
        # Renvoyer les tags comme réponse
        return TagsResponse(tags=tags_list)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
