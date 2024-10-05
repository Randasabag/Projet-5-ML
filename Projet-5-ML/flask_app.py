from flask import Flask, request, jsonify
import joblib
import pickle
#import tensorflow as tf
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
import os


app = Flask(__name__)

print("Flask utilisé")

@app.route("/generate_tags", methods=["POST"])
def generate_tags():

    my_dir = os.path.dirname(__file__)
    tfidf_path = os.path.join(my_dir, 'tfidf.joblib')
    classifier_path = os.path.join(my_dir, 'classifier.joblib')
    mlb_path = os.path.join(my_dir, 'mlb.joblib')

    # Charger le TfidfVectorizer
    with open(tfidf_path, 'rb') as file:
        tfidf = joblib.load(file)

    # Charger le classificateur
    with open(classifier_path, 'rb') as file1:
        classifier = joblib.load(file1)

    with open(mlb_path, 'rb') as file2:
        mlb = joblib.load(file2)  # Charger le MultiLabelBinarizer
    
    try:
        # Lire les données JSON de la requête
        data = request.json
        question = data.get("question")
        
        if not question:
            return jsonify({"error": "Question is required"}), 400
        
        # Vérification du type de la question
        if not isinstance(question, str):
            return jsonify({"error": "Question must be a string"}), 400
        
        # Prétraitement de la question si nécessaire
        input_data = (tfidf.transform([question]))  # Modifie ceci selon le format attendu par ton modèle
        # Faire une prédiction avec le modèle
        predicted_tags = classifier.predict(input_data)
        print("Pred",predicted_tags)

        # Exemple où tags_list est [['python']]
        tags_list = mlb.inverse_transform(predicted_tags)
        print("Tags list:", tags_list)

         # Vérifier si tags_list contient des éléments
        if not tags_list or not any(tags_list):
            return jsonify({"tags": []})  # Retourner une liste vide si aucun tag n'est trouvé
    
        formatted_tags = [f"#{tag}" for tag in tags_list[0]]  # Utiliser [0] pour accéder à la première sous-liste

        # Retourner les tags au format JSON
        return jsonify({'tags': formatted_tags})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
        

if __name__ == "__main__":
    app.run()
