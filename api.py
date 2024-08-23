import streamlit as st
import joblib
import tensorflow_hub as hub

# Charger le modèle enregistré
model = joblib.load("use_model.pkl")
mlb = joblib.load("mlb.pkl")  # Charger le MultiLabelBinarizer
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")


# Titre de l'application
st.title("Générateur de Tags")

# Champ de saisie pour la question
question = st.text_input("Posez votre question:")

# Si une question est posée, générer des tags
if st.button("Générer des tags"):
    if question:
        # Prétraitement si nécessaire
        input_data = embed([question]) # Ajuster selon le modèle
        
        # Prédire les tags avec le modèle
        predicted_tags = model.predict(input_data)
        
        # Convertir les prédictions en tags lisibles
        if predicted_tags.ndim == 2:  # Si les résultats sont binaires (multilabel)
            predicted_tags = mlb.inverse_transform(predicted_tags)
        else:
            predicted_tags = mlb.inverse_transform(predicted_tags.reshape(1, -1))
        
        # Afficher les tags générés
        st.write("Tags générés :", predicted_tags[0])
    else:
        st.write("Veuillez entrer une question.")
