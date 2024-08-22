import streamlit as st
import joblib

# Charger le modèle enregistré
model = joblib.load("use_model.pkl")

# Titre de l'application
st.title("Générateur de Tags")

# Champ de saisie pour la question
question = st.text_input("Posez votre question:")

# Si une question est posée, générer des tags
if st.button("Générer des tags"):
    if question:
        # Prétraitement si nécessaire
        input_data = [question]  # Ajuster selon le modèle
        
        # Prédire les tags avec le modèle
        predicted_tags = model.predict(input_data)
        
        # Afficher les tags
        st.write("Tags générés :", predicted_tags)
    else:
        st.write("Veuillez entrer une question.")
