import streamlit as st
import requests


# Titre de l'application
st.title("Générateur de Tags")


# Champ de saisie pour la question
question = st.text_input("Posez votre question:")


# Si une question est posée, générer des tags
if st.button("Générer des tags"):

   
    response = requests.post('http://127.0.0.1:8000/generate_tags', json={"question": question})
    data = response.json()

    # Afficher les hashtags
    if 'tags' in data:
                st.subheader("Tags générés :")
                tags = data['tags']
                for tag in tags:
                    hashtag = f"#{tag}"
                    st.markdown(f'<span style="color: #007BFF;">{hashtag}</span>', unsafe_allow_html=True)
    else:
                st.error("Les tags ne sont pas dans le format attendu.")
