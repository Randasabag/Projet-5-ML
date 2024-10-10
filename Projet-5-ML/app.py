import streamlit as st
import requests


# Titre de l'application
st.title("Générateur de Tags")

# Champ de saisie pour la question
question = st.text_input("Posez votre question:")

# Si une question est posée, générer des tags
if st.button("Générer des tags"):

    cloud_path = "https://randaals.pythonanywhere.com/generate_tags" 
    local_path = 'http://127.0.0.1:8000/generate_tags'
    response = requests.post(cloud_path, json={"question": question})

     # Vérifier le statut de la réponse
    if response.status_code != 200:
        st.error(f"Erreur: {response.status_code} - {response.text}")
    else:
        data = response.json()
        print("response",response.text)

        # Afficher les hashtags
        if 'tags' in data:
                    st.subheader("Tags générés :")
                    tags = data['tags']
                    for tag in tags:
                        hashtag = f"{tag}"
                        st.markdown(f'<span style="color: #007BFF;">{hashtag}</span>', unsafe_allow_html=True)
        else:
                    st.error("Les tags ne sont pas dans le format attendu.")
