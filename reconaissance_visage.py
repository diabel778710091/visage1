import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Charger le classifieur de visages
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detecter_visages(image, couleur_rectangles, scaleFactor, minNeighbors):
    # Convertir l'image en niveaux de gris
    gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Détecter les visages à l'aide du classifieur de visages
    visages = face_cascade.detectMultiScale(gris, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
    # Dessiner des rectangles autour des visages détectés
    couleur = tuple(int(couleur_rectangles[i:i + 2], 16) for i in (1, 3, 5))
    for (x, y, w, h) in visages:
        cv2.rectangle(image, (x, y), (x + w, y + h), couleur, 2)
    return image

def app():
    st.title("Détection de Visages")
    st.write("Utilisez les options ci-dessous pour configurer l'application.")

    # Choisir la couleur des rectangles
    couleur_rectangles = st.color_picker('Choisissez la couleur des rectangles', '#00FF00')

    # Ajuster les paramètres de détection
    scaleFactor = st.slider('Ajuster le scaleFactor', min_value=1.1, max_value=2.0, value=1.3, step=0.1)
    minNeighbors = st.slider('Ajuster minNeighbors', min_value=1, max_value=10, value=5, step=1)

    # Utiliser le composant de Streamlit pour capturer une image depuis la webcam
    image_file = st.camera_input("Capturez une image")

    if image_file is not None:
        # Lire l'image à partir du fichier téléchargé
        image = np.array(bytearray(image_file.read()), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        # Détecter les visages
        image = detecter_visages(image, couleur_rectangles, scaleFactor, minNeighbors)

        # Convertir l'image en format compatible avec Streamlit
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image_rgb, channels="RGB", caption="Image avec Détection de Visages")

        # Bouton pour enregistrer l'image
        if st.button("Enregistrer l'image"):
            # Convertir l'image en format PIL pour l'enregistrement
            pil_image = Image.fromarray(image_rgb)
            pil_image.save('image_detection_visages.png')
            st.write("Image enregistrée sous 'image_detection_visages.png'")

if __name__ == "__main__":
    app()