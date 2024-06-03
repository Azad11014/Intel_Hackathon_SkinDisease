import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained model (TensorFlow)
model = load_model('model_weights.h5')

st.title('Skin Disease Cassification App')

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png"])
classes = ['Actinic keratosis', 'Atopic Dermatitis', 'Benign keratosis', 'Dermatofibroma', 'Melanocytic nevus', 'Melanoma', 'Squamous cell carcinoma', 'Tinea Ringworm Candidiasis', 'Vascular lesion']
if uploaded_file is not None:
    # Load and display image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', width=200)
    st.write("")
    st.write("Classifying...")

    # Preprocess image
    img = image.resize((200, 200))
    img = np.array(img) / 255.0  # normalize to [0, 1]
    img = np.expand_dims(img, axis=0)  # add batch dimension

    # Predict
    pred_proba = model.predict(img)
    pred_class = np.argmax(pred_proba, axis = 1).item()
    st.write(f"The detectd disease is: {classes[pred_class]}")