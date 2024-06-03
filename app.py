import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the pre-trained model (TensorFlow)
model = load_model('model.h5')

st.title('Cataract Detection App')

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Load and display image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess image
    img = image.resize((224, 224))  # assuming model expects 224x224 input
    img = np.array(img) / 255.0  # normalize to [0, 1]
    img = np.expand_dims(img, axis=0)  # add batch dimension

    # Predict
    predictions = model.predict(img)
    score = predictions[0]

    if score < 0.5:
        st.write("The model predicts that the image does not show cataract.")
    else:
        st.write("The model predicts that the image shows cataract.")
