import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from matplotlib import figure
st.set_option('deprecation.showPyplotGlobalUse', False)

# Load the pre-trained model (TensorFlow)
model = keras.models.load_model('model_weights.h5')

#Extracting the significant feature extraction blocks in order to view some of the feature maps being generated
model_res1 = tf.keras.models.Model(inputs = model.input, outputs = model.get_layer("activation").output, name = "Filters_Res1")
model_res2 = tf.keras.models.Model(inputs = model.input, outputs = model.get_layer("activation_1").output, name = "Filters_Res2")
model_vgg3 = tf.keras.models.Model(inputs = model.input, outputs = model.get_layer("batch_normalization_4").output, name = "Filters_Vgg3")
model_res4 = tf.keras.models.Model(inputs = model.input, outputs = model.get_layer("activation_2").output, name = "Filters_Res4")
model_res5 = tf.keras.models.Model(inputs = model.input, outputs = model.get_layer("activation_3").output, name = "Filters_Res5")

st.title('Skin Disease Cassification App')
classes = ['Actinic keratosis', 'Atopic Dermatitis', 'Benign keratosis', 'Dermatofibroma', 'Melanocytic nevus', 'Melanoma', 'Squamous cell carcinoma', 'Tinea Ringworm Candidiasis', 'Vascular lesion']

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png"])

def displayFeatureMaps(feature_maps, figsize = (20,3)) -> figure.Figure:
    ix = 1
    fig = plt.figure(figsize=(20,3))
    fig.set_facecolor("#262730")
    fm_count = feature_maps.shape[-1]
    if(fm_count > 8):
        for _ in range(feature_maps.shape[-1]//(fm_count//2)):
            for _ in range((fm_count//2)):
                # Specify subplot and turn off axis
                ax = plt.subplot(feature_maps.shape[-1]//(fm_count//2), (fm_count//2), ix)
                ax.set_xticks([])
                ax.set_yticks([])
                # Plot filter channel in grayscale
                plt.imshow(feature_maps[0, :, :, ix-1], cmap='viridis')
                ix += 1
    else:
        ix = 1
        for _ in range(feature_maps.shape[-1]):
            ax = plt.subplot(1,fm_count,ix)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(feature_maps[0, :, :, ix-1], cmap='viridis')
            ix += 1 

    fig.tight_layout()
    return fig

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

    # Displaying the intermediate filters generated
    output_feature_maps = model_res1.predict(img)
    st.write("Visualizing the filters after  Residual Layer 1:")
    st.pyplot(displayFeatureMaps(output_feature_maps))

    output_feature_maps = model_res2.predict(img)
    st.write("Visualizing the filters after  Residual Layer 2:")
    st.pyplot(displayFeatureMaps(output_feature_maps))

    output_feature_maps = model_vgg3.predict(img)
    st.write("Visualizing the filters after  VGG Layer 3:")
    st.pyplot(displayFeatureMaps(output_feature_maps, figsize=(20,1)))

    output_feature_maps = model_res4.predict(img)
    st.write("Visualizing the filters after  Residual Layer 4:")
    st.pyplot(displayFeatureMaps(output_feature_maps, figsize=(20,1)))

    output_feature_maps = model_res5.predict(img)
    st.write("Visualizing the filters after  Residual Layer 5:")
    st.pyplot(displayFeatureMaps(output_feature_maps, figsize=(20,1)))

    # Predict
    pred_proba = model.predict(img)
    pred_class = np.argmax(pred_proba, axis = 1).item()
    st.warning(f"The detectd disease is: {classes[pred_class]}")