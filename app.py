import streamlit as st
import os
import re
import numpy as np
import rasterio
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tempfile import NamedTemporaryFile

st.set_page_config(page_title="Future Prediction Urban Expansion", layout="wide")
st.title("Urban Built-up Prediction Web App")

@st.cache_resource
def load_model(model_path):
    return tf.keras.models.load_model(model_path)

model = load_model("unet_lulc_model.h5")

def extract_year_from_filename(filename):
    match = re.search(r'(\d{4})', os.path.basename(filename))
    return int(match.group(1)) if match else -1

def save_uploaded_tiff(uploaded_file):
    temp = NamedTemporaryFile(delete=False, suffix=".tif")
    temp.write(uploaded_file.read())
    temp.flush()
    return temp.name

def preprocess_tiff(tiff_path, target_shape=(256, 256)):
    with rasterio.open(tiff_path) as src:
        data = src.read(1)
        original_shape = data.shape
        resized = cv2.resize(data, target_shape, interpolation=cv2.INTER_LINEAR)
        input_tensor = np.expand_dims(resized, axis=(0, -1))
    return input_tensor, original_shape, data

def predict_image(model, input_tensor):
    prediction = model.predict(input_tensor)[0, ..., 0]
    return prediction

def resize_back(prediction, original_shape):
    return cv2.resize(prediction, original_shape[::-1], interpolation=cv2.INTER_LINEAR)

def classify_binary(prediction, threshold=0.5):
    return np.where(prediction >= threshold, 1, 0)

def visualize_results(input1, input2, predicted, binary):
    cmap_binary = plt.cm.get_cmap('tab10', 2)
    fig, axs = plt.subplots(1, 4, figsize=(20, 6))

    axs[0].imshow(input1, cmap='gray')
    axs[0].set_title("Input TIFF 1")
    axs[0].axis('off')

    axs[1].imshow(input2, cmap='gray')
    axs[1].set_title("Input TIFF 2")
    axs[1].axis('off')

    im2 = axs[2].imshow(predicted, cmap='viridis')
    axs[2].set_title("Predicted Probability")
    axs[2].axis('off')
    fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

    im3 = axs[3].imshow(binary, cmap=cmap_binary, vmin=0, vmax=1)
    axs[3].set_title("Binary Classification")
    axs[3].axis('off')
    cbar = fig.colorbar(im3, ax=axs[3], ticks=[0, 1], fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels(['Non Built-up', 'Built-up'])

    st.pyplot(fig)

st.subheader("Upload Two Binary LULC Raster TIFFs (Built-up=1, Non Built-up=0)")
input1 = st.file_uploader("Upload earlier year TIFF", type=["tif", "tiff"], key="tiff1")
input2 = st.file_uploader("Upload recent year TIFF", type=["tif", "tiff"], key="tiff2")

if input1 and input2:
    path1 = save_uploaded_tiff(input1)
    path2 = save_uploaded_tiff(input2)

    input_tensor2, original_shape, img2 = preprocess_tiff(path2)
    _, _, img1 = preprocess_tiff(path1)

    st.subheader("Preview Inputs")
    col1, col2 = st.columns(2)
    with col1:
        st.image(img1, caption="Input TIFF 1", use_column_width=True, clamp=True)
    with col2:
        st.image(img2, caption="Input TIFF 2", use_column_width=True, clamp=True)

    year = st.number_input("Enter Target Year for Prediction (e.g., 2030)", min_value=2024, max_value=2100, value=2030)

    if st.button("Run Prediction"):
        prediction = predict_image(model, input_tensor2)
        prediction_resized = resize_back(prediction, original_shape)
        binary_class = classify_binary(prediction_resized)

        st.subheader(f"Predicted Built-up/Non Built-up Areas for {year}")
        visualize_results(img1, img2, prediction_resized, binary_class)

        np.save("predicted_binary.npy", binary_class)
        st.download_button("Download Binary Prediction (.npy)", data=open("predicted_binary.npy", "rb"), file_name=f"binary_builtup_{year}.npy")
