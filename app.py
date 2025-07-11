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

def save_uploaded_tiff(uploaded_file):
    temp = NamedTemporaryFile(delete=False, suffix=".tif")
    temp.write(uploaded_file.read())
    temp.flush()
    return temp.name

def preprocess_tiff(tiff_path, target_shape=(256, 256)):
    with rasterio.open(tiff_path) as src:
        data = src.read(1)
        original_shape = data.shape
        profile = src.profile
        resized = cv2.resize(data, target_shape, interpolation=cv2.INTER_LINEAR)
        input_tensor = np.expand_dims(resized, axis=(0, -1))
    return input_tensor, original_shape, data, profile

def predict_image(model, input_tensor):
    prediction = model.predict(input_tensor)[0, ..., 0]
    return prediction

def resize_back(prediction, original_shape):
    return cv2.resize(prediction, original_shape[::-1], interpolation=cv2.INTER_LINEAR)

def classify_binary(prediction, threshold=0.5):
    return np.where(prediction >= threshold, 1, 0)

def save_as_tif(array, profile, output_path, dtype=rasterio.uint8):
    profile.update(dtype=dtype, count=1)
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(array.astype(dtype), 1)

def visualize_results(input2, predicted, binary):
    cmap_binary = plt.cm.get_cmap('tab10', 2)
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))

    axs[0].imshow(input2, cmap='gray')
    axs[0].set_title("Input TIFF")
    axs[0].axis('off')

    im2 = axs[1].imshow(predicted, cmap='viridis')
    axs[1].set_title("Predicted Probability")
    axs[1].axis('off')
    fig.colorbar(im2, ax=axs[1], fraction=0.046, pad=0.04)

    im3 = axs[2].imshow(binary, cmap=cmap_binary, vmin=0, vmax=1)
    axs[2].set_title("Binary Classification")
    axs[2].axis('off')
    cbar = fig.colorbar(im3, ax=axs[2], ticks=[0, 1], fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels(['Non Built-up', 'Built-up'])

    st.pyplot(fig)

st.subheader("Upload Two Binary LULC Raster TIFFs (Built-up=1, Non Built-up=0)")
input_files = st.file_uploader("Upload any two years of raster TIFF files", type=["tif", "tiff"], accept_multiple_files=True)

if input_files and len(input_files) == 2:
    path1 = save_uploaded_tiff(input_files[0])
    path2 = save_uploaded_tiff(input_files[1])

    input_tensor2, original_shape, img2, profile = preprocess_tiff(path2)

    year = st.number_input("Enter Target Year for Prediction (e.g., 2030)", min_value=2024, max_value=2100, value=2030)

    if st.button("Run Prediction"):
        prediction = predict_image(model, input_tensor2)
        prediction_resized = resize_back(prediction, original_shape)
        binary_class = classify_binary(prediction_resized)

        pred_tif_path = f"predicted_{year}.tif"
        binary_tif_path = f"binary_classified_{year}.tif"

        save_as_tif(prediction_resized, profile, pred_tif_path, dtype=rasterio.float32)
        save_as_tif(binary_class, profile, binary_tif_path, dtype=rasterio.uint8)

        st.subheader(f"Predicted Built-up/Non Built-up Areas for {year}")
        visualize_results(img2, prediction_resized, binary_class)

        with open(pred_tif_path, "rb") as f:
            st.download_button("Download Predicted TIFF", f, file_name=pred_tif_path)

        with open(binary_tif_path, "rb") as f:
            st.download_button("Download Binary Classified TIFF", f, file_name=binary_tif_path)

else:
    st.info("Please upload exactly two TIFF files.")
