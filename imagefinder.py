import os
import pandas as pd
import numpy as np
import requests
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import Model
from PIL import Image
import streamlit as st

# Load pre-trained VGG16 model (without the top layer)
@st.cache_resource  # Cache the model to avoid reloading
def load_model():
    vgg16_model = VGG16(weights='imagenet', include_top=False)
    return Model(inputs=vgg16_model.input, outputs=vgg16_model.output)

model = load_model()

# Function to prepare an image
def prepare_image(image_path, target_size=(224, 224)):
    image = load_img(image_path, target_size=target_size)
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    return preprocess_input(image_array)

# Function to extract features using VGG16
def extract_features(image_path):
    preprocessed_image = prepare_image(image_path)
    features = model.predict(preprocessed_image)
    return features.flatten()

# Function to extract features from images in a folder
def extract_features_from_folder(folder_path, max_images=None):
    feature_list = []
    image_names = []
    
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        if image_name.lower().endswith(('png', 'jpg', 'jpeg')):
            features = extract_features(image_path)
            feature_list.append(features)
            image_names.append(image_name)
        if max_images and len(image_names) >= max_images:
            break
    
    return feature_list, image_names

# Function to find the top similar images
def find_top_similar_images(query_image, feature_list, image_names, folder_path, top_n=5):
    query_features = extract_features(query_image)
    similarities = []
    
    for i, features in enumerate(feature_list):
        similarity = cosine_similarity([query_features], [features])[0][0]
        similarities.append((image_names[i], similarity))
    
    # Sort by similarity score in descending order
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    
    # Get the paths of the top similar images
    similar_images = []
    for image_name, _ in similarities[:top_n]:
        similar_images.append(os.path.join(folder_path, image_name))
    
    return similar_images

# Streamlit app
def main():
    st.title("Interactive Image Similarity Finder")
    
    # Step 1: Download and extract features once
    st.subheader("Step 1: Preparing Data")
    output_dir = "downloaded_images"
    csv_path = "Data ID - Sheet1.csv"  # Replace with your actual file path
    
    if not os.path.exists(csv_path):
        st.error("CSV file not found! Please ensure the CSV file is available.")
        return
    
    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)
    
    if len(os.listdir(output_dir)) == 0:
        st.write("Downloading images...")
        for _, row in df.iterrows():
            image_url = row['image_link']
            response = requests.get(image_url, stream=True)
            if response.status_code == 200:
                output_path = os.path.join(output_dir, f"{row['Product ID']}.jpg")
                with open(output_path, "wb") as file:
                    file.write(response.content)
        st.success("Images downloaded successfully!")
    else:
        st.info("Images are already downloaded.")
    
    # Extract features once and cache them
    if "feature_list" not in st.session_state:
        st.write("Extracting features...")
        feature_list, image_names = extract_features_from_folder(output_dir)
        st.session_state.feature_list = feature_list
        st.session_state.image_names = image_names
        st.success("Features extracted successfully!")

    # Step 2: Interactive Image Upload and Similarity Search
    st.subheader("Step 2: Upload Query Image")
    uploaded_query_image = st.file_uploader("Upload an image to find similar images", type=["png", "jpg", "jpeg"])

    # Keep track of uploaded images and their results
    if "uploaded_images" not in st.session_state:
        st.session_state.uploaded_images = []
        st.session_state.results = []

    if uploaded_query_image:
        # Save and process uploaded image
        query_image_path = os.path.join("temp", uploaded_query_image.name)
        os.makedirs("temp", exist_ok=True)
        with open(query_image_path, "wb") as f:
            f.write(uploaded_query_image.getbuffer())
        
        # Find top similar images
        st.write("Finding similar images...")
        top_similar_images = find_top_similar_images(
            query_image=query_image_path,
            feature_list=st.session_state.feature_list,
            image_names=st.session_state.image_names,
            folder_path=output_dir
        )
        st.session_state.uploaded_images.append(uploaded_query_image.name)
        st.session_state.results.append(top_similar_images)
    
    # Display uploaded images and results
    for idx, (image_name, result_images) in enumerate(zip(st.session_state.uploaded_images, st.session_state.results)):
        with st.container():
            st.image(os.path.join("temp", image_name), caption=f"Uploaded Image {idx + 1}", use_column_width=True)
            st.write("Top Similar Images:")
            for result_image in result_images:
                st.image(result_image, use_column_width=True)
            if st.button(f"Remove Image {idx + 1}"):
                st.session_state.uploaded_images.pop(idx)
                st.session_state.results.pop(idx)
                break  # Rerun to refresh UI

if __name__ == "__main__":
    main()
