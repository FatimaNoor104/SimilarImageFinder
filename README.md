# Image Similarity Finder

Image Similarity Finder is a **Streamlit-based web application** that identifies the top 5 visually similar images from a dataset to a user-uploaded query image. It leverages a **pre-trained VGG16 deep learning model** to extract image features and compute similarity using **cosine similarity**.

---

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Technologies Used](#technologies-used)
4. [How It Works](#how-it-works)
5. [Application Walkthrough](#application-walkthrough)

---

## Overview
This application is designed to find images visually similar to a user-provided query image. It downloads images from a CSV file containing URLs, extracts image features using a **VGG16 model**, and saves them for efficient similarity searches.

**Key Benefits**:
- Handles pre-uploaded datasets of images.
- Real-time upload and similarity comparison.
- Optimized to extract features only once for performance.

---

## Features
- **Image Downloading**: Automatically downloads images from a provided CSV containing image links.
- **Feature Extraction**: Extracts features using the VGG16 model for dataset images.
- **Similarity Search**: Finds the top 5 most similar images to a user-uploaded query image.
- **Dynamic Updates**: Handles multiple uploads, displaying feature extraction status accordingly.

---

## Technologies Used
The project uses the following technologies and libraries:
- **Python**: Primary programming language.
- **TensorFlow/Keras**: For using the pre-trained VGG16 deep learning model.
- **Streamlit**: For creating an interactive web interface.
- **Pandas**: For handling CSV data.
- **NumPy**: For numerical computations.
- **scikit-learn**: For cosine similarity calculations.
- **Pillow (PIL)**: For image handling and display.
- **Requests**: For downloading images from URLs.

---

## How It Works

1. **Dataset Preparation**:
   - The app reads a CSV file containing image URLs and product IDs.
   - It downloads the images and saves them to a folder.

2. **Feature Extraction**:
   - Extracts features from dataset images using the VGG16 model (without its top layer).
   - The extracted features are stored for efficient comparison.

3. **Query Image Upload**:
   - Users can upload a query image via the interface.
   - Features for the uploaded image are extracted dynamically.

4. **Similarity Calculation**:
   - The app calculates the cosine similarity between the query image's features and the dataset images.
   - The top 5 similar images are displayed on the interface.

---

## Application Walkthrough

### Initial View
When the app is run for the first time:
- It reads the dataset, downloads images, and extracts their features.
![Initial App View](https://github.com/user-attachments/assets/d097aaa7-fcf2-4883-acdf-c64a4694bce8)

- Once the features are successfully extracted, user can upload an image to find the top 5 similar images from the dataset.
  
![App View](https://github.com/user-attachments/assets/3fd81f22-edc1-4d44-9897-c253256b38ec)


### Query Image Upload
After uploading the query image, the app displays similar images from the dataset.
![Query image 1](https://github.com/user-attachments/assets/9838c274-42bf-41ba-9280-77435e7ae523)
![Similar images 1 2 ](https://github.com/user-attachments/assets/e0eac3d7-7dc5-460c-944e-59c543235de8)
![Similar images 3 4](https://github.com/user-attachments/assets/59ba720e-941b-4a53-9154-bcae4e5f7a10)
![Similar image 5](https://github.com/user-attachments/assets/b8b111af-4ca7-4538-8091-f4a35af088ec)

### Subsequent Query Image Upload
For subsequent uploads, the app skips redundant steps and displays similar images for the newly uploaded query.

![Query image 2](https://github.com/user-attachments/assets/0657b99e-78d8-4a50-849a-f0e7109b4cdb)
![SI 1 2](https://github.com/user-attachments/assets/bf4c7fca-3d41-49c1-a937-45cc48f5fa1b)
![S1 3 4](https://github.com/user-attachments/assets/dc723940-debb-4415-bb8f-455dd76aa102)
![SI 5](https://github.com/user-attachments/assets/43ad949f-5116-4fb7-b249-3fb85022d428)


