import streamlit as st
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from transformers import MaskFormerFeatureExtractor, MaskFormerForInstanceSegmentation
import os

from huggingface_hub import login
# Access the Hugging Face token from secrets
hf_token = st.secrets["huggingface"]["api_token"]

# Log in to Hugging Face
login(token=hf_token)

# Set the page layout
st.set_page_config(page_title="Image Segmentation App", layout="wide")

# Load the model and feature extractor initially
@st.cache_resource
def load_model():
    feature_extractor = MaskFormerFeatureExtractor.from_pretrained("thiagohersan/maskformer-satellite-trees")
    model = MaskFormerForInstanceSegmentation.from_pretrained("thiagohersan/maskformer-satellite-trees")
    return feature_extractor, model

feature_extractor, model = load_model()

# Define preprocessing function
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((500, 1024)),  # Resize to (500, 1024)
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet normalization
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Define inference function
def predict(image):
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    masks = outputs.logits.argmax(dim=1)[0].numpy()  # Get predicted class masks
    return masks

# App Title
st.title("Image Segmentation App")

# Sidebar for sample images
st.sidebar.header("Sample Images")
sample_images_path = "./sample_images"  # Replace with your actual sample images directory
if not os.path.exists(sample_images_path):
    os.makedirs(sample_images_path)

sample_images = [f for f in os.listdir(sample_images_path) if f.endswith(('.png', '.jpeg', '.jpg'))]
if sample_images:
    st.sidebar.write("Drag and drop these sample images into the input window:")
    for img_name in sample_images:
        st.sidebar.image(os.path.join(sample_images_path, img_name), caption=img_name, use_column_width=True)
else:
    st.sidebar.write("No sample images found. Please add images to the 'sample_images' directory.")

# File upload and drag-and-drop box
uploaded_file = st.file_uploader("Upload your image or drag and drop", type=["png", "jpeg", "jpg"], accept_multiple_files=False)

# Main workflow
if uploaded_file is not None:
    try:
        # Load the image
        image = Image.open(uploaded_file).convert("RGB")

        # Preprocess the image
        preprocessed_image = preprocess_image(image)

        # Display the uploaded image
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Inference button
        if st.button("Run Segmentation"):
            with st.spinner("Processing..."):
                masks = predict(image)

                # Display the output mask
                st.write("Segmentation Result:")
                st.image(masks, caption="Predicted Segmentation", use_column_width=True, clamp=True)
    except Exception as e:
        st.error(f"Error processing the image: {e}")
else:
    st.write("Upload or drag and drop an image to start.")
