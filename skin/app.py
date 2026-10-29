import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Skin Cancer Classifier", layout="centered")

st.title("🩺 Skin Cancer Classification")
st.write("Upload an image of the skin lesion to predict its class using our trained CNN model.")

# 1. Load the model (cached so it only loads once)
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('skin_cancer_model.keras')

try:
    model = load_my_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.info("Make sure 'skin_cancer_model.keras' is in the same directory as this script.")

# 2. File Uploader
uploaded_file = st.file_uploader("Choose a skin image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # 3. Preprocess Image
    st.write("Processing and predicting...")
    
    # Convert image to RGB if grayscale, resize to match model inputs (128x128)
    img = image.convert("RGB")
    img = img.resize((128, 128))
    
    # Transform image to numpy array and normalize pixel values to [0, 1]
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    # Add batch dimension: shape becomes (1, 128, 128, 3)
    img_tensor = np.expand_dims(img_array, axis=0)
    
    # 4. Make prediction
    predictions = model.predict(img_tensor)
    confidence_scores = predictions[0]
    predicted_class_idx = np.argmax(confidence_scores)
    
    # Map indices back to Class names (adjust if you know the exact order of folders, e.g., Benign/Malignant)
    class_names = ["Class 0", "Class 1"]
    predicted_class = class_names[predicted_class_idx]
    confidence = confidence_scores[predicted_class_idx] * 100
    
    # 5. Display Results
    st.subheader("Prediction Results:")
    st.metric(label="Predicted Category", value=predicted_class)
    st.metric(label="Confidence Score", value=f"{confidence:.2f}%")
    
    # Detailed breakdown
    st.write("**Class Confidences:**")
    for i, name in enumerate(class_names):
        st.write(f"- {name}: {confidence_scores[i] * 100:.2f}%")
