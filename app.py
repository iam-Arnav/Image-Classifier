import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from PIL import Image
import numpy as np

st.set_page_config(page_title="Image Classifier", layout="wide")

@st.cache_resource
def load_model():
    """Load the pre-trained MobileNetV2 model"""
    model = MobileNetV2(weights='imagenet')
    return model

def prepare_image(image):
    """Prepare image for classification"""
    img = image.convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    return img_array

def classify_image(model, image):
    """Classify the uploaded image"""
    processed_image = prepare_image(image)
    predictions = model.predict(processed_image)
    decoded_predictions = decode_predictions(predictions, top=5)[0]
    
    return decoded_predictions

st.title("🖼️ Image Classification with Machine Learning")
st.write("Upload an image and let the AI classify what it sees!")

st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"],
        help="Upload a JPG, JPEG, or PNG image"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

with col2:
    st.subheader("Classification Results")
    
    if uploaded_file is not None:
        try:
            with st.spinner("Analyzing image..."):
                model = load_model()
                predictions = classify_image(model, image)
                
                st.success("Classification complete!")
                st.write("**Top 5 Predictions:**")
                
                for i, (imagenet_id, label, score) in enumerate(predictions, 1):
                    confidence = score * 100
                    st.write(f"{i}. **{label.replace('_', ' ').title()}** - {confidence:.2f}%")
                    st.progress(float(score))
                    
                st.markdown("---")
                st.info("This model uses MobileNetV2 trained on ImageNet dataset with 1000+ categories.")
        except Exception as e:
            st.error(f"Error classifying image: {str(e)}")
            st.info("Please try uploading a different image.")
    else:
        st.info("Upload an image to see the classification results here.")

st.markdown("---")
st.caption("Built with TensorFlow, Keras, and Streamlit")
