import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from PIL import Image
import numpy as np
import pandas as pd
import io
from datetime import datetime

st.set_page_config(page_title="Batch Classification", layout="wide")

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
    predictions = model.predict(processed_image, verbose=0)
    decoded_predictions = decode_predictions(predictions, top=5)[0]
    
    return decoded_predictions

st.title("📦 Batch Image Classification")
st.write("Upload multiple images and classify them all at once!")

st.markdown("---")

uploaded_files = st.file_uploader(
    "Choose images...",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    help="Upload multiple JPG, JPEG, or PNG images"
)

if uploaded_files:
    st.success(f"✅ {len(uploaded_files)} image(s) uploaded")
    
    if st.button("🚀 Classify All Images", type="primary"):
        model = load_model()
        
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, uploaded_file in enumerate(uploaded_files):
            try:
                status_text.text(f"Processing {idx + 1}/{len(uploaded_files)}: {uploaded_file.name}")
                
                image = Image.open(uploaded_file)
                predictions = classify_image(model, image)
                
                top_prediction = predictions[0]
                imagenet_id, label, score = top_prediction
                
                results.append({
                    'Filename': uploaded_file.name,
                    'Top Prediction': label.replace('_', ' ').title(),
                    'Confidence': f"{score * 100:.2f}%",
                    'Score': score
                })
                
            except Exception as e:
                results.append({
                    'Filename': uploaded_file.name,
                    'Top Prediction': 'Error',
                    'Confidence': str(e),
                    'Score': 0
                })
            
            progress_bar.progress((idx + 1) / len(uploaded_files))
        
        status_text.text("✅ Classification complete!")
        
        st.markdown("---")
        st.subheader("📊 Results")
        
        df = pd.DataFrame(results)
        df_display = df.drop('Score', axis=1)
        st.dataframe(df_display, use_container_width=True)
        
        csv = df_display.to_csv(index=False)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name=f"classification_results_{timestamp}.csv",
            mime="text/csv"
        )
        
        st.markdown("---")
        st.subheader("🖼️ Image Gallery with Predictions")
        
        cols_per_row = 3
        for idx in range(0, len(uploaded_files), cols_per_row):
            cols = st.columns(cols_per_row)
            
            for col_idx, col in enumerate(cols):
                file_idx = idx + col_idx
                if file_idx < len(uploaded_files):
                    with col:
                        image = Image.open(uploaded_files[file_idx])
                        st.image(image, use_container_width=True)
                        
                        result = results[file_idx]
                        if result['Top Prediction'] != 'Error':
                            st.write(f"**{result['Top Prediction']}**")
                            st.write(f"Confidence: {result['Confidence']}")
                        else:
                            st.error(f"Error: {result['Confidence']}")

else:
    st.info("👆 Upload multiple images above to classify them in batch!")

st.markdown("---")
st.caption("Built with TensorFlow, Keras, and Streamlit")
