import streamlit as st
import tensorflow as tf
from tensorflow import keras
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image
import zipfile
import shutil

st.set_page_config(page_title="Model Performance", layout="wide")

st.title("📊 Model Performance Dashboard")
st.write("Evaluate your trained models with detailed metrics and visualizations!")

st.markdown("---")

models_dir = "trained_models"

if os.path.exists(models_dir) and os.listdir(models_dir):
    available_models = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
    
    selected_model = st.selectbox(
        "Select a model to evaluate",
        available_models,
        format_func=lambda x: x.replace('custom_model_', 'Model ')
    )
    
    if selected_model:
        model_path = os.path.join(models_dir, selected_model)
        metadata_path = os.path.join(model_path, 'metadata.json')
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            st.markdown("---")
            st.subheader("📋 Model Information")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Number of Classes", metadata['num_classes'])
            with col2:
                st.metric("Image Size", f"{metadata['img_size']}x{metadata['img_size']}")
            with col3:
                st.metric("Training Epochs", metadata['epochs'])
            with col4:
                st.metric("Created", metadata['created_at'])
            
            st.markdown("**Classes:**")
            st.write(", ".join(metadata['classes']))
            
            st.markdown("---")
            st.subheader("📈 Training Performance")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "Training Accuracy",
                    f"{metadata['final_accuracy']:.2%}",
                    delta=None
                )
            
            with col2:
                st.metric(
                    "Validation Accuracy",
                    f"{metadata['final_val_accuracy']:.2%}",
                    delta=None
                )
            
            st.markdown("---")
            st.subheader("🧪 Test on New Data")
            
            st.info("""
            Upload a test dataset (ZIP file) to evaluate the model on unseen data.
            The ZIP should contain folders with class names matching the training classes.
            """)
            
            test_zip = st.file_uploader(
                "Upload Test Dataset (ZIP file)",
                type=["zip"],
                help="Upload a ZIP file with the same structure as training data"
            )
            
            if test_zip is not None:
                if st.button("🔍 Evaluate Model", type="primary"):
                    try:
                        temp_test_dir = "temp_test_dataset"
                        os.makedirs(temp_test_dir, exist_ok=True)
                        
                        with st.spinner("Extracting test dataset..."):
                            with zipfile.ZipFile(test_zip, 'r') as zip_ref:
                                zip_ref.extractall(temp_test_dir)
                        
                        test_path = temp_test_dir
                        subdirs = [d for d in os.listdir(temp_test_dir) if os.path.isdir(os.path.join(temp_test_dir, d))]
                        if len(subdirs) == 1:
                            test_path = os.path.join(temp_test_dir, subdirs[0])
                        
                        with st.spinner("Loading model..."):
                            model = keras.models.load_model(model_path)
                        
                        test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
                        
                        test_generator = test_datagen.flow_from_directory(
                            test_path,
                            target_size=(metadata['img_size'], metadata['img_size']),
                            batch_size=32,
                            class_mode='categorical',
                            shuffle=False
                        )
                        
                        st.info(f"📊 Test samples: {test_generator.samples}")
                        
                        with st.spinner("Evaluating model..."):
                            predictions = model.predict(test_generator, verbose=0)
                            predicted_classes = np.argmax(predictions, axis=1)
                            true_classes = test_generator.classes
                            class_labels = list(test_generator.class_indices.keys())
                        
                        test_loss, test_accuracy = model.evaluate(test_generator, verbose=0)
                        
                        st.markdown("---")
                        st.subheader("📊 Evaluation Results")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Test Accuracy", f"{test_accuracy:.2%}")
                        with col2:
                            st.metric("Test Loss", f"{test_loss:.4f}")
                        
                        st.markdown("---")
                        st.subheader("🔢 Confusion Matrix")
                        
                        cm = confusion_matrix(true_classes, predicted_classes)
                        
                        fig, ax = plt.subplots(figsize=(10, 8))
                        sns.heatmap(
                            cm,
                            annot=True,
                            fmt='d',
                            cmap='Blues',
                            xticklabels=class_labels,
                            yticklabels=class_labels,
                            ax=ax
                        )
                        ax.set_xlabel('Predicted')
                        ax.set_ylabel('Actual')
                        ax.set_title('Confusion Matrix')
                        
                        st.pyplot(fig)
                        
                        st.markdown("---")
                        st.subheader("📋 Classification Report")
                        
                        report = classification_report(
                            true_classes,
                            predicted_classes,
                            target_names=class_labels,
                            output_dict=True
                        )
                        
                        report_df = pd.DataFrame(report).transpose()
                        st.dataframe(report_df.round(4), use_container_width=True)
                        
                        shutil.rmtree(temp_test_dir)
                        
                    except Exception as e:
                        st.error(f"Error during evaluation: {str(e)}")
                        if os.path.exists(temp_test_dir):
                            shutil.rmtree(temp_test_dir)
        
        else:
            st.warning("No metadata found for this model")

else:
    st.info("No trained models found. Train a model first using the 'Train Model' page!")

st.markdown("---")
st.caption("Built with TensorFlow, Keras, and Streamlit")
