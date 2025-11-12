import streamlit as st
import tensorflow as tf
from tensorflow import keras
import json
import os
import shutil
import zipfile
from datetime import datetime
from io import BytesIO

st.set_page_config(page_title="Model Management", layout="wide")

st.title("💾 Model Management")
st.write("Export and import your trained models for reusability!")

st.markdown("---")

models_dir = "trained_models"

tab1, tab2, tab3 = st.tabs(["📤 Export Model", "📥 Import Model", "🗂️ Manage Models"])

with tab1:
    st.subheader("📤 Export Model")
    st.write("Download your trained model as a ZIP file for use in other projects or systems.")
    
    if os.path.exists(models_dir) and os.listdir(models_dir):
        available_models = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
        
        selected_model = st.selectbox(
            "Select a model to export",
            available_models,
            format_func=lambda x: x.replace('custom_model_', 'Model '),
            key="export_select"
        )
        
        if selected_model:
            model_path = os.path.join(models_dir, selected_model)
            metadata_path = os.path.join(model_path, 'metadata.json')
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                st.markdown("**Model Details:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"Classes: {metadata['num_classes']}")
                with col2:
                    st.write(f"Accuracy: {metadata['final_val_accuracy']:.2%}")
                with col3:
                    st.write(f"Created: {metadata['created_at']}")
                
                if st.button("📦 Create Export Package", type="primary"):
                    try:
                        with st.spinner("Creating export package..."):
                            zip_buffer = BytesIO()
                            
                            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                                for root, dirs, files in os.walk(model_path):
                                    for file in files:
                                        file_path = os.path.join(root, file)
                                        arcname = os.path.relpath(file_path, models_dir)
                                        zip_file.write(file_path, arcname)
                        
                        zip_buffer.seek(0)
                        
                        st.success("✅ Export package created!")
                        
                        st.download_button(
                            label="📥 Download Model Package",
                            data=zip_buffer.getvalue(),
                            file_name=f"{selected_model}.zip",
                            mime="application/zip"
                        )
                        
                    except Exception as e:
                        st.error(f"Error creating export: {str(e)}")
    else:
        st.info("No trained models found. Train a model first!")

with tab2:
    st.subheader("📥 Import Model")
    st.write("Upload a previously exported model to use it again.")
    
    uploaded_model = st.file_uploader(
        "Upload Model Package (ZIP file)",
        type=["zip"],
        help="Upload a ZIP file containing an exported model",
        key="import_upload"
    )
    
    if uploaded_model is not None:
        if st.button("📦 Import Model", type="primary"):
            try:
                os.makedirs(models_dir, exist_ok=True)
                
                temp_extract = "temp_import"
                os.makedirs(temp_extract, exist_ok=True)
                
                with st.spinner("Extracting model..."):
                    with zipfile.ZipFile(uploaded_model, 'r') as zip_ref:
                        zip_ref.extractall(temp_extract)
                
                extracted_items = os.listdir(temp_extract)
                
                if len(extracted_items) == 1 and os.path.isdir(os.path.join(temp_extract, extracted_items[0])):
                    model_folder = extracted_items[0]
                    source_path = os.path.join(temp_extract, model_folder)
                else:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    model_folder = f"imported_model_{timestamp}"
                    source_path = temp_extract
                
                dest_path = os.path.join(models_dir, model_folder)
                
                if os.path.exists(dest_path):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    model_folder = f"{model_folder}_{timestamp}"
                    dest_path = os.path.join(models_dir, model_folder)
                
                shutil.move(source_path, dest_path)
                
                metadata_path = os.path.join(dest_path, 'metadata.json')
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    st.success(f"✅ Model imported successfully as: {model_folder}")
                    
                    st.markdown("**Imported Model Details:**")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"Classes: {metadata.get('num_classes', 'N/A')}")
                    with col2:
                        st.write(f"Accuracy: {metadata.get('final_val_accuracy', 0):.2%}")
                    with col3:
                        st.write(f"Original Date: {metadata.get('created_at', 'N/A')}")
                else:
                    st.warning("Model imported but metadata not found. The model may still work.")
                
                if os.path.exists(temp_extract):
                    shutil.rmtree(temp_extract)
                    
            except Exception as e:
                st.error(f"Error importing model: {str(e)}")
                if os.path.exists(temp_extract):
                    shutil.rmtree(temp_extract)

with tab3:
    st.subheader("🗂️ Manage Models")
    st.write("View and manage all your trained models.")
    
    if os.path.exists(models_dir) and os.listdir(models_dir):
        available_models = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
        
        models_info = []
        
        for model_name in available_models:
            model_path = os.path.join(models_dir, model_name)
            metadata_path = os.path.join(model_path, 'metadata.json')
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                model_size = sum(
                    os.path.getsize(os.path.join(dirpath, filename))
                    for dirpath, dirnames, filenames in os.walk(model_path)
                    for filename in filenames
                ) / (1024 * 1024)
                
                models_info.append({
                    'Model Name': model_name.replace('custom_model_', 'Model '),
                    'Classes': metadata.get('num_classes', 'N/A'),
                    'Accuracy': f"{metadata.get('final_val_accuracy', 0):.2%}",
                    'Size (MB)': f"{model_size:.2f}",
                    'Created': metadata.get('created_at', 'N/A'),
                    '_path': model_name
                })
        
        if models_info:
            import pandas as pd
            df = pd.DataFrame(models_info)
            display_df = df.drop('_path', axis=1)
            st.dataframe(display_df, use_container_width=True)
            
            st.markdown("---")
            st.subheader("🗑️ Delete Model")
            
            delete_model = st.selectbox(
                "Select a model to delete",
                available_models,
                format_func=lambda x: x.replace('custom_model_', 'Model '),
                key="delete_select"
            )
            
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("🗑️ Delete", type="secondary"):
                    try:
                        model_path = os.path.join(models_dir, delete_model)
                        shutil.rmtree(model_path)
                        st.success(f"✅ Model deleted: {delete_model}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error deleting model: {str(e)}")
        else:
            st.info("No model metadata found.")
    else:
        st.info("No trained models found. Train a model first!")

st.markdown("---")
st.caption("Built with TensorFlow, Keras, and Streamlit")
