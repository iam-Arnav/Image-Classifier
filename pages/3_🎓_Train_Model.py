import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import numpy as np
import zipfile
import os
import shutil
import json
from datetime import datetime

st.set_page_config(page_title="Train Custom Model", layout="wide")

st.title("🎓 Train Custom Model")
st.write("Upload your labeled dataset and train a custom image classification model!")

st.markdown("---")

st.info("""
**Dataset Format Requirements:**
- Upload a ZIP file containing folders
- Each folder name should be a class label
- Put images of that class inside the corresponding folder

Example structure:
```
dataset.zip
├── cats/
│   ├── cat1.jpg
│   ├── cat2.jpg
├── dogs/
│   ├── dog1.jpg
│   ├── dog2.jpg
```
""")

uploaded_zip = st.file_uploader(
    "Upload Dataset (ZIP file)",
    type=["zip"],
    help="Upload a ZIP file with labeled image folders"
)

if uploaded_zip is not None:
    st.markdown("---")
    st.subheader("📋 Training Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        epochs = st.slider("Number of Epochs", min_value=1, max_value=50, value=10)
        batch_size = st.selectbox("Batch Size", [8, 16, 32], index=1)
        img_size = st.selectbox("Image Size", [128, 224], index=1)
    
    with col2:
        use_augmentation = st.checkbox("Enable Data Augmentation", value=True)
        if use_augmentation:
            st.write("**Augmentation Options:**")
            rotation = st.slider("Rotation Range (degrees)", 0, 40, 20)
            zoom = st.slider("Zoom Range", 0.0, 0.3, 0.2)
            flip = st.checkbox("Horizontal Flip", value=True)
    
    if st.button("🚀 Start Training", type="primary"):
        try:
            temp_dir = "temp_dataset"
            models_dir = "trained_models"
            
            os.makedirs(temp_dir, exist_ok=True)
            os.makedirs(models_dir, exist_ok=True)
            
            with st.spinner("Extracting dataset..."):
                with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
            
            dataset_path = temp_dir
            subdirs = [d for d in os.listdir(temp_dir) if os.path.isdir(os.path.join(temp_dir, d))]
            if len(subdirs) == 1:
                dataset_path = os.path.join(temp_dir, subdirs[0])
            
            classes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
            num_classes = len(classes)
            
            st.success(f"✅ Found {num_classes} classes: {', '.join(classes)}")
            
            total_images = sum([len(os.listdir(os.path.join(dataset_path, c))) for c in classes])
            st.info(f"📊 Total images: {total_images}")
            
            if use_augmentation:
                train_datagen = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=rotation,
                    zoom_range=zoom,
                    horizontal_flip=flip,
                    validation_split=0.2
                )
            else:
                train_datagen = ImageDataGenerator(
                    rescale=1./255,
                    validation_split=0.2
                )
            
            with st.spinner("Preparing data..."):
                train_generator = train_datagen.flow_from_directory(
                    dataset_path,
                    target_size=(img_size, img_size),
                    batch_size=batch_size,
                    class_mode='categorical',
                    subset='training'
                )
                
                val_generator = train_datagen.flow_from_directory(
                    dataset_path,
                    target_size=(img_size, img_size),
                    batch_size=batch_size,
                    class_mode='categorical',
                    subset='validation'
                )
            
            st.write(f"📦 Training samples: {train_generator.samples}")
            st.write(f"📦 Validation samples: {val_generator.samples}")
            
            with st.spinner("Building model..."):
                base_model = tf.keras.applications.MobileNetV2(
                    input_shape=(img_size, img_size, 3),
                    include_top=False,
                    weights='imagenet'
                )
                base_model.trainable = False
                
                model = keras.Sequential([
                    base_model,
                    layers.GlobalAveragePooling2D(),
                    layers.Dropout(0.2),
                    layers.Dense(num_classes, activation='softmax')
                ])
                
                model.compile(
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
            
            st.success("✅ Model built successfully!")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            metrics_placeholder = st.empty()
            
            class StreamlitCallback(keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    progress = (epoch + 1) / epochs
                    progress_bar.progress(progress)
                    status_text.text(f"Epoch {epoch + 1}/{epochs}")
                    
                    with metrics_placeholder.container():
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Train Loss", f"{logs['loss']:.4f}")
                        col2.metric("Train Accuracy", f"{logs['accuracy']:.4f}")
                        col3.metric("Val Loss", f"{logs['val_loss']:.4f}")
                        col4.metric("Val Accuracy", f"{logs['val_accuracy']:.4f}")
            
            with st.spinner("Training model..."):
                history = model.fit(
                    train_generator,
                    epochs=epochs,
                    validation_data=val_generator,
                    callbacks=[StreamlitCallback()],
                    verbose=0
                )
            
            status_text.text("✅ Training complete!")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"custom_model_{timestamp}"
            model_path = os.path.join(models_dir, model_name)
            
            model.save(model_path)
            
            metadata = {
                'model_name': model_name,
                'classes': classes,
                'num_classes': num_classes,
                'img_size': img_size,
                'epochs': epochs,
                'final_accuracy': float(history.history['accuracy'][-1]),
                'final_val_accuracy': float(history.history['val_accuracy'][-1]),
                'created_at': timestamp
            }
            
            with open(os.path.join(model_path, 'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            st.success(f"✅ Model saved as: {model_name}")
            
            st.markdown("---")
            st.subheader("📊 Training Results")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Final Training Accuracy", f"{history.history['accuracy'][-1]:.2%}")
            with col2:
                st.metric("Final Validation Accuracy", f"{history.history['val_accuracy'][-1]:.2%}")
            
            shutil.rmtree(temp_dir)
            
        except Exception as e:
            st.error(f"Error during training: {str(e)}")
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

else:
    st.info("👆 Upload a dataset ZIP file to begin training!")

st.markdown("---")
st.caption("Built with TensorFlow, Keras, and Streamlit")
