import streamlit as st

st.set_page_config(
    page_title="Image Classification ML Platform",
    page_icon="🖼️",
    layout="wide"
)

st.title("🖼️ Image Classification ML Platform")
st.write("Welcome to the Image Classification Platform!")

st.markdown("---")

st.markdown("""
### Features

**🔍 Single Image Classification**
Upload a single image and get instant predictions using our pre-trained MobileNetV2 model trained on 1000+ ImageNet categories.

**📦 Batch Classification**
Process multiple images at once and download results in a convenient format.

**🎓 Train Custom Models**
Upload your own labeled dataset to train a custom image classification model tailored to your specific needs.

**📊 Model Performance**
View detailed performance metrics, confusion matrices, and accuracy scores for your trained models.

**💾 Model Management**
Export and import trained models for reusability across different sessions.

---

### Get Started

Use the sidebar to navigate between different features!
""")

st.info("👈 Select a feature from the sidebar to begin")

st.markdown("---")
st.caption("Built with TensorFlow, Keras, and Streamlit")
