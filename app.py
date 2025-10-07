import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import pandas as pd
import os
from pathlib import Path
import zipfile
import io

# -----------------------------------------------------------
# 🎨 Page Configuration
# -----------------------------------------------------------
st.set_page_config(
    page_title="😷 Face Mask Detection",
    page_icon="😷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------
# 🎨 Custom CSS for Beautiful UI
# -----------------------------------------------------------
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    div.stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem;
        border-radius: 10px;
        font-weight: bold;
        transition: transform 0.2s;
    }
    div.stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
    .upload-box {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        margin: 1rem 0;
    }
    .result-box {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.15);
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    h1, h2, h3 {
        color: white;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stTabs [aria-selected="true"] {
        background: white !important;
        color: #667eea !important;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# 🔧 Load Model
# -----------------------------------------------------------
@st.cache_resource
def load_trained_model():
    try:
        # Load .h5 model only
        if os.path.exists("face_mask_detector.h5"):
            model = load_model("face_mask_detector.h5")
            st.info("✅ Loaded .h5 model successfully")
            return model
        else:
            st.error("❌ Model file 'face_mask_detector.h5' not found.")
            return None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

# -----------------------------------------------------------
# 🔍 Prediction Function
# -----------------------------------------------------------
def predict_mask(img, model):
    """Process image and return prediction"""
    img_resized = cv2.resize(img, (100, 100))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_normalized = img_rgb / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)
    
    prediction = model.predict(img_input, verbose=0)[0][0]
    
    if prediction >= 0.5:
        label = "With Mask"
        emoji = "😷"
        confidence = float(prediction)
        color = "success"
    else:
        label = "Without Mask"
        emoji = "❌"
        confidence = 1 - float(prediction)
        color = "error"
    
    return {
        "label": label,
        "emoji": emoji,
        "confidence": confidence,
        "raw_score": float(prediction),
        "color": color,
        "img_rgb": img_rgb
    }

# -----------------------------------------------------------
# 📊 Display Single Result
# -----------------------------------------------------------
def display_single_result(result, filename="Uploaded Image"):
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(result["img_rgb"], caption=filename, use_column_width=True)
    
    with col2:
        st.markdown(f"### {result['emoji']} {result['label']}")
        st.progress(result["confidence"])
        st.metric("Confidence", f"{result['confidence']*100:.2f}%")
        
        if result["color"] == "success":
            st.success("✅ Mask Detected!")
        else:
            st.error("⚠️ No Mask Detected!")

# -----------------------------------------------------------
# 📂 Process Multiple Images
# -----------------------------------------------------------
def process_multiple_images(files, model):
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, file in enumerate(files):
        status_text.text(f"Processing {idx+1}/{len(files)}: {file.name}")
        
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        
        if img is not None:
            result = predict_mask(img, model)
            results.append({
                "Filename": file.name,
                "Prediction": result["label"],
                "Confidence": f"{result['confidence']*100:.2f}%",
                "Raw Score": result["raw_score"]
            })
        
        progress_bar.progress((idx + 1) / len(files))
    
    status_text.empty()
    progress_bar.empty()
    
    return results

# -----------------------------------------------------------
# 🚀 Main App
# -----------------------------------------------------------
def main():
    # Header
    st.markdown("<h1 style='text-align: center;'>😷 Face Mask Detection System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: white; font-size: 1.2rem;'>AI-Powered Mask Detection using Deep Learning</p>", unsafe_allow_html=True)
    
    # Load Model
    with st.spinner("🔄 Loading AI Model..."):
        model = load_trained_model()
    
    if model is None:
        st.error("⚠️ Failed to load model. Please ensure 'face_mask_detector.h5' exists in the same directory.")
        return
    
    st.success("✅ Model Loaded Successfully!")
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📋 About")
        st.info("""
        This app uses a CNN model to detect face masks in images.
        
        **Features:**
        - Single Image Detection
        - Batch Processing
        - CSV Export
        """)
        
        st.markdown("### 📊 Model Info")
        st.write("**Input Size:** 100x100")
        st.write("**Model Type:** CNN")
        st.write("**Classes:** With/Without Mask")
    
    # Main Content
    tab1, tab2, tab3 = st.tabs(["📷 Single Image", "📁 Multiple Images", "📊 CSV Upload"])
    
    # -----------------------------------------------------------
    # Tab 1: Single Image Upload
    # -----------------------------------------------------------
    with tab1:
        st.markdown("### Upload a Single Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png"],
            key="single"
        )
        
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            
            if img is not None:
                with st.spinner("🔍 Analyzing image..."):
                    result = predict_mask(img, model)
                
                st.markdown("---")
                display_single_result(result, uploaded_file.name)
            else:
                st.error("❌ Could not read the image. Please try another file.")
    
    # -----------------------------------------------------------
    # Tab 2: Multiple Images Upload
    # -----------------------------------------------------------
    with tab2:
        st.markdown("### Upload Multiple Images")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📤 Upload Images")
            uploaded_files = st.file_uploader(
                "Choose multiple image files",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
                key="multiple"
            )
        
        with col2:
            st.markdown("#### 📦 Upload ZIP Folder")
            zip_file = st.file_uploader(
                "Upload a ZIP file containing images",
                type=["zip"],
                key="zip"
            )
        
        # Process uploaded files
        files_to_process = []
        
        if uploaded_files:
            files_to_process = uploaded_files
        
        if zip_file:
            with zipfile.ZipFile(io.BytesIO(zip_file.read())) as z:
                for filename in z.namelist():
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        with z.open(filename) as f:
                            files_to_process.append(type('obj', (object,), {
                                'name': filename,
                                'read': lambda: f.read()
                            })())
        
        if files_to_process:
            st.markdown(f"**📊 Total Images:** {len(files_to_process)}")
            
            if st.button("🚀 Start Batch Processing", key="process_batch"):
                results = process_multiple_images(files_to_process, model)
                
                if results:
                    st.markdown("---")
                    st.markdown("### 📊 Results Summary")
                    
                    df = pd.DataFrame(results)
                    
                    # Statistics
                    col1, col2, col3 = st.columns(3)
                    
                    with_mask = len(df[df["Prediction"] == "With Mask"])
                    without_mask = len(df[df["Prediction"] == "Without Mask"])
                    
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>😷 With Mask</h3>
                            <h1>{with_mask}</h1>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>❌ Without Mask</h3>
                            <h1>{without_mask}</h1>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>📈 Total</h3>
                            <h1>{len(df)}</h1>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    st.dataframe(df, use_column_width=True)
                    
                    # Download CSV
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="mask_detection_results.csv",
                        mime="text/csv"
                    )
    
    # -----------------------------------------------------------
    # Tab 3: CSV Upload
    # -----------------------------------------------------------
    with tab3:
        st.markdown("### Upload CSV with Image Paths")
        st.info("📝 CSV should have a column named 'image_path' or 'filename' with full paths to images")
        
        csv_file = st.file_uploader("Choose a CSV file", type=["csv"], key="csv")
        
        if csv_file is not None:
            try:
                df_csv = pd.read_csv(csv_file)
                st.dataframe(df_csv.head(), use_column_width=True)
                
                # Find image path column
                path_column = None
                for col in ['image_path', 'filename', 'path', 'file']:
                    if col in df_csv.columns:
                        path_column = col
                        break
                
                if path_column:
                    st.success(f"✅ Found image path column: '{path_column}'")
                    
                    if st.button("🚀 Process Images from CSV", key="process_csv"):
                        st.warning("⚠️ Note: This requires images to be accessible at the specified paths on the server.")
                else:
                    st.error("❌ Could not find image path column. Please ensure your CSV has 'image_path' or 'filename' column.")
            
            except Exception as e:
                st.error(f"❌ Error reading CSV: {str(e)}")

if __name__ == "__main__":
    main()