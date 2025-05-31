import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import json
from io import BytesIO
import base64
import time

# -- Page Config --
st.set_page_config(
    page_title="Crop AI | Plant Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -- Load JSON Disease Info --
@st.cache_resource
def load_disease_data():
    with open("treatments_final_cleaned.json", "r") as f:
        return json.load(f)

disease_info = load_disease_data()

# -- Base64 image helper for cover image --
def get_image_base64(path):
    img = Image.open(path)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

# -- Style --
st.markdown("""
    <style>
        :root {
            --primary: #2e7d32;
            --secondary: #81c784;
            --accent: #ff8f00;
            --light: #f1f8e9;
            --dark: #1b5e20;
        }
        
        html, body, [data-testid="stApp"] {
            background: linear-gradient(135deg, #e6f4ea, #028800);
            background-attachment: fixed;
            color: #333;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .main {
            background-color: rgba(255, 255, 255, 0.97);
            padding: 2rem;
            border-radius: 16px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        }
        header, footer {visibility: hidden;}
        .result-card {
            background-color: #e9f7ef;
            padding: 2rem;
            border-left: 6px solid var(--primary);
            border-radius: 12px;
            margin-bottom: 2rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }
        .footer {
            text-align: center;
            font-size: 0.9em;
            color: #555;
            margin-top: 4rem;
            padding-top: 1rem;
            border-top: 1px solid #ddd;
        }
        .stButton>button {
            background-color: var(--primary);
            color: white;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            border: none;
            transition: all 0.3s;
        }
        .stButton>button:hover {
            background-color: var(--dark);
            transform: translateY(-2px);
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .stFileUploader>div>div>div>div {
            color: var(--primary);
        }
        .info-card {
            background-color: #f5f5f5;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            border-left: 4px solid var(--accent);
        }
        .symptom-item {
            background-color: #fff8e1;
            padding: 0.5rem 1rem;
            border-radius: 6px;
            margin-bottom: 0.5rem;
            border-left: 3px solid var(--accent);
        }
        .treatment-item {
            background-color: #e3f2fd;
            padding: 0.5rem 1rem;
            border-radius: 6px;
            margin-bottom: 0.5rem;
            border-left: 3px solid #1976d2;
        }
        .tab-content {
            padding: 1rem 0;
        }
        .confidence-bar {
            height: 24px;
            background: linear-gradient(90deg, #e0e0e0, #e0e0e0);
            border-radius: 12px;
            margin: 0.5rem 0;
            position: relative;
            overflow: hidden;
        }
        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--secondary), var(--primary));
            border-radius: 12px;
            transition: width 0.5s ease-in-out;
        }
        .confidence-label {
            position: absolute;
            width: 100%;
            text-align: center;
            line-height: 24px;
            color: white;
            font-weight: bold;
            text-shadow: 0 1px 2px rgba(0,0,0,0.3);
        }
        .top-prediction {
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(46, 125, 50, 0.4); }
            70% { box-shadow: 0 0 0 10px rgba(46, 125, 50, 0); }
            100% { box-shadow: 0 0 0 0 rgba(46, 125, 50, 0); }
        }
    </style>
""", unsafe_allow_html=True)

# -- Cover Image (Centered) --
img_base64 = get_image_base64("cover1.png")
st.markdown(f"""
    <div style="text-align: center; margin-bottom: 2rem;">
        <img src="data:image/png;base64,{img_base64}" style="width: 100%; max-width: 1600px; border-radius: 16px; box-shadow: 0 8px 24px rgba(0,0,0,0.1);" />
    </div>
""", unsafe_allow_html=True)

# -- Class Names --
class_names = list(disease_info.keys())  # Ensure names match JSON keys

# -- Load Model --
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mobilenet_densenet_hybrid_subset_model.h5")

model = load_model()

# -- Title --
st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style='color: var(--dark); margin-bottom: 0.5rem;'>
            🌿 Crop AI
        </h1>
        <p style='font-size: 1.2rem; color: #444; max-width: 800px; margin: 0 auto;'>
            Advanced plant disease detection with treatment recommendations powered by deep learning
        </p>
    </div>
""", unsafe_allow_html=True)

# -- Sidebar --
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h3>🔍 How It Works</h3>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        1. **Upload** a clear image of a plant leaf
        2. Our AI analyzes the image in seconds
        3. Get **instant diagnosis** and treatment options
        4. Access detailed disease information
    """)
    
    st.markdown("---")
    
    st.markdown("""
        <div style="text-align: center;">
            <h4>📸 Image Tips</h4>
            <p>For best results:</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        - Use **natural lighting**
        - Capture **close-up** of affected leaves
        - Include both healthy and diseased areas
        - Avoid blurry or shadowed images
    """)
    
    st.markdown("---")
    
    st.markdown("""
        <div style="text-align: center;">
            <p>For agricultural professionals and home gardeners</p>
        </div>
    """, unsafe_allow_html=True)

# -- Upload Image --
uploaded_file = st.file_uploader(
    "Choose a plant leaf image (JPG, PNG)", 
    type=["jpg", "jpeg", "png", "JPG", "JPEG", "PNG"],
    help="Upload a clear image of a plant leaf for disease detection"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Display uploaded image with animation
    with st.container():
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        st.image(image, caption="📸 Uploaded Image", width=400, use_container_width='auto')
        st.markdown("</div>", unsafe_allow_html=True)

    # Process image
    img = image.resize((160, 160))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction with progress
    with st.spinner("🔍 Analyzing image..."):
        progress_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.02)  # Simulate processing time
            progress_bar.progress(percent_complete + 1)
        
        predictions = model.predict(img_array)[0]
        top_indices = predictions.argsort()[-3:][::-1]
        top_classes = [class_names[i] for i in top_indices]
        top_confidences = [predictions[i] for i in top_indices]
    
    predicted_disease = top_classes[0]
    info = disease_info.get(predicted_disease, {})

    # Display results with tabs
    tab1, tab2, tab3 = st.tabs(["📊 Diagnosis", "📋 Disease Details", "💊 Treatment Options"])

    with tab1:
        # Main prediction card
        st.markdown(f"""
        <div class="result-card top-prediction">
            <div style="text-align: center;">
                <h3 style="color: var(--dark); margin-bottom: 1rem;">✅ AI Diagnosis</h3>
                <h4 style="color: var(--primary);">{predicted_disease}</h4>
                <div style="margin: 1.5rem auto; max-width: 300px;">
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {top_confidences[0]*100:.0f}%"></div>
                        <div class="confidence-label">{top_confidences[0]*100:.2f}% Confidence</div>
                    </div>
                </div>
                <p style="font-size: 0.9rem; color: #666;">AI-powered disease detection</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Other potential matches
        st.markdown("#### Other Possible Matches")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if len(top_classes) > 1:
                st.markdown(f"""
                <div class="info-card">
                    <h5>🥈 {top_classes[1]}</h5>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {top_confidences[1]*100:.0f}%"></div>
                        <div class="confidence-label">{top_confidences[1]*100:.2f}%</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            if len(top_classes) > 2:
                st.markdown(f"""
                <div class="info-card">
                    <h5>🥉 {top_classes[2]}</h5>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {top_confidences[2]*100:.0f}%"></div>
                        <div class="confidence-label">{top_confidences[2]*100:.2f}%</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with tab2:
        if info:
            st.markdown("#### 📝 Disease Overview")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="info-card">
                    <p><strong>Disease Type:</strong> {info.get('disease_type', 'N/A')}</p>
                    <p><strong>Causal Agent:</strong> {info.get('causal_agent', 'N/A')}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="info-card">
                    <p><strong>Organic Approved:</strong> {"✅ Yes" if info.get("organic_approved") else "❌ No"}</p>
                    <p><strong>Region Suitability:</strong> {", ".join(info.get("region_suitability", ["N/A"]))}</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("#### 🚨 Symptoms")
            for s in info.get("symptoms", []):
                st.markdown(f'<div class="symptom-item">🔍 {s}</div>', unsafe_allow_html=True)
            
            st.markdown("#### ⚠️ Toxicity Information")
            tox = info.get('toxicity_info', {})
            st.markdown(f"""
            <div class="info-card">
                <p><strong>Human Safety:</strong> {tox.get('human_safety', 'N/A')}</p>
                <p><strong>Bee Safety:</strong> {tox.get('bee_safety', 'N/A')}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### 📌 Additional Notes")
            st.info(info.get("notes", "No additional notes available."))

    with tab3:
        if info:
            st.markdown("#### 🧪 Chemical Treatments")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### 🧴 Recommended Pesticide")
                st.code(info.get("recommended_pesticide", "N/A"), language="text")
            
            with col2:
                st.markdown("##### 🪲 Recommended Insecticide")
                st.code(info.get("recommended_insecticide", "N/A"), language="text")
            
            st.markdown("#### 🌿 Alternative & Organic Treatments")
            for alt in info.get("alternative_treatments", []):
                st.markdown(f'<div class="treatment-item">🌱 {alt}</div>', unsafe_allow_html=True)
            
            st.markdown("#### 🛡 Preventive Measures")
            for pm in info.get("preventive_measures", []):
                st.markdown(f'<div class="treatment-item">🛡️ {pm}</div>', unsafe_allow_html=True)
            
            st.markdown("#### 🏷 Brand Suggestions")
            st.write(", ".join(info.get("brand_suggestions", ["No specific brand recommendations"])) or "No specific brand recommendations")
            
            st.markdown("---")
            st.markdown("""
            <div style="background-color: #fff3e0; padding: 1rem; border-radius: 8px;">
                <p style="text-align: center; font-weight: bold;">⚠️ Important Notice</p>
                <p>Always follow manufacturer instructions when using pesticides and insecticides. 
                Wear proper protective equipment and observe safety precautions.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("No treatment information available for this disease.")

else:
    # Showcase when no image is uploaded
    st.markdown("""
    <div style="text-align: center; padding: 3rem; background-color: rgba(255,255,255,0.7); border-radius: 12px;">
        <h3>📤 Upload a Plant Leaf Image</h3>
        <p>Get instant disease diagnosis and treatment recommendations</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Example cases section
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center;">
        <h3>🌱 Common Plant Diseases We Detect</h3>
    </div>
    """, unsafe_allow_html=True)
    
    cols = st.columns(4)
    example_diseases = [
        ("Early Blight", "Common in tomatoes and potatoes"),
        ("Powdery Mildew", "Affects many garden plants"),
        ("Leaf Rust", "Fungal disease in cereals"),
        ("Bacterial Spot", "Affects peppers and tomatoes")
    ]
    
    for col, (disease, desc) in zip(cols, example_diseases):
        with col:
            st.markdown(f"""
            <div style="background-color: #f5f5f5; padding: 1rem; border-radius: 8px; height: 120px;">
                <h5>{disease}</h5>
                <p style="font-size: 0.8rem;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

# -- Footer --
st.markdown("""
<div class="footer">
    <p>© 2025 Crop AI | For educational and advisory purposes only</p>
    <p style="font-size: 0.8rem;">Consult a professional agronomist for critical agricultural decisions</p>
</div>
""", unsafe_allow_html=True)