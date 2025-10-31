import streamlit as st
from PIL import Image
import time

# --- DEMO DATA ---
FAKE_RESULT = {
    'display_name': "Healthy Leaf 🌿",
    'confidence': 98.99,
    'prevention': "Maintain regular watering, ensure good air circulation, and avoid over-fertilization.",
    'solution': "No treatment needed! Continue regular monitoring to maintain leaf health.",
    'extra_tip': "Ensure the soil has proper drainage and the plant receives adequate sunlight."
}

# --- PAGE CONFIG ---
st.set_page_config(page_title="AgriAI 🌱", page_icon="🌿", layout="centered")

# --- GLOBAL CSS STYLING ---
st.markdown("""
<style>
    body {
        background-color: #F5F7F5;
    }

    /* Center title */
    h1 {
        text-align: center;
        color: #2E7D32;
        font-weight: 800;
        margin-bottom: 0.2em;
    }

    /* Subtitle */
    .subtitle {
        text-align: center;
        font-size: 1.1em;
        color: #4A4A4A;
        margin-bottom: 1.5em;
    }

    /* File uploader style */
    .stFileUploader > div {
        border: 2px dashed #2E7D32;
        background-color: #F0FFF4;
        border-radius: 12px;
        padding: 1.2rem;
    }
    .stFileUploader label {
        color: #2E7D32 !important;
        font-weight: bold;
        font-size: 1.05em;
    }

    /* Custom button style */
    div.stButton > button:first-child {
        background: linear-gradient(to right, #2E7D32, #66BB6A);
        color: white;
        font-weight: bold;
        border-radius: 12px;
        height: 3em;
        width: 100%;
        border: none;
        transition: 0.3s;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(to right, #1B5E20, #4CAF50);
        transform: scale(1.02);
    }

    /* Card base style */
    .result-card {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 1.2em;
        margin: 0.8em 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border-left: 6px solid;
        display: flex;
        align-items: flex-start;
        gap: 1em;
    }

    .result-card h3 {
        margin: 0;
        font-size: 1.1em;
        font-weight: bold;
        color: #2E7D32;
    }

    .result-card p {
        margin: 4px 0 0 0;
        color: #333;
        font-size: 0.97em;
    }

    .card-disease { border-left-color: #43A047; }
    .card-confidence { border-left-color: #FBC02D; }
    .card-prevention { border-left-color: #29B6F6; }
    .card-solution { border-left-color: #7E57C2; }
    .card-tip { border-left-color: #8D6E63; }

    /* Footer */
    .footer {
        text-align: center;
        margin-top: 2em;
        color: #757575;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)


# --- APP HEADER ---
st.title("🌿 AgriAI")
st.markdown("<p class='subtitle'>AI-Powered Plant Disease Detection & Guidance</p>", unsafe_allow_html=True)
st.markdown("---")

# --- UPLOAD SECTION ---
st.markdown("<h4 style='text-align:center; color:#2E7D32;'>Upload a leaf image to detect potential diseases</h4>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

# --- Model Info Box ---
st.info("💡 **Powered by GNN (Graph Neural Networks)** — AgriAI studies the vein structure and leaf texture patterns to identify diseases with high precision.")

# --- IMAGE AND RESULT DISPLAY ---
image_col, result_col = st.columns(2, gap="large")

with image_col:
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Leaf Image", use_container_width=True)
    else:
        st.markdown("<div style='text-align:center; color:#90A4AE; border:2px dashed #B0BEC5; border-radius:10px; padding:50px;'>Upload a leaf image to begin analysis 🌿</div>", unsafe_allow_html=True)

with result_col:
    if uploaded_file:
        if st.button("🔍 Analyze Image"):
            with st.spinner("Analyzing leaf... please wait"):
                time.sleep(1.8)  # demo delay

            result = FAKE_RESULT

            # --- DISPLAY RESULT CARDS ---
            st.markdown(f"""
            <div class='result-card card-disease'>
                <h3>🩺 Predicted Disease</h3>
                <p>{result['display_name']}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class='result-card card-confidence'>
                <h3>🎯 Confidence</h3>
                <p>{result['confidence']:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class='result-card card-prevention'>
                <h3>🛡️ Prevention</h3>
                <p>{result['prevention']}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class='result-card card-solution'>
                <h3>💡 Recommended Solution</h3>
                <p>{result['solution']}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class='result-card card-tip'>
                <h3>🌞 Expert Tip</h3>
                <p>{result['extra_tip']}</p>
            </div>
            """, unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.title("About 🌱 AgriAI")
st.sidebar.markdown("""
AgriAI is a **smart agricultural assistant** powered by **Graph Neural Networks** to detect plant diseases from leaf images.

**Features:**
- High accuracy diagnosis  
- Real-time image analysis  
- Practical prevention & solution tips  

> ⚠️ Demo Mode: No real AI predictions yet. Connect your trained model to enable full functionality.
""")

# --- FOOTER ---
st.markdown("<div class='footer'>© 2025 AgriAI | Built with ❤️ using Streamlit</div>", unsafe_allow_html=True)
