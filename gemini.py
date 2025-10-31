import streamlit as st
from PIL import Image
import base64
import io
import json
import requests  # To make API calls
import time

# --- (NEW) AI Model API Configuration ---
# Using a generative AI model for its vision capabilities
# (FIXED) Corrected the typo in the API URL
MODEL_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key="

# (NEW) --- Define the JSON structure we want the model to return ---
RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "display_name": {"type": "STRING"},
        "confidence": {"type": "NUMBER"},
        "prevention": {"type": "STRING"},
        "solution": {"type": "STRING"},
        "extra_tip": {"type": "STRING"}
    },
    "required": ["display_name", "confidence", "prevention", "solution", "extra_tip"]
}

# (NEW) --- Helper function to convert image to base64 ---
def image_to_base64(image, format="JPEG"):
    """Converts a PIL Image to a base64 encoded string."""
    buffered = io.BytesIO()
    # Convert RGBA images (like PNGs with transparency) to RGB
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# (NEW) --- Helper function to call the AI API ---
def get_ai_analysis(api_key, image, retries=2, backoff_factor=2):
    """
    Calls the AI API with the image and a structured prompt.
    Implements exponential backoff for retries.
    """
    full_api_url = f"{MODEL_API_URL}{api_key}"
    
    # Convert PIL Image to base64
    base64_image_data = image_to_base64(image)
    
    # This is the "Agent" part:
    # 1. systemInstruction: Sets the persona and rules (JSON only!)
    # 2. user_prompt: Gives the task.
    # 3. generationConfig: Forces the JSON output.
    
    system_instruction = (
        "You are AgriAI, an expert botanist and plant pathologist. "
        "Your task is to analyze the provided image of a plant leaf. "
        "Identify the plant and any disease, or state if it is healthy. "
        "Provide a confidence score (0-100) for your diagnosis. "
        "Then, provide practical prevention steps, a recommended solution, and one expert tip. "
        "You MUST respond ONLY with a valid JSON object matching the requested schema. "
        "Do not include any other text, greetings, or explanations outside the JSON."
    )
    
    user_prompt = (
        "Analyze this plant leaf image. Provide your diagnosis (e.g., 'Tomato - Early Blight' or 'Healthy Rose Leaf'), "
        "confidence, prevention, solution, and an expert tip."
    )

    payload = {
        "systemInstruction": {
            "parts": [{"text": system_instruction}]
        },
        "contents": [
            {
                "parts": [
                    {"text": user_prompt},
                    {
                        "inlineData": {
                            "mimeType": "image/jpeg",
                            "data": base64_image_data
                        }
                    }
                ]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": RESPONSE_SCHEMA
        }
    }

    headers = {"Content-Type": "application/json"}
    
    delay = 1  # Initial delay of 1 second
    for attempt in range(retries + 1):
        try:
            response = requests.post(full_api_url, headers=headers, json=payload)
            
            # Handle successful response
            if response.status_code == 200:
                response_data = response.json()
                # Extract the JSON text from the candidate's content
                json_text = response_data['candidates'][0]['content']['parts'][0]['text']
                # Parse the JSON text into a Python dictionary
                return json.loads(json_text)
            
            # Handle non-OK status codes that might not be retried
            elif response.status_code >= 400 and response.status_code < 500:
                return {"error": f"Client error: {response.status_code} - {response.text}"}
            
            # For server errors (5xx) or rate limiting (429), we retry
            elif response.status_code == 429 or response.status_code >= 500:
                st.toast(f"API busy or error (Code {response.status_code}). Retrying in {delay}s...")
                # Fall through to retry logic
            
            else:
                 return {"error": f"Unexpected error: {response.status_code} - {response.text}"}

        except requests.exceptions.RequestException as e:
            st.toast(f"Network error: {e}. Retrying in {delay}s...")
            # Fall through to retry logic

        # Wait and increase delay for next retry
        if attempt < retries:
            time.sleep(delay)
            delay *= backoff_factor

    # If all retries fail
    return {"error": "Failed to get a response from the API after several attempts."}


# --- PAGE CONFIG ---
# --- (EDITED) Changed layout from "centered" to "wide" ---
st.set_page_config(page_title="AgriAI 🌱", page_icon="🌿", layout="wide")

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
    
    /* (NEW) Icon styling */
    .result-card .icon {
        font-size: 1.5rem;
        padding-top: 4px;
    }
    
    /* (NEW) Content styling */
    .result-card .content {
        flex: 1;
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

# --- (NEW) Check for API Key ---
api_key = ""
try:
    # Use a generic API_KEY name
    api_key = st.secrets["API_KEY"]
    if not api_key:
        st.error("🚨 API_KEY is not set in st.secrets. Please add it to .streamlit/secrets.toml")
except FileNotFoundError:
    st.error("🚨 .streamlit/secrets.toml file not found. Please create one and add your API_KEY.")
except KeyError:
    st.error("🚨 API_KEY not found in .streamlit/secrets.toml. Please add it.")


# --- UPLOAD SECTION ---
st.markdown("<h4 style='text-align:center; color:#2E7D32;'>Upload a leaf image to detect potential diseases</h4>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

# --- Model Info Box ---
st.info("— AgriAI analyzes the leaf image to identify diseases and provide expert guidance.")

# --- IMAGE AND RESULT DISPLAY ---
# This will now use the full width, making the columns appear more horizontal
image_col, result_col = st.columns(2, gap="large")

with image_col:
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Leaf Image", use_container_width=True)
    else:
        st.markdown("<div style='text-align:center; color:#90A4AE; border:2px dashed #B0BEC5; border-radius:10px; padding:50px;'>Upload a leaf image to begin analysis 🌿</div>", unsafe_allow_html=True)

with result_col:
    if uploaded_file and api_key: # Only show button if API key and file are present
        if st.button("🔍 Analyze Image"):
            with st.spinner("Analyzing leaf Image ... please wait"):
                
                # --- (NEW) CALL AI API ---
                result = get_ai_analysis(api_key, image)
            
            # --- (NEW) Handle API Response ---
            if "error" in result:
                st.error(f"**Analysis Failed:** {result['error']}")
            else:
                st.success("**Analysis Complete!**")
                
                # --- DISPLAY RESULT CARDS (Updated with icons) ---
                st.markdown(f"""
                <div class='result-card card-disease'>
                    <div class='icon'>🩺</div>
                    <div class='content'>
                        <h3>Predicted Disease</h3>
                        <p>{result.get('display_name', 'N/A')}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class='result-card card-confidence'>
                    <div class='icon'>🎯</div>
                    <div class='content'>
                        <h3>Confidence</h3>
                        <p>{result.get('confidence', 0.0):.2f}%</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class='result-card card-prevention'>
                    <div class='icon'>🛡️</div>
                    <div class='content'>
                        <h3>Prevention</h3>
                        <p>{result.get('prevention', 'N/A')}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class='result-card card-solution'>
                    <div class='icon'>💡</div>
                    <div class='content'>
                        <h3>Recommended Solution</h3>
                        <p>{result.get('solution', 'N/A')}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class='result-card card-tip'>
                    <div class='icon'>🌞</div>
                    <div class='content'>
                        <h3>Expert Tip</h3>
                        <p>{result.get('extra_tip', 'N/A')}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.title("About 🌱 AgriAI")
st.sidebar.markdown("""
AgriAI is a **smart agricultural assistant** powered by **Generative AI** to detect plant diseases from leaf images.

**Features:**
- High accuracy diagnosis
- Real-time image analysis
- Practical prevention & solution tips

> ⚠️ This app requires an **API Key**. Add it to `.streamlit/secrets.toml` to enable analysis.
""")

# --- FOOTER ---
st.markdown("<div class='footer'>© 2025 AgriAI | Powered by Generative AI</div>", unsafe_allow_html=True)

