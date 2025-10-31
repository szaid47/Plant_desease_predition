import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import os

# --- IMPORTANT ---
# Import the model-loading function from your model_definition.py file
# You MUST edit model_definition.py to include your actual model's class
try:
    from model_definition import load_model
except ImportError:
    st.error("FATAL ERROR: model_definition.py not found or contains errors.")
    st.stop()
except AttributeError:
    st.error("FATAL ERROR: 'load_model' function not found in model_definition.py.")
    st.stop()


# --- CONFIGURATION ---

# 1. (CRITICAL) REPLACE WITH YOUR CLASS NAMES
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# (NEW) --- DISEASE INFO DATABASE (Placeholder) ---
# You must populate this with real data for Prevention and Solution
DISEASE_INFO = {
    'Apple___Apple_scab': {
        'prevention': "Remove and destroy fallen leaves in autumn. Prune trees to improve air circulation. Plant disease-resistant varieties.",
        'solution': "Apply fungicides (e.g., captan, myclobutanil, or sulfur) starting in early spring and continuing through the rainy season."
    },
    'Apple___Black_rot': {
        'prevention': "Prune out dead or diseased branches. Remove mummified fruit. Ensure good air circulation.",
        'solution': "Apply fungicides during the growing season, similar to apple scab management."
    },
    'Apple___Cedar_apple_rust': {
        'prevention': "Remove nearby juniper trees (alternate host) if possible. Plant rust-resistant apple varieties.",
        'solution': "Apply fungicides (e.g., myclobutanil) from pink bud stage through bloom."
    },
    'Apple___healthy': {
        'prevention': "Maintain regular watering, proper fertilization, and good orchard sanitation.",
        'solution': "No treatment necessary. Keep up the good work!"
    },
    # --- ADD ALL YOUR OTHER CLASSES HERE ---
    
    # Default entry if a class is not found in the dictionary
    'default': {
        'prevention': "Information not available. Maintain good plant hygiene, proper watering, and fertilization.",
        'solution': "Information not available. Consult a local agricultural extension office for advice."
    }
}


# 2. (CRITICAL) SET YOUR MODEL PATH
MODEL_PATH = 'gat_model_best.pth'
NUM_CLASSES = len(CLASS_NAMES)

# 3. (VERIFY) SET IMAGE TRANSFORMATIONS
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- MODEL LOADING ---
@st.cache_resource
def get_model(model_path, num_classes):
    """Loads and returns the model."""
    if not os.path.exists(model_path):
        return None
    try:
        model = load_model(model_path, num_classes)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error("Please ensure 'model_definition.py' contains your correct model class "
                 "and that 'num_classes' matches your model's output.")
        return None

model = get_model(MODEL_PATH, NUM_CLASSES)

# --- (NEW) CSS STYLING ---
st.markdown("""
<style>
    /* Change button color */
    div.stButton > button:first-child {
        background-color: #3F704D;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        border: none;
    }
    div.stButton > button:first-child:hover {
        background-color: #31573B;
        color: white;
        border: none;
    }

    /* Style the file uploader */
    .stFileUploader > div {
        border: 2px dashed #3F704D;
        background-color: #F8FBF8;
        border-radius: 10px;
        padding: 1.5rem;
    }
    .stFileUploader label {
        color: #3F704D !important;
        font-weight: bold;
        font-size: 1.1em;
    }

    /* Main title */
    h1 {
        color: #3F704D !important;
        text-align: center;
        font-weight: bold;
    }
    
    /* Subtitle */
    .subtitle {
        text-align: center;
        color: #4A4A4A;
        margin-top: -15px;
        margin-bottom: 20px;
    }
    
    /* Section headers */
    .section-header {
        color: #3F704D;
        font-size: 1.5em;
        font-weight: bold;
        text-align: center;
        margin-top: 20px;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .section-header svg {
        margin-right: 10px;
        fill: #3F704D;
    }
    
    /* Card for displaying results */
    .result-card {
        background-color: #FFFFFF;
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border-left: 6px solid #3F704D; /* Default border color */
        display: flex;
        align-items: center;
    }
    
    .result-card-icon {
        font-size: 2.5em;
        margin-right: 20px;
    }
    
    .result-card-content h3 {
        margin: 0;
        padding: 0;
        font-size: 1.1em;
        color: #4A4A4A;
        font-weight: bold;
    }
    
    .result-card-content p {
        margin: 5px 0 0 0;
        padding: 0;
        font-size: 1em;
        color: #313131;
    }
    
    /* Specific card styles */
    .card-disease { border-left-color: #28a745; } /* Green */
    .card-confidence { border-left-color: #ffc107; } /* Yellow */
    .card-prevention { border-left-color: #17a2b8; } /* Blue/Info */
    .card-solution { border-left-color: #007bff; } /* Bright Blue */

    /* Image display */
    .stImage {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin-bottom: 10px;
    }
    .stImage img {
        border-radius: 15px;
    }
    
    /* Placeholder for no image */
    .image-placeholder {
        border: 2px dashed #B0BEC5;
        border-radius: 15px;
        display: flex;
        align-items: center;
        justify-content: center;
        height: 250px;
        background-color: #F8F9FA;
        color: #78909C;
        font-style: italic;
    }
    
</style>
""", unsafe_allow_html=True)

# --- STREAMLIT APP ---

st.set_page_config(
    page_title="Arecabandhu", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- (NEW) HEADER ---
st.title("Arecabandhu")
st.markdown("<p class='subtitle'>Your smart crop disease companion</p>", unsafe_allow_html=True)
st.markdown("---")


if model is None:
    st.error(f"Model file not found at '{MODEL_PATH}'.")
    st.write("Please make sure the model file is in the same directory as `app.py`.")
else:
    # --- (NEW) UPLOAD SECTION ---
    st.markdown("<h3 style='text-align: center; color: #313131;'>Detect Arecanut Diseases in Seconds</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #555;'>Upload an image to analyze the disease.</p>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    # --- (NEW) IMAGE & RESULTS LAYOUT ---
    
    # Placeholder for the image
    image_placeholder = st.empty()
    
    # Placeholder for the results
    results_placeholder = st.empty()

    if uploaded_file is None:
        # Show placeholder if no image is uploaded
        image_placeholder.markdown("<div class='image-placeholder'>Selected image will appear here</div>", unsafe_allow_html=True)

    if uploaded_file is not None:
        # 1. Display the uploaded image
        try:
            image = Image.open(uploaded_file).convert('RGB')
            # Display image in its placeholder
            with image_placeholder.container():
                st.image(image, caption="Selected Image", use_column_width=True)
        except Exception as e:
            st.error(f"Error opening image file: {e}")
            st.stop()
            
        # 2. Make prediction on button click
        if st.button("Analyze Image"):
            with st.spinner("Analyzing..."):
                try:
                    # 3. Preprocess the image
                    image_tensor = preprocess(image).unsqueeze(0) # Add batch dimension
                    
                    # 4. Make prediction
                    with torch.no_grad():
                        outputs = model(image_tensor)
                        probabilities = F.softmax(outputs, dim=1)
                        top_prob, top_catid = torch.max(probabilities, 1)
                    
                    # 5. Get results
                    class_name = CLASS_NAMES[top_catid.item()]
                    confidence = top_prob.item() * 100
                    display_name = class_name.replace('___', ' ')
                    
                    # Get prevention and solution from our dictionary
                    info = DISEASE_INFO.get(class_name, DISEASE_INFO['default'])

                    # 6. Display results in the results placeholder
                    with results_placeholder.container():
                        
                        # Analysis Results Header
                        st.markdown(
                            """
                            <div class='section-header'>
                                <svg xmlns="http://www.w3.org/2000/svg" height="24" viewBox="0 -960 960 960" width="24"><path d="M120-120v-80h80v-200h80v200h80v80H120Zm240 0v-240h80v-160h80v160h80v240H360Zm240 0v-400h80v-80h80v80h80v400H600Z"/></svg>
                                Analysis Results
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )

                        # Card 1: Predicted Disease
                        st.markdown(f"""
                        <div class="result-card card-disease">
                            <div class="result-card-icon">➕</div>
                            <div class="result-card-content">
                                <h3>Predicted Disease</h3>
                                <p>{display_name}</p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Card 2: Confidence
                        st.markdown(f"""
                        <div class="result-card card-confidence">
                            <div class="result-card-icon">🎯</div>
                            <div class="result-card-content">
                                <h3>Confidence</h3>
                                <p>{confidence:.2f}%</p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Card 3: Prevention
                        st.markdown(f"""
                        <div class="result-card card-prevention">
                            <div class="result-card-icon">🛡️</div>
                            <div class="result-card-content">
                                <h3>Prevention</h3>
                                <p>{info['prevention']}</p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Card 4: Solution
                        st.markdown(f"""
                        <div class="result-card card-solution">
                            <div class="result-card-icon">💡</div>
                            <div class="result-card-content">
                                <h3>Solution</h3>
                                <p>{info['solution']}</p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
                    st.error("This might be due to a mismatch between the model definition, "
                             "the saved weights, or the `CLASS_NAMES` list.")

# --- SIDEBAR ---
st.sidebar.title("About This App")
st.sidebar.info(
    "This app uses a PyTorch model to predict plant diseases from leaf images.\n\n"
    "**This is a template.** It requires you to provide your own:\n"
    "1. `gat_model_best.pth` (Your trained model weights)\n"
    "2. The correct model class in `model_definition.py`\n"
    "3. The correct `CLASS_NAMES` list in `app.py`\n\n"
    "Model predictions are for demonstration only and should not replace professional diagnosis."
)

