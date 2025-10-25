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
# The order MUST match the output indices of your trained model.
# (e.g., if your model outputs 0 for 'Apple_scab', it must be first)
# This example list is from the 38-class PlantVillage dataset.
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

# 2. (CRITICAL) SET YOUR MODEL PATH
MODEL_PATH = 'gat_model_best.pth'
NUM_CLASSES = len(CLASS_NAMES)

# 3. (VERIFY) SET IMAGE TRANSFORMATIONS
# These MUST be the same as the transformations used during training.
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- MODEL LOADING ---

# Use Streamlit's caching to load the model only once.
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

# --- STREAMLIT APP ---

st.set_page_config(page_title="Plant Disease Predictor", layout="centered")
st.title("🌱 Plant Disease Predictor")
st.write("Upload an image of a plant leaf, and the model will predict the disease.")

if model is None:
    st.error(f"Model file not found at '{MODEL_PATH}'.")
    st.write("Please make sure the model file is in the same directory as `app.py`.")
else:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # 1. Display the uploaded image
        try:
            image = Image.open(uploaded_file).convert('RGB')
        except Exception as e:
            st.error(f"Error opening image file: {e}")
            st.stop()
            
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        st.write("") # Add space
        
        # 2. Make prediction on button click
        if st.button("Classify", type="primary"):
            with st.spinner("Classifying..."):
                try:
                    # 3. Preprocess the image
                    image_tensor = preprocess(image).unsqueeze(0) # Add batch dimension
                    
                    # 4. Make prediction
                    with torch.no_grad():
                        outputs = model(image_tensor)
                        probabilities = F.softmax(outputs, dim=1)
                        top_prob, top_catid = torch.max(probabilities, 1)
                    
                    # 5. Display the result
                    class_name = CLASS_NAMES[top_catid.item()]
                    confidence = top_prob.item() * 100
                    
                    st.success(f"**Prediction:** {class_name.replace('___', ' ')}")
                    st.info(f"**Confidence:** {confidence:.2f}%")
                    
                    # Optional: Display top 3 predictions
                    st.write("---")
                    st.subheader("Top 3 Guesses:")
                    top_probs, top_indices = torch.topk(probabilities, 3, dim=1)
                    
                    for i in range(3):
                        prob = top_probs[0][i].item() * 100
                        name = CLASS_NAMES[top_indices[0][i].item()].replace('___', ' ')
                        st.write(f"{i+1}. {name} ({prob:.2f}%)")
                        
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
