Streamlit Plant Disease Prediction App

This is a template for creating a Streamlit web application to predict plant diseases from an image, using a PyTorch model.

🚨 CRITICAL SETUP 🚨

This template will not work out of the box. You must provide your specific model files and class definitions.

1. Place Your Model File

Make sure your trained PyTorch model file is named gat_model_best.pth and is in this same directory.

If it's named differently, update the MODEL_PATH variable in app.py.

2. Update model_definition.py (Most Important Step)

Open model_definition.py.

Delete the placeholder SimpleCNN class.

Paste in the exact Python class definition for your GATModel (or whatever it is named). The class definition is required for PyTorch to understand the structure of the weights in the .pth file.

In the load_model function (at the bottom of the file), change model = SimpleCNN(...) to model = YourModelClassName(...).

3. Update app.py

Open app.py.

Find the CLASS_NAMES list near the top.

Replace the example list with your exact list of class names.

The order of this list must match the output indices of your model (e.g., if your model predicts index 0 for "Apple Scab", "Apple Scab" must be the first item in the list).

Verify the preprocess variable. The image transformations (Resize, CenterCrop, Normalize) must be the same as those you used to train your model.

How to Run

Install Dependencies:

pip install -r requirements.txt


Run Streamlit:

streamlit run app.py


Open your browser to the local URL provided by Streamlit (usually http://localhost:8501).