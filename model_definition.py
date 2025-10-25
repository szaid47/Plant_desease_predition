import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 🚨 CRITICAL 🚨 ---
#
# This file is a PLACEHOLDER.
# You MUST replace the `SimpleCNN` class below with the
# *exact* Python class definition of your GAT model.
#
# The .pth file only contains weights, not the model's architecture.
# PyTorch needs the class definition to know how to load those weights.
#
# --- EXAMPLE ---
#
# 1. DELETE the `SimpleCNN` class.
# 2. PASTE your model class here, for example:
#
#    class GATModel(nn.Module):
#        def __init__(self, num_classes):
#            super(GATModel, self).__init__()
#            # ... all your layers (GATConv, Linear, etc.) ...
#            self.out = nn.Linear(..., num_classes)
#
#        def forward(self, x):
#            # ... your complete forward pass logic ...
#            return self.out(x)
#
# 3. UPDATE the `load_model` function below to use your class.
#
# --------------------


# --- PLACEHOLDER MODEL (REPLACE THIS) ---
class SimpleCNN(nn.Module):
    """
    This is a DUMMY model. Replace it with your GATModel class.
    This is based on a 224x224 input image.
    """
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        # Input: 3 x 224 x 224
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        # -> 16 x 224 x 224
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # -> 16 x 112 x 112
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        # -> 32 x 112 x 112
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # -> 32 x 56 x 56
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # -> 64 x 56 x 56
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # -> 64 x 28 x 28
        
        # Flatten the tensor
        self.fc1 = nn.Linear(64 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        
        # Flatten
        x = x.view(-1, 64 * 28 * 28)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# --- MODEL LOADING FUNCTION ---

def load_model(model_path, num_classes):
    """
    Loads the model weights into the model architecture.
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # --- 🚨 CRITICAL 🚨 ---
    # 1. Instantiate your *actual* model class here.
    #    REPLACE `SimpleCNN` with your model class (e.g., `GATModel`)
    model = SimpleCNN(num_classes=num_classes)
    
    # 2. Load the state dictionary
    #    This loads the weights from your .pth file.
    #    `map_location=device` ensures it works on CPU-only machines (like Streamlit sharing)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except RuntimeError as e:
        print("--- ERROR ---")
        print("Failed to load state_dict. This often means the model class in")
        print("`model_definition.py` does not match the architecture in the")
        print(f"`{model_path}` file. See error below:\n")
        raise e
        
    # Set model to evaluation mode (e.g., disable dropout)
    model.eval()
    
    return model.to(device)