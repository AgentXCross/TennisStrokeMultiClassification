import streamlit as st
import altair as alt
import torch
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import os
import random
from model_definition import create_convnext_tiny, create_mobilenet_v3, create_resnet18
from transforms import transforms

#Device setup
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

#Class names
CLASS_NAMES = ['Backhand', 'Forehand', 'Ready Position', 'Serve']

# Model Weights Paths
model1_weights = 'results-models/mobilenet_v3.pth'
model2_weights = 'results-models/resnet18.pth'
model3_weights = 'results-models/convnext.pth'

#Image Transformation
nothing, transform = transforms()

# Cached model loader
@st.cache_resource
def load_models():
    # Create Models
    model1 = create_mobilenet_v3(num_classes = len(CLASS_NAMES))
    model2 = create_resnet18(num_classes = len(CLASS_NAMES))
    model3 = create_convnext_tiny(num_classes = len(CLASS_NAMES))

    # Load my weights
    model1.load_state_dict(torch.load(model1_weights, map_location = device))
    model1.eval()
    model1.to(device)

    model2.load_state_dict(torch.load(model2_weights, map_location = device))
    model2.eval()
    model2.to(device)

    model3.load_state_dict(torch.load(model3_weights, map_location = device))
    model3.eval()
    model3.to(device)
    return model1, model2, model3

model1, model2, model3 = load_models()

def predict_image(image: Image.Image):
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits1 = model1(image_tensor)
        logits2 = model2(image_tensor)
        logits3 = model3(image_tensor)

        probs1 = torch.softmax(logits1, dim = 1)
        probs2 = torch.softmax(logits2, dim = 1)
        probs3 = torch.softmax(logits3, dim = 1)

        # Average ensemble
        avg_probs = (probs1 + probs2 + probs3) / 3
        avg_probs = avg_probs.squeeze().cpu().numpy()

    return avg_probs

st.set_page_config(
    page_title = "Tennis Stroke Classifier",
    page_icon = "🎾",  
    layout = "centered",
    initial_sidebar_state = "auto"
)

# CSS for centering
st.markdown("""
    <style>
        h1 {
            font-size: 50px !important;
        }
        .centered {
            text-align: center;
        }
        .stApp {
            max-width: 800px;
            margin: auto;
        }
    </style>
""", unsafe_allow_html = True)

# Streamlit UI
st.markdown("<h1 class = 'centered'>🎾 Tennis Stroke Classifier 🎾</h1>", unsafe_allow_html = True)

st.markdown("""
<p style='text-align: center; font-size:18px; color:#F5F5F5;'>
Upload a tennis stroke image taken from behind, and this CNN deep learning model made with Python PyTorch will predict whether it's a 
<strong>Forehand</strong>, <strong>Backhand</strong>, <strong>Serve</strong>, or <strong>Ready Position</strong>.  
<br><br>
</p>
""", unsafe_allow_html = True)

st.markdown("""
<p style='text-align: center; font-size:16px; color:#D3D3D3;'>
Image must either be a .jpeg/.jpg or .png file. After the image passes through the model, probabilities for each
class will be displayed in a bar chart. Made by Michael Liu.
<br><br>
</p>
""", unsafe_allow_html = True)

image_path = os.path.join("assets", "example.jpeg")
example_image = Image.open(image_path).convert("RGB")

st.markdown("<h4 class = 'centered'>Input Image Example</h4>", unsafe_allow_html = True)
st.image(example_image, width = 700)


# Upload section
uploaded_file = st.file_uploader("Upload a tennis stroke image from behind below (JPEG/PNG only)", type = ["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption = "Uploaded Image", width = 300)

    # Predict
    probs = predict_image(image)
    predicted_class = CLASS_NAMES[probs.argmax()]

    # Show results
    st.subheader(f"Prediction: **{predicted_class}**")

    # Create a DataFrame for Altair
    df = pd.DataFrame({
        "Class": CLASS_NAMES,
        "Probability": probs
    })

    # Build Altair Chart
    chart = (
        alt.Chart(df)
        .mark_bar(size = 80) 
        .encode(
            x = alt.X("Class", sort = None, title = None), 
            y = alt.Y("Probability", scale = alt.Scale(domain = [0, 1]), title = "Probability of Each Class"),
            color = alt.value("#FF0873") 
        )
        .properties(
            width = 600, 
            height = 500   
        )
        .configure_view(
            strokeWidth = 0 
        )
        .interactive(False) 
    )

    chart = chart.configure_axis(
        labelFontSize = 16,
        titleFontSize = 18,
        labelFont = "Source Sans Pro",
        titleFont = "Source Sans Pro"
    ).configure_title(
        fontSize = 22,
        font = "Source Sans Pro",
        anchor = "start",
        color = "#333"
    )

    st.altair_chart(chart, use_container_width = True)
    
