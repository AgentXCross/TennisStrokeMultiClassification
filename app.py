import streamlit as st
import altair as alt
import torch
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import os
import random
from model import TennisStrokeClassification

#Device setup
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

#Class names
CLASS_NAMES = ['Backhand', 'Forehand', 'Ready Position', 'Serve']

#Image Transformation
transform = transforms.Compose(
    [#Make sure every side length is at least 720 first
    transforms.Resize(720),
    #Crop the center square
    transforms.CenterCrop(720),
    #Resize
    transforms.Resize(size = (128, 128)),
    #Turn the image into a torch tensor
    transforms.ToTensor()]
)

# Cached model loader
@st.cache_resource
def load_model():
    model = TennisStrokeClassification(num_classes = 4)
    model.load_state_dict(torch.load("tennis_stroke_model.pth", map_location = "cpu"))
    model.eval()
    model.to(device)
    return model

model = load_model()

def predict_image(image: Image.Image):
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim = 1).squeeze().cpu().numpy()
    return probs

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
            y = alt.Y("Probability", scale = alt.Scale(domain = [0, 1]), title = "Probabilities of Each Class"),
            color = alt.value("#FF69B4") 
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
    
