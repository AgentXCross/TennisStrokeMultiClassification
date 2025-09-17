import streamlit as st
import altair as alt
import torch
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import os
import random
from torchvision import models

#Device setup
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

#Class names
CLASS_NAMES = ['Backhand', 'Forehand', 'Ready Position', 'Serve']

#Image Transformation
def center_crop_square(img: Image.Image) -> Image.Image:
    """Crops the center square from a PIL image."""
    width, height = img.size
    min_dim = min(width, height)
    left = (width - min_dim) // 2
    top = (height - min_dim) // 2
    right = left + min_dim
    bottom = top + min_dim
    return img.crop((left, top, right, bottom))

transform = transforms.Compose([
    transforms.Lambda(center_crop_square),
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
    ])

# Cached model loader
@st.cache_resource
def load_model():
    # Load pretrained EfficientNet
    model = models.efficientnet_b0(weights = models.EfficientNet_B0_Weights.DEFAULT)
    
    # Modify the classifier output number of classes
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 4)

    # Load my weights
    model.load_state_dict(torch.load("tennis_stroke_model_efficientnet.pth", map_location = "cpu"))
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
    
