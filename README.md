# **Tennis Stroke Multi-Class Classification**
A deep learning project to classify tennis images into four categories — **forehand, backhand, serve, and ready position** — using a 3 pretrained convolutional neural network (CNN). The chosen architectures used were ResNet-18, MobileNetV3-Small, and ConvNeXt-Tiny. 
This project was initially developed and evaluated in a **Jupyter Notebook**, then split into **Python** scripts.
Once models were trained and saved using **PyTorch**, model was deployed using **Streamlit**.

---

## **Project Table of Contents**
1. [Overview](#overview)  
2. [Dataset](#dataset)  
3. [Project Structure](#project-structure)  
4. [Model Architecture](#model-architecture)  
5. [Setup Instructions](#setup-instructions)  
6. [Training the Model](#training-the-model)  
7. [Saving & Loading the Model](#saving--loading-the-model)  
8. [Evaluation](#evaluation)  
9. [Results](#results)  
10. [Future Improvements](#future-improvements)

---

## **Overview**
The goal of this project is to fine tune Convolutional Neural Networks that can accurately classify tennis strokes from images. Images are exclusively taken from behind the player.

### **Key Features**
- Final model uses pretrained MobileNetV3-Small, ResNet-18, and ConvNeXt-Tiny models by averaging logits (prediction probabilities)
- Training Pipeline with:
  - Adam Optimizer
  - Focal Loss (Modified Cross-Entropy)
- Python scripts with clean separation of:
  - Dataset Preprocessing/Pipeline
  - Transformations
  - Model Architecture and Loss Function Definitions
  - Training and Testing Loop
  - Utility Functions
  - Streamlit App
- Supports **Apple MPS GPU acceleration** for Mac users. Nvidia GPU acceleration should also work by replacing `mps` with `cuda`. Also, replace `device = 'mps' if torch.mps.backends.is_available() else 'cpu'` with `device = 'cuda' if torch.cuda.is_available() else 'cpu'`
- Started with Jupyter Notebook exploration before moving to Python scripts.

---

## **Dataset**
The dataset used comes entirely from **Mendeley Data**:

**Source:**  
[Tennis Strokes Dataset on Mendeley Data](https://data.mendeley.com/datasets/nv3rpsxhhk/1)

### **Classes Included**
1. Forehand  
2. Backhand  
3. Serve
4. Ready Position

### **Dataset Structure and Data Splitting**
Images were split into `train` and `test` sets at a **75/25 ratio**.
This is done by using **scikit-learn** `train_test_split()` on a pandas `pd.DataFrame` containing all the images.

Dataset is structure as follows:

```
image_data/
│
├── backhand/
├── forehand/
├── ready_position/
└── serve/
```

---

## **Project Structure**
```
project_root/
│
├── main.py                     # Entry point for training and saving the model
├── app.py                      # Model deployment using Streamlit
├── model_definition.py         # Contains pretrained model definitions and loss function definition
├── dataset.py                  # Dataset Definition and DataLoader Creation
├── transforms.py               # Albumentation Transformations for training and testing
├── train_test_loop.py          # Training and evaluation loops
├── helper_functions.py         # Accuracy function, Model saving function, etc.
│
├── dataframe/                  # Folder containing training and testing dataframes
├── image_data/                 # Contains images
├── results-models/             # Model result dictionaries and model weights
├── assets/                     # Images for project results
│
├── model-development-2.ipynb   # Main project development notebook
├── old_model_development.ipynb # Original model development notebook
│
├── requirements.txt            # Library/Framework requirements for running Notebook and Scripts
└── README.md                   # Project documentation
```

---

## **Model Architectures**
The Overall Model is composed of 3 Models who have their logits/prediction probabilities averaged before making a decision:

| Model            | Details        |
|------------------|----------------|
| **MobileNetV3-Small** | Convolutional blocks with Conv2D → BatchNorm → ReLU Non-Linear Activation → MaxPool → Dropout |
| **ResNet-18**   | Flatten → ReLU Non-Linear Activation → Output (logits) |
| **ConvNeXt-Tiny**   | Flatten → ReLU Non-Linear Activation → Output (logits) |

---

## **Setup Instructions**

### **1. Clone the repository**
```bash
git clone git clone https://github.com/AgentXCross/TennisStrokeMultiClassification.git
cd TennisStrokeMultiClassification
```

### **2. Install dependencies**
Create a virtual environment and install requirements:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### **3. Organize the dataset**
Ensure your `dataset/` folder is structured as shown above with `train_set` and `test_set` directories.
When you first download the dataset from the website, all data is clumped under one folder. 
Use data_splitting.py to create the structured directories with a training and test set.
You may have to adjust the path/value of `INPUT_DIR` according to the name of the downloaded folder.

---

## **Training the Model**
Run the main training script:
```bash
python main.py
```

### **What happens during training**
- Model is initialized and moved to `mps` or `cpu` automatically (`cuda` should be supported). To move the model to `cuda`, when setting up device agnostic code:
```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

---

## **Saving & Loading the Model**

### **Saving after training**
The model automatically saves at the end of training:
```python
torch.save(model.state_dict(), "tennis_stroke_model.pth")
```

### **Loading for inference**
To load a trained model:
```python
from model import TennisStrokeClassification
import torch

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
model = TennisStrokeClassification().to(device)
model.load_state_dict(torch.load("tennis_stroke_model.pth", map_location = device))
model.eval()
```

---

## **Running the Streamlit App**
Once the model has been saved, run the app training script:
```bash
streamlit run app.py
```

---

## **Evaluation**
Model evaluation and initial experiments were done in **Jupyter Notebook** for:
- Data preprocessing and data loading
- Model architecture adjustments
- Image visualizations
- Sample predictions
- Loss/accuracy trend plots

Once the model was tuned, training was moved to Python scripts.

---

## **Original Custom Model Results**

| Metric       | Value (Final Epoch) |
|--------------|---------------------|
| Train Accuracy | **98.03%** |
| Test Accuracy  | **91.04%** |
| Test Loss      | **0.29** |

### **Performance Trend**
- Significant accuracy gains during the first 20 epochs.
- Steady improvement from 20–50 epochs with no signs of overfitting.
- Final plateau near **90% test accuracy**.

---

## **New EfficientNet Model Results**

| Metric       | Value (Final Epoch) |
|--------------|---------------------|
| Train Accuracy | **99.93%** |
| Test Accuracy  | **88.13%** |
| Test Loss      | **0.38** |

- Even though the results do not look as good as the original model, this model generalizes on new images much better

---

## **Future Improvements**
- Add early stopping to avoid unnecessary training or overfitting.
- Instead of using images, extract images frame-by-frame from videos.
- Get larger dataset with a larger variety of angles.

---

## **Acknowledgements**
Dataset provided by:
- [Mendeley Data - Tennis Strokes Dataset](https://data.mendeley.com/datasets/nv3rpsxhhk/1)

Citation:
    Wang, Chun-Yi; Lai, Kalin Guanlun; Huang, Hsu-Chun; Lin, Wei-Ting (2024), “Tennis Player Actions Dataset for Human Pose Estimation”, Mendeley Data, V1, doi: 10.17632/nv3rpsxhhk.1

This project was built as part of an exploration into **deep learning/computer vision for sports motion classification**.

---