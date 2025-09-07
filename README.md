# **Tennis Stroke Multi-Class Classification (Convolutional Neural Networks w/ PyTorch)**
A deep learning project to classify tennis images into four categories — **forehand, backhand, serve, and ready position** — using a custom convolutional neural network (CNN).  
This project was initially developed and evaluated in a **Jupyter Notebook**, then split into **Python** scripts for production.
Once model was trained and saved using **PyTorch**, model was deployed using **Streamlit**.

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
The goal of this project is to develop a Convolutional Neural Network that can accurately classify tennis strokes from images. Images are taken from behind the player.

### **Key Features**
- Custom CNN architecture with **Conv2d**, **BatchNorm**, **Dropout**, and **AdaptiveAvgPool2d** layers.
- Training pipeline with:
  - Adam optimizer + weight decay
  - Learning rate scheduler (`StepLR`)
  - Cross-entropy loss (`torch.nn.CrossEntropyLoss`)
- Data pipeline using `torchvision.datasets.ImageFolder`.
- Python scripts with clean separation of:
  - Dataset preprocessing and transformations
  - Model architecture
  - Training and testing loops
  - Utility functions
- Supports **Apple MPS GPU acceleration** for Mac users. Nvidia GPU acceleration should also work.
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

### **Structure After Processing**
Images were split into `train` and `test` sets at a **75/25 ratio**.
This is done using file data_splitting.py which uses **scikit-learn** `train_test_split`:

```
dataset/
│
├── train_set/
│   ├── forehand/
│   ├── backhand/
│   ├── serve/
│   └── ready_position/
│
└── test_set/
    ├── forehand/
    ├── backhand/
    ├── serve/
    └── ready_position/
```

---

## **Project Structure**
```
project_root/
│
├── main.py               # Entry point for training and saving the model
├── app.py                # Model deployment using Streamlit
├── model.py              # CNN model definition
├── dataset.py            # Dataloader and data transformation functions
├── train_test_loop.py    # Training and evaluation loops
├── extra_functions.py    # Accuracy function
├── data_splitting.py     # Used to create the structure after processing
│
├── tennis_stroke_model.pth  # (Generated after training, saved model weights)
│
├── requirements.txt      # Library/Framework requirements for running Notebook and Scripts
└── README.md             # Project documentation
```

---

## **Model Architecture**
The CNN is composed of:

| Layer Group      | Details |
|------------------|---------|
| **Feature Extractor** | 5 convolutional blocks with Conv2D → BatchNorm → ReLU Non-Linear Activation → MaxPool |
| **Adaptive Pooling**  | AdaptiveAvgPool2d output forces to **8x8** spatial size |
| **Classifier**   | Flatten → ReLU Non-Linear Activation → Output (logits) |

---

## **Setup Instructions**

### **1. Clone the repository**
```bash
git clone https://github.com/yourusername/tennis-stroke-classification.git
cd tennis-stroke-classification
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
- Training runs for **50 epochs**.
- `StepLR` scheduler halves the learning rate every 7 epochs.
- Final model weights and biases are saved to `tennis_stroke_model.pth`.

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

## **Results**

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

## **Future Improvements**
- Add early stopping to avoid unnecessary training or overfitting.
- Experiment with pretrained models like ResNet or EfficientNet.
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