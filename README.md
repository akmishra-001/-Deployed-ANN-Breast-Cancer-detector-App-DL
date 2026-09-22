# 🩺 Breast Cancer Detection App using Artificial Neural Network (ANN)

An end-to-end Deep Learning web application built using **Artificial Neural Networks (ANN)**, **Keras/TensorFlow**, and **Streamlit** to predict whether a breast tumor is **Malignant** (cancerous) or **Benign** (non-cancerous) based on cell feature measurements.

🚀 **Live Demo**: https://oqisahfrdytzoyp3fgbvnb.streamlit.app/

---

## 📌 Project Overview

Breast cancer is one of the most common health challenges globally. Early diagnosis plays a critical role in effective treatment and recovery. This application utilizes a Deep Learning model to assist in preliminary diagnostic classification using standard clinical cell measurements (such as radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, and fractal dimension).

### Key Features
- **Live Deployment**: Hosted and running live on **Streamlit Cloud** for instant accessibility.
- **Deep Learning Model**: Trained with TensorFlow/Keras using dense neural network layers with ReLU activation and Dropout regularization.
- **Interactive Web Interface**: Developed using Streamlit for instant predictions and a smooth user experience.
- **Data Preprocessing**: Features normalized using `StandardScaler` to ensure optimum model accuracy.
- **Binary Classification**: Output clearly categorizes tumors into **Malignant** or **Benign** along with confidence probabilities.

---

## 📁 Repository Structure

```text
├── Breast_Cancer_Classification_with_ANN.ipynb  # Jupyter Notebook for EDA & Model Training
├── app.py                                        # Streamlit Web Application Code
├── breast_cancer_model.keras                     # Trained ANN Model File
├── scaler.pkl                                    # Fitted StandardScaler Object
├── requirements.txt                              # Required Python Dependencies
├── .gitignore                                    # Files excluded from Git tracking
└── README.md                                     # Project Documentation

🛠️ Tech Stack & Deployment
Language: Python

Deep Learning: TensorFlow, Keras

Data Processing: Pandas, NumPy

Preprocessing & Evaluation: Scikit-Learn

Web UI: Streamlit

Deployment Platform: Streamlit Cloud

🚀 How to Run Locally
1. Clone the Repository
Bash
git clone [https://github.com/akmishra-001/-Deployed-ANN-Breast-Cancer-detector-App-DL.git](https://github.com/akmishra-001/-Deployed-ANN-Breast-Cancer-detector-App-DL.git)
cd -Deployed-ANN-Breast-Cancer-detector-App-DL
2. Create and Activate Virtual Environment
Bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS / Linux
python3 -m venv venv
source venv/bin/activate
3. Install Dependencies
Bash
pip install -r requirements.txt
4. Run the Streamlit Application
Bash
streamlit run app.py
📊 Model Architecture
Input Layer: Preprocessed normalized feature vector

Hidden Layers: Dense layers with ReLU activation and Dropout regularization to prevent overfitting

Output Layer: Single Dense node with Sigmoid activation function

Loss Function: Binary Crossentropy

Optimizer: Adam Optimizer

👤 Author
Abhishek Mishra

GitHub: @akmishra-001
