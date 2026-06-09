# 🛡️ SignaTrust AI: Smart Signature Authentication Platform

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg">
  <img src="https://img.shields.io/badge/Flask-Web%20Framework-green.svg">
  <img src="https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange.svg">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg">
</p>

## 📌 Overview

**SignaTrust AI** is an intelligent signature authentication platform that leverages Deep Learning to verify the authenticity of handwritten signatures. The system uses a Siamese Neural Network architecture to learn signature similarity patterns and determine whether two signatures belong to the same individual.

## 🚀 Key Features

- 🤖 AI-Powered Signature Authentication
- 🔒 Real-Time Signature Verification
- 📊 Interactive Similarity Score Dashboard
- 🌐 Flask-Based Web Application
- 🧩 Custom Dataset Support
- 🎨 Modern Responsive UI

## 🏗️ System Architecture

```text
User Upload
      │
      ▼
Image Preprocessing
      │
      ▼
Feature Extraction
      │
      ▼
Siamese Neural Network
      │
      ▼
Similarity Computation
      │
      ▼
Verification Result
```

## 📂 Project Structure

```text
SignaTrust-AI/
│
├── app.py
├── model_training.py
├── requirements.txt
├── README.md
│
├── model/
│   └── signature_model.h5
│
├── dataset/
│   ├── person1/
│   │   ├── genuine/
│   │   └── forged/
│
├── templates/
│   └── index.html
│
└── static/
    ├── script.js
    └── style.css
```

## 🛠️ Technology Stack

### Backend
- Python
- Flask
- TensorFlow
- Keras

### Frontend
- HTML5
- CSS3
- JavaScript
- Bootstrap

### Machine Learning
- Siamese Neural Network
- Contrastive Loss

## ⚙️ Installation

```bash
git clone https://github.com/yourusername/SignaTrust-AI.git
cd SignaTrust-AI

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

## 🧠 Train the Model

```bash
python model_training.py
```

## ▶️ Run the Application

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## 📊 API Endpoint

### POST /verify

Request:
- signature1 (file)
- signature2 (file)

Response:

```json
{
  "message": "Similarity Score: 0.82 (Authentic Signature)"
}
```

## 📸 Screenshots

### 🏠 Home Dashboard

<p align="center">
  <img src=""C:\Users\SARAS\Pictures\Screenshots\dashboard.png"" alt="Home Dashboard" width="900">
</p>

### 🔍 Verification Analysis

<p align="center">
  <img src=""C:\Users\SARAS\Pictures\Screenshots\verification-result.png"" alt="Verification Analysis" width="900">
</p>

## 🎯 Applications

- Banking Authentication
- Document Verification
- Financial Security
- Identity Verification
- Enterprise Authentication

## 🔮 Future Enhancements

- Explainable AI Visualizations
- Verification History Dashboard
- PDF Reports
- Cloud Deployment
- Mobile Application

## 👨‍💻 Author

**Saras Ugale**

AI/ML Engineer | Machine Learning Enthusiast

## ⭐ Support

If you found this project useful, please consider giving it a star ⭐ on GitHub.

## 📄 License

This project is licensed under the MIT License.
