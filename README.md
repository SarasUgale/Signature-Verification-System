# Signature-Verification-System
✒️ Signature Verification System
🔍 Deep Learning (Siamese Network) + Flask + Modern UI
A powerful Signature Verification System that uses a Siamese Neural Network to determine whether two signatures belong to the same person.
Built with Flask, TensorFlow/Keras, and a modern UI for seamless user experience.

🌟 Features

🧠 Deep Learning model (Siamese Network + Contrastive Loss)

🖥️ Flask backend API for real-time verification

📸 Upload two signatures and get similarity result

📊 Animated similarity progress bar

🔍 Clear output:

✔️ Signatures Match

❌ Signatures Do Not Match

📁 Custom dataset support

🧪 Easy model training

🎨 Beautiful, responsive UI

📂 Project Structure
```
signature-verification/
│
├── app.py                   # Flask backend
├── model_training.py        # Siamese model training
│
├── model/
│   └── signature_model.h5   # Saved model
│
├── dataset/
│   ├── person1/
│   │   ├── genuine/
│   │   └── forged/
│   ├── person2/
│       ├── genuine/
│       └── forged/
│
├── templates/
│   └── index.html           # Frontend UI
│
├── static/                  # CSS / JS, assets
│
├── requirements.txt
└── README.md
```
🧰 Tech Stack

```
Backend

Python

Flask

TensorFlow / Keras

NumPy

Pillow

Frontend

HTML5

CSS3

Bootstrap 5

JavaScript (Fetch API)
```

⚙️ Installation
1️⃣ Clone the Repository
git clone https://github.com/your-username/signature-verification.git
cd signature-verification

2️⃣ Create Virtual Environment
```
python -m venv venv


Activate:

Windows

venv\Scripts\activate

```
```
Mac/Linux

source venv/bin/activate
```
3️⃣ Install Dependencies
```
pip install -r requirements.txt
```
📁 Dataset Format
```
dataset/
├── person1/
│   ├── genuine/
│   └── forged/
├── person2/
    ├── genuine/
    └── forged/

```
Each folder contains multiple signature images.

🧠 Train the Model

Run:
```
python model_training.py

```
This will:

✔ Load dataset
✔ Train Siamese Network
✔ Save model to:

model/signature_model.h5

▶️ Run the Flask App

```
python app.py
```

Then open in browser:

👉 http://127.0.0.1:5000/

🖼️ How It Works

User uploads 2 signatures

Images are preprocessed:

Grayscale

Resized to 150×150

Normalized

Siamese Network predicts similarity

UI displays:

Percentage match

Result message

📊 Threshold System
THRESHOLD = 0.6


score > 0.6 → Match

score ≤ 0.6 → Not a Match

Adjust based on dataset quality.

📡 API Endpoint
POST /verify

Request

signature1 (file)

signature2 (file)

Response

{
  "message": "Similarity score: 0.82 (Signatures match!)"
}

🔮 Future Enhancements

📱 Mobile-friendly UI

🗄 Database to store signature history

🔧 Auto-denoise / thresholding

🌐 Cloud-hosted version

🖥 Dashboard for multiple signature comparison

📝 License

This project is licensed under the MIT License.

👨‍💻 Author

Saras Ugale
Signature Verification — Deep Learning + Flask

⭐ Like the Project?

If this project helped you, consider giving it a star ⭐ on GitHub!
