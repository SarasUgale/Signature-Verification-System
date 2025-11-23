from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

app = Flask(__name__)
model = load_model('model/signature_model.h5', compile=False)

# Dynamically set a validation threshold 
THRESHOLD = 0.6  

def decode_image(file):
    """Convert uploaded image file to numpy array."""
    image = Image.open(file).convert('L').resize((150, 150))
    return np.array(image).reshape(1, 150, 150, 1) / 255.0

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/verify', methods=['POST'])
def verify():
    try:
        # Get the uploaded files
        signature1 = request.files['signature1']        
        signature2 = request.files['signature2']

        # Decode images from files
        img1 = decode_image(signature1)
        img2 = decode_image(signature2)

        # Predict similarity
        similarity = model.predict([img1, img2])[0][0]
        message = f'Similarity score: {similarity:.2f}'

        if similarity > THRESHOLD:
            message += ' (Signatures match!)'
        else:
            message += ' (Signatures do not match.)'

        return jsonify({'message': message})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
