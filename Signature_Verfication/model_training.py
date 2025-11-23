import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda, Activation, Subtract
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
from PIL import Image

# Define constants
DATASET_PATH = "dataset/"
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 20  # Increased epochs

def load_image(path):
    """Load and preprocess an image."""
    img = Image.open(path).convert('L')  # Convert to grayscale
    img = img.resize(IMG_SIZE)
    img = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    return img.reshape(IMG_SIZE[0], IMG_SIZE[1], 1)

def load_data():
    """Load genuine and forged signature pairs."""
    X1, X2, Y = [], [], []

    for person in os.listdir(DATASET_PATH):
        person_path = os.path.join(DATASET_PATH, person)
        if os.path.isdir(person_path):
            genuine_dir = os.path.join(person_path, "genuine")
            forged_dir = os.path.join(person_path, "forged")

            genuine_imgs = [os.path.join(genuine_dir, f) for f in os.listdir(genuine_dir)]
            forged_imgs = [os.path.join(forged_dir, f) for f in os.listdir(forged_dir)]

            for i in range(min(len(genuine_imgs), len(forged_imgs))):
                # Genuine vs Forged pair
                X1.append(load_image(genuine_imgs[i]))
                X2.append(load_image(forged_imgs[i]))
                Y.append(0)  # Non-matching pair

                if i < len(genuine_imgs) - 1:
                    # Genuine vs Genuine pair
                    X1.append(load_image(genuine_imgs[i]))
                    X2.append(load_image(genuine_imgs[i + 1]))
                    Y.append(1)  # Matching pair

    return np.array(X1), np.array(X2), np.array(Y)

def contrastive_loss(y_true, y_pred):
    """Contrastive loss for pairwise learning."""
    margin = 1.0
    return tf.reduce_mean(y_true * tf.square(y_pred) + (1 - y_true) * tf.square(tf.maximum(margin - y_pred, 0)))

def build_siamese_network(input_shape):
    """Build the Siamese Neural Network."""
    def feature_extractor():
        model = tf.keras.Sequential([
            Conv2D(64, (3, 3), activation='relu', input_shape=input_shape),
            MaxPooling2D(),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(),
            Conv2D(256, (3, 3), activation='relu'),
            MaxPooling2D(),
            Conv2D(512, (3, 3), activation='relu'),
            MaxPooling2D(),
            Flatten(),
            Dense(256, activation='relu'),
            Dense(128, activation='relu')
        ])
        return model

    cnn = feature_extractor()

    input_1 = Input(input_shape)
    input_2 = Input(input_shape)

    features_1 = cnn(input_1)
    features_2 = cnn(input_2)

    diff = Subtract()([features_1, features_2])
    diff = Activation('relu')(diff)

    output = Dense(1, activation='sigmoid')(diff)
    return Model(inputs=[input_1, input_2], outputs=output)

# Train the model
X1, X2, Y = load_data()
input_shape = (150, 150, 1)

# Data augmentation
datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1)
augmented_X1 = datagen.flow(X1, batch_size=BATCH_SIZE, shuffle=False)
augmented_X2 = datagen.flow(X2, batch_size=BATCH_SIZE, shuffle=False)

model = build_siamese_network(input_shape)
model.compile(optimizer=Adam(), loss=contrastive_loss, metrics=['accuracy'])
model.fit([X1, X2], Y, epochs=EPOCHS, batch_size=BATCH_SIZE)

model.save("model/signature_model.h5")
print("Model trained and saved!")
