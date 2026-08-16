from flask import Flask, render_template, request
import os

# -----------------------------
# TensorFlow Optimization
# -----------------------------

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

import tensorflow as tf
import numpy as np

from database import (
    create_database,
    add_prediction,
    get_predictions,
    get_statistics
)

app = Flask(__name__, template_folder="templates")

# -----------------------------
# Paths
# -----------------------------

MODEL_PATH = "model/plant_disease_model.keras"
CLASS_NAMES_PATH = "model/class_names.txt"
UPLOAD_FOLDER = "static/uploads"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -----------------------------
# Create Database
# -----------------------------

create_database()

# -----------------------------
# Load AI Model
# -----------------------------

print("Loading AI model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

with open(CLASS_NAMES_PATH, "r") as file:
    class_names = [
        line.strip()
        for line in file.readlines()
        if line.strip()
    ]

print("AI model loaded successfully!")
print("Number of classes:", len(class_names))


# -----------------------------
# Home
# -----------------------------

@app.route("/")
def home():
    return render_template("index.html")


# -----------------------------
# Upload
# -----------------------------

@app.route("/upload")
def upload():
    return render_template("upload.html")


# -----------------------------
# Prediction
# -----------------------------

@app.route("/predict", methods=["POST"])
def predict():

    if "leaf_image" not in request.files:
        return "No image selected."

    image = request.files["leaf_image"]

    if image.filename == "":
        return "No image selected."

    # -----------------------------
    # Save Uploaded Image
    # -----------------------------

    filename = image.filename

    image_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    image.save(image_path)

    # -----------------------------
    # Prepare Image
    # -----------------------------

    img = tf.keras.utils.load_img(
        image_path,
        target_size=(128, 128)
    )

    img_array = tf.keras.utils.img_to_array(img)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    # Normalize image
    img_array = img_array / 255.0

    # -----------------------------
    # AI Prediction
    # -----------------------------

    predictions = model.predict(
        img_array,
        verbose=0
    )

    predicted_index = int(
        np.argmax(predictions[0])
    )

    confidence = float(
        np.max(predictions[0]) * 100
    )

    # Safety check
    if predicted_index >= len(class_names):
        return "Prediction class not found."

    disease = class_names[predicted_index]

    # -----------------------------
    # Status
    # -----------------------------

    if "healthy" in disease.lower():

        status = "Healthy Plant 🌿"

        information = (
            "The uploaded leaf appears healthy. "
            "Continue regular watering, nutrition and plant care."
        )

    else:

        status = "Possible Disease Detected ⚠️"

        information = (
            "The AI model detected a possible plant disease. "
            "Consider proper plant care and consult an agricultural "
            "expert for confirmation and treatment."
        )

    # -----------------------------
    # Save Prediction
    # -----------------------------

    add_prediction(
        filename,
        disease,
        confidence,
        status
    )

    # -----------------------------
    # Result Page
    # -----------------------------

    return render_template(
        "result.html",
        disease=disease,
        confidence=f"{confidence:.2f}",
        status=status,
        information=information,
        image_path="uploads/" + filename
    )


# -----------------------------
# Dashboard
# -----------------------------

@app.route("/dashboard")
def dashboard():

    predictions = get_predictions()

    total, healthy, diseased = get_statistics()

    return render_template(
        "dashboard.html",
        predictions=predictions,
        total=total,
        healthy=healthy,
        diseased=diseased
    )


# -----------------------------
# Run Flask
# -----------------------------

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )