import os

# =========================================================
# TensorFlow CPU / Render optimization
# =========================================================

os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from werkzeug.utils import secure_filename

from database import (
    create_database,
    add_prediction,
    get_predictions,
    get_statistics
)

# =========================================================
# Flask App
# =========================================================

app = Flask(__name__, template_folder="templates")

# =========================================================
# Paths
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "plant_disease_model.keras"
)

CLASS_NAMES_PATH = os.path.join(
    BASE_DIR,
    "model",
    "class_names.txt"
)

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "uploads"
)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Create upload folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =========================================================
# Database
# =========================================================

create_database()

# =========================================================
# Load AI Model
# =========================================================

print("Loading AI model...")

try:

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    print("AI model loaded successfully!")

except Exception as e:

    print("ERROR loading AI model:")
    print(e)
    model = None


# =========================================================
# Load Class Names
# =========================================================

try:

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:

        class_names = [
            line.strip()
            for line in file
            if line.strip()
        ]

    print("Number of classes:", len(class_names))

except Exception as e:

    print("ERROR loading class names:")
    print(e)

    class_names = []


# =========================================================
# Home
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================================================
# Upload Page
# =========================================================

@app.route("/upload")
def upload():

    return render_template(
        "upload.html"
    )


# =========================================================
# Prediction
# =========================================================

@app.route("/predict", methods=["POST"])
def predict():

    # -----------------------------------------------------
    # Check model
    # -----------------------------------------------------

    if model is None:

        return """
        <h2>AI model is not available.</h2>
        <p>Please try again later.</p>
        """, 500


    # -----------------------------------------------------
    # Check uploaded file
    # -----------------------------------------------------

    if "leaf_image" not in request.files:

        return """
        <h2>No image selected.</h2>
        <a href="/upload">Go Back</a>
        """, 400


    image = request.files["leaf_image"]


    if image.filename == "":

        return """
        <h2>No image selected.</h2>
        <a href="/upload">Go Back</a>
        """, 400


    # -----------------------------------------------------
    # Secure filename
    # -----------------------------------------------------

    filename = secure_filename(
        image.filename
    )


    # If filename becomes empty
    if not filename:

        return """
        <h2>Invalid image filename.</h2>
        <a href="/upload">Go Back</a>
        """, 400


    # -----------------------------------------------------
    # Save image
    # -----------------------------------------------------

    image_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    try:

        image.save(image_path)

    except Exception as e:

        print("Image save error:", e)

        return """
        <h2>Unable to save image.</h2>
        <a href="/upload">Go Back</a>
        """, 500


    # -----------------------------------------------------
    # Prepare image
    # -----------------------------------------------------

    try:

        img = tf.keras.utils.load_img(
            image_path,
            target_size=(128, 128)
        )

        img_array = tf.keras.utils.img_to_array(
            img
        )

        # Normalize pixel values
        img_array = img_array / 255.0

        img_array = np.expand_dims(
            img_array,
            axis=0
        )

    except Exception as e:

        print("Image processing error:", e)

        return """
        <h2>Invalid image.</h2>
        <p>Please upload a valid plant leaf image.</p>
        <a href="/upload">Try Again</a>
        """, 400


    # -----------------------------------------------------
    # AI Prediction
    # -----------------------------------------------------

    try:

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

    except Exception as e:

        print("Prediction error:", e)

        return """
        <h2>Prediction failed.</h2>
        <p>Please try another image.</p>
        <a href="/upload">Try Again</a>
        """, 500


    # -----------------------------------------------------
    # Get disease name
    # -----------------------------------------------------

    if predicted_index < len(class_names):

        disease = class_names[
            predicted_index
        ]

    else:

        disease = "Unknown"


    # -----------------------------------------------------
    # Plant status
    # -----------------------------------------------------

    if "healthy" in disease.lower():

        status = "Healthy Plant 🌿"

        information = (
            "The uploaded leaf appears healthy. "
            "Continue regular watering, proper nutrition "
            "and regular plant care."
        )

    else:

        status = "Possible Disease Detected ⚠️"

        information = (
            "The AI model detected a possible plant disease. "
            "For accurate confirmation and treatment, "
            "consult an agricultural expert."
        )


    # -----------------------------------------------------
    # Save prediction to database
    # -----------------------------------------------------

    try:

        add_prediction(
            filename,
            disease,
            confidence,
            status
        )

    except Exception as e:

        print("Database error:", e)


    # -----------------------------------------------------
    # Result Page
    # -----------------------------------------------------

    return render_template(

        "result.html",

        disease=disease,

        confidence=f"{confidence:.2f}",

        status=status,

        information=information,

        image_path="uploads/" + filename
    )


# =========================================================
# Dashboard
# =========================================================

@app.route("/dashboard")
def dashboard():

    try:

        predictions = get_predictions()

        total, healthy, diseased = get_statistics()

    except Exception as e:

        print("Dashboard database error:", e)

        predictions = []

        total = 0
        healthy = 0
        diseased = 0


    return render_template(

        "dashboard.html",

        predictions=predictions,

        total=total,

        healthy=healthy,

        diseased=diseased
    )


# =========================================================
# Health Check
# =========================================================

@app.route("/health")
def health():

    if model is not None:

        return {
            "status": "ok",
            "model": "loaded",
            "classes": len(class_names)
        }

    return {
        "status": "error",
        "model": "not loaded"
    }, 500


# =========================================================
# Run Application
# =========================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )