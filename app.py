from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
model = load_model("model_v3_mobilenet.h5")
classes = ["Kobe Bryant", "Maria Sharapova", "Cristiano Ronaldo"]

def prepare_image(img_path):
    img = load_img(img_path, target_size=(224, 224)) 
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No se encontró archivo"}), 400

    file = request.files['file']
    if file.filename == "":
        return jsonify({"error": "Nombre de archivo vacío"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join("uploads", filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(filepath)

    image = prepare_image(filepath)
    prediction = model.predict(image)
    predicted_class = classes[np.argmax(prediction)]

    os.remove(filepath)

    return jsonify({"label": predicted_class})

if __name__ == "__main__":
    app.run(debug=True)
