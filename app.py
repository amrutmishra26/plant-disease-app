from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

app = Flask(__name__)
model = load_model("model/final_model.h5")

CLASS_NAMES = ["Class 1", "Class 2", "Class 3"]  # Replace with actual class names

def predict_image(img_path):
    img = Image.open(img_path).resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    return CLASS_NAMES[class_index]

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filepath = os.path.join("static", file.filename)
            file.save(filepath)
            result = predict_image(filepath)
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
