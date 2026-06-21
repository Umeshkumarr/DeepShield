from flask import Flask, render_template, request
import tensorflow as tf
import cv2
import numpy as np
import os

app = Flask(__name__)

MODEL_PATH = os.path.join("model","model", "deepfake_model.h5")
model = tf.keras.models.load_model(MODEL_PATH)

UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def home():
    return render_template("home.html")

@app.route('/image')
def image():
    return render_template("image.html")

@app.route('/video')
def video():
    return render_template("video.html")

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    if 'file' not in request.files:
        return "No file uploaded"
    
    file = request.files['file']
    filename = file.filename.lower()

# IMAGE PROCESSING
    if filename.endswith(('.jpg', '.png', '.jpeg')):

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], "input.jpg")
        file.save(filepath)
        img = cv2.imread(filepath)
        img = cv2.resize(img, (224,224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        prediction = model.predict(img)
        prob = float(prediction[0][0])

# VIDEO PROCESSING
    elif filename.endswith(('.mp4', '.avi', '.mov')):

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], "input.mp4")
        file.save(filepath)
        cap = cv2.VideoCapture(filepath)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (224,224))
            frame = frame / 255.0
            frames.append(frame)
            if len(frames) >= 20:
                break

        cap.release()
        if len(frames) == 0:
            return "Error processing video"

        frames = np.array(frames)
        predictions = model.predict(frames)
        prob = float(np.median(predictions))  

    else:
        return "Unsupported file format"

    
    if prob > 0.5:
        result = "Fake"
        confidence = prob * 100
    else:
        result = "Real"
        confidence = (1 - prob) * 100

    confidence = round(confidence, 2)

    return render_template( 
    "result.html",
    result=result,
    confidence=confidence
    )

if __name__ == "__main__":
    app.run(debug=True)