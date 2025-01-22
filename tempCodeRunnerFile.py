import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DEBUG'] = True  # Enable Flask debug mode

# Load the pre-trained Keras model
model = load_model('my_model.keras')

# Define the class names for your model
class_names = ['A-', 'AB-', 'AB+','A+', 'B-', 'B+', 'O-', 'O+']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Load the image and preprocess it
            img = image.load_img(filepath, target_size=(64, 64))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0

            # Make a prediction
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions, axis=1)[0]
            predicted_class_name = class_names[predicted_class]

            os.remove(filepath)  # Clean up the uploaded file

            return jsonify({"predicted_class": predicted_class_name})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/match', methods=['POST'])
def match():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Process the image for fingerprint matching
            img = cv2.imread(filepath, 0)
            img = cv2.resize(img, (100, 100))

            st1 = '1,' + ','.join(str(int(img[i, j])) for i in range(img.shape[0]) for j in range(img.shape[1]))

            found = False
            with open('fingerprint_train.csv', 'r') as file1:
                lines = file1.readlines()
                for line in lines[1:]:
                    tokens = line.strip().split(',')
                    record_data = ','.join(tokens[:-2])
                    if record_data == st1:
                        group = tokens[-2]
                        found = True
                        break

            os.remove(filepath)  # Clean up the uploaded file

            if found:
                return jsonify({"result": f"Matched in group: {group}"})
            else:
                return jsonify({"result": "No match found"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)  # Enable debug mode
