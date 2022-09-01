from flask import Flask
from flask import send_file
from flask import send_from_directory
from flask import request
from flask import jsonify

import tensorflow as tf
import numpy as np

from model import compute_gradients
from model import SAVED_MODEL_PATH

app = Flask(__name__)

model = tf.keras.models.load_model(SAVED_MODEL_PATH)


@app.route("/")
def index():
    return send_file('index.html')


@app.route("/js/<path:path>")
def send_js(path):
    return send_from_directory('js', path)


@app.route("/classify", methods=['POST'])
def classify():
    json_body = request.get_json()
    image = np.array(json_body['imageData'], dtype=np.float32) / 255
    print(image)

    image = tf.image.resize(
        np.reshape(image, (1, 500, 500, 1)),
        (28, 28)
    )

    classification = model(image)[0]
    print(classification)
    print('Classification: {}'.format(np.argmax(classification)))
    response = {
        "classification": int(np.argmax(classification))
    }
    if json_body["returnAllProbabilities"]:
        response["probabilities"] = np.array(classification).tolist()
    if json_body["computeGradients"]:
        jacobian = np.array(compute_gradients(model, input_image=image)).astype(float)
        # Return the gradients as a dictionary of class to 28 x 28 matrix of gradients
        # Prior to returning, we need to convert the 2D numpy array into a nested list
        response["gradients"] = {
            i: list(map(list, jacobian[0, i, :, :, 0])) for i in range(10)
        }
    return jsonify(**response)
