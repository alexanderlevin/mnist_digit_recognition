from flask import Flask
from flask import send_file
from flask import send_from_directory
from flask import request
from flask import jsonify

import tensorflow as tf
import numpy as np

app = Flask(__name__)

model = tf.keras.models.load_model("saved_models")

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
    return jsonify(**response)