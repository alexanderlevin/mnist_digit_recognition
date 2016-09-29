from flask import Flask
from flask import send_file
from flask import send_from_directory
from flask import request
from flask import jsonify

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import scipy

app = Flask(__name__)

sess = tf.Session()

saver = tf.train.import_meta_graph('model.ckpt.meta')
print "Restoring saved model"
saver.restore(sess, 'model.ckpt')

@app.route("/")
def index():
    return send_file('index.html')

@app.route("/js/<path:path>")
def send_js(path):
    return send_from_directory('js', path)

@app.route("/classify", methods=['POST'])
def classify():
    json_body = request.get_json()
    image = np.array(json_body, dtype=np.float32) / 256
    print image

    image = np.reshape(image, (500, 500))
    with tf.Session():
        resized_image = tf.image.resize_images(
            image[:, :, None], 28, 28).eval()[:, :, 0]

    classification = sess.run("y_conv:0",
        feed_dict={"x:0": np.reshape(resized_image, (1, 28 * 28)), "keep_prob:0": 1.})[0]
    print classification
    print 'Classification: %s'%np.argmax(classification)
    return jsonify(classification=np.argmax(classification))
