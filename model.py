import tensorflow as tf

SAVED_MODEL_PATH = "saved_models"


def build_model(return_probabilities=False):
    """Build a simple convolutional neural network model

    This simple model has two convolutional layers and two fully-connected layers

    :param return_probabilities: Whether to return logits (default)
        or probabilities (computed via the softmax function)
    """

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Resizing(28, 28),
            tf.keras.layers.Conv2D(32, 5, padding="SAME", activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(64, 5, padding="SAME", activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Reshape((7 * 7 * 64,)),
            tf.keras.layers.Dense(1024, activation="relu"),
            tf.keras.layers.Dense(10, activation=None)
        ]
    )

    if return_probabilities:
        model.add(tf.keras.layers.Softmax())

    return model


def compute_gradients(model, input_image):
    """Compute gradients of output probabilities with respect to input data

    :param model: Keras model
    :param input_image: Input image, represented as an array of shape (1, 28, 28, 1)
    :return: The gradients, a tensor of shape (1, 10, 28, 28, 1).  If J is the output,
       then J[0, class, :, :, 0] is the gradient of the predicted probability of `class`
       with respect to the pixel values of the input image.
    """
    input_tensor = tf.convert_to_tensor(input_image)
    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        probabilities = model(input_tensor)

    return tape.batch_jacobian(probabilities, input_tensor)
