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
            tf.keras.layers.Conv2D(32, 5, padding="SAME"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(64, 5, padding="SAME"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Reshape((7 * 7 * 64,)),
            tf.keras.layers.Dense(1024),
            tf.keras.layers.Dense(10)
        ]
    )

    if return_probabilities:
        model.add(tf.keras.layers.Softmax())

    return model
