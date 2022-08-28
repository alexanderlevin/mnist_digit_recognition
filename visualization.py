from model import compute_gradients
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf


def find_image_to_visualize(model, dataset, desired_class, probability_threshold=0.9):
    """Find a promising image of a given class to visualize along with its gradients.

    Our criteria are:
    - The image is in the `desired_class`
    - Its probability is less than probability_threshold (0.9 by default).  This ensures that the
       model is at least somewhat uncertain about the class, which, in our experience, leads to more
       interesting gradients

    :param model: The keras digit classification model
    :param dataset: The dataset (un-batched)
    :param desired_class: We return an image in this class
    :param probability_threshold: The probability of the image being in `desired_class`, according to the `model`,
        must be below this threshold
    :return: The image with the required properties, in normalized form (entries between 0 and 1).
       The return value is of dimension (1, 28, 28, 1); the first dimension is the batch dimension, and the last
       dimension is the channel dimension.
    """
    filtered_dataset = dataset.filter(
        lambda _, label: label == desired_class
    ).map(
        lambda image, _: tf.cast(image, tf.float32) / 255.
    ).batch(128)

    dataset_and_predictions = tf.data.Dataset.zip(
        (filtered_dataset.flat_map(tf.data.Dataset.from_tensor_slices),
         tf.data.Dataset.from_tensor_slices(model.predict(filtered_dataset)))
    ).shuffle(1024)

    return dataset_and_predictions.filter(
        lambda image, predictions: predictions[desired_class] < probability_threshold
    ).as_numpy_iterator().next()


def visualize_gradients(model, image, classes_to_visualize, figure_height):
    """Visualize the input image, along with gradients corresponding to certain probability outputs.

    :param model: The keras model
    :param image: The input image; an array of rank 2, 3, or 4.  If the rank is less than 4, the batch dimension and/or
        the channel dimension are added as needed
    :param classes_to_visualize: Classes whose probability gradients we will analyze.  We assume this is
        an array of size 2
    :param figure_height: Height of the figure we display
    :return: None
    """

    if len(image.shape) == 2:
        image = image[None, :, :, None]
    elif len(image.shape) == 3:
        image = image[None, :, :, :]
    elif len(image.shape != 4):
        raise ValueError("`image` must be an array of rank 2, 3, or 4")

    jacobian = compute_gradients(model, image)
    min_gradient, max_gradient = (
        np.min(np.array(jacobian)[0, np.array(classes_to_visualize), :, :, 0]),
        np.max(np.array(jacobian)[0, np.array(classes_to_visualize), :, :, 0])
    )

    _, ax = plt.subplots(1, 3, figsize=(figure_height * 3, figure_height))
    sns.heatmap(image[0, :, :, 0], ax=ax[0])
    sns.heatmap(jacobian[0, classes_to_visualize[0], :, :, 0], vmin=min_gradient, vmax=max_gradient, ax=ax[1])
    sns.heatmap(jacobian[0, classes_to_visualize[1], :, :, 0], vmin=min_gradient, vmax=max_gradient, ax=ax[2])

