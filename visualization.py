from model import compute_gradients
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf


def find_image_to_visualize(model, dataset, desired_class, probability_threshold=0.9):
    """Find a promising image of a given class to visualize along with its gradients.

    Our criteria are:
    - The image is in the `desired_class`
    - Its probability is less than probability_threshold (0.9 by default).  This ensures that the
       model is at least somewhat uncertain about the class, which, in our experience, leads to more
       interesting gradients

    Example:
       >>> _, ds_test = extract_and_preprocess_dataset(128)
       >>> ds_test = ds_test.unbatch()
       >>> image, probabilities = find_image_to_visualize(model, ds_test, desired_class=1)

    :param model: The keras digit classification model
    :param dataset: The dataset (un-batched), consisting of images (normalized such that pixel values are between 0
        and 1) and labels
    :param desired_class: We return an image in this class
    :param probability_threshold: The probability of the image being in `desired_class`, according to the `model`,
        must be below this threshold
    :return: The image with the required properties, in normalized form (entries between 0 and 1).
       The return value is of dimension (28, 28, 1); the last dimension is the channel dimension.
    """
    filtered_dataset = dataset.filter(
        lambda _, label: label == desired_class
    ).map(
        lambda image, _: image
    ).batch(128)

    dataset_and_predictions = tf.data.Dataset.zip(
        (filtered_dataset.unbatch(),
         tf.data.Dataset.from_tensor_slices(model.predict(filtered_dataset)))
    ).shuffle(1024)

    return dataset_and_predictions.filter(
        lambda image, predictions: predictions[desired_class] < probability_threshold
    ).as_numpy_iterator().next()


def visualize_gradients(model, image, classes_to_visualize, figure_height):
    """Visualize the input image, along with gradients corresponding to certain probability outputs.

    :param model: The keras model
    :param image: The input image; an array of rank 2, 3, or 4.  If the rank is less than 4, the batch dimension and/or
        the channel dimension are added as needed.  The image must be normalized, with pixel values between 0 and 1
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


def find_image_optimizing_class_probability(model, initial_image, target_class, n_iterations, learning_rate):
    """Optimize an output class by gradient ascent

    Starting with an initial image, perform gradient ascent to maximize a model's output for a given class.
    :param model: The keras model
    :param initial_image: Initial image, of dimensions (1, 28, 28, 1)
    :param target_class: Class whose probability (or other model output) you want to optimize
    :param n_iterations: Total number of iterations
    :param learning_rate: Learning rate
    :return: The final image
    """
    current_image = initial_image
    for _ in range(n_iterations):
        gradients = compute_gradients(model, current_image)[:, target_class, :, :, :]
        current_image += learning_rate * gradients
        current_image = tf.clip_by_value(current_image, 0, 1)

    return np.array(current_image)
