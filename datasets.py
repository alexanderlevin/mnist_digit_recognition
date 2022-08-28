import tensorflow as tf
import tensorflow_datasets as tfds


def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255., label


def extract_and_preprocess_dataset(batch_size: int):
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    ds_train = ds_train.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE
    ).cache(
    ).shuffle(
        ds_info.splits["train"].num_examples
    ).batch(
        batch_size
    ).prefetch(
        tf.data.AUTOTUNE
    )

    ds_test = ds_test.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE
    ).batch(batch_size)

    return ds_train, ds_test