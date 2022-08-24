import argparse
import tensorflow as tf
import tensorflow_datasets as tfds
from model import build_model
from model import SAVED_MODEL_PATH

parser = argparse.ArgumentParser(description="Batch arguments")
parser.add_argument('--num-epochs', type=int, dest='num_epochs', default=5)
parser.add_argument('--batch-size', type=int, dest='batch_size', default=128)
args = parser.parse_args()


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
    )
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)
    return ds_train


def run():
    ds_train = extract_and_preprocess_dataset(args.batch_size)

    model = build_model(return_probabilities=True)

    model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False))
    model.fit(ds_train, epochs=args.num_epochs)
    model.save(SAVED_MODEL_PATH)


if __name__ == '__main__':
    run()
