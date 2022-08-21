import argparse
import tensorflow as tf
import tensorflow_datasets as tfds

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

    model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    model.fit(ds_train, epochs=args.num_epochs)
    model.save("saved_models")


if __name__ == '__main__':
    run()
