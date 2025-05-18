import argparse
import os
from datetime import datetime

from datasets import extract_and_preprocess_dataset
import tensorflow as tf
from model import build_model
from model import SAVED_MODEL_PATH

parser = argparse.ArgumentParser(description="Batch arguments")
parser.add_argument('--num-epochs', type=int, dest='num_epochs', default=5)
parser.add_argument('--batch-size', type=int, dest='batch_size', default=128)
args = parser.parse_args()


def run():
    now = datetime.now()
    model_save_path = os.path.join(SAVED_MODEL_PATH, now.strftime("%Y%m%d-%H%M"))

    ds_train, ds_test = extract_and_preprocess_dataset(args.batch_size)

    model = build_model(return_probabilities=True)

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics="sparse_categorical_accuracy"
    )
    model.fit(ds_train, epochs=args.num_epochs, validation_data=ds_test)
    model.save(model_save_path)


if __name__ == '__main__':
    run()
