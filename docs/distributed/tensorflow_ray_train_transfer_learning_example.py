import argparse
import json
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import tensorflow_hub as hub
from ray.train import Trainer

BATCH_SIZE = 32
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
AUTOTUNE = tf.data.AUTOTUNE
NUM_CLASSES = 5
os.environ["TFHUB_CACHE_DIR"] = f"{os.getcwd()}/tf_hub_modules"


class TrainReportCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        ray.train.report(**logs)


def create_dataset(BATCH_SIZE):
    data_root = tf.keras.utils.get_file(
        "flower_photos",
        "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz",
        untar=True,
    )

    ds_train = tf.keras.utils.image_dataset_from_directory(
        str(data_root),
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        batch_size=BATCH_SIZE,
    )

    normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
    ds_train = ds_train.map(lambda x, y: (normalization_layer(x), y))
    ds_train = ds_train.cache().prefetch(buffer_size=AUTOTUNE)

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.DATA
    )
    ds_train = ds_train.with_options(options)

    return ds_train


def build_and_compile_model(config):
    learning_rate = config.get("lr", 0.001)
    inception_v3 = "https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4"
    feature_extractor_model = inception_v3
    feature_extractor_layer = hub.KerasLayer(
        feature_extractor_model,
        input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3),
        trainable=False,
    )
    model = tf.keras.Sequential(
        [feature_extractor_layer, tf.keras.layers.Dense(NUM_CLASSES)]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def train_func(config):
    per_worker_batch_size = config.get("batch_size", 64)
    epochs = config.get("epochs", 3)
    steps_per_epoch = config.get("steps_per_epoch", 70)

    tf_config = json.loads(os.environ["TF_CONFIG"])
    num_workers = len(tf_config["cluster"]["worker"])

    strategy = tf.distribute.MultiWorkerMirroredStrategy()

    global_batch_size = per_worker_batch_size * num_workers
    multi_worker_dataset = create_dataset(global_batch_size)

    with strategy.scope():
        # Model building/compiling need to be within `strategy.scope()`.
        multi_worker_model = build_and_compile_model(config)

    history = multi_worker_model.fit(
        multi_worker_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=[TrainReportCallback()],
        verbose=False,
    )
    results = history.history
    return results


def train_tensorflow_transfer_learning(num_workers=2, use_gpu=False, epochs=4):
    trainer = Trainer(backend="tensorflow", num_workers=num_workers, use_gpu=use_gpu)
    trainer.start()
    results = trainer.run(
        train_func=train_func, config={"lr": 1e-3, "batch_size": 64, "epochs": epochs}
    )
    trainer.shutdown()
    print(f"Results: {results[0]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--address", required=False, type=str, help="the address to use for Ray"
    )
    parser.add_argument(
        "--num-workers",
        "-n",
        type=int,
        default=2,
        help="Sets number of workers for training.",
    )
    parser.add_argument(
        "--use-gpu", action="store_true", default=False, help="Enables GPU training"
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of epochs to train for."
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        default=False,
        help="Finish quickly for testing.",
    )

    args, _ = parser.parse_known_args()

    import ray

    if args.smoke_test:
        ray.init(num_cpus=2)
        train_tensorflow_transfer_learning()
    else:
        ray.init(address=args.address)
        train_tensorflow_transfer_learning(
            num_workers=args.num_workers, use_gpu=args.use_gpu, epochs=args.epochs
        )
