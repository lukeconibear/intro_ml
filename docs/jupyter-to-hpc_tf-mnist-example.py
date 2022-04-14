
import os

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

# global setup
tf.keras.utils.set_random_seed(42)
print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
AUTOTUNE = tf.data.AUTOTUNE
NUM_EPOCHS = 5

# download the data
def download_data():
    (ds_train, ds_val, ds_test) = tfds.load(
        "mnist",
        split=["train[:80%]", "train[80%:90%]", "train[90%:]"],
        shuffle_files=True,
        as_supervised=True,
        with_info=False,
    )
    return ds_train, ds_val, ds_test


# preprocess the data
def normalise_image(image, label):
    return tf.cast(image, tf.float32) / 255.0, label


# create data pipelines
def training_pipeline(ds_train):
    ds_train = ds_train.map(normalise_image, num_parallel_calls=AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(AUTOTUNE)
    return ds_train


def test_pipeline(ds_test):
    ds_test = ds_test.map(normalise_image, num_parallel_calls=AUTOTUNE)
    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(AUTOTUNE)
    return ds_test


# create and compile the model
def create_and_compile_model():
    inputs = tf.keras.Input(shape=(28, 28, 1), name="inputs")
    x = tf.keras.layers.Flatten(name="flatten")(inputs)
    x = tf.keras.layers.Dense(128, activation="relu", name="layer1")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="layer2")(x)
    outputs = tf.keras.layers.Dense(10, name="outputs")(x)

    model = tf.keras.Model(inputs, outputs, name="functional")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    return model


# train the model
def train_model():
    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=NUM_EPOCHS,
        verbose=False,
    )
    return history


# save the model
def save_model(model):
    path_models = f"{os.getcwd()}/models"
    model.save(f"{path_models}/model_tf_mnist")


# combine the functions in a call to main
def main():
    ds_train, ds_val, ds_test = download_data()

    ds_train = training_pipeline(ds_train)
    ds_val = training_pipeline(ds_val)
    ds_test = test_pipeline(ds_test)

    model = create_and_compile_model()
    history = train_model()
    save_model(model)


if __name__ == "__main__":
    # run the functions
    main()

    # view the model accuracy
    print(
        f"Training accuracy: {[round(num, 2) for num in history.history['accuracy']]}"
    )
    print(
        f"Validation accuracy: {[round(num, 2) for num in history.history['val_accuracy']]}"
    )
