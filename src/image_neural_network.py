from __future__ import annotations

from typing import Iterable

import keras
import numpy as np
import tensorflow as tf
from keras import layers


LABEL_ORDER = [
    "supported",
    "refuted",
    "not_enough_information",
]

LABEL_TO_INDEX = {
    label: index
    for index, label in enumerate(LABEL_ORDER)
}

INDEX_TO_LABEL = {
    index: label
    for label, index in LABEL_TO_INDEX.items()
}


def encode_labels(
    labels: Iterable[str],
) -> np.ndarray:
    """Convert text labels to integer class indices."""

    encoded = []

    for label in labels:
        if label not in LABEL_TO_INDEX:
            raise ValueError(
                f"Unknown label: {label}"
            )

        encoded.append(
            LABEL_TO_INDEX[label]
        )

    return np.asarray(
        encoded,
        dtype=np.int32,
    )


def decode_predictions(
    predicted_indices: Iterable[int],
) -> list[str]:
    """Convert class indices back to text labels."""

    decoded = []

    for index in predicted_indices:
        integer_index = int(index)

        if integer_index not in INDEX_TO_LABEL:
            raise ValueError(
                f"Unknown class index: {integer_index}"
            )

        decoded.append(
            INDEX_TO_LABEL[integer_index]
        )

    return decoded


def load_and_resize_image(
    image_path: tf.Tensor,
    label: tf.Tensor,
    image_size: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Load one PNG image and resize it."""

    image_bytes = tf.io.read_file(
        image_path
    )

    image = tf.io.decode_png(
        image_bytes,
        channels=3,
    )

    image = tf.image.resize(
        image,
        [image_size, image_size],
    )

    image = tf.cast(
        image,
        tf.float32,
    )

    return image, label


def create_image_dataset(
    image_paths: Iterable[str],
    labels: Iterable[int],
    image_size: int = 96,
    batch_size: int = 16,
    shuffle: bool = False,
    random_state: int = 42,
) -> tf.data.Dataset:
    """Create a small tf.data image dataset."""

    path_array = np.asarray(
        list(image_paths),
        dtype=str,
    )

    label_array = np.asarray(
        list(labels),
        dtype=np.int32,
    )

    if len(path_array) == 0:
        raise ValueError(
            "Image paths cannot be empty."
        )

    if len(path_array) != len(label_array):
        raise ValueError(
            "Image paths and labels must have equal length."
        )

    dataset = tf.data.Dataset.from_tensor_slices(
        (path_array, label_array)
    )

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(path_array),
            seed=random_state,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        lambda image_path, label: (
            load_and_resize_image(
                image_path,
                label,
                image_size,
            )
        ),
        num_parallel_calls=1,
    )

    dataset = dataset.batch(
        batch_size
    )

    dataset = dataset.prefetch(1)

    return dataset


def build_image_neural_network(
    image_size: int = 96,
    dropout_rate: float = 0.30,
    learning_rate: float = 0.001,
) -> keras.Model:
    """Build and compile a small CNN classifier."""

    inputs = keras.Input(
        shape=(image_size, image_size, 3),
        name="chart_image",
    )

    x = layers.Rescaling(
        1.0 / 255,
        name="pixel_rescaling",
    )(inputs)

    x = layers.Conv2D(
        16,
        kernel_size=3,
        padding="same",
        activation="relu",
        name="conv_1",
    )(x)

    x = layers.MaxPooling2D(
        name="pool_1",
    )(x)

    x = layers.Conv2D(
        32,
        kernel_size=3,
        padding="same",
        activation="relu",
        name="conv_2",
    )(x)

    x = layers.MaxPooling2D(
        name="pool_2",
    )(x)

    x = layers.Conv2D(
        64,
        kernel_size=3,
        padding="same",
        activation="relu",
        name="conv_3",
    )(x)

    x = layers.GlobalAveragePooling2D(
        name="global_average_pooling",
    )(x)

    x = layers.Dropout(
        dropout_rate,
        name="first_dropout",
    )(x)

    x = layers.Dense(
        32,
        activation="relu",
        name="hidden_dense",
    )(x)

    x = layers.Dropout(
        dropout_rate,
        name="second_dropout",
    )(x)

    outputs = layers.Dense(
        len(LABEL_ORDER),
        activation="softmax",
        name="class_probabilities",
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="image_neural_network",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=learning_rate,
        ),
        loss=(
            "sparse_categorical_crossentropy"
        ),
        metrics=["accuracy"],
    )

    return model
