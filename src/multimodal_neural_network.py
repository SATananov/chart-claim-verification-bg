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


def create_text_vectorizer(
    training_text: Iterable[str],
    max_tokens: int = 1000,
    sequence_length: int = 24,
) -> layers.TextVectorization:
    """Adapt a vectorizer using training text only."""

    text_array = np.asarray(
        list(training_text),
        dtype=str,
    )

    if len(text_array) == 0:
        raise ValueError(
            "Training text cannot be empty."
        )

    vectorizer = layers.TextVectorization(
        max_tokens=max_tokens,
        standardize=(
            "lower_and_strip_punctuation"
        ),
        output_mode="int",
        output_sequence_length=(
            sequence_length
        ),
        name="text_vectorization",
    )

    vectorizer.adapt(text_array)

    return vectorizer


def _load_multimodal_example(
    image_path: tf.Tensor,
    claim_text: tf.Tensor,
    label: tf.Tensor,
    image_size: int,
) -> tuple[dict[str, tf.Tensor], tf.Tensor]:
    """Load one image and keep its matching claim text."""

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

    inputs = {
        "claim_text": claim_text,
        "chart_image": image,
    }

    return inputs, label


def create_multimodal_dataset(
    image_paths: Iterable[str],
    claim_texts: Iterable[str],
    labels: Iterable[int],
    image_size: int = 96,
    batch_size: int = 16,
    shuffle: bool = False,
    random_state: int = 42,
) -> tf.data.Dataset:
    """Create a tf.data dataset for image and text inputs."""

    path_array = np.asarray(
        list(image_paths),
        dtype=str,
    )

    text_array = np.asarray(
        list(claim_texts),
        dtype=str,
    )

    label_array = np.asarray(
        list(labels),
        dtype=np.int32,
    )

    example_count = len(path_array)

    if example_count == 0:
        raise ValueError(
            "The multimodal dataset cannot be empty."
        )

    if not (
        example_count
        == len(text_array)
        == len(label_array)
    ):
        raise ValueError(
            "Image paths, claim texts and labels "
            "must have equal length."
        )

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            path_array,
            text_array,
            label_array,
        )
    )

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=example_count,
            seed=random_state,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        lambda image_path, claim_text, label: (
            _load_multimodal_example(
                image_path=image_path,
                claim_text=claim_text,
                label=label,
                image_size=image_size,
            )
        ),
        num_parallel_calls=1,
    )

    dataset = dataset.batch(
        batch_size
    )

    dataset = dataset.prefetch(1)

    return dataset


def build_multimodal_neural_network(
    vectorizer: layers.TextVectorization,
    image_size: int = 96,
    embedding_dimension: int = 32,
    text_units: int = 32,
    combined_units: int = 64,
    dropout_rate: float = 0.30,
    learning_rate: float = 0.001,
) -> keras.Model:
    """Build and compile a text-and-image classifier."""

    vocabulary_size = len(
        vectorizer.get_vocabulary()
    )

    text_input = keras.Input(
        shape=(),
        dtype=tf.string,
        name="claim_text",
    )

    text_features = vectorizer(
        text_input
    )

    text_features = layers.Embedding(
        input_dim=vocabulary_size,
        output_dim=embedding_dimension,
        mask_zero=True,
        name="token_embedding",
    )(text_features)

    text_features = (
        layers.GlobalAveragePooling1D(
            name="text_average_pooling",
        )(text_features)
    )

    text_features = layers.Dense(
        text_units,
        activation="relu",
        name="text_dense",
    )(text_features)

    text_features = layers.Dropout(
        dropout_rate,
        name="text_dropout",
    )(text_features)

    image_input = keras.Input(
        shape=(image_size, image_size, 3),
        name="chart_image",
    )

    image_features = layers.Rescaling(
        1.0 / 255,
        name="pixel_rescaling",
    )(image_input)

    image_features = layers.Conv2D(
        16,
        kernel_size=3,
        padding="same",
        activation="relu",
        name="image_conv_1",
    )(image_features)

    image_features = layers.MaxPooling2D(
        name="image_pool_1",
    )(image_features)

    image_features = layers.Conv2D(
        32,
        kernel_size=3,
        padding="same",
        activation="relu",
        name="image_conv_2",
    )(image_features)

    image_features = layers.MaxPooling2D(
        name="image_pool_2",
    )(image_features)

    image_features = layers.Conv2D(
        64,
        kernel_size=3,
        padding="same",
        activation="relu",
        name="image_conv_3",
    )(image_features)

    image_features = (
        layers.GlobalAveragePooling2D(
            name="image_average_pooling",
        )(image_features)
    )

    image_features = layers.Dropout(
        dropout_rate,
        name="image_dropout",
    )(image_features)

    combined_features = layers.Concatenate(
        name="multimodal_concatenation",
    )(
        [
            text_features,
            image_features,
        ]
    )

    combined_features = layers.Dense(
        combined_units,
        activation="relu",
        name="combined_dense",
    )(combined_features)

    combined_features = layers.Dropout(
        dropout_rate,
        name="combined_dropout",
    )(combined_features)

    outputs = layers.Dense(
        len(LABEL_ORDER),
        activation="softmax",
        name="class_probabilities",
    )(combined_features)

    model = keras.Model(
        inputs={
            "claim_text": text_input,
            "chart_image": image_input,
        },
        outputs=outputs,
        name="multimodal_neural_network",
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
