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
    """Adapt a text vectorizer using training text only."""

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

    text_array = np.asarray(
        list(training_text),
        dtype=str,
    )

    if len(text_array) == 0:
        raise ValueError(
            "Training text cannot be empty."
        )

    vectorizer.adapt(text_array)

    return vectorizer


def build_text_neural_network(
    vectorizer: layers.TextVectorization,
    embedding_dimension: int = 32,
    hidden_units: int = 32,
    dropout_rate: float = 0.30,
    learning_rate: float = 0.001,
) -> keras.Model:
    """Build and compile a small Keras text classifier."""

    vocabulary_size = len(
        vectorizer.get_vocabulary()
    )

    inputs = keras.Input(
        shape=(),
        dtype=tf.string,
        name="claim_text",
    )

    x = vectorizer(inputs)

    x = layers.Embedding(
        input_dim=vocabulary_size,
        output_dim=embedding_dimension,
        mask_zero=True,
        name="token_embedding",
    )(x)

    x = layers.GlobalAveragePooling1D(
        name="average_pooling",
    )(x)

    x = layers.Dropout(
        dropout_rate,
        name="first_dropout",
    )(x)

    x = layers.Dense(
        hidden_units,
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
        name="text_neural_network",
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
