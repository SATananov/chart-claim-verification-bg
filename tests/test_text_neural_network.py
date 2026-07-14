import keras
import numpy as np
from keras import layers

from src.text_neural_network import (
    LABEL_ORDER,
    build_text_neural_network,
    create_text_vectorizer,
    decode_predictions,
    encode_labels,
)


def test_label_encoding_round_trip():
    encoded = encode_labels(LABEL_ORDER)
    decoded = decode_predictions(encoded)

    assert encoded.dtype == np.int32
    assert decoded == LABEL_ORDER


def test_vectorizer_returns_fixed_length():
    vectorizer = create_text_vectorizer(
        [
            "The rate was higher in 2020.",
            "The rate was lower in 2021.",
            "The chart has no value for 2026.",
        ],
        sequence_length=12,
    )

    vectorized = vectorizer(
        [
            "The rate was higher.",
            "No value is shown.",
        ]
    )

    assert tuple(vectorized.shape) == (2, 12)


def test_text_model_has_required_layers():
    vectorizer = create_text_vectorizer(
        [
            "supported higher value",
            "refuted lower value",
            "unknown future year",
        ]
    )

    model = build_text_neural_network(
        vectorizer=vectorizer,
    )

    layer_types = {
        type(layer)
        for layer in model.layers
    }

    assert layers.TextVectorization in layer_types
    assert layers.Embedding in layer_types
    assert layers.Dropout in layer_types
    assert layers.Dense in layer_types
    assert model.output_shape == (
        None,
        len(LABEL_ORDER),
    )
    assert isinstance(model, keras.Model)
