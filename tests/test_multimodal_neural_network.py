from pathlib import Path

import keras
import numpy as np
import tensorflow as tf
from keras import layers

from src.multimodal_neural_network import (
    LABEL_ORDER,
    build_multimodal_neural_network,
    create_multimodal_dataset,
    create_text_vectorizer,
    decode_predictions,
    encode_labels,
)


def test_multimodal_label_round_trip():
    encoded = encode_labels(LABEL_ORDER)
    decoded = decode_predictions(encoded)

    assert encoded.dtype == np.int32
    assert decoded == LABEL_ORDER


def test_multimodal_dataset_shapes(
    tmp_path: Path,
):
    image_path = tmp_path / "chart.png"

    image = tf.zeros(
        shape=(40, 60, 3),
        dtype=tf.uint8,
    )

    tf.io.write_file(
        str(image_path),
        tf.io.encode_png(image),
    )

    dataset = create_multimodal_dataset(
        image_paths=[str(image_path)],
        claim_texts=[
            "The value was higher."
        ],
        labels=[0],
        image_size=96,
        batch_size=1,
    )

    input_batch, label_batch = next(
        iter(dataset)
    )

    assert set(input_batch) == {
        "claim_text",
        "chart_image",
    }

    assert tuple(
        input_batch["chart_image"].shape
    ) == (
        1,
        96,
        96,
        3,
    )

    assert tuple(
        input_batch["claim_text"].shape
    ) == (1,)

    assert tuple(label_batch.shape) == (1,)


def test_multimodal_model_has_two_inputs():
    vectorizer = create_text_vectorizer(
        [
            "supported higher value",
            "refuted lower value",
            "unknown future year",
        ]
    )

    model = (
        build_multimodal_neural_network(
            vectorizer=vectorizer,
            image_size=96,
        )
    )

    layer_types = {
        type(layer)
        for layer in model.layers
    }

    input_names = {
        tensor.name.split(":")[0]
        for tensor in model.inputs
    }

    assert input_names == {
        "claim_text",
        "chart_image",
    }

    assert layers.TextVectorization in layer_types
    assert layers.Embedding in layer_types
    assert layers.Conv2D in layer_types
    assert layers.Concatenate in layer_types
    assert layers.Dropout in layer_types
    assert layers.Dense in layer_types
    assert model.output_shape == (
        None,
        len(LABEL_ORDER),
    )
    assert isinstance(model, keras.Model)
