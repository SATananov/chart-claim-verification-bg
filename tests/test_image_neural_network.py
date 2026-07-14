from pathlib import Path

import keras
import numpy as np
import tensorflow as tf
from keras import layers

from src.image_neural_network import (
    LABEL_ORDER,
    build_image_neural_network,
    create_image_dataset,
    decode_predictions,
    encode_labels,
)


def test_image_label_round_trip():
    encoded = encode_labels(LABEL_ORDER)
    decoded = decode_predictions(encoded)

    assert encoded.dtype == np.int32
    assert decoded == LABEL_ORDER


def test_image_dataset_shape(
    tmp_path: Path,
):
    image_path = tmp_path / "chart.png"

    image = tf.zeros(
        shape=(40, 60, 3),
        dtype=tf.uint8,
    )

    encoded_image = tf.io.encode_png(
        image
    )

    tf.io.write_file(
        str(image_path),
        encoded_image,
    )

    dataset = create_image_dataset(
        image_paths=[str(image_path)],
        labels=[0],
        image_size=96,
        batch_size=1,
    )

    image_batch, label_batch = next(
        iter(dataset)
    )

    assert tuple(image_batch.shape) == (
        1,
        96,
        96,
        3,
    )

    assert tuple(label_batch.shape) == (1,)


def test_image_model_has_required_layers():
    model = build_image_neural_network(
        image_size=96,
    )

    layer_types = {
        type(layer)
        for layer in model.layers
    }

    assert layers.Conv2D in layer_types
    assert layers.MaxPooling2D in layer_types
    assert (
        layers.GlobalAveragePooling2D
        in layer_types
    )
    assert layers.Dropout in layer_types
    assert layers.Dense in layer_types
    assert model.output_shape == (
        None,
        len(LABEL_ORDER),
    )
    assert isinstance(model, keras.Model)
