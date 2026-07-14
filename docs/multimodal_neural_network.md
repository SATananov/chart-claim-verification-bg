# Multimodal neural network

Step 010 combines the text and image branches of the project.

## Inputs

The model receives two inputs for every example:

- the text claim;
- the chart image.

## Text branch

1. `TextVectorization`
2. `Embedding`
3. `GlobalAveragePooling1D`
4. `Dense`
5. `Dropout`

The vocabulary is learned only from the training claims.

## Image branch

1. Pixel rescaling
2. Three convolution layers
3. Two max-pooling layers
4. `GlobalAveragePooling2D`
5. `Dropout`

## Combined branch

The text and image features are joined with `Concatenate`.

The combined representation passes through a dense layer, dropout and a
three-class softmax output.

## Training rule

The model is trained only on the training split and checked on the
validation split. The test split remains untouched.

Early stopping monitors validation loss and restores the best weights.

## Interpretation

The main comparison is between:

- the text-only neural network;
- the image-only CNN;
- the multimodal neural network.

The multimodal model is the central model of the project, but it is not
guaranteed to outperform the text-only model on this small development
dataset. The generated claims follow a limited set of text templates,
and the chart images come from one real source series.

A result that is similar to the text-only model should be reported
honestly rather than presented as proof of broad chart understanding.
