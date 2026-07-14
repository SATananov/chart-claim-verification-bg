# Keras text neural network

Step 008 adds the first neural network in the project.

## Input

The model receives only the text claim. It does not receive a chart
image in this step.

## Architecture

1. `TextVectorization`
2. `Embedding`
3. `GlobalAveragePooling1D`
4. `Dropout`
5. `Dense` with ReLU
6. `Dropout`
7. `Dense` with softmax for three classes

## Training rule

The text vectorizer is adapted only with training claims.

The model is trained only with the training split and is checked with
the validation split. The test split remains untouched.

## Optimizer and regularization

The model uses Adam and dropout. Early stopping restores the weights
from the epoch with the lowest validation loss.

## Comparison

The validation accuracy and macro F1 score are compared with:

- the majority baseline;
- TF-IDF with logistic regression.

The result must be interpreted carefully because the development claims
use a small number of generated text templates. A strong text score does
not prove that the system understands chart images.
