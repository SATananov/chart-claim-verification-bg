# Keras CNN image model

Step 009 adds the computer vision branch of the project.

## Input

The model receives only the chart image.

It does not receive the text claim.

## Architecture

1. Pixel rescaling
2. Convolution layer
3. Max pooling
4. Convolution layer
5. Max pooling
6. Convolution layer
7. Global average pooling
8. Dropout
9. Dense layer
10. Three-class softmax output

## Important interpretation

Every chart image is paired with supported, refuted and
not-enough-information claims.

The image-only model cannot know which claim it is supposed to check
because the claim text is hidden from it. Therefore, a validation score
near the majority baseline is expected and is scientifically useful.

This experiment demonstrates that chart pixels alone are not enough for
the final task. The later multimodal model must combine the chart image
with the text claim.

## Training rule

The CNN is trained only on the training split and checked on the
validation split. The test split remains untouched.

All generated chart images are recreated locally when they are missing.
They remain ignored by Git.
