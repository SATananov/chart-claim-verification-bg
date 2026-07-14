# Simple baselines

Step 007 compares two simple models before neural networks are trained.

## Majority baseline

This classifier always predicts the most frequent training label.

The development data is balanced, so its validation accuracy is close
to random guessing for three classes.

## Text baseline

The second model uses:

- TF-IDF text features
- word unigrams and bigrams
- logistic regression

It uses only `claim_text`. It does not use chart images.

## Current validation results

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Majority baseline | 0.3333 | 0.1667 |
| TF-IDF + Logistic Regression | 0.7778 | 0.7723 |

The test set is not used in this step.

The text result is useful, but the generated claims follow a small
number of templates. A neural model must therefore be compared
carefully and should not be described as understanding charts until the
image input is tested separately.
