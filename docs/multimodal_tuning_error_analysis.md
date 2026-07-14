# Multimodal tuning and error analysis

Step 011 performs a small, controlled hyperparameter search for the
multimodal neural network and then studies its validation mistakes.

## Search space

Only three configurations are compared:

1. the Step 010 baseline;
2. lower dropout;
3. a wider text and fusion representation with a slightly lower
   learning rate.

The search is intentionally small because the development dataset is
small. A large search would increase validation overfitting without
providing reliable evidence.

## Selection rule

The best configuration is selected by:

1. validation macro F1;
2. validation accuracy;
3. fewer completed epochs.

The test split remains untouched.

## Error analysis

For every validation example, the notebook stores:

- chart group;
- image path;
- claim text;
- true label;
- predicted label;
- confidence;
- probability for each class;
- whether the prediction is correct;
- the error type.

The report also counts the most frequent true-label to predicted-label
confusions.

## Honest interpretation

The validation set is used both for model selection and development
analysis. Its score may therefore be optimistic.

The final test split must be evaluated only once after the model design
is complete. Step 011 does not open or evaluate the test split.
