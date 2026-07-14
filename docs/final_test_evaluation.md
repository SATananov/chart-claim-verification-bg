# Final test evaluation

Step 012 performs the one-time evaluation of the frozen multimodal
model on the untouched test split.

## Frozen decision

Step 011 selected the baseline multimodal configuration:

- embedding dimension: 32;
- text units: 32;
- combined units: 64;
- dropout: 0.30;
- learning rate: 0.001.

No architecture, threshold or hyperparameter is changed after the test
split is opened.

## Preconditions

Before evaluation, the notebook verifies:

1. train, validation and test data contain the required columns;
2. chart groups do not overlap across the three splits;
3. the selected configuration is the validation-selected baseline;
4. the locally saved tuned model exists;
5. no final-test completion marker already exists.

## One-time policy

After all final reports are written, the notebook creates:

`reports/final_test_evaluation_completed.json`

A later notebook run stops before evaluating the test set again.

This guard does not provide cryptographic protection. It documents and
enforces the intended scientific workflow inside the project.

## Final reports

The notebook saves:

- final test accuracy and macro F1;
- classification report;
- confusion matrix;
- row-level predictions and probabilities;
- final test errors;
- validation-to-test generalization gap;
- completion marker.

The test result must be reported honestly even when it is lower than
the validation score.
