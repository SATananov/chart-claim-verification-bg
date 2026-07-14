# Development dataset

Step 005 creates a reproducible development dataset from the first
real time series already stored in the project.

## Generated content

- 10 overlapping time windows
- 3 visual styles for each window
- 30 chart images
- 360 chart-and-claim examples
- 120 examples for each label

## Purpose

This dataset is large enough to test the image pipeline, claim pipeline,
class balance and the later train, validation and test code.

## Limitation

All windows come from one source series and some periods overlap.
The dataset is suitable for development, but it is not yet a strong
final evaluation dataset.

Before the final model comparison, more independent real statistical
series should be added. All visual styles of the same chart window must
stay in the same data split.
