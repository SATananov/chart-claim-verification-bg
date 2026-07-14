# Grouped train, validation and test split

The development dataset is divided into three parts:

- train: 70%
- validation: 15%
- test: 15%

The split is made by `chart_group_id`, not by individual rows.

All visual styles and claims from one chart window stay in the same
split. This prevents the same chart group from appearing during both
training and evaluation.

For the current 360 examples, the split is exact:

- train: 252 rows
- validation: 54 rows
- test: 54 rows

The labels remain balanced inside every split.

This protects against direct chart-group leakage. The development data
still contains overlapping time windows from one source series, so more
independent real series are needed before the final model comparison.
