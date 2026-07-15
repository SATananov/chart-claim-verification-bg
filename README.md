# Chart Claim Verification

A compact deep learning research project that checks whether a short
text claim is supported by a statistical chart.

The classifier predicts one of three labels:

- `supported`
- `refuted`
- `not_enough_information`

The central experiment compares text-only, image-only and multimodal
neural networks. The final multimodal model receives both the claim text
and the chart image.

## Final result

The frozen baseline multimodal model was evaluated once on the untouched
test split.

| Metric | Validation | Final test | Test minus validation |
|---|---:|---:|---:|
| Accuracy | 0.8889 | **0.7778** | -0.1111 |
| Macro F1 | 0.8885 | **0.7697** | -0.1188 |

The final test set contains 54 examples. The model classified 42
correctly and made 12 errors.

The result is substantially above the 0.3333 majority accuracy
baseline, but the validation-to-test drop shows that the small
development dataset does not support broad generalization claims.

See the complete analysis in
[`docs/final_project_report.md`](docs/final_project_report.md).

## Research question

Can a neural network classify a claim about a chart more accurately when
it combines text and image information than when it receives only one
of those inputs?

The project tests this question through controlled ablations:

1. majority baseline;
2. TF-IDF with logistic regression;
3. Keras text neural network;
4. Keras CNN image model;
5. Keras multimodal neural network.

## Data

The project starts from a real annual Eurostat series describing the EU
unemployment rate from 2019 to 2025.

From that source series, the data pipeline creates:

- 30 chart images;
- 360 chart-and-claim examples;
- 120 examples for each class;
- 252 training examples;
- 54 validation examples;
- 54 final test examples.

The split is grouped by `chart_group_id`. A chart group cannot appear in
more than one split, which prevents image variants from leaking between
training, validation and test data.

Generated PNG images and trained `.keras` models remain local and are
ignored by Git. The notebooks can rebuild the chart images from the
stored processed data.

## Model comparison

Validation results used during model development:

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Majority baseline | 0.3333 | 0.1667 |
| TF-IDF + Logistic Regression | 0.7778 | 0.7723 |
| Keras text neural network | 0.8333 | 0.8279 |
| Keras CNN image model | 0.3333 | 0.1667 |
| Keras multimodal neural network | **0.8889** | **0.8885** |
| Tuned Keras multimodal neural network | **0.8889** | **0.8885** |

The image-only model matches the majority baseline because the same
chart can be paired with supported, refuted and
not-enough-information claims. Without the claim text, the CNN does not
know which statement it must verify.

The multimodal model improves on the text-only model, showing that chart
features add useful information in this controlled dataset.

## Final test error analysis

Final test confusion matrix:

| True class | Predicted supported | Predicted refuted | Predicted not enough information |
|---|---:|---:|---:|
| Supported | 15 | 0 | 3 |
| Refuted | 0 | 18 | 0 |
| Not enough information | 0 | 9 | 9 |

The model recognizes all 18 refuted examples correctly. Its main
remaining limitation is separating `not_enough_information` from
`refuted`.

## Architecture

### Text branch

```text
Claim text
→ TextVectorization
→ Embedding
→ GlobalAveragePooling1D
→ Dense
→ Dropout
```

### Image branch

```text
Chart image
→ Rescaling
→ Conv2D
→ MaxPooling
→ Conv2D
→ MaxPooling
→ Conv2D
→ GlobalAveragePooling2D
→ Dropout
```

### Multimodal fusion

```text
Text features ──┐
                ├→ Concatenate → Dense → Dropout → Softmax
Image features ─┘
```

The output layer contains three softmax units.

## Hyperparameter tuning

A deliberately small search compared three configurations:

- baseline;
- lower dropout;
- wider fusion representation.

The baseline and lower-dropout models tied at 0.8889 validation accuracy
and 0.8885 macro F1. The wider model performed worse. The simpler
baseline configuration was therefore frozen for final evaluation.

This limited search reduces the risk of overfitting the small validation
set.

## Previous research

This project is related to chart question answering and claim
verification research:

- [FEVER](https://arxiv.org/abs/1803.05355) introduced the
  `Supported`, `Refuted` and `NotEnoughInfo` claim-verification labels
  for textual evidence.
- [DVQA](https://arxiv.org/abs/1801.08163) studied question answering
  over bar charts.
- [PlotQA](https://arxiv.org/abs/1909.00997) introduced large-scale
  reasoning over plots generated from real-world data.
- [ChartQA](https://arxiv.org/abs/2203.10244) combined visual and
  logical reasoning with human-written and generated chart questions.
- [DePlot](https://arxiv.org/abs/2212.10505) converted plots into
  structured text before language-model reasoning.

Unlike those large benchmarks, this repository uses a small,
three-class educational experiment. Its main contribution is a clear
comparison of text-only, image-only and multimodal models under a
grouped split and a one-time final test policy.

## Project structure

```text
chart-claim-verification-bg/
├── data/
│   ├── processed/              # source, dataset and split CSV files
│   └── generated/              # locally generated chart images
├── docs/                       # methodology and final report
├── models/                     # local ignored Keras models
├── notebooks/                  # numbered experimental workflow
├── reports/                    # metrics, predictions and error reports
├── src/                        # reusable project code
├── tests/                      # pytest test suite
├── README.md
├── requirements.txt
└── requirements-lock.txt
```

## Main reports

- [`reports/model_validation_comparison.csv`](reports/model_validation_comparison.csv)
- [`reports/multimodal_tuning_results.csv`](reports/multimodal_tuning_results.csv)
- [`reports/multimodal_validation_errors.csv`](reports/multimodal_validation_errors.csv)
- [`reports/final_test_metrics.csv`](reports/final_test_metrics.csv)
- [`reports/final_test_confusion_matrix.csv`](reports/final_test_confusion_matrix.csv)
- [`reports/final_test_errors.csv`](reports/final_test_errors.csv)
- [`reports/final_evaluation_summary.csv`](reports/final_evaluation_summary.csv)
- [`reports/final_test_evaluation_completed.json`](reports/final_test_evaluation_completed.json)

## Reproducible setup

The project was developed on Windows with Python 3.13, TensorFlow and
Keras.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
```

Run the automated tests:

```powershell
python -m pytest
```

The confirmed project state before this documentation step is:

```text
38 passed
```

Run the notebooks in numerical order for a full development workflow.

Do not rerun `notebooks/10_final_test_evaluation.ipynb` after the
completion marker exists. The final test set was intentionally evaluated
once after model selection.

## Scientific integrity

The project follows these rules:

- chart groups are isolated across train, validation and test splits;
- text vocabulary is adapted only on training claims;
- hyperparameters are selected with validation data;
- the final test split is opened only after model selection;
- the validation-to-test performance drop is reported;
- no result is presented as state of the art;
- limitations of the small generated dataset are stated explicitly.

## Limitations

- The dataset contains only 360 examples.
- Claims are generated from a limited set of templates.
- Charts are derived from one annual source series.
- The model may learn template and visual-layout shortcuts.
- The final test set contains only 54 examples.
- Results do not demonstrate general chart understanding.
- The project does not parse arbitrary charts from the internet.
- The classifier does not provide extracted numeric evidence.

## Conclusion

The experiment supports the main project hypothesis within this small
dataset: combining chart images with claim text performs better than
using either modality alone during validation.

The frozen multimodal model achieved 0.7778 accuracy and 0.7697 macro F1
on the untouched final test split. The generalization gap and error
analysis show that the problem is not solved, especially for
not-enough-information claims.

This repository should be understood as a transparent educational deep
learning study and a foundation for a larger, more diverse chart
verification dataset.

## Author

Stefan Tananov  
GitHub: [SATananov](https://github.com/SATananov)

## Visual Demonstration

A separate Streamlit application demonstrates the frozen multimodal model through a visual interface:

**Repository:**  
https://github.com/SATananov/chart-claim-verification

The application allows the user to:

- select a packaged statistical chart or upload an image;
- enter a short claim in English;
- classify the claim as `supported`, `refuted`, or `not_enough_information`;
- inspect the confidence scores for all three classes;
- review the model architecture and stored evaluation results.

The visual application is a presentation layer only. It does not retrain the model and does not evaluate the final test set again.

Because the training dataset is small and domain-specific, predictions for arbitrary uploaded charts should be treated as demonstrations rather than reliable universal fact-checking results.
