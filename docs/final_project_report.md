# Final Project Report

## Chart Claim Verification with a Multimodal Neural Network

**Author:** Stefan Tananov  
**Framework:** TensorFlow and Keras  
**Task:** Three-class chart claim verification

## 1. Executive summary

This project studies whether a neural network can determine if a short
text claim is supported by a statistical chart.

Each chart-and-claim pair receives one of three labels:

- `supported`;
- `refuted`;
- `not_enough_information`.

The experimental design compares classical, text-only, image-only and
multimodal approaches. The best validation model combines a text branch
with a convolutional image branch.

The frozen multimodal model achieved:

| Metric | Validation | Final test |
|---|---:|---:|
| Accuracy | 0.8889 | 0.7778 |
| Macro F1 | 0.8885 | 0.7697 |

The final test result is lower than the validation result. This gap is
reported directly rather than hidden. It shows that model development
on a small validation set produced an optimistic estimate.

The project demonstrates a complete deep learning workflow:

- problem definition;
- source-data preparation;
- chart and claim generation;
- grouped train, validation and test splitting;
- baselines;
- text and image ablations;
- multimodal fusion;
- controlled hyperparameter tuning;
- error analysis;
- one-time final test evaluation;
- automated testing and reproducibility documentation.

## 2. Problem statement

Charts communicate numerical relationships efficiently, but verifying a
natural-language statement against a chart requires more than image
classification.

A system must connect:

1. the entities and time references in the claim;
2. comparative language such as *higher* or *lower*;
3. the values represented by the chart;
4. missing information that cannot be verified from the chart.

The project therefore formulates chart understanding as a three-class
classification task:

\[
f(\text{chart image}, \text{claim text})
\rightarrow
\{\text{supported},\text{refuted},\text{not enough information}\}
\]

The central research question is:

> Does combining claim text and chart-image features improve
> classification compared with text-only and image-only models?

## 3. Objectives

The project has six objectives:

1. Build a transparent dataset from a real statistical series.
2. Prevent chart-group leakage across data splits.
3. Establish simple baselines before training neural networks.
4. Compare text-only, image-only and multimodal architectures.
5. Analyze errors and the validation-to-test generalization gap.
6. Preserve an untouched test set until the final model is frozen.

## 4. Previous research

### 4.1 FEVER

Thorne et al. introduced FEVER, a large textual fact-verification
dataset with `Supported`, `Refuted` and `NotEnoughInfo` labels.

Reference:

- James Thorne, Andreas Vlachos, Christos Christodoulopoulos and Arpit
  Mittal. *FEVER: a Large-scale Dataset for Fact Extraction and
  VERification*. 2018.
- https://arxiv.org/abs/1803.05355

Connection to this project:

- The current project uses a closely related three-class decision.
- FEVER verifies claims against text evidence, while this project uses a
  chart image and a short claim.
- The current dataset is much smaller and does not include evidence
  retrieval.

### 4.2 DVQA

Kafle et al. introduced a question-answering dataset for bar-chart
understanding and showed that ordinary visual question-answering models
struggle with chart-specific text and values.

Reference:

- Kushal Kafle, Brian Price, Scott Cohen and Christopher Kanan.
  *DVQA: Understanding Data Visualizations via Question Answering*.
  2018.
- https://arxiv.org/abs/1801.08163

Connection to this project:

- Both tasks require visual understanding of charts.
- DVQA predicts answers to questions; this project predicts a
  verification label.
- The image-only result in this project reinforces the need to connect
  visual information to language.

### 4.3 PlotQA

Methani et al. introduced PlotQA to support reasoning over plots with
real-valued answers and data originating from real-world sources.

Reference:

- Nitesh Methani, Pritha Ganguly, Mitesh M. Khapra and Pratyush Kumar.
  *PlotQA: Reasoning over Scientific Plots*. 2019.
- https://arxiv.org/abs/1909.00997

Connection to this project:

- Both projects use chart reasoning rather than ordinary object
  recognition.
- PlotQA is a very large question-answering benchmark.
- The present project is a small controlled classification experiment
  based on one real statistical series.

### 4.4 ChartQA

Masry et al. introduced ChartQA with human-written and generated
questions that require visual, logical and arithmetic reasoning.

Reference:

- Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty and Enamul Hoque.
  *ChartQA: A Benchmark for Question Answering about Charts with Visual
  and Logical Reasoning*. 2022.
- https://arxiv.org/abs/2203.10244

Connection to this project:

- ChartQA demonstrates the value of combining language and visual chart
  features.
- The current work uses simpler three-way classification rather than
  open question answering.
- Its multimodal ablation is easy to inspect and reproduce.

### 4.5 DePlot

Liu et al. proposed translating plots into linearized tables before
language reasoning.

Reference:

- Fangyu Liu et al. *DePlot: One-shot Visual Language Reasoning by
  Plot-to-Table Translation*. 2022.
- https://arxiv.org/abs/2212.10505

Connection to this project:

- DePlot explicitly reconstructs structured values before reasoning.
- The current CNN learns image features directly and does not recover a
  table.
- The final error pattern suggests that explicit value extraction could
  be a strong future improvement.

## 5. Data

### 5.1 Source

The project uses a stored annual Eurostat series for the EU unemployment
rate from 2019 through 2025.

The source data is small enough to inspect manually and provides clear
temporal comparisons for chart claims.

### 5.2 Generated chart-and-claim dataset

The project pipeline creates:

| Item | Count |
|---|---:|
| Generated chart images | 30 |
| Chart-and-claim examples | 360 |
| Supported examples | 120 |
| Refuted examples | 120 |
| Not-enough-information examples | 120 |

The charts vary visual presentation while preserving traceability to the
source series. Claims are produced through controlled templates.

### 5.3 Split design

| Split | Examples |
|---|---:|
| Training | 252 |
| Validation | 54 |
| Final test | 54 |
| Total | 360 |

The split is grouped by `chart_group_id`.

This is important because rows linked to the same chart must not be
distributed across multiple splits. Otherwise, a model could see nearly
identical chart content during training and evaluation.

Automated checks verify that train, validation and test chart groups do
not overlap.

## 6. Evaluation metrics

The project reports:

- accuracy;
- macro F1;
- per-class precision;
- per-class recall;
- per-class F1;
- confusion matrices;
- row-level predictions and probabilities.

Accuracy measures the proportion of correct predictions.

Macro F1 gives equal importance to all three classes. This is useful
even though the dataset is balanced because it reveals class-specific
failure patterns.

## 7. Experiments

### 7.1 Majority baseline

The majority baseline always predicts one class.

Validation result:

| Accuracy | Macro F1 |
|---:|---:|
| 0.3333 | 0.1667 |

This provides the minimum reference point for the balanced three-class
problem.

### 7.2 TF-IDF and logistic regression

The classical text baseline converts claims into TF-IDF features and
uses logistic regression.

Validation result:

| Accuracy | Macro F1 |
|---:|---:|
| 0.7778 | 0.7723 |

This strong result reveals that the claim templates contain substantial
predictive information.

### 7.3 Keras text neural network

Architecture:

```text
TextVectorization
→ Embedding
→ GlobalAveragePooling1D
→ Dropout
→ Dense with ReLU
→ Dropout
→ Three-class softmax
```

Validation result:

| Accuracy | Macro F1 |
|---:|---:|
| 0.8333 | 0.8279 |

The neural text model improves on the classical text baseline.

### 7.4 Keras CNN image model

Architecture:

```text
Rescaling
→ Conv2D
→ MaxPooling2D
→ Conv2D
→ MaxPooling2D
→ Conv2D
→ GlobalAveragePooling2D
→ Dropout
→ Dense
→ Three-class softmax
```

Validation result:

| Accuracy | Macro F1 |
|---:|---:|
| 0.3333 | 0.1667 |

This result is expected.

The same chart can be paired with claims from all three classes. The
image-only model has no information about which claim is being checked.
The experiment therefore demonstrates a task-design limitation of an
image-only system rather than a software failure.

### 7.5 Keras multimodal neural network

The multimodal model contains two branches.

Text branch:

```text
Claim
→ TextVectorization
→ Embedding
→ GlobalAveragePooling1D
→ Dense
→ Dropout
```

Image branch:

```text
Chart
→ Rescaling
→ Three convolution layers
→ Pooling
→ GlobalAveragePooling2D
→ Dropout
```

Fusion:

```text
Text features ──┐
                ├→ Concatenate → Dense → Dropout → Softmax
Image features ─┘
```

Validation result:

| Accuracy | Macro F1 |
|---:|---:|
| 0.8889 | 0.8885 |

The multimodal model is the best validation model. It improves on both
the text-only and image-only neural networks.

## 8. Hyperparameter tuning

A small controlled search evaluates three configurations:

| Configuration | Accuracy | Macro F1 |
|---|---:|---:|
| Baseline | 0.8889 | 0.8885 |
| Lower dropout | 0.8889 | 0.8885 |
| Wider fusion | 0.8333 | 0.8279 |

The baseline and lower-dropout configurations tie. The wider model is
worse.

The baseline is selected because it is the simpler tied model.

Frozen configuration:

| Hyperparameter | Value |
|---|---:|
| Embedding dimension | 32 |
| Text dense units | 32 |
| Combined dense units | 64 |
| Dropout | 0.30 |
| Learning rate | 0.001 |

A larger search is intentionally avoided because repeated optimization
against only 54 validation examples would increase validation
overfitting.

## 9. Validation error analysis

The selected model makes 6 errors out of 54 validation examples.

| True label | Predicted label | Count |
|---|---|---:|
| Refuted | Supported | 3 |
| Supported | Not enough information | 3 |

The errors are concentrated in repeated year-comparison templates and
are made with high confidence.

This indicates a systematic grounding limitation rather than random
uncertainty. The model can learn language and image correlations but
does not explicitly extract the exact chart values before comparing
them.

## 10. Final one-time test evaluation

After model selection, the frozen baseline multimodal model is evaluated
once on the untouched test split.

Final test result:

| Metric | Score |
|---|---:|
| Accuracy | **0.7778** |
| Macro F1 | **0.7697** |
| Correct examples | 42 of 54 |
| Errors | 12 of 54 |

Generalization gap:

| Metric | Validation | Test | Test minus validation |
|---|---:|---:|---:|
| Accuracy | 0.8889 | 0.7778 | -0.1111 |
| Macro F1 | 0.8885 | 0.7697 | -0.1188 |

The validation score was optimistic by approximately 11 percentage
points.

This is plausible because:

- the dataset is small;
- the validation split contains only 54 rows;
- validation was used for model and hyperparameter decisions;
- generated claims follow a limited number of templates.

The test result remains well above majority accuracy, but it should not
be interpreted as evidence of general chart understanding.

## 11. Final test confusion matrix

| True class | Supported | Refuted | Not enough information |
|---|---:|---:|---:|
| Supported | 15 | 0 | 3 |
| Refuted | 0 | 18 | 0 |
| Not enough information | 0 | 9 | 9 |

Interpretation:

- All 18 refuted examples are classified correctly.
- Three supported claims are classified as not enough information.
- Nine not-enough-information claims are incorrectly classified as
  refuted.
- The main final-test weakness is distinguishing contradiction from
  missing evidence.

## 12. Testing and software quality

The project separates reusable code from notebooks.

Reusable modules cover:

- chart generation;
- claim generation;
- dataset construction;
- grouped splitting;
- baselines;
- text neural networks;
- image neural networks;
- multimodal datasets and models;
- tuning and error analysis;
- final test safeguards.

The confirmed automated test result before the final documentation step
is:

```text
38 passed
```

Tests include:

- dataset schema and balance checks;
- split-overlap protection;
- model architecture checks;
- label encoding and decoding;
- TensorFlow dataset shape checks;
- tuning-selection logic;
- error-report construction;
- one-time final-test marker behavior.

## 13. Reproducibility and project integrity

The project records intermediate and final reports as CSV or JSON files.

Important safeguards include:

- fixed random seeds;
- grouped splitting;
- training-only text-vectorizer adaptation;
- validation-only model selection;
- ignored local `.keras` files;
- stored training histories;
- row-level prediction reports;
- a final-test completion marker;
- explicit validation-to-test comparison.

The completion marker documents that the test set has already been
evaluated:

```text
reports/final_test_evaluation_completed.json
```

The final test notebook should not be rerun for model selection.

## 14. Limitations

### 14.1 Small sample size

There are only 360 examples and 54 final test rows. Small changes in
predictions can produce noticeable metric changes.

### 14.2 Template-generated language

The claims use controlled templates. Text models may learn template
patterns rather than general semantic reasoning.

### 14.3 One source series

All charts originate from one annual unemployment series. The project
does not cover the diversity of public charts.

### 14.4 Limited chart types

The development charts do not represent arbitrary dashboards, stacked
charts, scatter plots, dual axes or complex legends.

### 14.5 No explicit value extraction

The CNN creates latent visual features but does not convert the chart
into a structured table. This may explain errors involving exact
comparisons and absent information.

### 14.6 No external benchmark evaluation

The model is not evaluated on FEVER, DVQA, PlotQA, ChartQA or another
public benchmark. Results cannot be compared directly with those
systems.

## 15. Future work

A stronger continuation would:

1. add many independent statistical series;
2. include multiple chart families and visual styles;
3. use manually written claims;
4. extract chart values into a structured table;
5. compare direct multimodal fusion with plot-to-table reasoning;
6. evaluate on a larger group-held-out test set;
7. measure calibration as well as classification accuracy;
8. return the evidence values that support each decision.

The most important technical improvement is explicit chart-to-table
extraction followed by numerical and logical claim comparison.

## 16. Conclusion

The project answers its research question positively within the limits
of the development dataset.

The multimodal neural network performs better on validation data than
the text-only and image-only models. This supports the idea that the
chart image contributes useful information when combined with the
claim.

The final test accuracy of 0.7778 and macro F1 of 0.7697 show meaningful
performance above the majority baseline. At the same time, the
validation-to-test gap and the confusion between refuted and
not-enough-information claims reveal clear limits.

The final conclusion is therefore balanced:

> Multimodal fusion is useful for this controlled chart-verification
> task, but the current dataset and architecture are not sufficient for
> robust general chart reasoning.

## References

1. Thorne, J., Vlachos, A., Christodoulopoulos, C. and Mittal, A.
   *FEVER: a Large-scale Dataset for Fact Extraction and VERification*.
   2018. https://arxiv.org/abs/1803.05355
2. Kafle, K., Price, B., Cohen, S. and Kanan, C.
   *DVQA: Understanding Data Visualizations via Question Answering*.
   2018. https://arxiv.org/abs/1801.08163
3. Methani, N., Ganguly, P., Khapra, M. M. and Kumar, P.
   *PlotQA: Reasoning over Scientific Plots*.
   2019. https://arxiv.org/abs/1909.00997
4. Masry, A., Long, D. X., Tan, J. Q., Joty, S. and Hoque, E.
   *ChartQA: A Benchmark for Question Answering about Charts with Visual
   and Logical Reasoning*.
   2022. https://arxiv.org/abs/2203.10244
5. Liu, F. et al.
   *DePlot: One-shot Visual Language Reasoning by Plot-to-Table
   Translation*.
   2022. https://arxiv.org/abs/2212.10505
