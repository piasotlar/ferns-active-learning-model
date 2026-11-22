# Viral Contig Classification and Active Learning

This project uses an XGBoost model to classify viral contigs as `Yes` or `No`.
`Yes` indicates that a contig is considered more likely to represent a virus
infecting the organisms from which the samples were collected.

The workflow uses active learning: after each iteration, the model selects
unlabeled contigs for manual annotation. These labels are then added to the
training pool for the next iteration.

## Requirements

Python 3.10 or newer is recommended. Install the required packages:

```bash
python -m pip install pandas numpy scikit-learn xgboost \
    matplotlib seaborn openpyxl
```

## Input data

The initial contig table is the final output of the
[fern-virus-discovery-pipeline](https://github.com/piasotlar/fern-virus-discovery-pipeline).

Before creating the training and test sets, the table must be extended with:

- `CONSISTENT LENGTH (tax)` – whether the contig length is consistent with the
  expected length for its taxonomy;
- `PLANT HOST (tax)` – whether the available taxonomy and host information
  indicates a plant host.

These features are added by
[`extending_table/table_extend.py`](extending_table/table_extend.py).

Run the preprocessing step first:

```bash
cd extending_table
python table_extend.py
```

This creates `table_extended.tsv`. 

With the extended table prepare an initial training pool with labeled `Yes` and `No`
examples and a separate labeled test set.

Then split the table into:

```text
train_pool.tsv
test.tsv
```

`train_pool.tsv` should contain both labeled and unlabeled contigs. `test.tsv` is a
separate labeled dataset used for evaluation.


## Active-learning workflow

### 1. Run the first iteration

From the project root run:

```bash
python model.py --iteration 0
```

The model trains on the currently labeled rows and creates an annotation
batch:

```text
batches/to_label_iteration_0.xlsx
```

Review this file and add the manual `Yes` or `No` labels.

### 2. Run iteration 1 (validation only)

Iteration 1 is a validation-only step. The manually annotated batch from iteration 0 is used to evaluate model performance without updating the training set. The validation contigs are excluded from the unlabeled pool during this iteration. They can be incorporated into the training set in a later iteration.

Run it with:

```bash
python model.py --iteration 1 \
    --validation-batch batches/to_label_iteration_0.xlsx
```

### 3. Run later iterations

For each later iteration, provide the annotated batch from the previous round.
For example:

```bash
python model.py --iteration 2 \
    --validation-batch batches/to_label_iteration_1.xlsx \
    --add-to-train-batch batches/to_label_iteration_0.xlsx
```

The script updates `train_pool.tsv` in place, so make a backup before starting
a new iteration. The next annotation batch is written to:

```text
batches/to_label_iteration_<n>.xlsx
```

## Sampling strategies

The strategy is selected in
[model.py](model.py):

```python
sampling_version = 4
```

- **Version 1** selects the most uncertain contigs, close to the decision
  boundary.
- **Version 2** selects uncertain and diverse contigs using clustering.
- **Version 3** combines highly confident and uncertain samples.
- **Version 4** selects 10 likely negative contigs
  (`prob_Yes < 0.05`) and 20 likely positive contigs
  (`prob_Yes > 0.95`). This is the current default.

Selected contigs include their `prob_No`, `prob_Yes`, and `uncertainty` values.

## Main outputs

The model creates:

- `batches/` – contigs selected for manual annotation;
- `predictions/` – predictions for the train, pool, and test datasets;
- `active_learning_stats.tsv` – metrics across active-learning iterations;
- `histograms/`, `heatmaps/`, `feature_importance/`, and
  `calibration_curves/` – diagnostic plots.

## Cross-validation

[cross_validation/cross_validation.py](cross_validation/cross_validation.py)
performs Leave-One-Out cross-validation on the labeled rows:

```bash
python cross_validation/cross_validation.py \
    --output-file cross_validation/cross_validation_results_final.csv
```
The output contains the true label, model prediction, class probabilities,
confidence, and whether each prediction was correct.

