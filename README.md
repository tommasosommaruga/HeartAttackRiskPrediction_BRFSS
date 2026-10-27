# CS-433 Project 1: Heart Attack Risk Prediction

Predict from the 2015 BRFSS survey whether someone has been told they had a
heart attack or coronary heart disease (MICHD, labels `-1` / `+1`). About 8.8%
of the labelled rows are positive. The leaderboard score is positive class F1.
Only NumPy and Matplotlib are used.

The main lesson from our runs is that **how features are built moves F1 much
more than which linear model sits on top**. We keep one preprocessing path and
compare a few solvers for reference. A neural net and a gradient boosted tree
are planned; their scores are not in yet.

## Layout

| Path | Role |
| --- | --- |
| `run.py` | Writes `submissions/weighted_logistic.csv`. |
| `run.ipynb` | EDA, preprocessing, model comparison, CV, submit. |
| `implementations.py` | Course training functions. |
| `helpers.py` | Load CSV data and write a submission. |
| `src/preprocessing.py` | Drop survey/leak columns → one hot levels → MI top → standardize. |
| `src/model.py` | Weighted logistic (FISTA), SGD logistic helper, F1 cutoff, scores. |
| `src/experiments.py` | Five fold CV, permutation importance, submit. |
| `src/eda.py` | Plots and missingness checks. |
| `config/experiments.json` | Folds, seed, penalties, `top_onehot`. |
| `latex-example-paper/report.tex` | Short report. |

## Usage

Put `x_train.csv`, `y_train.csv`, and `x_test.csv` in `data/dataset/`.

```bash
python run.py                 # submission
python -m src.experiments     # five fold CV → pipeline_experiments.csv
jupyter notebook run.ipynb
```

## Preprocessing

```python
data = prep.make_data(x, cfg)
state, x_fit = prep.fit(data, fit_rows, y[fit_rows] == 1, cfg)
x_out = prep.apply(data, other_rows, state)
```

1. Drop survey machinery (dates, phone checks, weights) and four label leaks.
2. Turn each remaining column into levels: frequent values, quantile bins for the rest, missing as its own level.
3. Keep the `top_onehot` levels with the highest mutual information with the label.
4. Standardize. Statistics are fit on training rows only.

Earlier we also tried reading codes as numbers with median fill, missing flags
only, a hand codebook recode, and keeping every one hot level. Those paths
scored lower; the gap was larger than the gap between logistic GD, SGD logistic,
and ridge on the same matrix.

## Models

The notebook compares three linear fits on the same prepared split: course
logistic GD, mini batch SGD logistic, and course ridge. The submission uses
weighted FISTA logistic with a small L2 penalty and a positive class cost of 2.
The F1 cutoff is chosen on a held out slice. Five fold mean F1 is about 0.427
on 300 columns. Neural net and gradient boosting results will be added later.

## Config knobs

Edit `config/experiments.json`: `n_folds`, `seed`, `tune_fraction`,
`positive_class_cost`, `max_iters`, `l2_penalty`, `min_level_share`,
`quantile_bins`, `top_onehot`, `use_leak_columns`, `eps`.
