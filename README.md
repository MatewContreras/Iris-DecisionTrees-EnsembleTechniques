# Iris Decision Trees and Ensemble Techniques

Exploring how ensemble methods improve decision tree classification on the Iris dataset. Covers the full pipeline from data exploration to model comparison using five classifiers.

## Overview

The Iris dataset (150 samples, 4 features, 3 classes) is used to study how aggregating multiple decision trees affects accuracy, stability, and generalization compared to a single tree.

Five models are compared:

| Model | Strategy | Effect |
|---|---|---|
| Decision Tree | Single tree, baseline | Reference |
| Bagging | 100 trees on bootstrap samples, majority vote | Reduces variance |
| Random Forest | Bagging + random feature subset per split | Reduces variance more |
| AdaBoost | Sequential stumps, misclassified samples get higher weight | Reduces bias |
| Gradient Boosting | Sequential trees fitted on residual errors | Reduces bias and variance |

## Analysis

- EDA: class distribution and feature distributions
- Depth sensitivity: how max_depth affects overfitting in a single tree
- Decision tree visualization at depth 3
- 5-fold stratified cross-validation accuracy across all five models
- Effect of number of estimators on accuracy
- Feature importances: single tree vs ensemble methods
- Confusion matrices on held-out test set
- Learning curves: train vs test accuracy gap
- Decision boundaries using petal length and petal width

## Results

- Every ensemble method matches or beats the single decision tree
- Random Forest has the lowest standard deviation — most stable across folds
- Gradient Boosting achieves the highest raw accuracy
- Bagging and Random Forest: adding more trees never hurts performance
- AdaBoost can overfit when `n_estimators` is too large
- Ensemble feature importances are more balanced and reliable than a single tree
- Ensemble decision boundaries are smoother and generalize better

## Visualizations

| Plot | Description |
|---|---|
| `q6_eda.png` | Class distribution and feature box plots |
| `q6_depth.png` | Depth vs accuracy for single decision tree |
| `q6_tree_viz.png` | Tree structure at max_depth=3 |
| `q6_bar_box.png` | Mean CV accuracy and fold-level spread per model |
| `q6_n_estimators.png` | Accuracy vs number of trees |
| `q6_feat_imp.png` | Feature importances across three models |
| `q6_confusion.png` | Confusion matrices on test set |
| `q6_learning_curves.png` | Learning curves for DT and Random Forest |
| `q6_boundaries.png` | Decision boundaries on petal features |

## Setup

```bash
pip install numpy pandas scikit-learn matplotlib
jupyter notebook question6.ipynb
```

## Dataset

Iris dataset — Fisher (1936), available via `sklearn.datasets.load_iris`

150 samples, 4 features (sepal length, sepal width, petal length, petal width), 3 classes (setosa, versicolor, virginica)
