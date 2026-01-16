# 🌧️ Aussie Rain Prediction

A machine learning project to predict whether it will rain tomorrow in Australia using Decision Trees and Random Forest classifiers.

## 📊 Dataset

- **Source**: Australian Weather Dataset (Rattle Package)
- **Target Variable**: `RainTomorrow` (Yes/No classification)
- **Features**: Temperature, humidity, wind, pressure, cloud cover, and more
- **Time Split**:
  - Training: Before 2015
  - Validation: 2015
  - Test: After 2015

## 🛠️ Tech Stack

```
pandas, numpy, scikit-learn, matplotlib, seaborn, plotly, joblib, pyarrow
```

## 🔄 Data Preprocessing Pipeline

### 1. Handle Missing Values
```python
# Drop rows where target columns are missing
raw_df.dropna(subset=['RainToday', 'RainTomorrow'], inplace=True)

# Impute numeric columns with mean
imputer = SimpleImputer(strategy='mean')
```

### 2. Feature Scaling
```python
# MinMax scaling for numeric features (0-1 range)
scaler = MinMaxScaler()
```

### 3. Categorical Encoding
```python
# One-hot encoding for categorical variables
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
```

## 📈 Models & Results

### Decision Tree Classifier

| Configuration | Train Accuracy | Validation Accuracy |
|--------------|----------------|---------------------|
| No constraints | ~100% | ~78% |
| max_depth=3 | ~83% | ~83% |
| max_depth=11 | ~88% | ~85% |
| max_leaf_nodes=128 | ~87% | ~85% |

### Random Forest Classifier

| Configuration | Train Accuracy | Validation Accuracy |
|--------------|----------------|---------------------|
| Default (100 trees) | ~100% | ~85% |
| n_estimators=10 | ~99% | ~85% |
| n_estimators=50, max_features='log2' | ~95% | ~85% |
| n_estimators=200, min_samples_split=3, min_samples_leaf=2 | ~99% | ~86% |

## 🎓 Key Learnings & Takeaways

### 1. **Overfitting is Real**
- An unconstrained Decision Tree achieves **100% training accuracy** but poor validation accuracy
- This is a classic sign of overfitting - the model memorizes the training data

### 2. **Regularization Techniques for Trees**
| Parameter | What it does | Effect |
|-----------|--------------|--------|
| `max_depth` | Limits tree depth | Reduces overfitting |
| `max_leaf_nodes` | Limits number of leaves | Controls model complexity |
| `min_samples_split` | Minimum samples to split a node | Prevents overly specific splits |
| `min_samples_leaf` | Minimum samples in a leaf | Ensures generalizable leaves |
| `min_impurity_decrease` | Minimum impurity reduction for split | Prunes insignificant splits |

### 3. **Finding Optimal Hyperparameters**
- Plot **Training vs Validation Error** across different parameter values
- Look for the "sweet spot" where validation error is minimized
- Optimal `max_depth` was around **9-11** for this dataset

### 4. **Random Forest Advantages**
- **Ensemble of trees** reduces variance and overfitting
- **Bootstrap sampling** (bagging) creates diverse trees
- **Feature randomization** (`max_features`) decorrelates trees
- More robust than single decision trees

### 5. **Feature Importance**
- Top predictors for rain: **Humidity3pm**, **Sunshine**, **Cloud3pm**, **Pressure**
- Use `model.feature_importances_` to understand what drives predictions

### 6. **Key Random Forest Parameters**
```python
RandomForestClassifier(
    n_estimators=100,      # Number of trees (more = better, but slower)
    max_features='sqrt',   # Features per split (try 'log2' for less correlation)
    max_depth=None,        # Tree depth limit
    min_samples_split=2,   # Min samples to split
    min_samples_leaf=1,    # Min samples per leaf
    bootstrap=True,        # Use bootstrap sampling
    n_jobs=-1,             # Parallelize across all cores
    class_weight='balanced' # Handle imbalanced classes
)
```

## 💡 Things to Remember

### Data Preprocessing
- ✅ Always split data **before** fitting imputers/scalers/encoders
- ✅ Fit preprocessing on **training data only**, transform all sets
- ✅ Use `handle_unknown='ignore'` in OneHotEncoder for unseen categories

### Model Evaluation
- ✅ Compare **training vs validation** accuracy to detect overfitting
- ✅ High training accuracy + low validation accuracy = **OVERFITTING**
- ✅ Use `model.score()` for quick accuracy checks

### Decision Trees
- ✅ Visualize with `plot_tree()` or `export_text()` for interpretability
- ✅ Start with constraints to avoid overfitting
- ✅ Check `model.tree_.max_depth` to see actual tree depth

### Random Forests
- ✅ Default 100 trees is usually a good starting point
- ✅ Use `n_jobs=-1` to parallelize training
- ✅ Access individual trees via `model.estimators_[i]`
- ✅ `bootstrap=False` uses all data per tree (no bagging)

## 📁 Project Structure

```
SciKit-Learn/
├── models/
│   └── random_forest.ipynb    # Main notebook
├── MedicalBill(LR)/
│   └── weather-dataset-rattle-package/
│       └── weatherAUS.csv     # Dataset
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook models/random_forest.ipynb
```

## 📚 Resources

- [Scikit-learn Decision Trees](https://scikit-learn.org/stable/modules/tree.html)
- [Scikit-learn Random Forests](https://scikit-learn.org/stable/modules/ensemble.html#forest)
- [Understanding Overfitting](https://scikit-learn.org/stable/auto_examples/model_selection/plot_underfitting_overfitting.html)

---

*Built with ❤️ using Scikit-Learn*
