# 🤖 Scikit-Learn Machine Learning Playground

A comprehensive collection of machine learning notebooks covering **supervised learning**, **unsupervised learning**, **gradient boosting**, and **recommender systems** using Scikit-Learn and related libraries.

## � Notebooks Overview

| Notebook | Algorithm | Dataset | Task Type |
|----------|-----------|---------|-----------|
| `LinearRegression.ipynb` | Linear Regression, SGD | Medical Insurance Charges | Regression |
| `logistic_regression.ipynb` | Logistic Regression | Australian Weather | Binary Classification |
| `random_forest.ipynb` | Decision Trees, Random Forest | Australian Weather | Binary Classification |
| `XGboost.ipynb` | XGBoost, K-Fold CV | Rossmann Store Sales | Regression |
| `Unsupervised.ipynb` | KMeans, DBSCAN, Agglomerative | Iris Dataset | Clustering |
| `Reccomender.ipynb` | Collaborative Filtering (FastAI) | MovieLens | Recommendation |

---

## 🛠️ Tech Stack

```
pandas, numpy, scikit-learn, matplotlib, seaborn, plotly, 
xgboost, lightgbm, fastai, joblib, pyarrow, opendatasets
```

---

## � Notebook Summaries

### 1️⃣ Linear Regression (`LinearRegression.ipynb`)
**Goal**: Predict medical insurance charges based on patient attributes

**Key Concepts**:
- Simple linear regression: `charges = w * age + b`
- Multi-variable regression with age, BMI, children, smoker status
- Manual RMSE calculation vs sklearn's built-in methods
- Encoding categorical variables (smoker → 0/1)

**Models Used**:
```python
LinearRegression()      # Closed-form solution
SGDRegressor()          # Stochastic Gradient Descent
```

---

### 2️⃣ Logistic Regression (`logistic_regression.ipynb`)
**Goal**: Predict if it will rain tomorrow in Australia

**Key Concepts**:
- Binary classification with probability outputs
- Time-based train/val/test split (before 2015 / 2015 / after 2015)
- Complete preprocessing pipeline:
  - `SimpleImputer` for missing values
  - `MinMaxScaler` for normalization
  - `OneHotEncoder` for categorical features
- Saving processed data with **Parquet** format

---

### 3️⃣ Decision Trees & Random Forest (`random_forest.ipynb`)
**Goal**: Rain prediction with tree-based models

**Key Concepts**:
- **Overfitting**: Unconstrained trees = 100% train accuracy, poor validation
- **Regularization**: `max_depth`, `max_leaf_nodes`, `min_samples_split`
- **Ensemble Learning**: Random Forest combines multiple trees
- **Feature Importance**: Identify top predictors

**Hyperparameter Tuning**:
```python
# Finding optimal depth
errors_df = pd.DataFrame([max_depth_error(md) for md in range(1, 21)])
```

---

### 4️⃣ XGBoost (`XGboost.ipynb`)
**Goal**: Predict daily sales for Rossmann stores

**Key Concepts**:
- **Feature Engineering**: 
  - Date extraction (Year, Month, Day, WeekOfYear)
  - Competition duration calculation
  - Promo period tracking
- **XGBoost Regressor** with gradient boosting
- **K-Fold Cross Validation** for robust evaluation
- Tree visualization with `plot_tree()`

**Model Training**:
```python
model = XGBRegressor(random_state=42, n_jobs=-1, n_estimators=20, max_depth=4)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
score = cross_val_score(model, X, targets, cv=kf, scoring='neg_root_mean_squared_error')
```

---

### 5️⃣ Unsupervised Learning (`Unsupervised.ipynb`)
**Goal**: Cluster iris flowers without labels

**Algorithms Compared**:
| Algorithm | Pros | Cons |
|-----------|------|------|
| **KMeans** | Fast, simple, works well on spherical clusters | Must specify K, sensitive to initialization |
| **DBSCAN** | Finds arbitrary shapes, handles noise | Struggles with varying densities |
| **Agglomerative** | Hierarchical, no K needed upfront | Computationally expensive |

**Elbow Method**:
```python
for n_clusters in range(2, 11):
    model = KMeans(n_clusters, random_state=42).fit(X)
    inertias.append(model.inertia_)
```

---

### 6️⃣ Recommender System (`Reccomender.ipynb`)
**Goal**: Movie recommendations using collaborative filtering

**Key Concepts**:
- **Matrix Factorization** with latent factors
- **FastAI's collab_learner** for embedding-based recommendations
- Learning rate finder for optimization

```python
data = CollabDataBunch.from_df(ratings_df, valid_pct=0.1)
learn = collab_learner(data, n_factors=40, y_range=[0,5.5], wd=.1)
learn.fit_one_cycle(5, 0.01)
```

---

## 🎓 Key Learnings & Takeaways

### Data Preprocessing
| Step | Tool | Purpose |
|------|------|---------|
| Missing Values | `SimpleImputer(strategy='mean')` | Fill NaN with column mean |
| Scaling | `MinMaxScaler()` | Normalize to [0,1] range |
| Encoding | `OneHotEncoder(handle_unknown='ignore')` | Convert categories to binary |

### Model Selection Guide
| Problem Type | Recommended Models |
|--------------|-------------------|
| Regression | Linear Regression → XGBoost |
| Binary Classification | Logistic Regression → Random Forest |
| Clustering | KMeans → DBSCAN |
| Recommendations | Collaborative Filtering |

### Avoiding Overfitting
- ✅ Use train/validation/test splits
- ✅ Apply regularization (max_depth, min_samples)
- ✅ Use cross-validation for robust estimates
- ✅ Monitor train vs validation metrics

### Feature Engineering Tips
- 📅 Extract date components (year, month, day, week)
- 🔢 Create duration features (months since event)
- 🏷️ Encode categorical variables properly
- 📊 Check feature importance after training

---

## 📁 Project Structure

```
SciKit-Learn/
├── models/
│   ├── LinearRegression.ipynb    # Regression basics
│   ├── logistic_regression.ipynb # Classification basics  
│   ├── random_forest.ipynb       # Tree-based models
│   ├── XGboost.ipynb             # Gradient boosting
│   ├── Unsupervised.ipynb        # Clustering algorithms
│   ├── Reccomender.ipynb         # Collaborative filtering
│   └── medical.csv
├── MedicalBill(LR)/
│   ├── ML_flow.ipynb             # EDA & baseline models
│   ├── weather-dataset-rattle-package/
│   │   └── weatherAUS.csv        # Australian weather data
│   ├── *.parquet                 # Preprocessed data files
│   └── aussie_rain.joblib        # Saved model
├── rossman_xgbmodel              # Trained XGBoost model
├── submission.csv                # Kaggle submission
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/Sw4ppleneel/Aussie-Rains.git
cd Aussie-Rains

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

---

## 📊 Datasets Used

| Dataset | Source | Size | Task |
|---------|--------|------|------|
| Medical Insurance | Kaggle/JovianML | 1.3K rows | Predict charges |
| Australian Weather | Kaggle | 145K rows | Predict rain |
| Rossmann Store Sales | Kaggle | 1M+ rows | Predict sales |
| Iris | Seaborn built-in | 150 rows | Clustering |
| MovieLens | GroupLens | 100K ratings | Recommendations |

---

## 📚 Resources

- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [FastAI Collaborative Filtering](https://docs.fast.ai/collab.html)
- [Kaggle Competitions](https://www.kaggle.com/)

---

## 🔮 Future Improvements

- [ ] Add hyperparameter tuning with GridSearchCV/Optuna
- [ ] Implement neural network models with PyTorch
- [ ] Add model deployment examples (Flask/FastAPI)
- [ ] Include more advanced feature engineering techniques

---

*Built with ❤️ while learning Machine Learning*
