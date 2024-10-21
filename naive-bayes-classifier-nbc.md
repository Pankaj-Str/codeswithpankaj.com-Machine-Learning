# Naive Bayes Classifier (NBC)

#### Naive Bayes Classifier (NBC): A Step-by-Step Tutorial

Naive Bayes Classifier (NBC) is a simple yet powerful supervised machine learning algorithm used for classification tasks. In this tutorial, codeswithpankaj will guide you through the steps to perform Naive Bayes classification using Python.

**Table of Contents**

1. Introduction to Naive Bayes Classifier
2. Types of Naive Bayes Classifiers
3. Naive Bayes Intuition
4. Naive Bayes Assumptions
5. Naive Bayes Scikit-Learn Libraries
6. Dataset Description
7. Import Libraries
8. Import Dataset
9. Exploratory Data Analysis
10. Declare Feature Vector and Target Variable
11. Split Data into Separate Training and Test Set
12. Feature Scaling (if necessary)
13. Run Naive Bayes Classifier
14. Confusion Matrix
15. Classification Metrics
16. Stratified K-Fold Cross Validation
17. Hyperparameter Optimization Using GridSearchCV
18. Results and Conclusion

***

#### 1. Introduction to Naive Bayes Classifier

Naive Bayes Classifier is a probabilistic classifier based on Bayes' theorem with the "naive" assumption of conditional independence between every pair of features given the class label.

**Key Features**:

* Simple and easy to implement.
* Works well with small datasets.
* Handles both binary and multi-class classification problems.

#### 2. Types of Naive Bayes Classifiers

* **Gaussian Naive Bayes**: Assumes that the features follow a normal distribution.
* **Multinomial Naive Bayes**: Suitable for discrete data, often used for text classification.
* **Bernoulli Naive Bayes**: Suitable for binary/boolean features.

#### 3. Naive Bayes Intuition

Naive Bayes classifiers work by calculating the probability of each class based on the given features and selecting the class with the highest probability. It applies Bayes' theorem with strong (naive) independence assumptions.

#### 4. Naive Bayes Assumptions

The primary assumption of Naive Bayes is that all features are conditionally independent given the class label. While this assumption is rarely true in real-world data, Naive Bayes often performs well in practice.

#### 5. Naive Bayes Scikit-Learn Libraries

Scikit-learn provides easy-to-use implementations of Naive Bayes classifiers through `GaussianNB`, `MultinomialNB`, and `BernoulliNB` classes.

#### 6. Dataset Description

We'll use the Iris dataset for this tutorial. The dataset contains three classes of iris plants, each with four features: sepal length, sepal width, petal length, and petal width.

#### 7. Import Libraries

First, we need to import the necessary libraries.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
```

#### 8. Import Dataset

We'll load the Iris dataset directly from Scikit-learn.

```python
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['target'] = iris.target
```

#### 9. Exploratory Data Analysis

Let's take a look at the first few rows of the dataset to understand its structure.

```python
# Display the first few rows of the dataset
print(data.head())

# Summary statistics
print(data.describe())

# Information about data types and non-null values
print(data.info())

# Visualize the data
pd.plotting.scatter_matrix(data.iloc[:, :-1], c=data['target'], figsize=(15, 15), marker='o',
                           hist_kwds={'bins': 20}, s=60, alpha=.8)
plt.show()
```

#### 10. Declare Feature Vector and Target Variable

```python
# Define the feature columns and target column
features = iris.feature_names
target = 'target'

# Split the data into features (X) and target (y)
X = data[features]
y = data[target]
```

#### 11. Split Data into Separate Training and Test Set

```python
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
```

#### 12. Feature Scaling (if necessary)

For Naive Bayes, feature scaling is generally not required, but it can be beneficial in some cases, especially when using GaussianNB.

```python
# Initialize the StandardScaler
scaler = StandardScaler()

# Fit and transform the training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform the test data
X_test_scaled = scaler.transform(X_test)
```

#### 13. Run Naive Bayes Classifier

We'll start with the Gaussian Naive Bayes classifier.

```python
# Initialize the Gaussian Naive Bayes model
gnb = GaussianNB()

# Train the model
gnb.fit(X_train_scaled, y_train)

# Make predictions
y_pred = gnb.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
```

#### 14. Confusion Matrix

```python
# Compute the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)
```

#### 15. Classification Metrics

```python
# Generate the classification report
class_report = classification_report(y_test, y_pred)
print("Classification Report:")
print(class_report)
```

#### 16. Stratified K-Fold Cross Validation

```python
# Initialize StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

# Perform cross-validation
cross_val_scores = []
for train_index, test_index in skf.split(X, y):
    X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
    y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
    
    # Scale the data
    X_train_fold_scaled = scaler.fit_transform(X_train_fold)
    X_test_fold_scaled = scaler.transform(X_test_fold)
    
    # Train and evaluate the model
    gnb = GaussianNB()
    gnb.fit(X_train_fold_scaled, y_train_fold)
    y_pred_fold = gnb.predict(X_test_fold_scaled)
    accuracy_fold = accuracy_score(y_test_fold, y_pred_fold)
    cross_val_scores.append(accuracy_fold)

print("Cross-validation scores:", cross_val_scores)
print("Mean cross-validation score:", np.mean(cross_val_scores))
```

#### 17. Hyperparameter Optimization Using GridSearchCV

For Gaussian Naive Bayes, there aren't many hyperparameters to tune. For Multinomial and Bernoulli Naive Bayes, we can tune the `alpha` parameter.

```python
# Define the parameter grid for MultinomialNB
param_grid = {'alpha': [0.1, 0.5, 1, 5, 10]}

# Create the GridSearchCV object for MultinomialNB
grid_search = GridSearchCV(MultinomialNB(), param_grid, cv=5)

# Perform the grid search
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print(f"Best parameters for MultinomialNB: {best_params}")

# Train the best model
best_mnb_model = grid_search.best_estimator_

# Evaluate the best model
best_y_pred = best_mnb_model.predict(X_test)
best_accuracy = accuracy_score(y_test, best_y_pred)
print(f"Best Model Accuracy for MultinomialNB: {best_accuracy}")
```

#### 18. Results and Conclusion

In this tutorial by codeswithpankaj, we've covered the basics of Naive Bayes Classifier (NBC) and how to implement it using Python. We walked through setting up the environment, loading and exploring the data, preparing the data, building the model, evaluating the model, making predictions, and tuning the model. Naive Bayes is a simple yet powerful tool in data science for classification tasks.

***



For more tutorials and resources, visit [codeswithpankaj.com](https://codeswithpankaj.com).
