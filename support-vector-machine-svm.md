# Support Vector Machine (SVM)

#### Support Vector Machines Classifier Tutorial with Python

Support Vector Machines (SVM) are powerful supervised machine learning algorithms used for both classification and regression tasks. In this tutorial, codeswithpankaj will guide you through a detailed step-by-step process to perform SVM analysis using Python.

**Table of Contents**

1. Introduction to Support Vector Machines
2. Support Vector Machines Intuition
3. Kernel Trick
4. SVM Scikit-Learn Libraries
5. Dataset Description
6. Import Libraries
7. Import Dataset
8. Exploratory Data Analysis
9. Declare Feature Vector and Target Variable
10. Split Data into Separate Training and Test Set
11. Feature Scaling
12. Run SVM with Default Hyperparameters
13. Run SVM with Linear Kernel
14. Run SVM with Polynomial Kernel
15. Run SVM with Sigmoid Kernel
16. Confusion Matrix
17. Classification Metrics
18. ROC - AUC
19. Stratified K-Fold Cross Validation with Shuffle Split
20. Hyperparameter Optimization Using GridSearchCV
21. Results and Conclusion

***

#### 1. Introduction to Support Vector Machines

Support Vector Machine (SVM) is a supervised learning algorithm that finds a hyperplane that best divides a dataset into classes. It can handle both linear and non-linear data using the kernel trick.

**Key Features**:

* Effective in high-dimensional spaces.
* Uses a subset of training points in the decision function (support vectors).
* Versatile with different kernel functions (linear, polynomial, RBF, sigmoid).

#### 2. Support Vector Machines Intuition

SVM works by finding the hyperplane that best separates the data points of different classes. The points closest to the hyperplane are called support vectors. The distance between the hyperplane and the support vectors is the margin, and SVM aims to maximize this margin.

#### 3. Kernel Trick

The kernel trick allows SVM to create non-linear decision boundaries. By applying a kernel function, SVM maps the original data into a higher-dimensional space where a linear separator can be found.

Common Kernel Functions:

* Linear Kernel
* Polynomial Kernel
* Radial Basis Function (RBF) Kernel
* Sigmoid Kernel

#### 4. SVM Scikit-Learn Libraries

Scikit-learn provides an easy-to-use implementation of SVM through the `SVC` class. It supports various kernel functions and hyperparameters for fine-tuning the model.

#### 5. Dataset Description

The Pulsar Star dataset contains features extracted from the integrated profile and DM-SNR curve. The dataset contains 17,898 samples and 9 attributes.

#### 6. Import Libraries

First, we need to import the necessary libraries.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
```

#### 7. Import Dataset

Download the dataset from [this link](https://archive.ics.uci.edu/ml/machine-learning-databases/00372/HTRU2.zip) and extract it. We'll load it into a Pandas DataFrame.

```python
# Load the dataset from a CSV file
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00372/HTRU_2.csv'
data = pd.read_csv(url, header=None)

# Rename the columns
data.columns = ['Mean_IP', 'Std_IP', 'Kurt_IP', 'Skew_IP', 'Mean_DM', 'Std_DM', 'Kurt_DM', 'Skew_DM', 'Class']
```

#### 8. Exploratory Data Analysis

Let's take a look at the first few rows of the dataset to understand its structure.

```python
# Display the first few rows of the dataset
print(data.head())

# Summary statistics
print(data.describe())

# Information about data types and non-null values
print(data.info())

# Visualize the data
plt.scatter(data['Mean_IP'], data['Std_IP'], c=data['Class'], cmap='viridis')
plt.xlabel('Mean Integrated Profile')
plt.ylabel('Standard Deviation Integrated Profile')
plt.title('Scatter plot of Mean vs Standard Deviation Integrated Profile')
plt.show()
```

#### 9. Declare Feature Vector and Target Variable

```python
# Define the feature columns and target column
features = ['Mean_IP', 'Std_IP', 'Kurt_IP', 'Skew_IP', 'Mean_DM', 'Std_DM', 'Kurt_DM', 'Skew_DM']
target = 'Class'

# Split the data into features (X) and target (y)
X = data[features]
y = data[target]
```

#### 10. Split Data into Separate Training and Test Set

```python
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

#### 11. Feature Scaling

Feature scaling is important for SVM as it is sensitive to the magnitudes of the features.

```python
# Initialize the StandardScaler
scaler = StandardScaler()

# Fit and transform the training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform the test data
X_test_scaled = scaler.transform(X_test)
```

#### 12. Run SVM with Default Hyperparameters

```python
# Initialize the SVM model with default hyperparameters
svm_default = SVC()

# Train the model
svm_default.fit(X_train_scaled, y_train)

# Make predictions
y_pred_default = svm_default.predict(X_test_scaled)

# Evaluate the model
accuracy_default = accuracy_score(y_test, y_pred_default)
print(f"Accuracy with default hyperparameters: {accuracy_default}")
```

#### 13. Run SVM with Linear Kernel

```python
# Initialize the SVM model with a linear kernel
svm_linear = SVC(kernel='linear')

# Train the model
svm_linear.fit(X_train_scaled, y_train)

# Make predictions
y_pred_linear = svm_linear.predict(X_test_scaled)

# Evaluate the model
accuracy_linear = accuracy_score(y_test, y_pred_linear)
print(f"Accuracy with linear kernel: {accuracy_linear}")
```

#### 14. Run SVM with Polynomial Kernel

```python
# Initialize the SVM model with a polynomial kernel
svm_poly = SVC(kernel='poly')

# Train the model
svm_poly.fit(X_train_scaled, y_train)

# Make predictions
y_pred_poly = svm_poly.predict(X_test_scaled)

# Evaluate the model
accuracy_poly = accuracy_score(y_test, y_pred_poly)
print(f"Accuracy with polynomial kernel: {accuracy_poly}")
```

#### 15. Run SVM with Sigmoid Kernel

```python
# Initialize the SVM model with a sigmoid kernel
svm_sigmoid = SVC(kernel='sigmoid')

# Train the model
svm_sigmoid.fit(X_train_scaled, y_train)

# Make predictions
y_pred_sigmoid = svm_sigmoid.predict(X_test_scaled)

# Evaluate the model
accuracy_sigmoid = accuracy_score(y_test, y_pred_sigmoid)
print(f"Accuracy with sigmoid kernel: {accuracy_sigmoid}")
```

#### 16. Confusion Matrix

```python
# Compute the confusion matrix for the linear kernel model
conf_matrix = confusion_matrix(y_test, y_pred_linear)
print("Confusion Matrix:")
print(conf_matrix)
```

#### 17. Classification Metrics

```python
# Generate the classification report for the linear kernel model
class_report = classification_report(y_test, y_pred_linear)
print("Classification Report:")
print(class_report)
```

#### 18. ROC - AUC

```python
# Compute ROC-AUC for the linear kernel model
y_pred_prob = svm_linear.decision_function(X_test_scaled)
roc_auc = roc_auc_score(y_test, y_pred_prob)

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Plot ROC curve
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='best')
plt.show()
```

#### 19. Stratified K-Fold Cross Validation with Shuffle Split

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
    svm = SVC(kernel='linear')
    svm.fit(X_train_fold_scaled, y_train_fold)
    y_pred_fold = svm.predict(X_test_fold_scaled)
    accuracy_fold = accuracy_score(y_test_fold, y_pred_fold)
    cross_val_scores.append(accuracy_fold)

print("Cross-validation scores:", cross_val_scores)


print("Mean cross-validation score:", np.mean(cross_val_scores))
```

#### 20. Hyperparameter Optimization Using GridSearchCV

```python
# Define the parameter grid
param_grid = {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf', 'poly', 'sigmoid']}

# Create the GridSearchCV object
grid_search = GridSearchCV(SVC(), param_grid, cv=5)

# Perform the grid search
grid_search.fit(X_train_scaled, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")

# Train the best model
best_svm_model = grid_search.best_estimator_

# Evaluate the best model
best_y_pred = best_svm_model.predict(X_test_scaled)
best_accuracy = accuracy_score(y_test, best_y_pred)
print(f"Best Model Accuracy: {best_accuracy}")
```

#### 21. Results and Conclusion

In this tutorial by codeswithpankaj, we've covered the basics of Support Vector Machine (SVM) and how to implement it using Python with the Pulsar Star dataset. We walked through setting up the environment, loading and exploring the data, preparing the data, building the model, evaluating the model, making predictions, and tuning the model. SVM is a powerful tool in data science for both classification and regression tasks.

***



For more tutorials and resources, visit [codeswithpankaj.com](https://codeswithpankaj.com).
