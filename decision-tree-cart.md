# Decision Tree (CART)

#### Decision Tree (CART): A Step-by-Step Tutorial

Decision Tree (Classification and Regression Tree, CART) is a powerful and popular machine learning algorithm used for both classification and regression tasks. In this tutorial, codeswithpankaj will guide you through the steps to perform decision tree analysis using Python, ensuring that it is easy to understand for students.

**Table of Contents**

1. Introduction to Decision Trees
2. Setting Up the Environment
3. Loading the Dataset
4. Exploring the Data
5. Preparing the Data
6. Building the Decision Tree Model
7. Evaluating the Model
8. Visualizing the Tree
9. Making Predictions
10. Conclusion

***

#### 1. Introduction to Decision Trees

A Decision Tree is a flowchart-like structure where each internal node represents a decision based on a feature, each branch represents the outcome of the decision, and each leaf node represents a class label (for classification) or a continuous value (for regression).

**Advantages**:

* Easy to understand and interpret.
* Can handle both numerical and categorical data.
* Requires little data preprocessing.

**Disadvantages**:

* Prone to overfitting.
* Can be unstable with small changes in data.

#### 2. Setting Up the Environment

First, we need to install the necessary libraries. We'll use `numpy`, `pandas`, `matplotlib`, and `scikit-learn`.

```python
# Install the libraries (uncomment the lines below if you haven't installed them yet)
# !pip install numpy pandas matplotlib scikit-learn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
```

**Explanation of Libraries**:

* **Numpy**: Used for numerical operations.
* **Pandas**: Used for data manipulation and analysis.
* **Matplotlib**: Used for data visualization.
* **Scikit-learn**: Provides tools for machine learning, including decision trees.

#### 3. Loading the Dataset

We'll use a simple dataset for this tutorial. You can use any dataset, but for simplicity, we'll create a synthetic dataset.

```python
# Create a synthetic dataset for classification
np.random.seed(0)
X = np.random.rand(100, 2)  # 100 samples, 2 features
y = (X[:, 0] + X[:, 1] > 1).astype(int)  # Binary outcome

# Convert to pandas DataFrame
data = pd.DataFrame(data=np.hstack((X, y.reshape(-1, 1))), columns=['X1', 'X2', 'y'])
```

**Understanding the Data**:

* **X1, X2**: Independent variables (features).
* **y**: Dependent variable (binary target).
* **Synthetic Dataset**: Created using random numbers to simulate real-world data.

#### 4. Exploring the Data

Let's take a look at the first few rows of the dataset to understand its structure.

```python
# Display the first few rows of the dataset
print(data.head())
```

**Data Exploration Techniques**:

* **Head Method**: Shows the first few rows.
* **Describe Method**: Provides summary statistics.
* **Info Method**: Gives information about data types and non-null values.

```python
# Summary statistics
print(data.describe())

# Information about data types and non-null values
print(data.info())

# Visualize the data
plt.scatter(data['X1'], data['X2'], c=data['y'], cmap='viridis')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Scatter plot of X1 vs X2')
plt.show()
```

#### 5. Preparing the Data

We'll split the data into training and testing sets to evaluate the model's performance.

```python
# Split the data into training and testing sets
X = data[['X1', 'X2']]
y = data['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

**Importance of Data Splitting**:

* **Training Set**: Used to train the model.
* **Testing Set**: Used to evaluate the model's performance.
* **Test Size**: Proportion of the dataset used for testing (e.g., 20%).

#### 6. Building the Decision Tree Model

Now, let's build the decision tree model using the training data.

**Classification**

```python
# Create the Decision Tree Classifier model
clf = DecisionTreeClassifier(random_state=0)

# Train the model
clf.fit(X_train, y_train)
```

**Regression**

For regression tasks, you can use `DecisionTreeRegressor` instead.

```python
# Create the Decision Tree Regressor model (for regression tasks)
reg = DecisionTreeRegressor(random_state=0)

# Train the model
reg.fit(X_train, y_train)
```

**Steps in Model Building**:

1. **Model Creation**: Instantiate the decision tree model.
2. **Model Training**: Fit the model to the training data using the `fit` method.

#### 7. Evaluating the Model

We'll evaluate the model by calculating accuracy for classification and mean squared error (MSE) for regression.

**Classification**

```python
# Make predictions on the test data
y_pred = clf.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)

# Generate the classification report
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(class_report)
```

**Regression**

```python
# Make predictions on the test data
y_pred = reg.predict(X_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
```

**Evaluation Metrics**:

* **Accuracy (Classification)**: Proportion of correctly predicted instances.
* **Classification Report**: Provides precision, recall, F1-score, and support for each class.
* **Mean Squared Error (MSE) (Regression)**: Measures the average squared difference between predicted and actual values.

#### 8. Visualizing the Tree

Visualizing the decision tree helps in understanding the model's decisions.

```python
# Plot the decision tree (for classification)
plt.figure(figsize=(20,10))
plot_tree(clf, filled=True, feature_names=['X1', 'X2'], class_names=['0', '1'])
plt.show()
```

#### 9. Making Predictions

Finally, let's use the model to make predictions.

**Classification**

```python
# Example: Predicting the outcome for new data points
new_data = np.array([[0.5, 0.8], [0.2, 0.1]])
predictions = clf.predict(new_data)
print(f"Predictions for new data points: {predictions}")
```

**Regression**

```python
# Example: Predicting the outcome for new data points (regression)
new_data = np.array([[0.5, 0.8], [0.2, 0.1]])
predictions = reg.predict(new_data)
print(f"Predictions for new data points: {predictions}")
```

**Prediction Process**:

* **New Data**: Input data for which we want to make predictions.
* **Model Prediction**: Use the `predict` method to get the predicted outcome.

#### 10. Conclusion

In this tutorial by codeswithpankaj, we've covered the basics of decision tree (CART) and how to implement it using Python. We walked through setting up the environment, loading and exploring the data, preparing the data, building the model, evaluating the model, visualizing the tree, and making predictions. Decision trees are powerful tools in data science for both classification and regression tasks.

***

For more tutorials and resources, visit [codeswithpankaj.com](https://codeswithpankaj.com).
