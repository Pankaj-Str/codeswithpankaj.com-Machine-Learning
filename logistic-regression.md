# Logistic Regression

#### Logistic Regression: A Step-by-Step Tutorial

Logistic Regression is a statistical method used for binary classification problems, where the outcome is a binary variable (e.g., yes/no, true/false). In this tutorial, codeswithpankaj will guide you through the steps to perform logistic regression using Python, ensuring that it is easy to understand for students.

**Table of Contents**

1. Introduction to Logistic Regression
2. Setting Up the Environment
3. Loading the Dataset
4. Exploring the Data
5. Preparing the Data
6. Building the Logistic Regression Model
7. Evaluating the Model
8. Making Predictions
9. Conclusion

***

#### 1. Introduction to Logistic Regression

Logistic Regression is used for predicting the probability of a binary outcome. Unlike linear regression, which predicts a continuous outcome, logistic regression predicts a probability that maps to two possible outcomes.

The logistic regression equation is:



$$
[ P(Y=1|X) = \frac{1}{1 + e^{-(b_0 + b_1X_1 + b_2X_2 + ... + b_nX_n)}} ]
$$

<figure><img src=".gitbook/assets/image (3).png" alt=""><figcaption></figcaption></figure>

**Applications**:

* Predicting if an email is spam or not.
* Determining if a customer will buy a product.
* Diagnosing diseases based on symptoms.

#### 2. Setting Up the Environment

First, we need to install the necessary libraries. We'll use `numpy`, `pandas`, `matplotlib`, and `scikit-learn`.

```python
# Install the libraries (uncomment the lines below if you haven't installed them yet)
# !pip install numpy pandas matplotlib scikit-learn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
```

**Explanation of Libraries**:

* **Numpy**: Used for numerical operations.
* **Pandas**: Used for data manipulation and analysis.
* **Matplotlib**: Used for data visualization.
* **Scikit-learn**: Provides tools for machine learning, including logistic regression.

#### 3. Loading the Dataset

We'll use a simple dataset for this tutorial. You can use any dataset, but for simplicity, we'll create a synthetic dataset.

```python
# Create a synthetic dataset
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

#### 6. Building the Logistic Regression Model

Now, let's build the logistic regression model using the training data.

```python
# Create the Logistic Regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)
```

**Steps in Model Building**:

1. **Model Creation**: Instantiate the logistic regression model.
2. **Model Training**: Fit the model to the training data using the `fit` method.

#### 7. Evaluating the Model

We'll evaluate the model by calculating accuracy, confusion matrix, and classification report.

```python
# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)

# Generate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Generate the classification report
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)
```

**Evaluation Metrics**:

* **Accuracy**: Proportion of correctly predicted instances.
* **Confusion Matrix**: Table showing the true positives, true negatives, false positives, and false negatives.
* **Classification Report**: Provides precision, recall, F1-score, and support for each class.

#### 8. Making Predictions

Finally, let's use the model to make predictions.

```python
# Example: Predicting the outcome for new data points
new_data = np.array([[0.5, 0.8], [0.2, 0.1]])
predictions = model.predict(new_data)
print(f"Predictions for new data points: {predictions}")
```

**Prediction Process**:

* **New Data**: Input data for which we want to make predictions.
* **Model Prediction**: Use the `predict` method to get the predicted outcome.

#### 9. Conclusion

In this tutorial by codeswithpankaj, we've covered the basics of logistic regression and how to implement it using Python. We walked through setting up the environment, loading and exploring the data, preparing the data, building the model, evaluating the model, and making predictions. Logistic regression is a powerful tool in data science for binary classification problems.

***

For more tutorials and resources, visit [codeswithpankaj.com](https://codeswithpankaj.com).
