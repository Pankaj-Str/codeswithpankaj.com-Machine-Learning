# Multiple Linerar Regression

#### Multiple Linear Regression: A Step-by-Step Tutorial

Multiple Linear Regression extends simple linear regression by modeling the relationship between multiple independent variables and a dependent variable. In this tutorial, codeswithpankaj will guide you through the steps to perform multiple linear regression using Python, ensuring that it is easy to understand for students.

**Table of Contents**

1. Introduction to Multiple Linear Regression
2. Setting Up the Environment
3. Loading the Dataset
4. Exploring the Data
5. Preparing the Data
6. Building the Multiple Linear Regression Model
7. Evaluating the Model
8. Making Predictions
9. Conclusion

***

#### 1. Introduction to Multiple Linear Regression

Multiple Linear Regression models the relationship between a dependent variable and multiple independent variables. The equation for multiple linear regression is:

\[ y = b\_0 + b\_1x\_1 + b\_2x\_2 + ... + b\_nx\_n ]



$$
[ y = b_0 + b_1x_1 + b_2x_2 + ... + b_nx_n ]
$$

<figure><img src="../.gitbook/assets/image (1) (1).png" alt=""><figcaption></figcaption></figure>

**Applications**:

* Predicting house prices based on features like size, location, and number of rooms.
* Estimating sales based on advertising spend across different media channels.
* Forecasting crop yield based on various environmental factors.

#### 2. Setting Up the Environment

First, we need to install the necessary libraries. We'll use `numpy`, `pandas`, `matplotlib`, and `scikit-learn`.

```python
# Install the libraries (uncomment the lines below if you haven't installed them yet)
# !pip install numpy pandas matplotlib scikit-learn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
```

**Explanation of Libraries**:

* **Numpy**: Used for numerical operations.
* **Pandas**: Used for data manipulation and analysis.
* **Matplotlib**: Used for data visualization.
* **Scikit-learn**: Provides tools for machine learning, including linear regression.

#### 3. Loading the Dataset

We'll use a simple dataset for this tutorial. You can use any dataset, but for simplicity, we'll create a synthetic dataset.

```python
# Create a synthetic dataset
np.random.seed(0)
X = 2 * np.random.rand(100, 3)  # 100 samples, 3 features
y = 4 + 3 * X[:, 0] + 2 * X[:, 1] + X[:, 2] + np.random.randn(100)

# Convert to pandas DataFrame
data = pd.DataFrame(data=np.hstack((X, y.reshape(-1, 1))), columns=['X1', 'X2', 'X3', 'y'])
```

**Understanding the Data**:

* **X1, X2, X3**: Independent variables (features).
* **y**: Dependent variable (target).
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
```

#### 5. Preparing the Data

We'll split the data into training and testing sets to evaluate the model's performance.

```python
# Split the data into training and testing sets
X = data[['X1', 'X2', 'X3']]
y = data['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

**Importance of Data Splitting**:

* **Training Set**: Used to train the model.
* **Testing Set**: Used to evaluate the model's performance.
* **Test Size**: Proportion of the dataset used for testing (e.g., 20%).

#### 6. Building the Multiple Linear Regression Model

Now, let's build the multiple linear regression model using the training data.

```python
# Create the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)
```

**Steps in Model Building**:

1. **Model Creation**: Instantiate the linear regression model.
2. **Model Training**: Fit the model to the training data using the `fit` method.

#### 7. Evaluating the Model

We'll evaluate the model by calculating the mean squared error (MSE) and the coefficient of determination (R²).

```python
# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)

# Calculate the coefficient of determination
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")
```

**Evaluation Metrics**:

* **Mean Squared Error (MSE)**: Measures the average squared difference between predicted and actual values.
* **R² Score**: Indicates the proportion of the variance in the dependent variable that is predictable from the independent variable(s).

#### 8. Making Predictions

Finally, let's use the model to make predictions.

```python
# Example: Predicting the value for new data points
new_data = np.array([[1.5, 2.0, 3.0]])
prediction = model.predict(new_data)
print(f"Prediction for X1=1.5, X2=2.0, X3=3.0: {prediction[0]}")
```

**Prediction Process**:

* **New Data**: Input data for which we want to make predictions.
* **Model Prediction**: Use the `predict` method to get the predicted value.

#### 9. Conclusion

In this tutorial by codeswithpankaj, we've covered the basics of multiple linear regression and how to implement it using Python. We walked through setting up the environment, loading and exploring the data, preparing the data, building the model, evaluating the model, and making predictions. Multiple linear regression is a powerful tool in data science, and understanding it will help you tackle many predictive modeling problems.

***

For more tutorials and resources, visit [codeswithpankaj.com](https://codeswithpankaj.com).

