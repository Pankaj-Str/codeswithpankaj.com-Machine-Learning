# Polynomial Regression

#### Polynomial Regression: A Step-by-Step Tutorial

Polynomial Regression is a type of regression analysis where the relationship between the independent variable (x) and the dependent variable (y) is modeled as an (n)th degree polynomial. In this tutorial, codeswithpankaj will guide you through the steps to perform polynomial regression using Python, ensuring that it is easy to understand for students.

**Table of Contents**

1. Introduction to Polynomial Regression
2. Setting Up the Environment
3. Loading the Dataset
4. Exploring the Data
5. Preparing the Data
6. Building the Polynomial Regression Model
7. Evaluating the Model
8. Making Predictions
9. Conclusion

***

#### 1. Introduction to Polynomial Regression

Polynomial Regression is used when the relationship between the independent variable (x) and the dependent variable (y) is not linear. The polynomial regression equation is:



$$
[ y = b_0 + b_1x + b_2x^2 + ... + b_nx^n ]
$$

<figure><img src=".gitbook/assets/image (2) (1).png" alt=""><figcaption></figcaption></figure>

**Applications**:

* Predicting growth rates.
* Modeling complex relationships in data.
* Estimating non-linear trends.

#### 2. Setting Up the Environment

First, we need to install the necessary libraries. We'll use `numpy`, `pandas`, `matplotlib`, and `scikit-learn`.

```python
# Install the libraries (uncomment the lines below if you haven't installed them yet)
# !pip install numpy pandas matplotlib scikit-learn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
```

**Explanation of Libraries**:

* **Numpy**: Used for numerical operations.
* **Pandas**: Used for data manipulation and analysis.
* **Matplotlib**: Used for data visualization.
* **Scikit-learn**: Provides tools for machine learning, including polynomial regression.

#### 3. Loading the Dataset

We'll use a simple dataset for this tutorial. You can use any dataset, but for simplicity, we'll create a synthetic dataset.

```python
# Create a synthetic dataset
np.random.seed(0)
X = 2 - 3 * np.random.normal(0, 1, 100)
y = X - 2 * (X ** 2) + np.random.normal(-3, 3, 100)

# Convert to pandas DataFrame
data = pd.DataFrame(data={'X': X, 'y': y})
```

**Understanding the Data**:

* **X**: Independent variable (feature).
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

# Visualize the data
plt.scatter(data['X'], data['y'], color='blue')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Scatter plot of X vs y')
plt.show()
```

#### 5. Preparing the Data

We'll split the data into training and testing sets to evaluate the model's performance.

```python
# Split the data into training and testing sets
X = data[['X']]
y = data['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

**Importance of Data Splitting**:

* **Training Set**: Used to train the model.
* **Testing Set**: Used to evaluate the model's performance.
* **Test Size**: Proportion of the dataset used for testing (e.g., 20%).

#### 6. Building the Polynomial Regression Model

We'll transform the data to include polynomial features and then fit a linear regression model.

```python
# Create polynomial features
poly = PolynomialFeatures(degree=2)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)

# Create the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_poly_train, y_train)
```

**Steps in Model Building**:

1. **Polynomial Transformation**: Convert the original features to polynomial features.
2. **Model Creation**: Instantiate the linear regression model.
3. **Model Training**: Fit the model to the training data using the `fit` method.

#### 7. Evaluating the Model

We'll evaluate the model by calculating the mean squared error (MSE) and the coefficient of determination (R²).

```python
# Make predictions on the test data
y_pred = model.predict(X_poly_test)

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
new_data = np.array([[1.5]])
new_data_poly = poly.transform(new_data)
prediction = model.predict(new_data_poly)
print(f"Prediction for X=1.5: {prediction[0]}")
```

**Prediction Process**:

* **New Data**: Input data for which we want to make predictions.
* **Polynomial Transformation**: Convert the new data to polynomial features.
* **Model Prediction**: Use the `predict` method to get the predicted value.

#### 9. Conclusion

In this tutorial by codeswithpankaj, we've covered the basics of polynomial regression and how to implement it using Python. We walked through setting up the environment, loading and exploring the data, preparing the data, building the model, evaluating the model, and making predictions. Polynomial regression is a powerful tool in data science for modeling non-linear relationships.

***

For more tutorials and resources, visit [codeswithpankaj.com](https://codeswithpankaj.com).
