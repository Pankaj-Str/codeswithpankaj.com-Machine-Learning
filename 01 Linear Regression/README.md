# Linear Regression

### Linear Regression in Machine Learning

Linear Regression is a fundamental machine learning algorithm used for predicting a continuous dependent variable based on one or more independent variables. It's a type of supervised learning algorithm and one of the simplest algorithms in machine learning.

#### Key Concepts

1. **Dependent Variable**: The variable we are trying to predict.
2. **Independent Variable**: The variable(s) used to predict the dependent variable.
3. **Linear Relationship**: The relationship between the dependent and independent variables is assumed to be linear.

### Step-by-Step Explanation

#### 1. **Understanding the Equation of a Line**

The equation of a simple linear regression line is:

\[ y = mx + c \]

- \( y \) is the dependent variable.
- \( x \) is the independent variable.
- \( m \) is the slope of the line.
- \( c \) is the y-intercept.

In multiple linear regression with multiple independent variables, the equation is:

\[ y = b_0 + b_1x_1 + b_2x_2 + \cdots + b_nx_n \]

- \( y \) is the dependent variable.
- \( x_1, x_2, \cdots, x_n \) are the independent variables.
- \( b_0 \) is the intercept.
- \( b_1, b_2, \cdots, b_n \) are the coefficients of the independent variables.

#### 2. **Dataset Example**

Suppose we have a dataset of house prices. The dataset includes features such as the size of the house (in square feet) and the price of the house. Here, the price is the dependent variable (\( y \)), and the size is the independent variable (\( x \)).

| Size (sq ft) | Price ($) |
|--------------|-----------|
| 1500         | 300,000   |
| 1600         | 320,000   |
| 1700         | 340,000   |
| 1800         | 360,000   |
| 1900         | 380,000   |

#### 3. **Visualizing the Data**

Plotting the data points on a graph helps visualize the relationship between the size and the price of the houses.

#### 4. **Finding the Best Fit Line**

The goal is to find the line that best fits the data. This line minimizes the difference between the actual data points and the predicted data points. This is done using the **Least Squares Method**.

The formula for the slope \( m \) and intercept \( c \) are derived using calculus to minimize the error.

#### 5. **Implementing Linear Regression**

Let's implement this in Python using the `sklearn` library.

```python
# Importing the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Dataset
X = np.array([1500, 1600, 1700, 1800, 1900]).reshape(-1, 1)  # Independent variable (Size)
y = np.array([300000, 320000, 340000, 360000, 380000])  # Dependent variable (Price)

# Creating the linear regression model
model = LinearRegression()
model.fit(X, y)

# Making predictions
y_pred = model.predict(X)

# Plotting the results
plt.scatter(X, y, color='blue')  # Original data points
plt.plot(X, y_pred, color='red')  # Regression line
plt.xlabel('Size (sq ft)')
plt.ylabel('Price ($)')
plt.title('House Prices Prediction')
plt.show()

# Printing the coefficients
print(f'Intercept: {model.intercept_}')
print(f'Slope: {model.coef_[0]}')
```

#### 6. **Interpreting the Output**

- **Intercept (\( c \))**: The value of the dependent variable when all the independent variables are zero.
- **Slope (\( m \))**: The change in the dependent variable for a one-unit change in the independent variable.

For example, if the intercept is 100,000 and the slope is 150, then the equation of our line is:

\[ \text{Price} = 100,000 + 150 \times \text{Size} \]

#### 7. **Evaluating the Model**

The performance of the linear regression model can be evaluated using metrics like:

- **Mean Squared Error (MSE)**
- **R-squared (R²)**

```python
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
```

#### 8. **Making Predictions**

Once the model is trained, it can be used to make predictions on new data.

```python
# Predicting the price of a house with 2000 sq ft
new_size = np.array([[2000]])
predicted_price = model.predict(new_size)
print(f'Predicted price for a house with 2000 sq ft: ${predicted_price[0]}')
```

### Conclusion

Linear Regression is a powerful and easy-to-understand algorithm for predicting continuous variables. By following these steps, you can implement and understand linear regression and apply it to real-world datasets.

For more detailed tutorials and examples, visit [codeswithpankaj.com](https://codeswithpankaj.com).
