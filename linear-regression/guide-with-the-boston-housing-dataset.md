# Guide with the Boston Housing Dataset

#### Linear Regression: A Step-by-Step Guide with the Boston Housing Dataset

Linear regression is a powerful tool used to model the relationship between one or more independent variables and a dependent variable. In this guide, we will use the Boston Housing Dataset to illustrate how to perform linear regression in Python.

**Step 1: Load and Explore the Dataset**

We begin by loading the Boston Housing Dataset, which contains information about various factors affecting house prices in Boston.

```python
import pandas as pd
from sklearn.datasets import load_boston

# Load the Boston Housing Dataset
boston = load_boston()
data = pd.DataFrame(boston.data, columns=boston.feature_names)
data['PRICE'] = boston.target

# Display the first few rows of the dataset
print(data.head())
```

**Explanation**:

* We use `load_boston()` to load the dataset.
* The dataset is converted to a Pandas DataFrame for easier manipulation.
* The column names are set to the feature names provided by the dataset.
* We add a new column called `PRICE` which contains the target variable (house prices).
* `print(data.head())` displays the first few rows of the dataset.

**Step 2: Define the Features and Target**

Next, we define which columns we will use as features (independent variables) and which column is the target (dependent variable).

```python
# Define features and target
X = data[['RM', 'LSTAT', 'PTRATIO']]  # You can choose other features as well
Y = data['PRICE']
```

**Explanation**:

* `X` contains the features we will use to predict the house prices. Here, we use `RM` (average number of rooms per dwelling), `LSTAT` (percentage of lower status of the population), and `PTRATIO` (pupil-teacher ratio by town).
* `Y` is the target variable, which is the house prices (`PRICE`).

**Step 3: Split the Dataset**

We split the dataset into training and testing sets to evaluate our model's performance on unseen data.

```python
from sklearn.model_selection import train_test_split

# Split the dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
```

**Explanation**:

* `train_test_split` splits the data into training and testing sets.
* `test_size=0.2` indicates that 20% of the data will be used for testing, and 80% for training.
* `random_state=42` ensures reproducibility by fixing the random seed.

**Step 4: Create and Train the Model**

We create a linear regression model and train it using the training set.

```python
from sklearn.linear_model import LinearRegression

# Create and train the model
model = LinearRegression()
model.fit(X_train, Y_train)
```

**Explanation**:

* `LinearRegression()` creates an instance of the linear regression model.
* `model.fit(X_train, Y_train)` trains the model using the training data.

**Step 5: Make Predictions**

We use the trained model to make predictions on the test set.

```python
# Predictions
Y_pred = model.predict(X_test)
```

**Explanation**:

* `model.predict(X_test)` uses the trained model to predict house prices for the test set.

**Step 6: Evaluate the Model**

We evaluate the model's performance using Mean Squared Error (MSE) and R-squared (( R^2 )).

```python
from sklearn.metrics import mean_squared_error, r2_score

# Evaluation
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print(f'MSE: {mse}')
print(f'R2 Score: {r2}')
```

**Explanation**:

* `mean_squared_error(Y_test, Y_pred)` calculates the MSE, which measures the average of the squared differences between actual and predicted values.
* `r2_score(Y_test, Y_pred)` calculates the ( R^2 ) score, which indicates the proportion of the variance in the dependent variable that is predictable from the independent variables.
* `print(f'MSE: {mse}')` and `print(f'R2 Score: {r2}')` display the MSE and ( R^2 ) score.

**Step 7: Plot the Results**

We plot the actual vs. predicted values for one of the features to visualize the model's performance.

```python
import matplotlib.pyplot as plt

plt.scatter(X_test['RM'], Y_test, color='blue', label='Actual')
plt.scatter(X_test['RM'], Y_pred, color='red', label='Predicted')
plt.xlabel('Average number of rooms per dwelling (RM)')
plt.ylabel('House Price')
plt.title('Linear Regression')
plt.legend()
plt.show()
```

**Explanation**:

* `plt.scatter(X_test['RM'], Y_test, color='blue', label='Actual')` plots the actual house prices against the average number of rooms per dwelling.
* `plt.scatter(X_test['RM'], Y_pred, color='red', label='Predicted')` plots the predicted house prices against the average number of rooms per dwelling.
* `plt.xlabel('Average number of rooms per dwelling (RM)')` and `plt.ylabel('House Price')` label the axes.
* `plt.title('Linear Regression')` adds a title to the plot.
* `plt.legend()` adds a legend to differentiate between actual and predicted values.
* `plt.show()` displays the plot.

#### Complete Code

Here is the complete code for the linear regression example using the Boston Housing Dataset:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the Boston Housing Dataset
boston = load_boston()
data = pd.DataFrame(boston.data, columns=boston.feature_names)
data['PRICE'] = boston.target

# Display the first few rows of the dataset
print(data.head())

# Define features and target
X = data[['RM', 'LSTAT', 'PTRATIO']]  # You can choose other features as well
Y = data['PRICE']

# Split the dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, Y_train)

# Predictions
Y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print(f'MSE: {mse}')
print(f'R2 Score: {r2}')

# Plot the results
plt.scatter(X_test['RM'], Y_test, color='blue', label='Actual')
plt.scatter(X_test['RM'], Y_pred, color='red', label='Predicted')
plt.xlabel('Average number of rooms per dwelling (RM)')
plt.ylabel('House Price')
plt.title('Linear Regression')
plt.legend()
plt.show()
```

#### Key Concepts

* **Linear Regression**: A method to model the relationship between dependent and independent variables using a linear equation.
* **Features**: Independent variables used to predict the target variable.
* **Target**: Dependent variable we are trying to predict.
* **Training Set**: Subset of the data used to train the model.
* **Testing Set**: Subset of the data used to evaluate the model.
* **Mean Squared Error (MSE)**: Measures the average squared difference between actual and predicted values.
* **R-squared (( R^2 ))**: Indicates the proportion of the variance in the dependent variable predictable from the independent variables.

This step-by-step guide helps you understand how to perform linear regression using a real-world dataset. For more detailed explanations and examples, visit [codeswithpankaj.com](http://codeswithpankaj.com).
