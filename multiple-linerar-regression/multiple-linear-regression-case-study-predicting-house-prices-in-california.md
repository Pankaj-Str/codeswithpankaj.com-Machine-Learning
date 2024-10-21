# Multiple Linear Regression Case Study: Predicting House Prices in California

#### Multiple Linear Regression Case Study: Predicting House Prices in California

**Case Study Question**

**Objective**: Build a multiple linear regression model to predict the median house value in California based on various factors such as median income, average number of rooms, average number of occupants, latitude, and longitude.

**Dataset**: We will use the California Housing Dataset, which includes the following features:

* **MedInc**: Median income in block group
* **AveRooms**: Average number of rooms per household
* **AveOccup**: Average number of household members
* **Latitude**: Latitude of the block
* **Longitude**: Longitude of the block

**Steps**:

1. Load and explore the dataset.
2. Define the features and the target variable.
3. Split the dataset into training and testing sets.
4. Create and train the multiple linear regression model.
5. Make predictions using the test set.
6. Evaluate the model's performance.
7. Visualize the results.

***

#### Step-by-Step Solution

**Step 1: Import Libraries**

First, we need to import the necessary libraries for data manipulation, modeling, and visualization.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
```

**Step 2: Load and Explore the Dataset**

Load the California Housing Dataset and take a quick look at the first few rows.

```python
# Load the California Housing Dataset
california = fetch_california_housing()
data = pd.DataFrame(california.data, columns=california.feature_names)
data['MedHouseVal'] = california.target

# Display the first few rows of the dataset
print(data.head())
```

**Explanation**:

* We use `fetch_california_housing()` to load the dataset.
* Convert the dataset to a Pandas DataFrame for easier manipulation.
* Add a new column `MedHouseVal` to represent the target variable (median house value).
* Display the first few rows of the dataset to get an overview of the data.

**Step 3: Define Features and Target**

Select the relevant features (independent variables) and define the target (dependent variable).

```python
# Define features and target
X = data[['MedInc', 'AveRooms', 'AveOccup', 'Latitude', 'Longitude']]
Y = data['MedHouseVal']
```

**Explanation**:

* `X` contains the features used to predict the house prices: `MedInc` (median income), `AveRooms` (average number of rooms), `AveOccup` (average number of occupants per household), `Latitude`, and `Longitude`.
* `Y` is the target variable, which is the median house value (`MedHouseVal`).

**Step 4: Split the Dataset**

Split the dataset into training and testing sets to evaluate the model's performance on unseen data.

```python
# Split the dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
```

**Explanation**:

* `train_test_split` splits the data into training and testing sets.
* `test_size=0.2` indicates that 20% of the data will be used for testing, and 80% for training.
* `random_state=42` ensures reproducibility by fixing the random seed.

**Step 5: Create and Train the Model**

Create a linear regression model and train it using the training set.

```python
# Create and train the model
model = LinearRegression()
model.fit(X_train, Y_train)
```

**Explanation**:

* `LinearRegression()` creates an instance of the linear regression model.
* `model.fit(X_train, Y_train)` trains the model using the training data.

**Step 6: Make Predictions**

Use the trained model to make predictions on the test set.

```python
# Predictions
Y_pred = model.predict(X_test)
```

**Explanation**:

* `model.predict(X_test)` uses the trained model to predict house prices for the test set.

**Step 7: Evaluate the Model**

Evaluate the model's performance using Mean Squared Error (MSE) and R-squared (( R^2 )).

```python
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

**Step 8: Plot the Results**

Visualize the actual vs. predicted values for one of the features to assess the model's performance.

```python
plt.scatter(X_test['MedInc'], Y_test, color='blue', label='Actual')
plt.scatter(X_test['MedInc'], Y_pred, color='red', label='Predicted')
plt.xlabel('Median Income')
plt.ylabel('Median House Value')
plt.title('Multiple Linear Regression')
plt.legend()
plt.show()
```

**Explanation**:

* `plt.scatter(X_test['MedInc'], Y_test, color='blue', label='Actual')` plots the actual house prices against the median income.
* `plt.scatter(X_test['MedInc'], Y_pred, color='red', label='Predicted')` plots the predicted house prices against the median income.
* `plt.xlabel('Median Income')` and `plt.ylabel('Median House Value')` label the axes.
* `plt.title('Multiple Linear Regression')` adds a title to the plot.
* `plt.legend()` adds a legend to differentiate between actual and predicted values.
* `plt.show()` displays the plot.

#### Complete Case Study Code

Here is the complete code for the multiple linear regression case study using the California Housing Dataset:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the California Housing Dataset
california = fetch_california_housing()
data = pd.DataFrame(california.data, columns=california.feature_names)
data['MedHouseVal'] = california.target

# Display the first few rows of the dataset
print(data.head())

# Define features and target
X = data[['MedInc', 'AveRooms', 'AveOccup', 'Latitude', 'Longitude']]
Y = data['MedHouseVal']

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
plt.scatter(X_test['MedInc'], Y_test, color='blue', label='Actual')
plt.scatter(X_test['MedInc'], Y_pred, color='red', label='Predicted')
plt.xlabel('Median Income')
plt.ylabel('Median House Value')
plt.title('Multiple Linear Regression')
plt.legend()
plt.show()
```

#### Summary

In this detailed case study, we walked through the process of implementing multiple linear regression using the California Housing Dataset. Here are the key steps we covered:

1. **Importing Libraries**: We imported necessary libraries for data manipulation, modeling, and visualization.
2. **Loading and Exploring the Dataset**: We loaded the California Housing Dataset and took a quick look at the data.
3. **Defining Features and Target**: We selected relevant features and defined the target variable.
4. **Splitting the Dataset**: We split the data into training and testing sets.
5. **Creating and Training the Model**: We created a linear regression model and trained it using the training data.
6. **Making Predictions**: We used the trained model to make predictions on the test set.
7. **Evaluating the Model**: We evaluated the model's performance using MSE and ( R^2 ) score.
8. **Plotting the Results**: We visualized the actual vs. predicted values for one of the features.

By following these steps, we built a multiple linear regression model that can predict house prices in California based on various features. This comprehensive approach helps in understanding the relationship between different factors and house prices, providing valuable insights for various applications such as real estate and economics.

For more detailed explanations and examples, visit [codeswithpankaj.com](http://codeswithpankaj.com).
