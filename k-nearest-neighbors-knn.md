# K Nearest Neighbors (KNN)

#### K-Nearest Neighbors (KNN): A Step-by-Step Tutorial

K-Nearest Neighbors (KNN) is a simple and effective machine learning algorithm used for both classification and regression tasks. It is a non-parametric, instance-based learning method. In this tutorial, codeswithpankaj will guide you through the steps to perform KNN analysis using Python.

**Table of Contents**

1. Introduction to K-Nearest Neighbors
2. Setting Up the Environment
3. Loading the Dataset
4. Exploring the Data
5. Preparing the Data
6. Building the KNN Model
7. Evaluating the Model
8. Making Predictions
9. Tuning the Model
10. Conclusion

***

#### 1. Introduction to K-Nearest Neighbors

K-Nearest Neighbors (KNN) is an algorithm that classifies a data point based on how its neighbors are classified. It is based on the idea that similar things exist in close proximity.

**Key Features**:

* Simple to understand and implement.
* Can be used for both classification and regression tasks.
* Non-parametric and instance-based.

**Applications**:

* Recommender systems.
* Image recognition.
* Medical diagnosis.

#### 2. Setting Up the Environment

First, we need to install the necessary libraries. We'll use `numpy`, `pandas`, `matplotlib`, and `scikit-learn`.

```python
# Install the libraries (uncomment the lines below if you haven't installed them yet)
# !pip install numpy pandas matplotlib scikit-learn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
```

**Explanation of Libraries**:

* **Numpy**: Used for numerical operations.
* **Pandas**: Used for data manipulation and analysis.
* **Matplotlib**: Used for data visualization.
* **Scikit-learn**: Provides tools for machine learning, including KNN.

#### 3. Loading the Dataset

For this tutorial, we'll use a CSV dataset. You can download the dataset from [this link](https://raw.githubusercontent.com/justmarkham/scikit-learn-videos/master/data/fruit\_data\_with\_colors.txt).

```python
# Load the dataset from a CSV file
url = 'https://raw.githubusercontent.com/justmarkham/scikit-learn-videos/master/data/fruit_data_with_colors.txt'
data = pd.read_csv(url, delimiter='\t')
```

**Understanding the Data**:

* **fruit\_label**: Dependent variable (class labels for different fruits).
* **mass, width, height, color\_score**: Independent variables (features).

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
plt.scatter(data['width'], data['height'], c=data['fruit_label'], cmap='viridis')
plt.xlabel('Width')
plt.ylabel('Height')
plt.title('Scatter plot of Width vs Height')
plt.show()
```

#### 5. Preparing the Data

We'll split the data into training and testing sets to evaluate the model's performance.

```python
# Define the feature columns and target column
features = ['mass', 'width', 'height', 'color_score']
target = 'fruit_label'

# Split the data into training and testing sets
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

**Importance of Data Splitting**:

* **Training Set**: Used to train the model.
* **Testing Set**: Used to evaluate the model's performance.
* **Test Size**: Proportion of the dataset used for testing (e.g., 20%).

#### 6. Building the KNN Model

Now, let's build the KNN model using the training data.

```python
# Create the KNN model
knn_model = KNeighborsClassifier(n_neighbors=5)

# Train the model
knn_model.fit(X_train, y_train)
```

**Steps in Model Building**:

1. **Model Creation**: Instantiate the KNN model.
2. **Model Training**: Fit the model to the training data using the `fit` method.

#### 7. Evaluating the Model

We'll evaluate the model by calculating accuracy and generating a classification report.

```python
# Make predictions on the test data
y_pred = knn_model.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)

# Generate the classification report
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(class_report)
```

**Evaluation Metrics**:

* **Accuracy**: Proportion of correctly predicted instances.
* **Classification Report**: Provides precision, recall, F1-score, and support for each class.

#### 8. Making Predictions

Finally, let's use the model to make predictions.

```python
# Example: Predicting the outcome for new data points
new_data = pd.DataFrame({
    'mass': [150, 160],
    'width': [7.0, 6.5],
    'height': [8.5, 8.0],
    'color_score': [0.75, 0.80]
})
predictions = knn_model.predict(new_data)
print(f"Predictions for new data points: {predictions}")
```

**Prediction Process**:

* **New Data**: Input data for which we want to make predictions.
* **Model Prediction**: Use the `predict` method to get the predicted outcome.

#### 9. Tuning the Model

Tuning the hyperparameters of the KNN model can improve its performance. One of the main parameters to tune is the number of neighbors.

```python
from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {'n_neighbors': np.arange(1, 10)}

# Create the GridSearchCV object
grid_search = GridSearchCV(knn_model, param_grid, cv=5)

# Perform the grid search
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")

# Train the best model
best_knn_model = grid_search.best_estimator_

# Evaluate the best model
best_y_pred = best_knn_model.predict(X_test)
best_accuracy = accuracy_score(y_test, best_y_pred)
print(f"Best Model Accuracy: {best_accuracy}")
```

#### 10. Conclusion

In this tutorial by codeswithpankaj, we've covered the basics of K-Nearest Neighbors (KNN) and how to implement it using Python. We walked through setting up the environment, loading and exploring the data, preparing the data, building the model, evaluating the model, making predictions, and tuning the model. KNN is a simple yet effective tool in data science for both classification and regression tasks.

***

For more tutorials and resources, visit [codeswithpankaj.com](https://codeswithpankaj.com).
