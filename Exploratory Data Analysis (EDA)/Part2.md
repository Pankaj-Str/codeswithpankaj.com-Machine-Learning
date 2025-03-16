#  Exploratory Data Analysis (EDA) 

By **Pankaj Chouhan** 

Hello, data lovers! I’m Pankaj Chouhan, and welcome to an in-depth tutorial on [www.codeswithpankaj.com](https://www.codeswithpankaj.com). Today, we’re diving into **Exploratory Data Analysis (EDA)**—the essential first step in any data science journey. Whether you’re a beginner or a pro, this guide will equip you with the skills to explore, clean, and prepare data using Python.

We’ll use the `students.csv` dataset (student performance data) as our example, and I’ll provide complete, runnable code. By the end, you’ll master EDA techniques like handling skewness, encoding data, and scaling features. Let’s get started!

---

## What is Exploratory Data Analysis (EDA)?

EDA is your data’s first impression. It’s the process of investigating a dataset to understand its structure, spot patterns, and fix issues before diving into modeling. Think of it as a treasure hunt—unearthing insights with stats and visuals.

In Python, EDA helps you:
- Summarize key characteristics.
- Identify anomalies or missing values.
- Prepare data for advanced analysis.

The payoff? A clean, well-understood dataset ready for action.

---

## Why Perform EDA?

EDA is the backbone of data science. Skipping it is like driving blind—you might crash! Here’s why it’s vital:

1. **Data Quality**: Catches errors like missing values or outliers.
2. **Insights**: Reveals trends and relationships.
3. **Model Prep**: Sets the stage for machine learning.

Let’s break it down into steps and apply it to real data.

---

## The EDA Process: Step-by-Step

EDA is a multi-faceted process. We’ll cover:
1. **EDA-Info and Shape**: Understand the dataset’s structure.
2. **Handling Missing Values**: Deal with gaps in the data.
3. **Handling Outliers**: Manage extreme values.
4. **Handling Skewness**: Address distribution imbalances.
5. **Data Encoding**: Convert categorical data for modeling.
6. **Feature Scaling**: Normalize or standardize numerical data.
7. **Feature Engineering**: Create new variables for deeper insights.
8. **Analyze Relationships**: Explore variable interactions.

We’ll use `students.csv`—a dataset of 1,000 students with exam scores and demographics—to demonstrate each step.

---

## Example: EDA on Student Performance Data

Our dataset tracks student performance with variables like gender, lunch type, test preparation, and scores. If you don’t have `students.csv`, download a similar dataset from Kaggle (e.g., "Students Performance in Exams").

### Step 1: Setting Up the Environment

Let’s import our Python toolkit. I recommend a Jupyter Notebook for interactivity.

```python
# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
plt.style.use('seaborn')
```

- **Pandas/NumPy**: Data manipulation and math.
- **Seaborn/Matplotlib**: Visualizations.
- **SciPy**: For skewness.
- **Scikit-learn**: For encoding and scaling.

---

### Step 2: EDA-Info and Shape

#### 2.1 Load the Dataset
```python
# Load the CSV file
data = pd.read_csv('students.csv')

# First look
print("First 5 Rows:")
print(data.head())
```

**Output**:
```
   gender race/ethnicity parental level of education         lunch test preparation course  math score  reading score  writing score
0  female        group B           bachelor's degree      standard                    none          72             72             74
1  female        group C                some college      standard               completed          69             90             88
2  female        group B             master's degree      standard                    none          90             95             93
3    male        group A          associate's degree  free/reduced                    none          47             57             44
4    male        group C                some college      standard                    none          76             78             75
```

#### 2.2 Shape
```python
# Rows and columns
print("Shape (Rows, Columns):", data.shape)
```

**Output**: `(1000, 8)`

We’ve got 1,000 students and 8 features.

#### 2.3 Info
```python
# Dataset info
print("Dataset Info:")
print(data.info())
```

**Output**:
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1000 entries, 0 to 999
Data columns (total 8 columns):
 #   Column                       Non-Null Count  Dtype 
---  ------                       --------------  ----- 
 0   gender                       1000 non-null   object
 1   race/ethnicity              1000 non-null   object
 2   parental level of education  1000 non-null   object
 3   lunch                        1000 non-null   object
 4   test preparation course      1000 non-null   object
 5   math score                   1000 non-null   int64 
 6   reading score                1000 non-null   int64 
 7   writing score                1000 non-null   int64 
dtypes: int64(3), object(5)
memory usage: 62.6+ KB
```

**Insight**: No nulls; 5 categorical and 3 numerical columns.

---

### Step 3: Handling Missing Values

```python
# Check for missing values
print("Missing Values:")
print(data.isnull().sum())
```

**Output**: All zeros—no missing data!

**If Missing**: 
- Drop: `data.dropna()`.
- Fill: `data['math score'].fillna(data['math score'].mean(), inplace=True)`.

---

### Step 4: Handling Outliers

Outliers can skew analysis. Let’s check `math score`:

```python
# Box plot
plt.figure(figsize=(8, 5))
sns.boxplot(x=data['math score'])
plt.title('Box Plot of Math Scores')
plt.show()

# IQR method
Q1 = data['math score'].quantile(0.25)
Q3 = data['math score'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
data_no_outliers = data[(data['math score'] >= lower_bound) & (data['math score'] <= upper_bound)]
print("Rows after outlier removal:", data_no_outliers.shape[0])
```

**Insight**: Low scores (e.g., 0) are outliers but valid. We’ll keep them.

---

### Step 5: Handling Skewness

Skewed data can affect models. Check `math score`:

```python
# Calculate skewness
print("Skewness of Math Score:", skew(data['math score']))

# Visualize
plt.figure(figsize=(8, 5))
sns.histplot(data['math score'], bins=10, kde=True)
plt.title('Math Score Distribution')
plt.show()
```

**Output**: Skewness ~ -0.3 (slightly left-skewed).

**Fixing Skewness** (if needed):
```python
# Log transform (if positive skew)
data['math_score_log'] = np.log1p(data['math score'])
print("Skewness after log transform:", skew(data['math_score_log']))
```

Here, skewness is mild, so no transform is necessary.

---

### Step 6: Data Encoding

Categorical variables need numbers for modeling. Let’s encode `gender`:

```python
# Label encoding
le = LabelEncoder()
data['gender_encoded'] = le.fit_transform(data['gender'])
print("Gender Encoded (First 5):")
print(data[['gender', 'gender_encoded']].head())
```

**Output**:
```
   gender  gender_encoded
0  female               0
1  female               0
2  female               0
3    male               1
4    male               1
```

**Alternative**: One-hot encoding for non-binary categories (e.g., `parental level of education`):
```python
data_encoded = pd.get_dummies(data, columns=['lunch'], prefix='lunch')
print(data_encoded.head())
```

---

### Step 7: Feature Scaling - Normalization and Standardization

Scaling ensures numerical features are comparable.

#### Normalization (0 to 1)
```python
# Min-Max Scaling
scaler = MinMaxScaler()
data['math_score_normalized'] = scaler.fit_transform(data[['math score']])
print("Normalized Math Score (First 5):")
print(data['math_score_normalized'].head())
```

**Output**: Values between 0 and 1.

#### Standardization (mean=0, std=1)
```python
# Standard Scaling
scaler = StandardScaler()
data['math_score_standardized'] = scaler.fit_transform(data[['math score']])
print("Standardized Math Score (First 5):")
print(data['math_score_standardized'].head())
```

**Output**: Centered around 0 with unit variance.

---

### Step 8: Feature Engineering

Create new features for richer insights:

```python
# Average score
data['average_score'] = data[['math score', 'reading score', 'writing score']].mean(axis=1)
print("Average Score (First 5):")
print(data['average_score'].head())
```

**Output**:
```
0    72.666667
1    82.333333
2    92.666667
3    49.333333
4    76.333333
```

This could predict overall performance!

---

### Step 9: Analyze Relationships

#### 9.1 Correlation Matrix
```python
# Correlation
correlation = data[['math score', 'reading score', 'writing score']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
```

**Insight**: Scores correlate strongly (0.8–0.95).

#### 9.2 Scatter Plot
```python
# Math vs. Reading by Lunch
plt.figure(figsize=(8, 6))
sns.scatterplot(x='math score', y='reading score', hue='lunch', data=data)
plt.title('Math vs. Reading Score by Lunch')
plt.show()
```

**Insight**: “Standard” lunch students score higher.

#### 9.3 Box Plot
```python
# Math scores by Test Prep
plt.figure(figsize=(8, 6))
sns.boxplot(x='test preparation course', y='math score', data=data)
plt.title('Math Scores by Test Preparation')
plt.show()
```

**Insight**: Test prep boosts scores.

---

### Key Insights

1. **Structure**: 1,000 rows, 8 columns; no missing values.
2. **Outliers**: Low scores exist but are valid.
3. **Skewness**: Mild, no major fixes needed.
4. **Relationships**: Scores correlate; lunch and prep impact performance.

---

### What’s Next?

EDA opens doors to:
1. **Predict Scores**: Model scores using `lunch` or `test preparation`.
2. **Categorical Conversion**: Encode `gender` (0/1) for ML.
3. **Feature Engineering**: Use `average_score` for new insights.
4. **Advanced Modeling**: Cluster students or predict outcomes.
5. **Model Tuning**: Apply scaling to optimize ML performance.

Stay tuned for more at [www.codeswithpankaj.com](https://www.codeswithpankaj.com)!

---

### Full Code

```python
# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
plt.style.use('seaborn')

# Load data
data = pd.read_csv('students.csv')

# EDA-Info and Shape
print("First 5 Rows:\n", data.head())
print("Shape:", data.shape)
print("Info:\n", data.info())

# Handling Missing Values
print("Missing Values:\n", data.isnull().sum())

# Handling Outliers
plt.figure(figsize=(8, 5))
sns.boxplot(x=data['math score'])
plt.title('Box Plot of Math Scores')
plt.show()

# Handling Skewness
print("Skewness of Math Score:", skew(data['math score']))
plt.figure(figsize=(8, 5))
sns.histplot(data['math score'], bins=10, kde=True)
plt.title('Math Score Distribution')
plt.show()

# Data Encoding
le = LabelEncoder()
data['gender_encoded'] = le.fit_transform(data['gender'])
print("Gender Encoded:\n", data[['gender', 'gender_encoded']].head())

# Feature Scaling
scaler_minmax = MinMaxScaler()
data['math_score_normalized'] = scaler_minmax.fit_transform(data[['math score']])
scaler_std = StandardScaler()
data['math_score_standardized'] = scaler_std.fit_transform(data[['math score']])
print("Scaled Math Scores:\n", data[['math score', 'math_score_normalized', 'math_score_standardized']].head())

# Feature Engineering
data['average_score'] = data[['math score', 'reading score', 'writing score']].mean(axis=1)
print("Average Score:\n", data['average_score'].head())

# Analyze Relationships
correlation = data[['math score', 'reading score', 'writing score']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(x='math score', y='reading score', hue='lunch', data=data)
plt.title('Math vs. Reading Score by Lunch')
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(x='test preparation course', y='math score', data=data)
plt.title('Math Scores by Test Preparation')
plt.show()
```

---

## Conclusion

EDA is your data science superpower. With Python tools like Pandas, Seaborn, and Scikit-learn, you can transform raw data into a launchpad for insights. I’m Pankaj Chouhan, and I hope this guide has empowered you to tackle any dataset.

Visit [www.codeswithpankaj.com](https://www.codeswithpankaj.com) for more tutorials, and subscribe to my YouTube channel. Questions? Drop them below. Happy coding!
