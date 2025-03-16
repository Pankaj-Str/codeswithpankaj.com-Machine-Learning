# Exploratory Data Analysis (EDA)

Hello, data enthusiasts! I’m Pankaj Chouhan, and welcome to another tutorial on [www.codeswithpankaj.com](https://www.codeswithpankaj.com). Today, we’re taking an in-depth journey into **Exploratory Data Analysis (EDA)**—a cornerstone of data science that unlocks the secrets within your data. Whether you’re new to coding or a seasoned analyst, this guide will teach you what EDA is, why it’s essential, and how to perform it in Python with a hands-on example.

We’ll use the `students.csv` dataset, which tracks student performance, and I’ll provide complete code to follow along. By the end, you’ll be ready to load, clean, analyze, and visualize data like a pro. Let’s dive in!

---

## What is Exploratory Data Analysis (EDA)?

EDA is your first step in understanding a dataset. It’s like exploring a new city—mapping out the streets (variables), spotting landmarks (patterns), and avoiding potholes (errors). In Python, EDA uses statistical summaries and visualizations to reveal the data’s structure and quirks.

Simply put, EDA helps you:
- Summarize key features.
- Detect anomalies or missing values.
- Understand relationships between variables.

The result? A clean dataset and actionable insights for deeper analysis.

---

## Why Perform EDA?

Think of EDA as the foundation of a data science project. Skipping it is like cooking without tasting the ingredients—you might end up with a mess! Here’s why EDA matters:

1. **Data Quality**: Identifies issues like missing values or outliers that could skew results.
2. **Insights**: Uncovers trends and relationships to guide your analysis.
3. **Model Readiness**: Prepares data for machine learning by highlighting key features.

Let’s break this down into steps and apply it to real data.

---

## The EDA Process: Step-by-Step

EDA follows three core phases:
1. **Understand the Data**: Load and explore the dataset’s structure.
2. **Clean the Data**: Remove noise, handle missing values, and fix inconsistencies.
3. **Analyze Relationships**: Use stats and visuals to find patterns.

We’ll demonstrate this with the `students.csv` dataset—perfect for learning EDA hands-on!

---

## Example: EDA on Student Performance Data

We’ll analyze a dataset of 1,000 students, featuring variables like gender, lunch type, test preparation, and exam scores. If you don’t have `students.csv`, grab a similar dataset from Kaggle (e.g., "Students Performance in Exams") or use any CSV you’ve got.

### Step 1: Setting Up the Environment

Let’s start by importing our Python tools. I recommend using a Jupyter Notebook for this—it’s interactive and great for EDA.

```python
# Import essential libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Set a clean plotting style
plt.style.use('seaborn')
```

- **Pandas**: For data wrangling.
- **NumPy**: For numerical operations.
- **Seaborn**: For stunning visualizations.
- **Matplotlib**: For plotting control.

---

### Step 2: Loading and Understanding the Data

Time to load the dataset and get a feel for it.

#### 2.1 Load the Dataset
```python
# Load the CSV file
data = pd.read_csv('students.csv')

# Peek at the first 5 rows
print("First 5 Rows of the Dataset:")
print(data.head())
```

**Sample Output**:
```
   gender race/ethnicity parental level of education         lunch test preparation course  math score  reading score  writing score
0  female        group B           bachelor's degree      standard                    none          72             72             74
1  female        group C                some college      standard               completed          69             90             88
2  female        group B             master's degree      standard                    none          90             95             93
3    male        group A          associate's degree  free/reduced                    none          47             57             44
4    male        group C                some college      standard                    none          76             78             75
```

This dataset has 8 columns:
- **Categorical**: `gender`, `race/ethnicity`, `parental level of education`, `lunch`, `test preparation course`.
- **Numerical**: `math score`, `reading score`, `writing score`.

#### 2.2 Check the Shape
```python
# Number of rows and columns
print("Dataset Shape (Rows, Columns):", data.shape)
```

**Output**: `(1000, 8)`

We’re working with 1,000 students and 8 features—a nice size for practice.

#### 2.3 Inspect Data Types
```python
# Data types of each column
print("Data Types:")
print(data.dtypes)
```

**Output**:
```
gender                         object
race/ethnicity                object
parental level of education   object
lunch                         object
test preparation course       object
math score                     int64
reading score                  int64
writing score                  int64
dtype: object
```

- `object`: Categorical/string data.
- `int64`: Numerical integers.

#### 2.4 Summary Statistics
```python
# Stats for numerical columns
print("Summary Statistics:")
print(data.describe())
```

**Output**:
```
       math score  reading score  writing score
count  1000.00000    1000.000000    1000.000000
mean     66.08900      69.169000      68.054000
std      15.16308      14.600192      15.195657
min       0.00000      17.000000      10.000000
25%      57.00000      59.000000      57.750000
50%      66.00000      70.000000      69.000000
75%      77.00000      79.000000      79.000000
max     100.00000     100.000000     100.000000
```

Key observations:
- Average scores are around 66–69.
- Minimums (e.g., 0 in math) hint at possible outliers or valid lows.

#### 2.5 Explore Unique Values
```python
# Unique values in categorical columns
print("Unique Values:")
print("Gender:", data['gender'].unique())
print("Lunch:", data['lunch'].unique())
print("Test Prep:", data['test preparation course'].unique())
print("Total Unique Values per Column:")
print(data.nunique())
```

**Output**:
```
Gender: ['female' 'male']
Lunch: ['standard' 'free/reduced']
Test Prep: ['none' 'completed']
Total Unique Values per Column:
gender                          2
race/ethnicity                 5
parental level of education    6
lunch                          2
test preparation course        2
math score                    81
reading score                 72
writing score                 77
dtype: int64
```

This shows binary categories (e.g., `gender`: 2) and more varied ones (e.g., `parental level of education`: 6).

---

### Step 3: Cleaning the Data

A clean dataset ensures accurate insights. Let’s check for issues and tidy up.

#### 3.1 Check for Missing Values
```python
# Count null values
print("Missing Values:")
print(data.isnull().sum())
```

**Output**:
```
gender                         0
race/ethnicity                 0
parental level of education    0
lunch                          0
test preparation course        0
math score                     0
reading score                  0
writing score                  0
dtype: int64
```

No missing values—lucky us! If there were, options include:
- Drop rows: `data.dropna()`.
- Fill with mean: `data['math score'].fillna(data['math score'].mean(), inplace=True)`.

#### 3.2 Remove Redundant Columns
For this analysis, `race/ethnicity` and `parental level of education` may not be critical. Let’s drop them:

```python
# Drop columns
student = data.drop(['race/ethnicity', 'parental level of education'], axis=1)
print("Updated Dataset (First 5 Rows):")
print(student.head())
```

**Output**:
```
   gender         lunch test preparation course  math score  reading score  writing score
0  female      standard                    none          72             72             74
1  female      standard               completed          69             90             88
2  female      standard                    none          90             95             93
3    male  free/reduced                    none          47             57             44
4    male      standard                    none          76             78             75
```

Now we have 6 focused columns.

#### 3.3 Handle Outliers
Outliers can distort analysis. Let’s check `math score` with a box plot:

```python
# Box plot for math score
plt.figure(figsize=(8, 5))
sns.boxplot(x=student['math score'])
plt.title('Box Plot of Math Scores')
plt.show()
```

**Explanation**:
- The box is the IQR (25th–75th percentile).
- Whiskers extend to min/max, with dots as outliers.
- Scores like 0 stand out but make sense (e.g., failing a test).

To remove outliers (optional):
```python
# IQR method
Q1 = student['math score'].quantile(0.25)
Q3 = student['math score'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
student_clean = student[(student['math score'] >= lower_bound) & (student['math score'] <= upper_bound)]
print("Rows after outlier removal:", student_clean.shape[0])
```

We’ll keep all data for now, as low scores are plausible.

---

### Step 4: Analyzing Relationships

This is where EDA shines—revealing how variables interact.

#### 4.1 Correlation Matrix
For numerical variables:

```python
# Correlation between scores
correlation = student[['math score', 'reading score', 'writing score']].corr()

# Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', xticklabels=correlation.columns, yticklabels=correlation.columns)
plt.title('Correlation Matrix of Scores')
plt.show()
```

**Output**: A heatmap with:
- `math score` vs. `reading score`: ~0.8.
- `reading score` vs. `writing score`: ~0.95.

**Insight**: Strong positive correlations—good readers are often good writers.

#### 4.2 Scatter Plots
Explore scores with categorical variables:

```python
# Math vs. Reading Score by Gender
plt.figure(figsize=(8, 6))
sns.scatterplot(x='math score', y='reading score', hue='gender', data=student)
plt.title('Math vs. Reading Score by Gender')
plt.show()

# Math vs. Reading Score by Lunch Type
plt.figure(figsize=(8, 6))
sns.scatterplot(x='math score', y='reading score', hue='lunch', data=student)
plt.title('Math vs. Reading Score by Lunch Type')
plt.show()
```

**Insights**:
- Gender: Similar patterns for males and females.
- Lunch: “Standard” lunch students score higher—a socio-economic clue.

#### 4.3 Histogram
See the distribution of `math score`:

```python
# Histogram of math scores
plt.figure(figsize=(8, 6))
sns.histplot(data=student, x='math score', bins=10, kde=True, color='blue')
plt.title('Distribution of Math Scores')
plt.xlabel('Math Score')
plt.ylabel('Frequency')
plt.show()
```

**Insight**: Scores peak at 60–80, slightly right-skewed.

#### 4.4 Box Plot by Category
Compare across groups:

```python
# Math scores by test preparation
plt.figure(figsize=(8, 6))
sns.boxplot(x='test preparation course', y='math score', data=student)
plt.title('Math Scores by Test Preparation')
plt.show()
```

**Insight**: Test prep completers have higher median scores—prep works!

---

### Step 5: Key Insights

Here’s what we’ve uncovered:
1. **Data Quality**: No missing values; streamlined to 6 columns.
2. **Scores**: High correlations (0.8–0.95) between math, reading, and writing.
3. **Categorical Impact**: “Standard” lunch and test prep boost scores.
4. **Distribution**: Scores cluster at 60–80.

Scatter and box plots were stars here, blending numerical and categorical insights.

---

### What’s Next?

EDA sets the stage for exciting next steps:
1. **Build a Model to Predict Scores**: Use `lunch` or `test preparation` to predict scores with a regression model.
2. **Convert Categorical Variables**: Transform `gender` (e.g., 0/1) for machine learning.
3. **Feature Engineering**: Create new variables, like average score, to enhance analysis.
4. **Advanced Modeling**: Try clustering or classification to group students or predict outcomes.

Check out my upcoming tutorial on predictive modeling at [www.codeswithpankaj.com](https://www.codeswithpankaj.com) for more!

---

### Full Code

Run this complete script to replicate the EDA:

```python
# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('seaborn')

# Load data
data = pd.read_csv('students.csv')

# Step 1: Understand the data
print("First 5 Rows:\n", data.head())
print("Shape:", data.shape)
print("Data Types:\n", data.dtypes)
print("Summary Stats:\n", data.describe())
print("Unique Values:\n", data.nunique())

# Step 2: Clean the data
print("Missing Values:\n", data.isnull().sum())
student = data.drop(['race/ethnicity', 'parental level of education'], axis=1)
print("Updated Data:\n", student.head())

# Step 3: Analyze relationships
# Correlation matrix
correlation = student[['math score', 'reading score', 'writing score']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Scatter plots
plt.figure(figsize=(8, 6))
sns.scatterplot(x='math score', y='reading score', hue='gender', data=student)
plt.title('Math vs. Reading Score by Gender')
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(x='math score', y='reading score', hue='lunch', data=student)
plt.title('Math vs. Reading Score by Lunch')
plt.show()

# Histogram
plt.figure(figsize=(8, 6))
sns.histplot(data=student, x='math score', bins=10, kde=True, color='blue')
plt.title('Distribution of Math Scores')
plt.show()

# Box plot
plt.figure(figsize=(8, 6))
sns.boxplot(x='test preparation course', y='math score', data=student)
plt.title('Math Scores by Test Preparation')
plt.show()
```

---

## Conclusion

EDA is your key to mastering data. With Python tools like Pandas and Seaborn, you can turn raw data into a goldmine of insights. I’m Pankaj Chouhan, and I hope this guide has empowered you to explore datasets with confidence.

For more tutorials, visit [www.codeswithpankaj.com](https://www.codeswithpankaj.com) or subscribe to my YouTube channel. Have questions? Leave them below. Happy coding!

---

