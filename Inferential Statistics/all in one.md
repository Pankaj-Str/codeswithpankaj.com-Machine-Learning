# **Complete Tutorial on Inferential Statistics**  
**Hosted on [codeswithpankaj.com](https://codeswithpankaj.com)**  
*Master Sampling, Hypothesis Testing, CLT, and More with Practical Examples*  

---

## **1. Introduction to Inferential Statistics**  
**Objective**: Use sample data to draw conclusions about populations.  
**Why It Matters**: From predicting election results to A/B testing in tech, inferential stats power data-driven decisions.  

---

## **2. Sampling Techniques**  
### **a. Random Sampling**  
**Steps** (with Python Example):  
1. **Define Population**: A list of 10,000 employee IDs.  
2. **Randomly Select**: Use Python’s `random` library.  
```python  
import random  
population = range(1, 10001)  # Employee IDs from 1 to 10,000  
sample = random.sample(population, 500)  # Select 500 IDs  
print("Random Sample:", sample)  
```  

### **b. Stratified Sampling**  
**Example**: A university with 70% Engineering and 30% Arts students.  
```python  
import pandas as pd  
# Create a DataFrame of students  
data = {'Department': ['Engineering']*7000 + ['Arts']*3000}  
df = pd.DataFrame(data)  
# Stratified sampling  
engineering_sample = df[df['Department'] == 'Engineering'].sample(140)  
arts_sample = df[df['Department'] == 'Arts'].sample(60)  
stratified_sample = pd.concat([engineering_sample, arts_sample])  
```  

---

## **3. Central Limit Theorem (CLT)**  
### **Simulation with Python**  
**Task**: Roll a die 40 times, calculate the mean, repeat 50 times, and plot results.  
```python  
import numpy as np  
import matplotlib.pyplot as plt  

means = []  
for _ in range(50):  
    rolls = np.random.randint(1, 7, 40)  # Simulate 40 die rolls  
    means.append(np.mean(rolls))  

plt.hist(means, bins=10, edgecolor='black')  
plt.title("CLT Demonstration: Distribution of Sample Means")  
plt.xlabel("Mean of 40 Die Rolls")  
plt.ylabel("Frequency")  
plt.show()  
```  
**Result**: The histogram will approximate a normal distribution!  

---

## **4. Estimating Population Parameters**  
### **Confidence Interval (CI) Calculation**  
**Formula**:  
\[
\text{95% CI} = \bar{x} \pm 1.96 \times \frac{s}{\sqrt{n}}  
\]  
**Python Example**:  
```python  
import scipy.stats as stats  

sample_mean = 50  
sample_std = 5  
n = 100  
confidence_level = 0.95  

# Calculate CI  
ci = stats.norm.interval(confidence_level, loc=sample_mean, scale=sample_std/np.sqrt(n))  
print("95% CI:", ci)  # Output: (49.02, 50.98)  
```  

---

## **5. Hypothesis Testing**  
### **Step-by-Step Example**  
**Scenario**: Test if a new algorithm reduces website load time (α = 0.05).  
- **Data**: Sample of 40 users, mean load time = 3.2 sec, SD = 0.5 sec.  
- **Claim**: Previous average load time = 3.5 sec.  

**Python Code**:  
```python  
from scipy.stats import ttest_1samp  

sample_data = np.random.normal(3.2, 0.5, 40)  # Simulated data  
t_stat, p_value = ttest_1samp(sample_data, 3.5)  # Compare to population mean  

print(f"t-statistic: {t_stat:.2f}, p-value: {p_value:.4f}")  
if p_value < 0.05:  
    print("Reject H₀: The algorithm reduces load time.")  
else:  
    print("Fail to reject H₀: No significant improvement.")  
```  

---

## **6. p-values, Type 1, and Type 2 Errors**  
### **Real-World Analogy**  
- **Type 1 Error (False Positive)**:  
  - *Example*: A spam filter marking a valid email as spam.  
- **Type 2 Error (False Negative)**:  
  - *Example*: A COVID test failing to detect an infected patient.  

---

## **7. Practical Assignments**  
### **Problem 1: Stratified Sampling**  
**Task**: Use the [Titanic Dataset](https://www.kaggle.com/c/titanic) to create a stratified sample based on passenger class (1st, 2nd, 3rd).  

### **Problem 2: CLT with Python**  
**Task**: Simulate the average height of 50 people (assume heights are uniformly distributed between 150–200 cm). Repeat 1,000 times and plot the distribution.  

### **Problem 3: Hypothesis Testing**  
**Task**: Test if the average sepal length in the Iris dataset differs between species (use ANOVA).  

---

## **8. Tools & Resources**  
- **Python Libraries**:  
  - `Pandas` for data manipulation.  
  - `SciPy`/`statsmodels` for statistical tests.  
  - `Matplotlib`/`Seaborn` for visualization.  
- **Datasets**: Practice with [Kaggle](https://www.kaggle.com/) or [UCI Machine Learning Repository](https://archive.ics.uci.edu/).  

---

## **Key Takeaways**  
1. **Sampling**: Ensures your data represents the population.  
2. **CLT**: The backbone of confidence intervals and hypothesis tests.  
3. **Hypothesis Testing**: A structured way to validate claims.  
4. **Errors**: Balance risks of false positives (Type 1) and false negatives (Type 2).  

---

**Explore More Tutorials**:  
Visit **[codeswithpankaj.com](https://codeswithpankaj.com)** for coding guides, data science projects, and interactive quizzes!  

---  
**Happy Learning, Pankaj Chouhan!** 🚀  