# **Hypothesis Testing Examples**  

---

## **Example 1: Does a New Fertilizer Make Plants Grow Taller?**  
**Scenario**: You want to test if a new fertilizer increases plant height.  
- **H₀**: The fertilizer has **no effect** (average height = 20 cm).  
- **H₁**: The fertilizer **increases height** (average height > 20 cm).  

**Test**: One-sample t-test (compare sample mean to a known value).  
**Data**: After using the fertilizer, 15 plants have an average height of 23 cm (SD = 4 cm).  
**Result**: If p-value < 0.05, reject H₀ → Fertilizer likely works!  

```python  
from scipy.stats import ttest_1samp  

heights = [22, 24, 21, 25, 23, 20, 24, 22, 25, 23, 24, 22, 23, 24, 25]  
t_stat, p_value = ttest_1samp(heights, popmean=20)  # Compare to H₀: mean=20  
print("p-value:", p_value)  # If p < 0.05, reject H₀!  
```  

---

## **Example 2: Are Men and Women Equally Likely to Own Cats?**  
**Scenario**: Test if gender (Male/Female) is related to pet preference (Cat/Dog).  

|            | **Cat** | **Dog** |  
|------------|---------|---------|  
| **Male**   | 30      | 50      |  
| **Female** | 45      | 25      |  

**H₀**: Gender and pet preference are **independent** (no relationship).  
**H₁**: Gender and pet preference are **related**.  

**Test**: Chi-square test (for categorical data).  
**Result**: If p-value < 0.05, reject H₀ → Gender and pet preference are linked.  

```python  
from scipy.stats import chi2_contingency  

data = [[30, 50], [45, 25]]  
chi_stat, p_value, _, _ = chi2_contingency(data)  
print("p-value:", p_value)  # If p < 0.05, reject H₀!  
```  

---

## **Example 3: Does a New Math Program Improve Pass Rates?**  
**Scenario**: A school claims 80% of students pass math. After a new teaching program, 85 out of 100 students pass.  

**H₀**: Pass rate = 80% (p = 0.8).  
**H₁**: Pass rate > 80% (p > 0.8).  

**Test**: Z-test for proportions.  
**Result**: If p-value < 0.05, the program likely improved pass rates.  

```python  
from statsmodels.stats.proportion import proportions_ztest  

pass_count = 85  
total_students = 100  
z_stat, p_value = proportions_ztest(pass_count, total_students, value=0.8, alternative='larger')  
print("p-value:", p_value)  # If p < 0.05, reject H₀!  
```  

---

## **Example 4: Is a New Painkiller Faster Than the Old One?**  
**Scenario**: Compare two painkillers.  
- **Old Drug**: 10 patients, average relief time = 30 mins (SD = 5).  
- **New Drug**: 10 patients, average relief time = 25 mins (SD = 4).  

**H₀**: Both drugs work equally fast (μ₁ = μ₂).  
**H₁**: New drug is faster (μ₁ > μ₂).  

**Test**: Two-sample t-test (compare two groups).  
**Result**: If p-value < 0.05, the new drug is faster!  

```python  
from scipy.stats import ttest_ind  

old_drug = [28, 32, 30, 29, 31, 33, 27, 30, 29, 31]  
new_drug = [24, 26, 25, 23, 27, 25, 24, 26, 25, 24]  
t_stat, p_value = ttest_ind(old_drug, new_drug)  
print("p-value:", p_value)  # If p < 0.05, reject H₀!  
```  

---

## **Example 5: Is a Die Fair?**  
**Scenario**: Roll a die 60 times. Do all numbers (1-6) have equal probability?  

| **Number** | 1 | 2 | 3 | 4 | 5 | 6 |  
|------------|---|---|---|---|---|---|  
| **Count**  |8 |12 |9 |11 |10 |10 |  

**H₀**: The die is fair (all numbers have equal probability).  
**H₁**: The die is unfair.  

**Test**: Chi-square goodness-of-fit test.  
**Result**: If p-value < 0.05, the die is likely unfair.  

```python  
from scipy.stats import chisquare  

observed = [8, 12, 9, 11, 10, 10]  
expected = [10, 10, 10, 10, 10, 10]  # Fair die expects 10 per number  
chi_stat, p_value = chisquare(observed, expected)  
print("p-value:", p_value)  # If p < 0.05, reject H₀!  
```  

---

## **Key Takeaways**  
1. **Null Hypothesis (H₀)**: Always assume "no effect" first.  
2. **Alternative Hypothesis (H₁)**: What you’re trying to prove.  
3. **p-value < α (e.g., 0.05)**: Reject H₀ → Your finding is significant!  

---

## **Summary of Tests**  
| **Scenario**                     | **Test**                | **Purpose**                              |  
|-----------------------------------|-------------------------|------------------------------------------|  
| Compare means to a known value    | One-sample t-test       | Does fertilizer increase plant height?   |  
| Compare two groups                | Two-sample t-test       | Is the new painkiller faster?            |  
| Test relationships in categories  | Chi-square test         | Are gender and pet preference linked?    |  
| Test proportions                  | Z-test for proportions  | Did the math program improve pass rates? |  
| Check fairness of a die           | Chi-square goodness-of-fit | Is the die fair?                      |  

🔍 **Pro Tip**: Use tools like Python or Excel to calculate p-values – no manual math needed!