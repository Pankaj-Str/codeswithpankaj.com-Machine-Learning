# **Predicting with Data Using Random vs. Stratified Sampling**  
*(Perfect for Students!)*  

---

## **1. What is Sampling?**  
**Sampling** means picking a small group (a "sample") from a larger group (a "population") to study or make predictions.  
- **Example**: Imagine you have a jar of 1000 candies (population). Instead of tasting all 1000, you pick 100 (sample) to guess the flavors.  

---

## **2. Two Easy Sampling Techniques**  
### **A. Random Sampling**  
- **What?** Everyone in the population has an equal chance of being picked.  
- **Example**: Closing your eyes and grabbing 100 candies from the jar.  

### **B. Stratified Sampling**  
- **What?** Split the population into groups (*strata*), then pick samples from each group.  
- **Example**: If the jar has 70% red candies and 30% green candies, pick 70 red and 30 green candies for your sample.  

---

## **3. Why Does It Matter?**  
- **Imbalanced Data Problem**: If 95% of your data is "cats" and 5% is "dogs," random sampling might pick almost all cats. Your model won’t learn about dogs!  
- **Stratified Sampling Fixes This**: Makes sure the sample has the same mix as the population (e.g., 95% cats, 5% dogs).  

---

## **4. Simple Example: Predicting Student Grades**  
Let’s predict if students pass (1) or fail (0) an exam.  

### **Step 1: Create Fake Data**  
```python
import pandas as pd

# Create a dataset of 1000 students (900 pass, 100 fail)
data = {
    'hours_studied': [4, 3, 5, 2, 6] * 200,  # Fake study hours
    'passed': [1, 1, 1, 0, 1] * 200          # 1=Pass, 0=Fail
}
df = pd.DataFrame(data)
print("Total Pass/Fail:", df['passed'].value_counts())
# Output: 1 (800 students), 0 (200 students)
```

### **Step 2: Split Data with Both Methods**  
```python
from sklearn.model_selection import train_test_split

# Random Sampling (may underrepresent "fail" students)
X_rand_train, X_rand_test, y_rand_train, y_rand_test = train_test_split(
    df[['hours_studied']], df['passed'], test_size=0.2, random_state=42
)

# Stratified Sampling (keeps pass/fail ratio)
X_strat_train, X_strat_test, y_strat_train, y_strat_test = train_test_split(
    df[['hours_studied']], df['passed'], test_size=0.2, stratify=df['passed'], random_state=42
)
```

### **Step 3: Check the Samples**  
```python
print("Random Test Samples:", y_rand_test.value_counts())
# Output: Pass (160), Fail (40) → Good balance by luck? Not guaranteed!

print("Stratified Test Samples:", y_strat_test.value_counts())
# Output: Pass (160), Fail (40) → Perfect balance (matches original 80-20 ratio)
```

---

## **5. Train a Model & See the Difference**  
```python
from sklearn.linear_model import LogisticRegression

# Train with random sampling
model_rand = LogisticRegression()
model_rand.fit(X_rand_train, y_rand_train)

# Train with stratified sampling
model_strat = LogisticRegression()
model_strat.fit(X_strat_train, y_strat_train)

# Predict on the test data
print("Random Sampling Accuracy:", model_rand.score(X_rand_test, y_rand_test))
print("Stratified Sampling Accuracy:", model_strat.score(X_strat_test, y_strat_test))
```

### **What Happened?**  
- **Random Sampling**: Might predict "pass" all the time (since most students pass). Accuracy looks good, but it’s bad at spotting failures.  
- **Stratified Sampling**: Better at predicting both passes and failures because it saw enough examples of both.  

---

## **6. When to Use Which?**  
| **Scenario**               | **Random Sampling** | **Stratified Sampling** |  
|-----------------------------|---------------------|--------------------------|  
| Balanced Data (50-50 split) | ✅                  | ✅                       |  
| Imbalanced Data (95-5 split)| ❌                  | ✅                       |  
| Quick Experiments           | ✅                  | ❌                       |  

---

## **7. Key Takeaways**  
1. **Random Sampling** = Easy, but risky for imbalanced data.  
2. **Stratified Sampling** = Smarter for imbalanced data (keeps the same ratio).  
3. Always check if your training data has enough examples of all categories!  

💡 **Pro Tip**: In Python, use `stratify=y` in `train_test_split` to do stratified sampling.