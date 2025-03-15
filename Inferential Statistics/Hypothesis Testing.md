# **Hypothesis Testing**  

---

## **1. What is Hypothesis Testing?**  
Imagine you have a **guess** about something, and you want to prove if it’s true using data.  
- **Example**: You think a coin is **biased** (lands on heads more than tails). Hypothesis testing helps you check if your guess is likely correct.  

---

## **2. Two Guesses: Null vs. Alternative**  
Every test starts with two opposite guesses:  

### **Null Hypothesis (H₀)**  
- The default guess that says **"nothing is happening"** or **"no difference."**  
- *Example*: "The coin is fair (50% heads, 50% tails)."  

### **Alternative Hypothesis (H₁)**  
- Your **claim** that something is different.  
- *Example*: "The coin is biased (lands on heads more than tails)."  

---

## **3. How Does It Work?**  
1. **Collect Data**: Flip the coin 100 times.  
2. **Calculate Results**: Suppose you get 65 heads.  
3. **Ask**: *"Is 65 heads surprising if the coin is fair?"*  
   - If it’s **very unlikely**, reject the null hypothesis (H₀).  
   - If it’s **not surprising**, keep H₀.  

---

## **4. p-value and Significance Level (α)**  
- **p-value**: The chance of seeing your results *if H₀ is true*.  
  - *Example*: If the coin is fair, the chance of getting 65+ heads is p = 0.02 (2%).  
- **α (Significance Level)**: The threshold you set (usually 5% or 0.05).  

### **Decision Rule**  
- **If p < α**: Reject H₀. Your result is "statistically significant."  
- **If p ≥ α**: Don’t reject H₀.  

---

## **5. Real-Life Example**  
**Scenario**: A teacher claims students who study with flashcards score higher (average score = 75). You test 20 students using flashcards and find their average score = 80.  

### **Hypotheses**  
- H₀: Flashcards make **no difference** (true average = 75).  
- H₁: Flashcards **help** (true average > 75).  

### **Testing**  
- Calculate the p-value (probability of getting 80+ if H₀ is true).  
- Suppose p = 0.03. Since p < 0.05, **reject H₀** → Flashcards likely work!  

---

## **6. Types of Errors**  
| **Error**       | **What Happens**                     | **Example**                          |  
|------------------|--------------------------------------|--------------------------------------|  
| **Type I Error** | Reject H₀ when it’s *actually true*. | Saying the coin is biased when it’s fair. |  
| **Type II Error**| Keep H₀ when it’s *actually false*.  | Saying the coin is fair when it’s biased. |  

---

## **7. Common Tests (Simplified)**  
1. **t-test**: Compare averages.  
   - *Example*: Do men and women earn different salaries?  
2. **Chi-square test**: Check relationships between categories.  
   - *Example*: Is there a link between eye color and glasses?  

---

## **8. Key Takeaways**  
1. **H₀** = "Nothing’s happening."  
2. **H₁** = "Something’s happening!"  
3. **p-value** = Chance of your result if H₀ is true.  
4. **Reject H₀** if p < α (usually 0.05).  

---

## **9. Try It Yourself!**  
**Scenario**: A pizza place claims delivery takes 30 minutes on average. You time 10 deliveries:  
`[28, 32, 29, 31, 35, 34, 27, 33, 30, 36]`  

### **Steps**  
1. **H₀**: Average time = 30 minutes.  
2. **H₁**: Average time ≠ 30 minutes.  
3. **Calculate p-value** (use Python code below).  
4. **If p < 0.05**, the claim is likely wrong!  

```python
from scipy.stats import ttest_1samp

data = [28, 32, 29, 31, 35, 34, 27, 33, 30, 36]
t_stat, p_value = ttest_1samp(data, popmean=30)  # Compare to H₀: mean=30
print("p-value:", p_value)  # If p < 0.05, reject H₀!
```  

**Result**: If p-value = 0.07 → **Don’t reject H₀**. If p-value = 0.03 → **Reject H₀**.  

--- 

💡 **Remember**: Hypothesis testing is like a trial. H₀ is "innocent until proven guilty"!