# 🏆 MUN Award Prediction using Logistic Regression

This project aims to predict whether a delegate will win an award in a Model United Nations (MUN) conference based on their previous experience, performance metrics, and committee/delegation information. The analysis includes data preprocessing, visualization, and applying a logistic regression model using both `scikit-learn` and `statsmodels`.

---

## 📂 Dataset

The dataset `mun_participants_dataset.csv` contains 100 records with the following features:

| Column                          | Description                                         |
|---------------------------------|-----------------------------------------------------|
| `Delegation`                   | Country represented by the delegate                |
| `Committee`                    | Committee in which the delegate participated       |
| `Previous_MUNs`                | Number of MUNs attended prior                      |
| `Previous_Resolutions_Sponsored` | Number of resolutions sponsored before             |
| `Previous_Awards`              | Number of awards won previously                    |
| `Experience_Level`             | Rated experience level (1–5)                       |
| `Speeches_Given`              | Number of speeches delivered                       |
| `Points_of_Information`       | Number of points of information made              |
| `Presence`                    | Attendance percentage                              |
| `Chairs_Penalties`            | Number of penalties assigned by the chairs        |
| `Award_Won`                   | Target variable (1 if award won, 0 otherwise)      |

---

## 📊 Exploratory Data Analysis (EDA)

- Count plot and pie chart of `Award_Won` to understand class balance.
- Correlation heatmap for all numeric features.
- Histogram of `Speeches_Given`.
- Committee distribution pie chart.

---

## 🧠 Modeling: Logistic Regression

### 🔄 Preprocessing

- One-hot encoding for categorical features (`Committee`, `Delegation`).
- Boolean columns converted to binary (0/1).
- Train-test split: 80% training, 20% testing.

### 🔍 Model Training & Evaluation (Scikit-learn)
```python
LogisticRegression(max_iter=1000)
```
### 📈 Model Summary (Statsmodels)

- Used `sm.Logit()` for statistical interpretation.
- Outputs coefficients, p-values, confidence intervals.
- Odds ratios calculated and visualized.
- Notable predictors: `Speeches_Given`, `Delegation_France`, `Delegation_Germany`, `Committee_UNSC`.

---
## Testing Random Forest Modeling
-The accuracy was higher than the one resulted from the regression Model.
-The features importance plot had  more logical insights.

## 📉 Key Findings

- Feature `Speeches_Given` was statistically significant (p < 0.05).
- Some country delegations showed higher odds of winning awards.
- Logistic regression performance was limited due to potential class imbalance or non-linearity.
