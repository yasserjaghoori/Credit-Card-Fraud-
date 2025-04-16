# 💳 Credit Card Fraud Detection — AIT 736 Group Project
**Yasser Jaghoori**  

---

## 📌 Project Overview

In today’s digital economy, credit card fraud poses a significant financial and security risk. This project aims to develop a machine learning–based fraud detection system that accurately identifies fraudulent transactions. We explored multiple classification models and combined them into an ensemble to improve performance and robustness.

The dataset used contains anonymized transaction data from European cardholders across two days:  
- **Total Transactions**: 284,807  
- **Fraudulent Transactions**: 492 (≈ 0.17%)  

This made the task a classic **imbalanced classification** problem, further complicated by anonymized features and evolving fraud patterns.

---

## 🧠 Modeling Approach

### Baseline Models
We trained and evaluated three common classifiers:
- **Logistic Regression** — simple and interpretable
- **Decision Tree** — capable of capturing nonlinear relationships
- **Random Forest** — ensemble of decision trees for improved generalization

Each model was trained on a **downsampled** version of the dataset with a 5:1 ratio between non-fraud and fraud cases.

### Ensemble Model
We implemented a **Voting Classifier** with soft voting that averages probabilities across all three base models to improve predictive reliability.

To further improve detection:
- **SMOTE** (Synthetic Minority Over-sampling Technique) was applied inside the pipeline during each cross-validation fold.
- **GridSearchCV** with stratified 5-fold cross-validation was used for **hyperparameter tuning**, optimizing for **F1 score**.

---

## 🧼 Data Collection & Processing

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Features**:
  - `Time`: Seconds since the first transaction
  - `Amount`: Transaction amount in euros
  - `V1–V28`: PCA-transformed features (original features withheld)
  - `Class`: Target (1 = fraud, 0 = non-fraud)

### Preprocessing:
- **Standardized** `Time` and `Amount` using `StandardScaler`
- **Log-transformed** `Amount` to reduce skew
- **Outliers** in fraud cases were retained due to their informative nature
- **No missing values** in the dataset

### Sampling:
- **Downsampling** used to balance data for baseline training
- **SMOTE** applied dynamically during CV in the ensemble pipeline
- **Stratified train/test split** (80/20) to preserve class ratios

---

## 📊 Results Summary

| Model               | Precision | Recall | F1 Score | AUPRC | ROC AUC |
|---------------------|-----------|--------|----------|--------|---------|
| Logistic Regression | 0.928     | 0.918  | 0.923    | 0.951  | 0.979   |
| Decision Tree       | 0.827     | 0.878  | 0.851    | 0.862  | 0.921   |
| Random Forest       | 0.967     | 0.898  | 0.931    | 0.951  | 0.981   |
| **Voting Ensemble** | 0.891     | 0.918  | 0.905    | 0.949  | **0.983** |

### Key Insights:
- The **Random Forest** achieved the highest individual F1 score.
- The **Ensemble model** achieved the **best balance** overall — highest ROC AUC (0.983), strong recall (0.918), and excellent precision (0.891).
- The use of **SMOTE and soft voting** significantly contributed to performance gains.

---

## ✅ Conclusion

This project demonstrated that:
- Combining multiple classifiers with **ensemble learning** enhances fraud detection.
- Handling **class imbalance** with **SMOTE** and **careful sampling** strategies is critical.
- Hyperparameter tuning and cross-validation improve model fairness and robustness.

The final ensemble model effectively balances recall (detecting true fraud) and precision (avoiding false alarms), making it well-suited for real-world deployment.

---

## 📚 References

- Kaggle. (n.d.). [Credit card fraud detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR, 12, 2825–2830.
- Lemaître, G., et al. (2017). *Imbalanced-learn: A Python Toolbox for Imbalanced Datasets*. JMLR, 18(17), 1–5.
- [Scikit-learn documentation](https://scikit-learn.org/)
- [Jupyter Notebooks](https://jupyter.org/)
- Marasco, E. (2025). *AIT 736 course materials*. George Mason University.
