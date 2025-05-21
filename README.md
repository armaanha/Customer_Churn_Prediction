# 🛒 Customer Churn Prediction in E-Commerce

Customer churn is a critical issue in the e-commerce industry, as retaining customers is far more cost-effective than acquiring new ones. This project leverages machine learning techniques to predict which customers are likely to churn, helping businesses implement proactive retention strategies.

## 📊 Dataset

The dataset was sourced from [Kaggle](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction) and contains 5,630 observations and 20 customer-related features such as:

- Tenure
- Payment Mode
- City Tier
- Device Usage
- Complaints
- Cashback & Coupons

Due to a significant class imbalance (83% non-churn, 17% churn), SMOTE was used to oversample the minority class for fair training.

## 🔍 Exploratory Data Analysis (EDA)

Key findings:
- **Tenure < 2 months** is the strongest churn predictor.
- Complaints and low satisfaction scores are strongly associated with churn.
- Feature correlations helped guide feature engineering and model choice.

## ⚙️ Models Used

Four supervised learning models were evaluated:
- **Logistic Regression** (with Elastic Net)
- **Decision Tree**
- **Random Forest** ✅ *Best Performer*
- **XGBoost**

### 🧪 Evaluation Metrics:
- Accuracy
- AUC-ROC
- Precision, Recall, F1-Score
- Confusion Matrix

| Model              | Accuracy | AUC  | Precision | Recall | F1 Score |
|-------------------|----------|------|-----------|--------|----------|
| Logistic Regression | 81.5%   | 0.885 | 78.8%    | 80.1%  | 79.5%    |
| Decision Tree      | 86.2%   | 0.908 | 90.0%    | 77.8%  | 83.5%    |
| **Random Forest**  | **98.0%**| **0.999**| **97.5%** | **98.1%** | **97.8%** |
| XGBoost            | 95.6%   | 0.993 | 95.3%    | 94.8%  | 95.1%    |

## 🌟 Key Insights

- **Random Forest** was the best performer, offering near-perfect precision and recall.
- **Tenure**, **Complaints**, and **Marital Status** were the top predictive features.
- Model findings can directly guide retention campaigns through CRM integration.



