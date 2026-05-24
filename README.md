# Telco Customer Churn Analysis

This project is an end-to-end customer churn analysis using the Telco Customer Churn dataset.  
The goal is to understand which customer characteristics are associated with churn and to build a machine learning model that can estimate the probability of customer churn.

The project was designed as a clear and explainable data science workflow suitable for a final-year statistics student who wants to demonstrate practical skills in exploratory data analysis, feature engineering, model building, and model evaluation.

---

## Project Objective

Customer churn is an important problem for subscription-based businesses.  
In this project, the target variable is `Churn`, which indicates whether a customer left the company.

The main questions are:

- What type of customers are more likely to churn?
- Which variables are most related to churn behavior?
- Can we build a model that identifies customers with high churn risk?
- Which model performs best under common classification metrics?

---

## Dataset Summary

The raw dataset contains:

| Item | Value |
|---|---:|
| Number of observations | 7,043 |
| Raw number of columns | 21 |
| Processed number of columns | 26 |
| Target variable | `Churn` |
| Problem type | Binary classification |

Target distribution:

| Churn | Count | Percentage |
|---|---:|---:|
| No | 5,174 | 73.46% |
| Yes | 1,869 | 26.54% |

The dataset is not extremely imbalanced, but the churn class is still the minority class.  
For this reason, model evaluation should not rely only on accuracy. Recall, F1-score and ROC-AUC are also important.

---

## Project Structure

```text
telco-churn-analysis/
│
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   └── telco_student_processed.csv
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modelling.ipynb
│
├── outputs/
│   ├── figures/
│   │   ├── eda/
│   │   └── modeling/
│   ├── tables/
│   │   ├── eda/
│   │   ├── feature_engineering/
│   │   └── modeling/
│   └── models/
│       └── best_telco_churn_model.joblib
│
├── README.md
└── requirements.txt
```

---

## Methodology

The project is divided into three main notebooks.

### 1. Exploratory Data Analysis

Notebook:

```text
notebooks/01_eda_student.ipynb
```

In this step, the dataset is examined before modeling.  
The analysis includes:

- General data structure
- Missing value checks
- Data type inspection
- Target variable distribution
- Numerical variable analysis
- Categorical variable churn rate analysis

Important EDA outputs:

```text
outputs/figures/eda/01_churn_distribution.png
outputs/figures/eda/02_hist_tenure.png
outputs/figures/eda/04_churn_rate_Contract.png
outputs/figures/eda/04_churn_rate_PaymentMethod.png
outputs/figures/eda/04_churn_rate_TechSupport.png
```

### 2. Feature Engineering

Notebook:

```text
notebooks/02_feature_engineering_student.ipynb
```

In this step, the raw dataset is prepared for modeling.

Main operations:

- Removed `customerID`
- Converted `TotalCharges` from text to numeric
- Filled missing `TotalCharges` values with 0
- Converted `Churn` into binary format
- Created new explanatory features

Created features:

| Feature | Description |
|---|---|
| `AvgMonthlyCharge` | Average monthly charge calculated from total charges and tenure |
| `HasInternetService` | Indicates whether the customer has internet service |
| `AutomaticPayment` | Indicates whether the customer uses automatic payment |
| `IsMonthToMonth` | Indicates whether the contract type is month-to-month |
| `NumInternetServices` | Number of additional internet-related services |
| `TenureGroup` | Grouped version of tenure |

The processed dataset is saved as:

```text
data/telco_student_processed.csv
```

### 3. Modeling

Notebook:

```text
notebooks/03_modeling_student.ipynb
```

In this step, three classification models are compared:

- Logistic Regression
- Random Forest
- Gradient Boosting

The modeling workflow uses a pipeline structure.  
Numerical variables are scaled with `StandardScaler`, and categorical variables are encoded with `OneHotEncoder`.

This approach helps prevent data leakage because preprocessing is fitted only on the training data inside the pipeline.

---

## Exploratory Data Analysis Findings

Some important churn patterns were observed during EDA.

### Contract Type

Customers with month-to-month contracts have a much higher churn rate.

| Contract Type | Churn Rate |
|---|---:|
| Month-to-month | 42.71% |
| One year | 11.27% |
| Two year | 2.83% |

This suggests that longer contract duration is strongly associated with lower churn.

### Internet Service

| Internet Service | Churn Rate |
|---|---:|
| Fiber optic | 41.89% |
| DSL | 18.96% |
| No internet service | 7.40% |

Fiber optic customers show a higher churn rate. This does not necessarily mean fiber optic causes churn; it may be related to pricing, customer expectations, service issues or other customer characteristics.

### Online Security and Tech Support

Customers without online security or technical support have higher churn rates.

| Variable | Category | Churn Rate |
|---|---|---:|
| OnlineSecurity | No | 41.77% |
| OnlineSecurity | Yes | 14.61% |
| TechSupport | No | 41.64% |
| TechSupport | Yes | 15.17% |

This indicates that support and security-related services may be associated with customer retention.

### Payment Method

| Payment Method | Churn Rate |
|---|---:|
| Electronic check | 45.29% |
| Mailed check | 19.11% |
| Bank transfer automatic | 16.71% |
| Credit card automatic | 15.24% |

Customers using electronic checks have the highest churn rate among payment groups.

---

## Model Evaluation

The data was split into training and test sets using stratified sampling.

- Test size: 20%
- Random state: 42
- Stratification: based on `Churn`

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

### Cross-Validation Results

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.7604 | 0.5333 | 0.7819 | 0.6340 | 0.8479 |
| Gradient Boosting | 0.7980 | 0.6552 | 0.5104 | 0.5730 | 0.8461 |
| Logistic Regression | 0.7490 | 0.5177 | 0.7967 | 0.6274 | 0.8458 |

### Test Set Results

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.7644 | 0.5380 | 0.7941 | 0.6415 | 0.8451 |
| Gradient Boosting | 0.8041 | 0.6701 | 0.5160 | 0.5831 | 0.8433 |
| Logistic Regression | 0.7331 | 0.4983 | 0.7914 | 0.6116 | 0.8420 |

---

## Best Model

The best model was selected as **Random Forest** based on ROC-AUC and recall performance.

Random Forest achieved:

| Metric | Value |
|---|---:|
| Accuracy | 0.7644 |
| Precision | 0.5380 |
| Recall | 0.7941 |
| F1-score | 0.6415 |
| ROC-AUC | 0.8451 |

The model captures a high proportion of actual churn customers.  
This is important because in churn prediction, missing a customer who is likely to leave can be costly for the business.

### Confusion Matrix for Random Forest

|  | Predicted No Churn | Predicted Churn |
|---|---:|---:|
| Actual No Churn | 780 | 255 |
| Actual Churn | 77 | 297 |

The model correctly identified 297 churn customers and missed 77 churn customers in the test set.

---

## Threshold Analysis

The default classification threshold is 0.50.  
However, in churn prediction, the threshold can be changed depending on the business goal.

For the Random Forest model:

| Threshold | Precision | Recall | F1-score | Accuracy |
|---:|---:|---:|---:|---:|
| 0.40 | 0.4812 | 0.8556 | 0.6160 | 0.7168 |
| 0.50 | 0.5380 | 0.7941 | 0.6415 | 0.7644 |
| 0.60 | 0.5807 | 0.6925 | 0.6317 | 0.7857 |
| 0.70 | 0.6580 | 0.5401 | 0.5932 | 0.8034 |

A lower threshold increases recall but reduces precision.  
A higher threshold increases precision but misses more churn customers.

---

## Feature Importance

Permutation importance was used to interpret which variables affected the best model most.

Top features:

| Feature | Importance |
|---|---:|
| `IsMonthToMonth` | 0.0309 |
| `tenure` | 0.0236 |
| `InternetService` | 0.0170 |
| `TotalCharges` | 0.0103 |
| `Contract` | 0.0100 |
| `PaymentMethod` | 0.0028 |
| `TenureGroup` | 0.0026 |
| `AvgMonthlyCharge` | 0.0015 |
| `OnlineSecurity` | 0.0011 |
| `MonthlyCharges` | 0.0010 |

The most important features are related to contract type, customer tenure and internet service.  
This is consistent with the EDA findings.

---

## Main Insights

The project produced several useful insights:

1. Customers with month-to-month contracts are much more likely to churn.
2. Longer tenure is associated with lower churn risk.
3. Customers without online security or technical support show higher churn rates.
4. Electronic check users have the highest churn rate among payment methods.
5. Random Forest achieved the best overall balance between ROC-AUC and recall.
6. Threshold selection is important because churn prediction often prioritizes catching high-risk customers.

---

## How to Run the Project

Clone the repository:

```bash
git clone https://github.com/ferhataydemir/telco-churn-analysis.git
cd telco-churn-analysis
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

For Windows:

```bash
.venv\Scripts\activate
```

Install requirements:

```bash
pip install -r requirements.txt
```

Open Jupyter Notebook:

```bash
jupyter notebook
```

Run the notebooks in this order:

```text
1. notebooks/01_eda_student.ipynb
2. notebooks/02_feature_engineering_student.ipynb
3. notebooks/03_modeling_student.ipynb
```

---

## Requirements

Main libraries used in this project:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
jupyter
```

If you see warnings about `numexpr` or `bottleneck`, update them with:

```bash
python -m pip install --upgrade numexpr bottleneck
```

---

## Outputs

The project saves outputs automatically.

EDA figures:

```text
outputs/figures/eda/
```

Modeling figures:

```text
outputs/figures/modeling/
```

Result tables:

```text
outputs/tables/
```

Saved model:

```text
outputs/models/best_telco_churn_model.joblib
```

---

## Future Improvements

Possible improvements for this project:

- Add hyperparameter tuning with GridSearchCV or RandomizedSearchCV
- Compare additional models such as XGBoost or LightGBM
- Add SHAP analysis for more detailed model interpretability
- Use probability calibration
- Add cost-sensitive evaluation for false positives and false negatives
- Build a simple Streamlit dashboard for churn risk prediction

---

## Author

**Ferhat Aydemir**  

