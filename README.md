# 📊 Churn Customer Analysis Project

This project focuses on building an end-to-end **customer churn analytics pipeline**, starting from raw survey data ingestion into SQL Server (via Docker), followed by data cleaning and validation in SQL, exploratory data analysis (EDA) in Python, and machine learning-based churn prediction for identifying high-risk customer segments and key churn drivers.

---

## 🚀 Project Workflow

### 1. Data Infrastructure (Docker + SQL Server)

* SQL Server is containerized using **Docker** and accessed via **VS Code**
* Data is ingested from multiple CSV files into SQL Server tables using Python
* Multiple datasets are merged into a unified analytical dataset for churn analysis

### 2. Data Engineering (SQL)

* Data cleaning (handling missing values, standardization)
* Feature encoding and transformation
* Data validation and consistency checks
* Creation of aggregated tables (e.g., visit behavior, need states)
* Construction of customer-level analytical datasets

### 3. Exploratory Data Analysis (Python)

EDA is performed using:

* **Pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**

Focus areas:

* Customer behavior patterns (weekday vs weekend visits)
* Brand perception and satisfaction (NPS, Likability)
* Usage segmentation
* Missing value analysis
* Behavioral feature engineering
* Spending and visit frequency analysis
* Demographic segmentation analysis

### 4. Visualization (Tableau)

* Interactive dashboards
* Market Overview
* Brand Positioning (Brand Equity Positioning, Usage Positioning, Brand Funnel Analysis, Brand Health Matrix)
* Customer segmentation analysis (Age, Gender, Occupation)
* Executive summary dashboards for decision-making


### 5. Machine Learning & Predictive Analytics

A classification pipeline is developed to:

* Predict **customer churn probability**
* Identify high-risk churn segments
* Compare multiple machine learning algorithms
* Evaluate model performance using ROC-AUC analysis

#### Churn Definition

A customer is considered churned when:

* **P3M = 1** (visited in the past 3 months)
* **P1M = 0** (did not visit in the past month)

#### Models Implemented

* Logistic Regression
  * Standard Logistic Regression
  * L1-Regularized Logistic Regression
  * L2-Regularized Logistic Regression
* Decision Trees
  * Gini Index
  * Entropy
* Neural Network (MLPClassifier)
* Random Forest Classifier

#### Model Evaluation

Models are evaluated using:

* ROC-AUC Score
* ROC Curve Comparison
* Classification Reports
* Confusion Matrix
* Cross-validation

The goal is to:

* Identify key drivers of churn
* Translate model outputs into actionable business insights
* Compare model interpretability vs predictive performance
* Support future integration into Power BI dashboards for strategic decision-making

---

## 🧰 Tech Stack

### Data Engineering

* SQL Server
* Docker

### Data Processing & EDA

* Python 3.11
* Pandas
* NumPy
* Matplotlib
* Seaborn

### Machine Learning & Predictive Analytics

* Scikit-learn

  * Logistic Regression
  * Decision Trees
  * Neural Networks (MLPClassifier)
  * Random Forest
  * ROC-AUC Evaluation
  * Cross Validation
  * Feature Scaling & Encoding

### Database Connection

* SQLAlchemy
* pyodbc

### Development Environment

* VS Code
* Jupyter Notebook

### Visualization

* Tableau

* Power BI (Upcoming)
