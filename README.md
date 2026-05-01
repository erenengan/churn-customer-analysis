# 📊 Churn Customer Analysis Project

This project focuses on building an end-to-end **customer churn analytics pipeline**, starting from raw survey data ingestion into SQL Server (via Docker), followed by data cleaning and validation in SQL, exploratory data analysis (EDA) in Python, and future extensions into machine learning-based churn prediction and Power BI dashboards.

---

## 🚀 Project Workflow

### 1. Data Infrastructure (Docker + SQL Server)
- SQL Server is containerized using **Docker** and accessed via **VS Code**
- Data is ingested from multiple CSV files into SQL Server tables using Python

### 2. Data Engineering (SQL)
- Data cleaning (handling missing values, standardization)
- Feature encoding and transformation
- Data validation and consistency checks
- Creation of aggregated tables (e.g., visit behavior, need states)

### 3. Exploratory Data Analysis (Python)
EDA is performed using:
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Seaborn**

Focus areas:
- Customer behavior patterns (weekday vs weekend visits)
- Brand perception and satisfaction (NPS, Likability)
- Usage segmentation
- Missing value analysis
- Behavioral feature engineering

### 4. Visualization (Power BI - Upcoming)
- Interactive dashboards
- Customer segmentation analysis
- Brand performance comparison
- Behavioral insights (usage, visit patterns)
- Executive summary dashboards for decision-making

### 5. Machine Learning (Future Work)
A classification model will be developed to:
- Predict **customer churn probability**
- Identify high-risk churn segments
- Evaluate model performance (Precision, Recall, F1-score, ROC-AUC)

The goal is to:
- Identify key drivers of churn
- Translate model outputs into actionable business insights
- Integrate results into Power BI for strategic decision-making

---

## 🧰 Tech Stack

### Data Engineering
- SQL Server (Dockerized)
- T-SQL

### Data Processing & EDA
- Python 3.11
- Pandas
- NumPy
- Matplotlib
- Seaborn

### Database Connection
- SQLAlchemy
- pyodbc
