# **Income Classification Using Machine Learning**

This project focuses on predicting whether an individual's income is **≤50K or >50K** based on demographic and work-related factors. Through **data preprocessing, feature engineering, model selection, and deployment**, we develop a robust **machine learning pipeline**. The final model is **deployed using FastAPI** and **containerized with Docker** for real-world predictions.

---

## **Table of Contents**
1. [Introduction](#introduction)  
2. [Dataset Description](#dataset-description)  
3. [Project Workflow](#project-workflow)  
4. [Results](#results)  
5. [Installation & Running the Project](#installation--running-the-project)  
6. [Future Enhancements](#future-enhancements)  
7. [Acknowledgments](#acknowledgments)  
8. [Contact](#contact)  

---

## **Introduction**
Income levels are influenced by **demographic and occupational factors** such as education, age, and hours worked per week.  
This project applies **machine learning techniques** to classify individuals into two income categories (`<=50K` or `>50K`) based on these attributes.

### **Objectives**
- Perform **data preprocessing and feature engineering** to optimize model performance.
- Evaluate different **machine learning algorithms** and select the best-performing model.
- Deploy the final model as a **FastAPI web service**, containerized with **Docker**, for real-world usage.

---

## **Dataset Description**
The dataset is sourced from the **UCI Machine Learning Repository** and consists of **48,842 instances** with **14 attributes**.  
The **target variable** is income, labeled as **≤50K or >50K**.

### **Key Features**
- **Demographics**: Age, Sex, Race, Marital Status, Education Level  
- **Work & Financial Attributes**: Workclass, Occupation, Capital Gain/Loss, Hours Per Week  
- **Categorical variables were one-hot encoded** to ensure compatibility with machine learning models.

---

## **Project Workflow**

### **1. Exploratory Data Analysis (EDA)**
- Analyzed **feature distributions** and relationships to identify key patterns.
- Discovered **strong correlations between education, occupation, and income level**.

### **2. Data Preprocessing**
- **One-hot encoded** categorical features.
- **Handled missing values** and verified data consistency.
- Ensured feature alignment between **training and inference** data.

### **3. Model Selection & Hyperparameter Tuning**
- Evaluated multiple models:
  - **Gradient Boosting**
  - **XGBoost**
  - **LightGBM**
  - **CatBoost** (Final Model)
- Used **GridSearchCV & RandomizedSearchCV** for hyperparameter tuning.
- **CatBoost achieved the highest overall performance** and was selected as the final model.

### **4. Model Calibration & Decision Threshold Optimization**
- **Plotted calibration curves** to assess probability estimation accuracy.
- Adjusted the **decision threshold from 0.50 to 0.52** to optimize the balance between **precision and recall**.

### **5. Model Deployment**
- Developed a **FastAPI application** for serving predictions.
- **Containerized the API using Docker** to ensure portability and scalability.

---

## **Results**

### **Best Model: CatBoost**
| Metric     | Default Threshold (0.50) | Optimized Threshold (0.52) |
|------------|----------------|----------------|
| **Accuracy** | 85.72% | 86.03% |
| **F1-Score** | 0.7157 | 0.7170 |
| **ROC-AUC** | 0.9245 | 0.9245 |

### **Key Findings**
- **Education, Occupation, and Capital Gain** were the strongest predictors of income level.
- **Feature interactions revealed nuanced effects**, particularly among different occupation groups.
- **Adjusting the decision threshold improved precision**, reducing false positives while maintaining recall.

---

## **Installation & Running the Project**

### **1. Clone the Repository**
``` bash
git close https://github.com/Devin-Shrode/Census-Income-Analysis-and-Modeling
cd income-prediction
```

### **2. Create a Virtual Environment**
``` bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### **3. Install Dependencies**
``` bash
pip install -r requirements.txt
```

### **4. Run the API Locally**
Start the FastAPI server:
``` bash
uvicorn census_income_api:app --host 0.0.0.0 --port 8000 --reload
```

### **5. Sending a Prediction Request**
Use **Postman** or **cURL** to send a `POST` request:
``` bash
curl -X 'POST' 'http://127.0.0.1:8000/predict' -H 'Content-Type: application/json' -d '{"features": [39, 13, 0, 0, 40, 1, 0, 1, 1, 0, 1, 1, ... ]}'
```

The API returns:
``` json
{
    "predicted_income": ">50K"
}
```

### **6. Running the API in Docker**
Build and run the container:
``` bash
docker build -t census_income_api .
docker run -p 8000:8000 census_income_api
```

---

## **Future Enhancements**
- **Deploy to Cloud Services** (AWS/GCP/Azure) for broader accessibility.
- **Develop a Frontend UI** for users to interact with predictions.
- **Implement Model Monitoring** to track real-world performance and detect drift.

---

## **Acknowledgments**
This project was built using publicly available data from the **UCI Machine Learning Repository** (https://archive.ics.uci.edu/dataset/20/census+income).  
Libraries used include **pandas, NumPy, scikit-learn, CatBoost, FastAPI, and Docker**.

---

## **Contact**
For any questions or collaboration opportunities, reach out at:
- **Email**: devin.shrode@proton.me  
- **LinkedIn**: [linkedin.com/in/DevinShrode](https://www.linkedin.com/in/DevinShrode)  
- **GitHub**: [github.com/Devin-Shrode/Wine-Quality](https://github.com/Devin-Shrode/Wine-Quality)  

