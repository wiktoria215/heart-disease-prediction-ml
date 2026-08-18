# Heart Disease Prediction Model 🫀

## Overview
This repository contains a collaborative machine learning project focused on predicting cardiovascular disease risk (1 - risk, 0 - no risk) based on clinical data from the [Heart Disease UCI dataset](https://www.kaggle.com/datasets/ronitf/heart-disease-uci). The project demonstrates a structured, engineering-focused approach to data processing, feature scaling, and predictive modeling.

## Tech Stack
* **Language:** Python
* **Libraries:** `pandas`, `scikit-learn`

## Project Pipeline
1. **Data Wrangling:** Loading and cleaning the dataset, which involves dropping duplicates, identifying missing values, and formatting target labels.
2. **Data Preprocessing:** 
   * Segregating features (`X`) and target variables (`y`).
   * Applying One-Hot Encoding (`pd.get_dummies`) for categorical variables.
   * Standardizing numerical features utilizing `StandardScaler` to ensure optimal algorithm performance and avoid data leakage during the `train_test_split`.
3. **Machine Learning Models:** The project evaluates multiple algorithms including Random Forest, KNN, and Logistic Regression. My primary contribution was initializing and training the `LogisticRegression` classifier.
4. **Evaluation:** Rigorous model validation utilizing industry-standard classification metrics:
   * Accuracy Score
   * ROC AUC Score
   * Confusion Matrix

## Setup Instructions
To run this project locally:

### 1. Clone the repository
` ` `bash
git clone <repository-url>
cd <repository-folder>
` ` `

### 2. Create and activate a virtual environment
**Linux / macOS:**
` ` `bash
python3 -m venv venv
source venv/bin/activate
` ` `
**Windows:**
` ` `cmd
python -m venv venv
venv\Scripts\activate
` ` `

### 3. Install dependencies
` ` `bash
pip install -r requirements.txt
` ` `

### 4. Run the model pipeline
` ` `bash
python main.py
` ` `
