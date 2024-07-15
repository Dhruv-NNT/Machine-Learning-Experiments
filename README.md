# 📊 Machine Learning Experiments

Welcome to the **Machine Learning Experiments** repository. This repository contains various machine learning experiments exploring different techniques and algorithms across multiple datasets. Each notebook demonstrates practical applications of machine learning concepts and provides insights into different methodologies.

## 📂 Repository Structure

```
📦Machine Learning Experiments
 ┣ 📜categorical-feature-transformations.ipynb
 ┣ 📜credit-card-fraud-detection.ipynb
 ┣ 📜Stacking_Ensemble.ipynb
 ┗ 📜titanic-competition.ipynb
```

## 📝 Notebooks Overview

### 1. Categorical Feature Transformations

This notebook focuses on different techniques for transforming categorical features for use in machine learning models. It covers methods such as:

- **Label Encoding**: Converting categorical values to numerical labels.
- **One-Hot Encoding**: Creating binary columns for each category.
- **Target Encoding**: Encoding categories based on the mean target value for each category.
- **Frequency Encoding**: Encoding categories based on their frequency in the dataset.

**Key Insights:**

- Understanding various encoding techniques and their impact on model performance.
- Comparing the effectiveness of each encoding method using different datasets.

### 2. Credit Card Fraud Detection

This notebook addresses the problem of credit card fraud detection using various machine learning algorithms. The goal is to identify fraudulent transactions from a dataset of credit card transactions.

**Techniques Utilized:**

- **Data Preprocessing**: Handling imbalanced data using techniques like oversampling and undersampling.
- **Feature Engineering**: Creating new features and selecting important ones.
- **Model Training and Evaluation**: Implementing models such as Logistic Regression, Decision Trees, Random Forest, and XGBoost.
- **Model Evaluation**: Using metrics such as Accuracy, Precision, Recall, F1-Score, and ROC-AUC to evaluate model performance.

**Key Insights:**

- Handling imbalanced datasets effectively.
- Comparing the performance of different machine learning models for fraud detection.
- Importance of evaluation metrics in assessing model performance for imbalanced datasets.

### 3. Titanic Survivor Prediction (Kaggle Competition)

This notebook explores the classic Titanic survivor prediction problem. The objective is to predict whether a passenger survived based on features such as age, gender, and class.

**Techniques Utilized:**

- **Data Preprocessing**: Handling missing values and feature scaling.
- **Feature Engineering**: Creating new features from existing ones.
- **Model Training and Evaluation**: Implementing models such as Logistic Regression, Decision Trees, Random Forest, and Support Vector Machines (SVM).
- **Model Evaluation**: Using metrics such as Accuracy, Precision, and Recall to evaluate model performance.

**Key Insights:**

- Effective data preprocessing and feature engineering techniques.
- Comparing different machine learning models for binary classification.
- Insights into feature importance and model interpretability.

### 4. Stacking Ensemble

This notebook demonstrates the use of stacking ensemble methods to improve model performance. Stacking involves training multiple base models and combining their predictions using a meta-model.

**Techniques Utilized:**

- **Base Models**: Training various base models like Logistic Regression, Decision Trees, and Random Forest.
- **Stacking**: Combining the predictions of base models using a meta-model, typically a simple model like Logistic Regression.
- **Model Evaluation**: Assessing the performance of the stacking ensemble compared to individual base models.

**Key Insights:**

- Understanding the concept of stacking ensembles.
- Implementing stacking ensembles to enhance model performance.
- Comparing the effectiveness of stacking ensembles with individual models.

## 🛠️ Installation

1. **Clone the repository:**

```bash
git clone https://github.com/your-username/Machine-Learning-Experiments.git
cd Machine-Learning-Experiments
```

2. **Create a virtual environment and activate it:**

```bash
python -m venv env
source env/bin/activate  # On Windows, use `env\Scripts\activate`
```

3. **Install the required dependencies:**

```bash
pip install -r requirements.txt
```

## 🚀 Running the Notebooks

1. **Open the Jupyter Notebooks:**

```bash
jupyter notebook
```

2. **Select and run the desired notebook to explore the experiments and results.**

## 📧 Contact

For any questions or feedback, feel free to reach out to:

- **Name**: Aradhya Dhruv
- **Email**: aradhya.dhruv@example.com

## 📝 Acknowledgements

This repository contains experiments conducted as part of my learning and exploration in the field of machine learning. Special thanks to the creators of the datasets and the open-source community for their invaluable resources and support.

---
