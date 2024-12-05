# Credit Card Fraud Detection Project
By Shubhi Mohta

**Overview**
This project aims to detect fraudulent credit card transactions using machine learning techniques. Both supervised and unsupervised approaches are implemented to handle fraud detection effectively. The focus is on preprocessing, class imbalance handling, model training, evaluation, and explainability.

**Dataset**
- Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download
- Description: This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.
- Features: Time, Amount, and PCA-transformed features (V1 to V28).
- Target: Class (0 = Non-Fraudulent, 1 = Fraudulent).
- Total Records: 284,807 transactions, with only ~0.17% being fraudulent.

**Setup Instructions**
- Install Required Libraries: Ensure the following Python libraries are installed:
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost shap
- Dataset Placement: Download the dataset from Kaggle and save it as creditcard.csv in the project directory.
- Run the Project:
1. Open the Jupyter Notebook file "AIML Task (Nirmitee.io).ipynb".
2. Execute the cells sequentially to preprocess the data, train models, and visualize results.

**Key Steps and Choices**
- Preprocessing:
1. Scaled the Amount feature using StandardScaler.
2. Addressed class imbalance using SMOTE to oversample the minority class.

- Supervised Models:
1. Baseline Model: Logistic Regression for simple fraud classification.
2. Advanced Model: XGBoost for improved accuracy and recall.

- Unsupervised Model:
1. Isolation Forest: Anomaly detection algorithm to identify potential fraudulent transactions.

- Evaluation Metrics:
1. Supervised Models: Accuracy, Precision, Recall, F1-Score, Confusion Matrix, ROC-AUC Curve.
2. Unsupervised Model: Number of anomalies detected and overlap with actual fraud cases.

- Explainability:
Used SHAP (SHapley Additive ExPlanations) to interpret model decisions and highlight feature importance.

**Results**
- Supervised Models:
1. Logistic Regression and XGBoost achieved high precision and recall, with XGBoost outperforming in overall metrics.
2. Confusion Matrices and ROC-AUC curves were generated for both models.

- Unsupervised Model:
1. The Isolation Forest successfully detected anomalies, including several actual fraud cases.

**Visualizations:**

- Confusion Matrices for both supervised models.
- ROC-AUC Curve for XGBoost.
- SHAP summary plot showing the most important features influencing predictions.

**Project Deliverables**
- Notebook: Includes the entire workflow, from preprocessing to evaluation.
- Metrics and Visualizations: Confusion matrices, ROC-AUC curves, SHAP plots, and anomaly detection results.
- README.md: Brief overview, setup instructions, and key findings.