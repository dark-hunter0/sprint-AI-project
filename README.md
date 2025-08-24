# Comprehensive Machine Learning Pipeline for Heart Disease Prediction

This project implements a full end-to-end machine learning pipeline to predict the presence of heart disease
based on the UCI Heart Disease Dataset. The workflow covers data preprocessing, feature selection
, model training, hyperparameter tuning, and deployment of an interactive web application using Streamlit
and Ngrok.

## Features
- **Data Preprocessing:** Handles missing values, encodes categorical features, and scales numerical data.
- **Dimensionality Reduction:** Applies Principal Component Analysis (PCA) to visualize data structure.
- **Feature Selection:** Utilizes Random Forest Importance, RFE, and Chi-Square tests to identify the most relevant predictors.
- **Supervised Learning:** Trains and evaluates multiple classification models, including Logistic Regression, Decision Trees, Random Forest, and SVM.
- **Unsupervised Learning:** Employs K-Means and Hierarchical Clustering to discover natural groupings in the data.
- **Model Optimization:** Fine-tunes the best-performing model using GridSearchCV for enhanced accuracy.
- **Interactive UI:** A user-friendly web interface built with Streamlit for real-time predictions.
- **Deployment:** Instructions for making the local web application publicly accessible via Ngrok.

## Dataset
This project uses the **Heart Disease UCI Dataset**. It is a widely-used dataset for classification tasks, containing 76 attributes, but only 14 are typically used for analysis.

- **Link to Dataset:** [UCI Machine Learning Repository: Heart Disease Data Set](https://archive.ics.uci.edu/ml/datasets/heart+disease)

## File Structure
Heart_Disease_Project/
├── data/
│ └── heart_disease.csv
├── deployment/
│ └── ngrok_setup.txt
├── models/
│ └── final_model.pkl
├── notebooks/
│ ├── 01_data_preprocessing.ipynb
│ ├── 02_pca_analysis.ipynb
│ ├── 03_feature_selection.ipynb
│ ├── 04_supervised_learning.ipynb
│ ├── 05_unsupervised_learning.ipynb
│ └── 06_hyperparameter_tuning.ipynb
├── results/
│ └── evaluation_metrics.txt
├── ui/
│ └── app.py
├── .gitignore
├── README.md
└── requirements.txt

## Github link
https://github.com/dark-hunter0/sprint-AI-project.git