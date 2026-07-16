# Predictive Maintenance Dashboard

A machine learning-powered web application that predicts industrial equipment failure risk in real time based on sensor readings, enabling proactive maintenance instead of reactive repairs.

Live App: https://predictive-maintenance-ml-shalini.streamlit.app/
Dataset: AI4I 2020 Predictive Maintenance Dataset (UCI) — https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

- Problem Statement
Unplanned equipment failure in manufacturing leads to costly downtime and safety risks. This project uses historical sensor data (temperature, torque, rotational speed, tool wear) to predict failure risk before it happens, supporting predictive maintenance strategies used in Industry 4.0.

- Approach
- Performed exploratory data analysis and identified severe class imbalance — only 3.39% of 10,000 records represent actual failures.
- Preprocessed data by encoding categorical machine type, removing data-leakage columns (failure-mode indicators), scaling features, and applying a stratified train/test split.
- Trained and compared two models — Logistic Regression and Random Forest — both with class-balanced weighting to address the imbalance.
- Selected Logistic Regression over Random Forest despite its lower overall accuracy, because it achieves significantly higher recall (84% vs 51%) on the failure class — critical in a maintenance context where missing a real failure is far costlier than a false alarm.
- Built an interactive dashboard using Streamlit and deployed it on Streamlit Community Cloud.

- Tech Stack
- Language: Python
- ML/Data: scikit-learn, pandas, NumPy
- Visualization: Plotly
- Web App: Streamlit
- Deployment: Streamlit Community Cloud

- Model Performance
- Logistic Regression: 0.14 precision / 0.84 recall / 0.24 F1-score (failure class)
- Random Forest: 0.90 precision / 0.51 recall / 0.65 F1-score (failure class)

- Key Insight
Feature importance analysis revealed Torque and Rotational Speed as the strongest predictors of failure, followed by Tool Wear — aligning with known mechanical failure patterns.

- Future Improvements
- Threshold tuning on Random Forest to improve recall while retaining higher precision
- Incorporating time-series/sequential sensor data for early trend detection
- Adding SHAP-based explainability for individual predictions

Author: Shalini C P — B.E. Electronics and Communication Engineering, KPR Institute of Engineering and Technology
