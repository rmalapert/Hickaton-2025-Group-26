# Hi!ckathon #6 - AI & Education Challenge (Group 26)

**Date:** November 28 – December 1, 2025 (Final Sprint)  
**Location:** ENSTA, Palaiseau  
**Role:** Data Scientist & Product Designer  
**Associated with:** HEC Paris, Institut Polytechnique de Paris  
**Organizers:** Hi! PARIS  
**Corporate Partners:** L'Oréal, Capgemini, TotalEnergies, VINCI, Schneider Electric

## About this Repository

This repository contains the source code, analysis notebooks, and submission files developed by **Group 26** during the final sprint of the **Hi!ckathon #6**.

The Hi!ckathon is a flagship AI and Data Science challenge organized by Hi! PARIS, bringing together students to tackle real-world AI problems with significant business and societal impact. This 6th edition focused on the **AI & Education Challenge**, leveraging the massive PISA dataset (1.7 million student records, 300+ variables) to transform education through Artificial Intelligence.

## The Challenge

The competition involved a 3-day intensive sprint to address two main tracks:

### 1. Technical Task: Predictive Modeling
**Objective:** Predict student math scores using the PISA dataset.  
**Achievement:** Achieved an RMSE of **82.17**.

*   **Pipeline:** Developed a high-performance regression pipeline using CatBoost.
*   **Data Engineering:** Implemented complex missing data handling (imputation, pruning), feature selection via correlation analysis, and One-Hot Encoding.
*   **Optimization:** Fine-tuned hyperparameters (learning rate, max depth, estimators) using GridSearchCV to outperform baseline models like Linear Regression and Random Forest.

### 2. Business Task: Innovation Track - Student Mental Health
**Focus:** Addressing Issue #4 - Student Mental Health.

*   **Solution:** Designed a data-driven solution to detect early signs of distress.
*   **Value Proposition:** An AI-powered tool connecting students, teachers, and counselors to optimize resource allocation in schools.
*   **Ethical AI:** Prioritized a Frugal AI approach (low-compute CatBoost) and strict data privacy compliance.

## Repository Structure

*   `model/`: Contains the predictive models and training pipelines (CatBoost, etc.).
*   `dataviz/`: Notebooks for data visualization and analysis (PISA data, bullying impact).
*   `feature_selection/`: Results and analysis of feature selection processes.
*   `submission/`: Final submission files (CSVs) for the competition.
*   `prototype/`: Code for the prototype application (Streamlit/Python).
*   `rapport/`: Synthesis and data quality reports.

## Skills & Technologies

*   **Programming:** Python
*   **Machine Learning & AI:** CatBoost, Scikit-learn, Predictive Modeling
*   **Data Science:** Data Visualization, Statistical Programming, Feature Engineering
*   **Soft Skills:** Project Management, Presentations
*   **Language:** English
