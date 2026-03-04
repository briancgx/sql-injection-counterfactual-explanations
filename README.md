# SQL Injection Counterfactual Explanations
This repository contains the code and experimental resources related to the research work:
**Counterfactual Explanation of a Classification Model for Detecting SQL Injection Attacks**

The project explores how machine learning models can be used to classify SQL queries and how **counterfactual explanations** may help analyze the factors that influence the model's predictions.

<img width="697" height="308" alt="image" src="https://github.com/user-attachments/assets/a385de46-1ac0-4662-88b2-4a1aa6504527" />


---
# Overview
SQL Injection is one of the most well-known vulnerabilities affecting database-driven applications. Machine learning techniques have been proposed as an alternative approach for detecting malicious queries by analyzing their structural and syntactic characteristics.

This project investigates how classification models behave when analyzing SQL queries and how changes in query features can affect prediction outcomes.

The experiments focus on exploring how small modifications in query characteristics may lead to different classification results.

---
# Repository Contents
This repository includes:
• Python scripts used for data preprocessing and experimentation  
• Scripts for feature analysis and query processing  
• Experimental implementations related to model behavior analysis  
• Datasets containing benign and malicious SQL queries  
• Processed datasets used during the experiments

The code reflects the experimental workflow used during the development of the research project.

---
# Methodology
The experiments generally involve:
1. Preparing datasets containing SQL queries
2. Processing and analyzing query features
3. Training classification models
4. Studying how modifications in query features affect model predictions
5. Analyzing model behavior using explainability techniques

---
# Technologies Used

The project was developed using:
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

These tools were used for data processing, experimentation, and visualization.

---
# Authors
Brian A. Cumi-Guzman  
Alejandro D. Espinosa-Chim  
Mauricio G. Orozco-del-Castillo  
Juan A. Recio-García

---
# License
This repository is provided for academic and research purposes.

---
