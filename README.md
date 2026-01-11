# Logistic Regression from Scratch – Breast Cancer Classification

##  Overview
This project implements **binary logistic regression entirely from scratch** using NumPy, without relying on high-level machine learning libraries for model training, optimization, or evaluation. The objective is to demonstrate a **strong understanding of the mathematical and algorithmic foundations** behind logistic regression rather than using prebuilt APIs.

The implementation is evaluated on the **Breast Cancer Wisconsin dataset**, a standard benchmark for binary classification problems.

---

## What This Project Demonstrates
- Internal working of logistic regression
- Optimization techniques used in real ML systems
- Numerical stability and regularization handling
- End-to-end model evaluation from first principles

This repository is designed as a **machine learning foundations project**.

---

##  Key Features
- Manual train–test split and feature standardization  
- Logistic regression core implemented from scratch:
  - Numerically stable sigmoid function  
  - Negative log-likelihood loss  
  - Gradient and Hessian computation  
- Optimization algorithms implemented manually:
  - Batch Gradient Descent (learning-rate decay + early stopping)  
  - Mini-batch Stochastic Gradient Descent (SGD)  
  - Newton’s Method (IRLS with damping)  
- L2 regularization with optional bias exclusion  
- Evaluation metrics implemented from scratch:
  - Accuracy, Precision, Recall, F1-score  
  - ROC curve and AUC (trapezoidal integration)  
- PCA via SVD for 2D visualization and decision surface plotting  
- Comparative analysis of convergence and performance across optimizers  

---

##  Experiments & Visualizations
- Loss convergence comparison between BGD, SGD, and Newton’s method  
- ROC curves with AUC comparison  
- PCA-based 2D visualization of class separation  
- Decision surface visualization on PCA-reduced features  
- Effect of L2 regularization on weight norms and training loss  

---

##  Tech Stack
- Python  
- NumPy  
- Pandas  
- Matplotlib  
- Seaborn  
- scikit-learn (dataset loading only)


pip install -r requirements.txt
python breast_logistic_from_scratch.py
