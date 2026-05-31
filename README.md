# Optimizing Churn Prediction: From Model Selection to Hyperparameter Tuning

## Project Overview

This project demonstrates best practices for building reliable machine learning models by addressing a critical challenge in predictive modeling: **How do we know if our model's performance is trustworthy, or did we just get lucky with our data split?**

We work with a telecom churn dataset to predict which customers are likely to cancel their service. By the end of this project, you'll understand how to:

- ✅ Obtain reliable performance estimates using **cross-validation**
- ✅ Systematically tune hyperparameters with **GridSearchCV** and **RandomizedSearchCV**
- ✅ Prevent data leakage by integrating preprocessing into pipelines
- ✅ Build production-ready models with proper evaluation practices

---

## The Problem

A colleague built a simple model and achieved **83% accuracy** on a single train-test split. Management is excited! But there's a hidden risk:

**A single train-test split can be misleading.** Depending on which data points end up in the test set, you might get a "lucky" score that doesn't reflect true model performance. Change the random seed, and the accuracy might drop significantly—even though the model hasn't changed.

**The Solution:** Use cross-validation and systematic hyperparameter tuning to get honest, reliable performance estimates.

---

## Dataset Description

The dataset contains **customer demographics, account information, and service usage patterns**:

### Features
- **customerID**: Unique identifier for each customer
- **gender**: Male/Female
- **SeniorCitizen**: 0 = No, 1 = Yes
- **Partner**: Yes/No
- **Dependents**: Yes/No
- **tenure**: Number of months with the company
- **MonthlyCharges**: Customer's monthly bill
- **TotalCharges**: Total amount billed so far
- **Contract**: Month-to-month, One year, Two year
- **PaperlessBilling**: Yes/No
- **PaymentMethod**: Various payment methods (categorical)
- **Service Features**: Binary Yes/No indicators
  - Phone Service
  - MultipleLines
  - InternetService (DSL, Fiber optic, No internet service)
  - OnlineSecurity
  - OnlineBackup
  - DeviceProtection
  - TechSupport
  - StreamingTV
  - StreamingMovies
- **Churn** (Target): 0 = Customer stayed, 1 = Customer left

---

## Project Structure

### 1. **Loading and Exploring the Dataset**
   - Load the telecom churn dataset
   - Understand target variable distribution
   - Identify feature types (numerical vs. categorical)

### 2. **Building a Baseline Model**
   - Create a preprocessing pipeline with:
     - StandardScaler for numerical features
     - OneHotEncoder for categorical features
   - Train a Logistic Regression baseline model
   - Evaluate on a single train-test split (baseline: ~83% accuracy)

### 3. **Robust Evaluation with Cross-Validation**
   - Implement **5-Fold Cross-Validation** using `cross_val_score`
   - Obtain a more reliable performance estimate by averaging results across multiple splits
   - Understand why a single split can be misleading

### 4. **Understanding Hyperparameters**
   - **Learned Parameters**: Values the model learns from data (e.g., coefficients)
   - **Hyperparameters**: Settings you choose before training (e.g., C, max_depth)
   - Focus on the **C hyperparameter** in Logistic Regression:
     - Small C → Strong regularization → Simpler model (risk of underfitting)
     - Large C → Weak regularization → Complex model (risk of overfitting)

### 5. **Automated Tuning with GridSearchCV**
   - Create a preprocessing + model pipeline
   - Define a grid of hyperparameter values to test
   - Use GridSearchCV to exhaustively search all combinations
   - Best for small, well-defined hyperparameter spaces

### 6. **Efficient Tuning with RandomizedSearchCV**
   - Use continuous distributions instead of fixed lists
   - Randomly sample hyperparameter combinations
   - Control search iterations with `n_iter` parameter
   - Best for large, continuous hyperparameter spaces

### 7. **Final Evaluation**
   - Evaluate the tuned model on the held-out test set
   - Compare three scores:
     1. Single split baseline accuracy
     2. Best cross-validation score (during tuning)
     3. Final test set score
   - Verify no data leakage occurred

---

## Key Concepts

### Cross-Validation
Instead of relying on a single random train-test split, cross-validation:
- Splits data into k folds (typically k=5)
- Trains and evaluates the model k times
- Averages the results for a more robust estimate
- Reduces variance in performance estimates

### GridSearchCV vs. RandomizedSearchCV

| Aspect | GridSearchCV | RandomizedSearchCV |
|--------|--------------|-------------------|
| **Search Strategy** | Tests every combination | Randomly samples combinations |
| **Parameter Space** | Fixed discrete values | Discrete or continuous distributions |
| **Computational Cost** | Higher (exhaustive search) | Lower (controlled sampling) |
| **Use Case** | Small hyperparameter spaces | Large or continuous spaces |
| **Best For** | When you want guaranteed coverage | When efficiency matters more |
