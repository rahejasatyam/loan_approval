# 📌 Two-Stage Loan Approval System

## 📖 Introduction

The Two-Stage Loan Approval System is a machine learning–based application that helps banks and financial institutions make better loan decisions. It works in two steps:

1. First, it decides whether a loan should be approved or rejected.
2. Then, if approved, it predicts the optimal loan amount.

This two-step approach follows real-world banking workflows and makes the system more accurate, reliable, and realistic.

This project is suitable for beginners who want to understand how machine learning can be used in financial decision-making systems.

---

## 🎯 Project Objective

The main objectives of this project are:

- To automate the loan approval process.
- To reduce manual effort in evaluating applicants.
- To provide data-driven and fair decisions.
- To predict loan amounts only for eligible users.
- To build a scalable and production-ready ML system.

---

## 🧠 System Overview (Two-Stage Pipeline)

This project uses two separate machine learning models.

---

### ✅ Stage 1: Loan Approval (Classification)

In the first stage, a Random Forest Classifier predicts whether the applicant should get a loan.

Output:
- 1 → Approved
- 0 → Rejected

This stage analyzes the applicant’s financial and personal information.

---

### 💰 Stage 2: Loan Amount Prediction (Regression)

If the applicant is approved in Stage 1, the system moves to Stage 2.

A Random Forest Regressor predicts the suitable loan amount.

This stage runs only when the loan is approved. If rejected, no amount is predicted.

---

### 📌 Why Two Stages?

Using two models provides several benefits:

- Prevents invalid predictions
- Avoids unnecessary calculations
- Improves accuracy
- Matches real banking systems
- Reduces data leakage

---

## ✨ Key Features

- Two-stage machine learning pipeline
- Classification + Regression approach
- Random Forest models
- Conditional prediction logic
- Feature consistency using feature_names_in_
- Modular, class-based Python design
- Pre-trained model loading using Joblib
- Easy to extend and deploy

---

## 🛠️ Technology Stack

| Category | Tools / Libraries |
|----------|------------------|
| Language | Python |
| Data Handling | Pandas |
| ML Library | Scikit-learn |
| Model Storage | Joblib |
| Models | Random Forest |

---

## 📂 Project Structure

