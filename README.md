# 📌 Two-Stage Loan Approval System

##  Introduction

The Two-Stage Loan Approval System is a machine learning–based application that helps banks and financial institutions make better loan decisions. It works in two steps:

1. First, it decides whether a loan should be approved or rejected.
2. Then, if approved, it predicts the optimal loan amount.

This two-step approach follows real-world banking workflows and makes the system more accurate, reliable, and realistic.

This project is suitable for beginners who want to understand how machine learning can be used in financial decision-making systems.

---

##  Project Objective

The main objectives of this project are:

- To automate the loan approval process.
- To reduce manual effort in evaluating applicants.
- To provide data-driven and fair decisions.
- To predict loan amounts only for eligible users.
- To build a scalable and production-ready ML system.

---

##  System Overview (Two-Stage Pipeline)

This project uses two separate machine learning models.

---

###  Stage 1: Loan Approval (Classification)

In the first stage, a Random Forest Classifier predicts whether the applicant should get a loan.

Output:
- 1 → Approved
- 0 → Rejected

This stage analyzes the applicant’s financial and personal information.

---

###  Stage 2: Loan Amount Prediction (Regression)

If the applicant is approved in Stage 1, the system moves to Stage 2.

A Random Forest Regressor predicts the suitable loan amount.

This stage runs only when the loan is approved. If rejected, no amount is predicted.

---

###  Why Two Stages?

Using two models provides several benefits:

- Prevents invalid predictions
- Avoids unnecessary calculations
- Improves accuracy
- Matches real banking systems
- Reduces data leakage

---

##  Key Features

- Two-stage machine learning pipeline
- Classification + Regression approach
- Random Forest models
- Conditional prediction logic
- Feature consistency using feature_names_in_
- Modular, class-based Python design
- Pre-trained model loading using Joblib
- Easy to extend and deploy

---

##  Technology Stack

| Category | Tools / Libraries |
|----------|------------------|
| Language | Python |
| Data Handling | Pandas |
| ML Library | Scikit-learn |
| Model Storage | Joblib |
| Models | Random Forest |

---
##  Project Structure

loan_approval/
│
├── models/
│ ├── stage_1_rf_classifier_pipeline.pkl
│ └── stage_2_rf_regression_pipeline.pkl
│
├── loan_approval_app.py
├── main.py
├── README.md
└── loan_approval_dataset.csv



---

##  Folder Explanation

- models/  
  Contains trained machine learning models.

- stage_1_rf_classifier_pipeline.pkl  
  Classifier model for approval prediction.

- stage_2_rf_regression_pipeline.pkl  
  Regression model for loan amount prediction.

- loan_approval_app.py / main.py  
  Main application files.

- README.md  
  Project documentation.

- loan_approval_dataset.csv  
  Dataset used for training.

---

##  Input Features

The system uses the following applicant details:

| Feature | Description |
|---------|-------------|
| no_of_dependents | Number of dependents |
| education | Graduate / Not Graduate |
| self_employed | Yes / No |
| income_annum | Annual income |
| loan_amount | Requested loan amount |
| loan_term | Loan duration |
| cibil_score | Credit score |
| residential_assets_value | Property value |
| commercial_assets_value | Business assets |
| luxury_assets_value | Luxury assets |
| bank_asset_value | Bank savings |

These features help measure the applicant’s financial stability.

---

##  Application Workflow

The system follows these steps:

1. User enters applicant details.
2. Data is converted into a Pandas DataFrame.
3. Stage 1 predicts approval or rejection.
4. If approved, Stage 2 predicts loan amount.
5. Result is displayed on the screen.

---

##  Complete Implementation

```python
import joblib
import os
import pandas as pd


class LoanApprovalApp:

    def __init__(
        self,
        classifier="stage_1_rf_classifier_pipeline.pkl",
        regressor="stage_2_rf_regression_pipeline.pkl"
    ):

        model_path = os.path.join(os.getcwd(), "models")

        self.clf = joblib.load(
            os.path.join(model_path, classifier)
        )

        self.reg = joblib.load(
            os.path.join(model_path, regressor)
        )

        self.cols = [
            "no_of_dependents",
            "education",
            "self_employed",
            "income_annum",
            "loan_amount",
            "loan_term",
            "cibil_score",
            "residential_assets_value",
            "commercial_assets_value",
            "luxury_assets_value",
            "bank_asset_value"
        ]


    def get_user_input(self):

        print("\n--- Enter Applicant Details ---\n")

        data = {
            "no_of_dependents": 3,
            "education": "Graduate",
            "self_employed": "Yes",
            "income_annum": 8300000,
            "loan_amount": 3001400000,
            "loan_term": 60,
            "cibil_score": 4,
            "residential_assets_value": 1000000,
            "commercial_assets_value": 1600000,
            "luxury_assets_value": 17200000,
            "bank_asset_value": 6100000
        }

        return pd.DataFrame([data])


    def two_stage_predict(self, applicant_df):

        output = {}

        applicant_df = applicant_df[self.clf.feature_names_in_]

        approve = self.clf.predict(applicant_df)[0]

        output["approve"] = int(approve)

        if approve == 1:

            applicant_df_reg = applicant_df.copy()

            applicant_df_reg["loan_status"] = "Approve"

            prediction = self.reg.predict(applicant_df_reg)[0]

            output["regression_prediction"] = float(prediction)

        else:

            output["regression_prediction"] = None

        return output


    def run(self):

        print("\n============================")
        print("  Loan Approval Application ")
        print("============================\n")

        applicant_df = self.get_user_input()

        result = self.two_stage_predict(applicant_df)

        print("\n------ RESULT ------")

        if result["approve"] == 1:

            print("Loan Status : APPROVED")

            print(
                f"Predicted Loan Amount / Value : "
                f"{result['regression_prediction']:.2f}"
            )

        else:

            print("Loan Status : REJECTED")

        print("\n---------------------\n")

        return result


if __name__ == "__main__":

    app = LoanApprovalApp()

    app.run()
```
How to Run the Project
1️ Install Dependencies
pip install pandas scikit-learn joblib

2️ Setup Models

Place trained models inside the models folder:

models/
 ├── stage_1_rf_classifier_pipeline.pkl
 └── stage_2_rf_regression_pipeline.pkl

3️ Run the Program
python loan_approval_app.py


or

python main.py

 Sample Output
Approved Case
Loan Status : APPROVED
Predicted Loan Amount / Value : 2450000.50

Rejected Case
Loan Status : REJECTED

 Real-World Use Cases

Banking systems

Credit scoring platforms

Loan management software

FinTech applications

Financial decision systems

 Future Enhancements

REST API using FastAPI or Flask

Web interface

Model explainability

Database integration

Cloud deployment

Security features

Learning Outcomes

Machine learning pipelines

Classification and regression

Model deployment

Production-level Python

Financial ML systems

Data leakage prevention


