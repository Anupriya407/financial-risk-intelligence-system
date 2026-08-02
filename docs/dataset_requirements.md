# Financial Risk Intelligence System (FRIS)

# Dataset Requirements Specification

**Version:** 1.0

**Project:** Financial Risk Intelligence System (FRIS)

**Phase:** Dataset Engineering

---

# 1. Purpose

This document defines the minimum data requirements for the Financial Risk Intelligence System (FRIS) Version 1.

The purpose of this specification is to ensure that all selected datasets provide sufficient information to build an enterprise-grade Loan Risk Intelligence Platform.

Datasets selected for FRIS must satisfy these requirements before being accepted into the project.

---

# 2. Dataset Objectives

The selected datasets must enable the system to:

- Analyze customer financial information.
- Assess customer financial health.
- Predict loan default probability.
- Classify financial risk.
- Explain model predictions.
- Generate lending recommendations.
- Evaluate model generalization using an independent dataset.

---

# 3. Dataset Strategy

FRIS Version 1 will use two datasets.

## Dataset 1

Primary Training Dataset

Purpose

- Model training
- Feature engineering
- Hyperparameter tuning
- Cross validation
- Explainability
- Model evaluation

---

## Dataset 2

External Validation Dataset

Purpose

- Independent evaluation
- Generalization testing
- Performance comparison
- Robustness assessment

This dataset will NOT be used for model training.

---

# 4. Mandatory Feature Categories

Every candidate dataset should be evaluated against the following business requirements.

## 4.1 Customer Profile

Required Features

- Age
- Gender (if available)
- Marital Status
- Education
- Number of Dependents

Importance

High

---

## 4.2 Employment Information

Required Features

- Employment Status
- Occupation
- Employer Type
- Years Employed

Importance

Critical

---

## 4.3 Income Information

Required Features

- Annual Income
- Monthly Income
- Household Income
- Income Type

Importance

Critical

---

## 4.4 Loan Information

Required Features

- Loan Amount
- Loan Purpose
- Loan Term
- Interest Rate
- Installment Amount

Importance

Critical

---

## 4.5 Credit History

Required Features

- Previous Defaults
- Credit Score
- Late Payments
- Number of Credit Accounts
- Credit Utilization

Importance

Critical

---

## 4.6 Existing Financial Obligations

Required Features

- Current Debt
- Mortgage
- Credit Card Debt
- Debt-to-Income Ratio

Importance

Critical

---

## 4.7 Assets

Preferred Features

- Savings
- Property Ownership
- Vehicle Ownership
- Investment Assets

Importance

Medium

---

## 4.8 Financial Behaviour

Preferred Features

- Payment History
- Missed Payments
- Account Age
- Credit Behaviour

Importance

High

---

## 4.9 Temporal Information

Preferred Features

- Application Date
- Loan Issue Date
- Employment Start Date

Importance

Medium

---

## 4.10 Target Variable

Mandatory

The dataset must contain a clearly defined target variable representing loan repayment outcome.

Examples

- Loan Default
- Charged Off
- Fully Paid
- Default Flag

Importance

Mandatory

---

# 5. Dataset Quality Requirements

Candidate datasets should satisfy the following criteria.

## Size

Preferred

- More than 10,000 observations

Excellent

- More than 100,000 observations

---

## Missing Values

- Missing values should be documented.
- Excessive missingness should be avoided.

---

## Data Types

The dataset should contain:

- Numerical features
- Categorical features
- Boolean features (if available)

---

## Documentation

The dataset should include:

- Feature descriptions
- Target description
- Data dictionary

---

## Licensing

The dataset must have a license permitting educational and research use.

---

# 6. Dataset Evaluation Criteria

Candidate datasets will be evaluated using the following dimensions.

- Feature Completeness
- Data Quality
- Sample Size
- Target Quality
- Documentation
- Real-World Relevance
- Licensing
- Ease of Integration

---

# 7. Expected Deliverables

After dataset selection, the project should contain:

- Primary Training Dataset
- External Validation Dataset
- Dataset Metadata
- Data Dictionary
- Dataset Version Information
- Dataset Quality Report

---

# 8. Future Expansion

Future versions of FRIS may incorporate:

- Credit Bureau Data
- Banking Transaction Data
- Financial Statement Data
- Macroeconomic Indicators
- Fraud Detection Data
- Portfolio-Level Risk Data

These datasets are outside the scope of FRIS Version 1.

---

# End of Document