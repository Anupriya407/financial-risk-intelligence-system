# Financial Risk Intelligence System (FRIS)
# Dataset Architecture
# Version 1

## Overview

FRIS Version 1 predicts customer loan default risk.

The dataset is based on the Home Credit Default Risk dataset and consists of seven relational tables.

The dataset is designed around one central entity:

Customer (SK_ID_CURR)

Every other table either directly or indirectly relates to the customer.

---

# Dataset Relationship Diagram

                    Customer
                (SK_ID_CURR)
                       │
     ┌─────────────────┼──────────────────┐
     ▼                 ▼                  ▼
Current Loan      Credit Bureau     Previous Applications
(application)       (bureau)        (previous_application)
      │                 │                    │
      │                 ▼                    ▼
      │         bureau_balance      installments_payments
      │
      ▼
credit_card_balance

      ▼
POS_CASH_balance

---

# Table 1
application_train.csv

Purpose

Contains the current loan application submitted by each customer.

Primary Key

SK_ID_CURR

Target

TARGET

Information

- Customer demographics
- Employment
- Income
- Loan request
- Housing
- Family
- Education
- External risk scores

Contribution to FRIS

Represents the customer's current financial profile.

---

# Table 2
bureau.csv

Purpose

Historical loans obtained from external financial institutions.

Primary Key

SK_ID_BUREAU

Foreign Key

SK_ID_CURR

Contribution

Provides external credit history.

Contains

- Previous loans
- Loan types
- Credit status
- Credit amounts
- Active vs Closed loans

---

# Table 3
bureau_balance.csv

Purpose

Monthly history of every bureau loan.

Primary Key

Composite

SK_ID_BUREAU
MONTHS_BALANCE

Contribution

Monthly repayment behaviour.

Contains

- Loan status
- Delinquency history
- Closed history

---

# Table 4
previous_application.csv

Purpose

Customer's previous Home Credit applications.

Primary Key

SK_ID_PREV

Foreign Key

SK_ID_CURR

Contribution

Historical lending relationship.

Contains

- Approved applications
- Refused applications
- Loan amounts
- Contract type
- Loan timing

---

# Table 5
installments_payments.csv

Purpose

Actual repayment history.

Primary Key

Composite

SK_ID_PREV
NUM_INSTALMENT_NUMBER

Contribution

Repayment behaviour.

Contains

- Scheduled payments
- Actual payments
- Payment delays
- Partial payments

---

# Table 6
credit_card_balance.csv

Purpose

Monthly credit card history.

Primary Key

Composite

SK_ID_PREV
MONTHS_BALANCE

Contribution

Credit card usage behaviour.

Contains

- Credit utilization
- Monthly balances
- Cash withdrawals
- Payments
- Credit limits

---

# Table 7
POS_CASH_balance.csv

Purpose

Monthly POS and cash loan history.

Primary Key

Composite

SK_ID_PREV
MONTHS_BALANCE

Contribution

Consumer loan behaviour.

Contains

- Remaining installments
- Loan status
- Days past due

---

# Customer 360° View

FRIS combines information from all seven tables to create a unified customer profile.

Current Profile

↓

External Credit History

↓

Internal Loan History

↓

Repayment Behaviour

↓

Credit Card Behaviour

↓

POS & Cash Loan Behaviour

↓

Machine Learning Feature Engineering

↓

Loan Default Prediction

---

# Planned Feature Groups

1. Customer Demographics
2. Financial Information
3. Employment Features
4. Credit Bureau Features
5. Previous Loan Features
6. Installment Behaviour Features
7. Credit Card Behaviour Features
8. POS Loan Behaviour Features
9. Aggregated Customer Features
10. Model Input Features

---

# Dataset Pipeline (Future Phases)

Raw CSV

↓

Validation

↓

Cleaning

↓

Relationship Validation

↓

Missing Value Analysis

↓

Feature Engineering

↓

Feature Store

↓

Model Training

↓

Prediction