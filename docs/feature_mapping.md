# Financial Risk Intelligence System (FRIS)

# Feature Mapping Specification

Version: 1.0

---

# Purpose

This document maps FRIS business requirements to the Home Credit dataset.

It defines how customer information, financial information, historical credit behavior, and loan information will be transformed into machine learning features.

---

# Dataset

Primary Dataset

Home Credit Default Risk

---

# Customer Profile

Business Requirement

- Customer Age
- Gender
- Family Status
- Education
- Housing

Dataset

application_train.csv

Columns

(To be filled after studying the data dictionary)

---

# Employment

Business Requirement

- Employment Status
- Occupation
- Organization
- Employment Length

Dataset

application_train.csv

Columns

(To be filled)

---

# Income

Business Requirement

- Income
- Income Source

Dataset

application_train.csv

Columns

(To be filled)

---

# Loan Information

Business Requirement

- Loan Amount
- Loan Type
- Loan Purpose
- Installments

Dataset

application_train.csv

Columns

(To be filled)

---

# Credit History

Business Requirement

- Previous Loans
- Previous Defaults
- Credit History
- Active Loans

Datasets

bureau.csv

bureau_balance.csv

Columns

(To be filled)

---

# Previous Applications

Business Requirement

- Previous Loan Applications
- Approved Loans
- Rejected Loans

Dataset

previous_application.csv

Columns

(To be filled)

---

# Payment Behaviour

Business Requirement

- Missed Payments
- Late Payments
- Installment History

Dataset

installments_payments.csv

Columns

(To be filled)

---

# Credit Card Behaviour

Business Requirement

- Credit Utilization
- Outstanding Balance
- Revolving Credit

Dataset

credit_card_balance.csv

Columns

(To be filled)

---

# Cash Loans

Business Requirement

- Existing Cash Loans
- Existing Consumer Loans

Dataset

POS_CASH_balance.csv

Columns

(To be filled)

---

# Target

Business Requirement

Loan Default

Dataset

application_train.csv

Column

TARGET