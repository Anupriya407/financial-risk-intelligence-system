# FRIS V1 Architecture

## High-Level Architecture

Loan Officer

↓

Streamlit Dashboard

↓

FastAPI Backend

↓

Business Services

↓

Machine Learning Pipeline

↓

PostgreSQL Database

## Description

- Streamlit provides the user interface.
- FastAPI handles requests from the dashboard.
- Business Services contain the application logic.
- The ML Pipeline predicts loan default risk.
- PostgreSQL stores customer, loan, and prediction data.