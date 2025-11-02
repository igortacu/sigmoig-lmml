import pandas as pd
import numpy as np

# 1. load raw data
df = pd.read_csv("loan_data.csv")

# 2. fix leak 1: last_audit_team_id
# this field reveals outcome → freeze it to a constant
df["last_audit_team_id"] = -1

# 3. fix leak 2: monthly_payment_capacity
# recompute from allowed info only
# monthly income from annual_income
monthly_income = df["annual_income"] / 12
# expected monthly payment from loan_amount and term
monthly_payment = df["loan_amount"] / df["loan_term_months"]
# capacity = income - payment, floor at 0
df["monthly_payment_capacity"] = (monthly_income - monthly_payment).clip(lower=0)

# 4. fix leak 3: interest_rate
# rebuild a rate that depends only on info available at application
# simple rule: base on term + credit_score, no target info
base_rate = 0.03 + (60 - df["loan_term_months"]) * 0.0005
risk_adj = (850 - df["credit_score"]) * 0.0001
df["interest_rate"] = (base_rate + risk_adj).round(4)

# 5. keep column order EXACTLY as in the statement
cols = [
    "age",
    "annual_income",
    "employment_years",
    "loan_amount",
    "loan_term_months",
    "interest_rate",
    "credit_score",
    "num_credit_lines",
    "num_delinquent_accounts",
    "monthly_payment_capacity",
    "last_audit_team_id",
    "loan_defaulted",
]
df = df[cols]

# 6. no scaling, no extra index
df.to_csv("loan_data_preprocessed.csv", index=False)
