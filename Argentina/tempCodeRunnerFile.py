import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

# 1) Load
df = pd.read_csv("loan_data.csv")  # adjust path if needed

# 2) FIX #1 — Remove duplicates BEFORE splitting (prevents contamination)
df = df.drop_duplicates()

# 3) FIX #2 — Drop post-outcome/target-leaking features
leak_cols = [c for c in ["last_audit_team_id"] if c in df.columns]
df = df.drop(columns=leak_cols)

# 4) Features/Target
target_col = "loan_defaulted"
y = df[target_col].astype(int).values
X = df.drop(columns=[target_col]).copy()

# 5) Split (stratified for class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6) FIX #3 — Put scaler inside a Pipeline (fit on TRAIN only)
num_features = list(X.columns)  # all numeric here
preprocess = ColumnTransformer(
    transformers=[("num", StandardScaler(), num_features)],
    remainder="drop",
)

clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1,
)

pipe = Pipeline(steps=[("preprocess", preprocess), ("model", clf)])
pipe.fit(X_train, y_train)

# 7) Evaluate on proper held-out test
y_pred = pipe.predict(X_test)
y_proba = pipe.predict_proba(X_test)[:, 1]
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC :", roc_auc_score(y_test, y_proba))

# 8) Save the leak-free preprocessed dataset:
#    Use the train-fitted scaler to transform the ENTIRE feature matrix
X_all_scaled = pipe.named_steps["preprocess"].fit(X_train).transform(X)  # fit on TRAIN, transform ALL
X_scaled_df = pd.DataFrame(X_all_scaled, columns=num_features, index=X.index)
X_scaled_df[target_col] = y
X_scaled_df.to_csv("loan_data_preprocessed.csv", index=False)
print("Saved: loan_data_preprocessed.csv")