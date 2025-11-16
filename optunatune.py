import optuna
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Load the preprocessed dataset
df = pd.read_csv("final_dataset_no_leakage.csv")



# Split features and target
X = df.drop(columns=["los_category"])
y = df["los_category"]

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Define Optuna objective
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "use_label_encoder": False,
        "eval_metric": "logloss"
    }
    model = XGBClassifier(**params)
    score = cross_val_score(model, X_train, y_train, scoring="f1", cv=3).mean()
    return score

# Run hyperparameter tuning
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

# Train final model with best parameters
best_params = study.best_params
best_params["use_label_encoder"] = False
best_params["eval_metric"] = "logloss"

model = XGBClassifier(**best_params)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("✅ Final Evaluation (Tuned):")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
print(f"AUC: {roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]):.4f}")
