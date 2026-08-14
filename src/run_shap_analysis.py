import os
import pandas as pd
import matplotlib.pyplot as plt
import shap

from xgboost import XGBRegressor


# ============================================================
# CREATE OUTPUT FOLDER IF NEEDED
# ============================================================

os.makedirs("plots", exist_ok=True)


# ============================================================
# LOAD SAME PROCESSED DATA USED BY OTHER MODELS
# ============================================================

X_train = pd.read_csv("data/X_train_ue1.csv")
X_test = pd.read_csv("data/X_test_ue1.csv")

y_train = pd.read_csv("data/y_train_ue1.csv").squeeze()


# ============================================================
# SELECT SAME INPUT FEATURES
# ============================================================

selected_features = [
    "UE1_throughput_lag1",
    "UE1_throughput_lag2",
    "UE1_throughput_lag3",
    "UE1-Jitter",
    "UE1-CQI",
]

X_train = X_train[selected_features]
X_test = X_test[selected_features]


# ============================================================
# FINAL XGBOOST MODEL
#
# Best parameters found during hyperparameter tuning
# ============================================================

model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=1.0,
    objective="reg:squarederror",
    random_state=42
)


# Train final XGBoost model
model.fit(X_train, y_train)


# ============================================================
# CREATE SHAP EXPLAINER
# ============================================================

explainer = shap.TreeExplainer(model)


# ============================================================
# CALCULATE SHAP VALUES
#
# Use a subset of the test set to keep SHAP analysis fast.
# 2000 samples is usually enough for a clear summary plot.
# ============================================================

X_shap = X_test.iloc[:2000].copy()

shap_values = explainer(X_shap)


# Rename features for cleaner plots
feature_names = {
    "UE1_throughput_lag1": "Lag 1",
    "UE1_throughput_lag2": "Lag 2",
    "UE1_throughput_lag3": "Lag 3",
    "UE1-Jitter": "Jitter",
    "UE1-CQI": "CQI",
}

X_shap = X_shap.rename(
    columns=feature_names
)

shap_values.feature_names = list(
    X_shap.columns
)


# ============================================================
# SHAP BEESWARM SUMMARY PLOT
#
# Shows:
# - overall feature importance
# - whether high/low feature values push predictions
#   higher or lower
# ============================================================

plt.figure()

shap.plots.beeswarm(
    shap_values,
    show=False
)

plt.tight_layout()

plt.savefig(
    "plots/xgboost_shap_beeswarm.png",
    dpi=600,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SHAP FEATURE IMPORTANCE BAR PLOT
#
# Shows average absolute SHAP value for each feature
# ============================================================

plt.figure()

shap.plots.bar(
    shap_values,
    show=False
)

plt.tight_layout()

plt.savefig(
    "plots/xgboost_shap_feature_importance.png",
    dpi=600,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# PRINT MEAN ABSOLUTE SHAP VALUES
# ============================================================

mean_abs_shap = abs(
    shap_values.values
).mean(axis=0)

shap_importance = pd.DataFrame({
    "Feature": X_shap.columns,
    "Mean Absolute SHAP Value": mean_abs_shap
}).sort_values(
    "Mean Absolute SHAP Value",
    ascending=False
)


print(
    "\n========== XGBoost SHAP Feature Importance =========="
)

print(
    shap_importance.to_string(
        index=False
    )
)


print(
    "\nSHAP analysis completed successfully."
)

print(
    "Saved plots:"
)

print(
    "plots/xgboost_shap_beeswarm.png"
)

print(
    "plots/xgboost_shap_feature_importance.png"
)
