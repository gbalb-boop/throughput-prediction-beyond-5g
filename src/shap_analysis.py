from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import shap

from xgboost import XGBRegressor


DATA_PATH = Path("data")
PLOTS_PATH = Path("plots")

PLOTS_PATH.mkdir(exist_ok=True)

SELECTED_FEATURES = [
    "UE1_throughput_lag1",
    "UE1_throughput_lag2",
    "UE1_throughput_lag3",
    "UE1-Jitter",
    "UE1-CQI",
]

BEST_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 1.0,
}


# --------------------------------------------------
# Load processed data
# --------------------------------------------------

X_train = pd.read_csv(DATA_PATH / "X_train_ue1.csv")
X_test = pd.read_csv(DATA_PATH / "X_test_ue1.csv")

y_train = pd.read_csv(DATA_PATH / "y_train_ue1.csv").squeeze()

X_train = X_train[SELECTED_FEATURES]
X_test = X_test[SELECTED_FEATURES]


# --------------------------------------------------
# Train final XGBoost model
# --------------------------------------------------

model = XGBRegressor(
    n_estimators=BEST_PARAMS["n_estimators"],
    learning_rate=BEST_PARAMS["learning_rate"],
    max_depth=BEST_PARAMS["max_depth"],
    subsample=BEST_PARAMS["subsample"],
    colsample_bytree=BEST_PARAMS["colsample_bytree"],
    objective="reg:squarederror",
    random_state=42,
)

model.fit(X_train, y_train)


# --------------------------------------------------
# Calculate SHAP values
# --------------------------------------------------

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)


# --------------------------------------------------
# Rename features for cleaner plots
# --------------------------------------------------

DISPLAY_NAMES = {
    "UE1_throughput_lag1": "Lag 1",
    "UE1_throughput_lag2": "Lag 2",
    "UE1_throughput_lag3": "Lag 3",
    "UE1-Jitter": "Jitter",
    "UE1-CQI": "CQI",
}

X_test_display = X_test.rename(
    columns=DISPLAY_NAMES
)


# --------------------------------------------------
# SHAP mean absolute feature importance
# --------------------------------------------------

plt.figure()

shap.summary_plot(
    shap_values,
    X_test_display,
    plot_type="bar",
    show=False,
)

plt.xlabel(
    "Mean Absolute SHAP Value",
    fontsize=11,
)

plt.tight_layout()

plt.savefig(
    PLOTS_PATH / "shap_feature_importance.png",
    dpi=600,
    bbox_inches="tight",
)

plt.close()


# --------------------------------------------------
# SHAP summary plot
# --------------------------------------------------

plt.figure()

shap.summary_plot(
    shap_values,
    X_test_display,
    show=False,
)

plt.xlabel(
    "SHAP Value",
    fontsize=11,
)

plt.tight_layout()

plt.savefig(
    PLOTS_PATH / "shap_summary.png",
    dpi=600,
    bbox_inches="tight",
)

plt.close()


# --------------------------------------------------
# Print mean absolute SHAP values
# --------------------------------------------------

mean_abs_shap = (
    pd.DataFrame(
        {
            "Feature": X_test_display.columns,
            "Mean Absolute SHAP Value": abs(
                shap_values
            ).mean(axis=0),
        }
    )
    .sort_values(
        "Mean Absolute SHAP Value",
        ascending=False,
    )
)

print(
    "\n========== XGBoost SHAP Feature Importance =========="
)

print(
    mean_abs_shap.to_string(
        index=False
    )
)

print(
    "\nSHAP plots saved to:"
)

print(
    PLOTS_PATH / "shap_feature_importance.png"
)

print(
    PLOTS_PATH / "shap_summary.png"
)
