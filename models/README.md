# Trained Model Artifacts

This directory contains the trained model and preprocessing artifacts used for the final experiments in **Throughput Prediction in Beyond 5G Networks Using Machine Learning**.

All models were trained using the chronological training split described in the main project README. The saved artifacts were independently reloaded and verified against the test set to confirm that they reproduce the reported final results.

## Files

| File | Description |
| --- | --- |
| `feature_scaler.joblib` | Min-Max scaler fitted on the five UE1 training features and used to transform validation and test inputs. |
| `linear_regression.joblib` | Final trained Linear Regression baseline. |
| `random_forest.joblib` | Final trained Random Forest model using 200 estimators, maximum depth 10, and minimum leaf size 2. |
| `xgboost.joblib` | Serialized Python version of the final trained XGBoost model. |
| `xgboost.json` | XGBoost's native representation of the final trained model. |
| `lstm.keras` | Final trained LSTM neural network saved in Keras format. |
| `lstm_target_scaler.joblib` | Min-Max scaler fitted to the training target and used to transform and inverse-transform LSTM outputs. |

## Model Performance

The saved artifacts reproduce the following test-set results:

| Model | MAE (Bytes/s) | RMSE (Bytes/s) | R² |
| --- | ---: | ---: | ---: |
| Linear Regression | 6,098.99 | 8,575.28 | 0.5658 |
| Random Forest | **4,956.63** | 7,624.63 | 0.6567 |
| XGBoost | 4,987.59 | **7,621.46** | **0.6570** |
| LSTM | 5,367.59 | 7,867.20 | 0.6345 |

XGBoost achieved the strongest overall performance based on RMSE and R², while Random Forest achieved the lowest MAE.

## Reproducing the Artifacts

The final model artifacts can be regenerated from the project root using:

```bash
python src/generate_model_artifacts.py