# Throughput Prediction in Beyond 5G Networks Using Machine Learning

Machine learning pipeline for **short-term wireless throughput prediction in a Beyond 5G (B5G) network**. This project compares Linear Regression, Random Forest, XGBoost, and LSTM models for predicting the next total throughput measurement of a user equipment (UE) using recent network conditions.

This work was completed through the **NSF AI-EDGE Summer Research Program at The Ohio State University**.

**Authors:** Gabriel Balbaneda and Rakshu Sankarraman  
**Faculty Mentor:** Biswajit Kumar Dash  
**Program:** AI-EDGE Summer Research Program, Summer 2026

---

## Project Overview

Modern wireless networks experience rapid changes in traffic demand and channel conditions, making reliable resource allocation increasingly difficult. Predicting near-future throughput can help network controllers anticipate these changes and support Quality of Service (QoS) management.

This project investigates whether a small set of recent network measurements can predict **total UE1 throughput at the next time step**.

Four regression approaches were evaluated:

- Linear Regression
- Random Forest
- XGBoost
- Long Short-Term Memory (LSTM)

The models were trained and evaluated on measurements from a cloud-native Beyond 5G testbed.

---

## Dataset

The project uses the publicly available dataset associated with:

> T. Tsourdinis, I. Chatzistefanidis, N. Makris, T. Korakis, N. Nikaein, and S. Fdida, "Service-aware real-time slicing for virtualized beyond 5G networks," *Computer Networks*, vol. 247, 110445, 2024.

The original dataset contains **48,600 time-series observations** from three User Equipments (UE1, UE2, and UE3). This project focuses on **UE1**.

Total UE1 throughput was calculated as:

```text
Total UE1 Throughput =
    UE1 WebRTC Throughput
  + UE1 SIPp Throughput
  + UE1 Web Server Throughput
```

After generating lag features, **48,597 observations** remained.

### Input Features

The models use five features:

```text
UE1 Throughput Lag 1
UE1 Throughput Lag 2
UE1 Throughput Lag 3
UE1 Jitter
UE1 Channel Quality Indicator (CQI)
```

**Target:** Total UE1 throughput at the next time step.

---

## Data Preprocessing

Because the measurements are time-series data, the dataset was split **chronologically rather than randomly**:

| Dataset | Samples | Percentage |
| --- | ---: | ---: |
| Training | 34,017 | 70% |
| Validation | 7,290 | 15% |
| Testing | 7,290 | 15% |

Input features were normalized using **Min-Max scaling**.

To prevent data leakage, the scaler was fit **only on the training data** and then applied to the validation and test sets.

Three lag features were generated from total UE1 throughput at `t-1`, `t-2`, and `t-3`.

---

## Models

### Linear Regression

Linear Regression serves as the baseline model for determining how well a simple linear relationship can predict future throughput.

### Random Forest

Thirty Random Forest hyperparameter configurations were evaluated using the validation set.

The final configuration was:

```python
n_estimators = 200
max_depth = 10
min_samples_leaf = 2
```

### XGBoost

Thirty XGBoost hyperparameter configurations were also evaluated using the validation set.

The final configuration was:

```python
n_estimators = 200
learning_rate = 0.05
max_depth = 6
subsample = 0.8
colsample_bytree = 1.0
```

### LSTM

The LSTM processes the three historical throughput measurements as a sequence while incorporating jitter and CQI as additional network features.

Thirty configurations were evaluated, with early stopping used to reduce overfitting. The final model used:

```python
lstm_units = 50
dense_units = 32
learning_rate = 0.001
batch_size = 32
```

Training stopped after **32 epochs**.

---

## Results

Performance was evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Coefficient of Determination (R²)

### Test Set Performance

| Model | MAE (Bytes/s) | RMSE (Bytes/s) | R² |
| --- | ---: | ---: | ---: |
| Linear Regression | 6,098.99 | 8,575.28 | 0.5658 |
| **Random Forest** | **4,956.63** | 7,624.63 | 0.6567 |
| **XGBoost** | 4,987.59 | **7,621.46** | **0.6570** |
| LSTM | 5,367.59 | 7,867.20 | 0.6345 |

**XGBoost achieved the strongest overall performance**, producing the highest R² and lowest RMSE.

Random Forest performed nearly identically and achieved the **lowest MAE**.

Both tree-based models substantially outperformed the Linear Regression baseline. The LSTM improved on Linear Regression but did not outperform Random Forest or XGBoost.

---

## Model Interpretability with SHAP

Because XGBoost achieved the strongest overall performance, **SHAP (SHapley Additive exPlanations)** was used to interpret its predictions.

The feature ranking based on mean absolute SHAP values was:

1. **Throughput Lag 1**
2. **Jitter**
3. **Throughput Lag 2**
4. **Throughput Lag 3**
5. **CQI**

The most recent throughput measurement had the greatest influence on the model's predictions.

SHAP analysis also showed that higher Lag 1 throughput values generally increased predicted throughput, while lower values decreased it. Jitter was the second-most influential feature.

These results suggest that **recent traffic behavior was more informative for next-step throughput prediction than CQI in this dataset**.

---

## Key Findings

- Tree-based models captured nonlinear throughput behavior better than Linear Regression.
- XGBoost achieved the best overall test performance with an **R² of 0.6570**.
- Random Forest achieved the lowest prediction MAE at **4,956.63 Bytes/s**.
- The LSTM did not outperform the tree-based models despite explicitly modeling the throughput sequence.
- SHAP identified the most recent throughput measurement as the strongest predictor.
- Sudden throughput spikes remained difficult for all models to predict accurately.

---

## Repository Structure

```text
wireless_congestion_prediction/
├── data/
│   └── README.md
├── plots/
├── src/
│   ├── data_preprocessing.py
│   ├── visualization.py
│   ├── train_regression.py
│   ├── train_random_forest.py
│   ├── train_random_forest_tuning.py
│   ├── train_xgboost.py
│   ├── train_xgboost_tuning.py
│   ├── train_lstm.py
│   ├── train_lstm_tuning.py
│   └── shap_analysis.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/gbal-boop/wireless_congestion_prediction.git
cd wireless_congestion_prediction
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment on Windows:

```bash
venv\Scripts\activate
```

On macOS or Linux:

```bash
source venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Dataset Setup

Download `ue-lte-network-traffic-stats.csv` from the original App-Aware 5G dataset repository:

```text
https://github.com/teo-tsou/app_aware_5g/tree/master/dataset
```

Place the file in:

```text
data/ue-lte-network-traffic-stats.csv
```

The original and generated datasets are not required to be stored in version control because the processed datasets can be reproduced using the preprocessing script.

---

## Usage

### 1. Preprocess the Dataset

```bash
python src/data_preprocessing.py
```

This creates the chronological training, validation, and testing datasets used by the models.

### 2. Train and Evaluate Linear Regression

```bash
python src/train_regression.py
```

### 3. Train and Evaluate Random Forest

```bash
python src/train_random_forest.py
```

### 4. Train and Evaluate XGBoost

```bash
python src/train_xgboost.py
```

### 5. Train and Evaluate LSTM

```bash
python src/train_lstm.py
```

### 6. Generate Exploratory Visualizations

```bash
python src/visualization.py
```

### 7. Run SHAP Analysis

```bash
python src/shap_analysis.py
```

Generated figures are saved in the `plots/` directory.

---

## Hyperparameter Tuning

Separate tuning scripts reproduce the hyperparameter searches used during model development.

### Random Forest

```bash
python src/train_random_forest_tuning.py
```

### XGBoost

```bash
python src/train_xgboost_tuning.py
```

### LSTM

```bash
python src/train_lstm_tuning.py
```

Each tuning script evaluates **five configurations per tuning part**, with six parts corresponding to **30 evaluated configurations** for each model.

The tuning scripts use the **training and validation sets only**. The held-out test set is reserved for final evaluation in the corresponding final model scripts.

The `RUN_PART` variable inside each tuning script can be changed from `1` through `6` to reproduce each portion of the hyperparameter search.

---

## Technologies

### Languages and Libraries

- Python
- pandas
- NumPy
- scikit-learn
- XGBoost
- TensorFlow / Keras
- SHAP
- Matplotlib

### Machine Learning

- Linear Regression
- Random Forest Regression
- Gradient Boosting
- LSTM Neural Networks
- Time-Series Feature Engineering
- Hyperparameter Tuning
- Model Interpretability

---

## Limitations and Future Work

This study focuses on one UE, five input features, and offline evaluation on a single dataset.

Future work could include:

- Extending prediction to UE2 and UE3
- Evaluating longer throughput histories
- Adding latency, packet loss, and signal-strength features
- Testing models under additional network conditions
- Evaluating inference latency in real time
- Integrating predictions with network resource-allocation or congestion-management systems

---

## Research Context

This project was completed as part of the **NSF AI-EDGE Summer Research Program at The Ohio State University**.

AI-EDGE focuses on advancing artificial intelligence technologies for next-generation wireless networks. This project explored the application of machine learning to short-term throughput prediction as a step toward data-driven network management.

---

## Authors

**Gabriel Balbaneda**  
Department of Electrical and Computer Engineering  
University of Illinois Chicago

**Rakshu Sankarraman**  
Department of Computer Science and Engineering  
The Ohio State University

**Faculty Mentor: Biswajit Kumar Dash**  
Department of Computer Science and Engineering  
The Ohio State University

---

## References

1. T. Tsourdinis, I. Chatzistefanidis, N. Makris, T. Korakis, N. Nikaein, and S. Fdida, "Service-aware real-time slicing for virtualized beyond 5G networks," *Computer Networks*, vol. 247, 110445, 2024. DOI: `10.1016/j.comnet.2024.110445`

2. S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," *Advances in Neural Information Processing Systems*, vol. 30, 2017.

---

## License

The source code in this repository is available under the **MIT License**. See `LICENSE` for details.

The external dataset used in this project is attributed to its original authors and is not covered by this repository's MIT License.

---

## Acknowledgments

This research was conducted through the **NSF AI-EDGE Institute Summer Research Program at The Ohio State University**. We thank our faculty mentor, Biswajit Kumar Dash, for his guidance throughout the project.
