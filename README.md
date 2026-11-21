# Breast Cancer Diagnosis Prediction

Classifies breast tumors as **benign** or **malignant** using the Breast Cancer Wisconsin (Diagnostic) dataset (569 samples, 30 numeric features).

## Approach
- EDA: missing values, duplicates, class balance, distributions, outliers, correlation heatmap
- Baseline: Logistic Regression with `StandardScaler`
- Final model: `StandardScaler` → `PCA` → `SVC (RBF)`, tuned with `GridSearchCV` (5-fold stratified CV, scored on recall)
- Best params: `C=10`, `gamma='scale'`, `PCA n_components=10`
- Includes a simple `predict_cancer()` function for inference on new samples

## Results (test set, 20% stratified split)
| Model | Accuracy |
|---|---|
| Logistic Regression (baseline) | 96.49% |
| PCA + SVC (tuned) | 96.49% (malignant precision 0.97, recall 0.93) |

## Dataset
The CSV is not included in this repo. Download the [Breast Cancer Wisconsin (Diagnostic) dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data) and save it as `breast_cancer_dataset.csv` in the project folder.

## Run
```bash
pip install -r requirements.txt
jupyter notebook breast_cancer_diagnosis.ipynb
```

## Tech Stack
Python, Pandas, Scikit-learn, Matplotlib, Seaborn
