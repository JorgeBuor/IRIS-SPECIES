# 🌸 Iris Species Classification Dashboard

**Data Mining — Universidad de la Costa**  
**Author:** Jorge Estiiven Burgos Ortega

---

## Purpose

This project applies a complete data mining pipeline to the classic Iris dataset: data understanding, preprocessing, classification modeling with Random Forest, evaluation, and interactive visualization. The goal is to predict the species of an Iris flower (*setosa*, *versicolor*, or *virginica*) from four petal and sepal measurements, and communicate the results through an interactive Streamlit dashboard.

---

## Dataset

`Iris.csv` — contains 150 flower samples with the following columns:

| Column | Description |
|---|---|
| `Id` | Sample identifier |
| `SepalLengthCm` | Sepal length in centimeters |
| `SepalWidthCm` | Sepal width in centimeters |
| `PetalLengthCm` | Petal length in centimeters |
| `PetalWidthCm` | Petal width in centimeters |
| `Species` | Target class: Iris-setosa, Iris-versicolor, or Iris-virginica |

---

## Workflow

1. **Data Understanding** — descriptive statistics, null-value check, class distribution.
2. **Preprocessing** — label encoding of the target variable; feature standardization with `StandardScaler`; 80/20 stratified train/test split.
3. **Modeling** — `RandomForestClassifier` (100 estimators, `random_state=42`).
4. **Evaluation** — Accuracy, Precision (weighted), Recall (weighted), F1-Score (weighted), and Confusion Matrix.
5. **Visualization** — interactive Streamlit dashboard with prediction panel and 3D scatter plot.

---

## Repository Structure

```
├── Proyect.py        # Streamlit dashboard
├── Iris.csv          # Dataset
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

---

## How to Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/JorgeBuor/IRIS-SPECIES.git
   cd <repo-folder>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   streamlit run Proyect.py
   ```

---

## Dashboard Features

- **Sidebar filters:** species selection for exploration views, color palette, and grid toggle
- **KPI cards:** Accuracy, Precision, Recall, F1 Score, and number of training samples
- **Tab 1 — Model Performance:** confusion matrix heatmap + classification metrics bar chart
- **Tab 2 — Predict Species:** interactive sliders for the four features → predicted species + probability breakdown + 3D scatter plot showing the new sample relative to the full dataset
- **Tab 3 — Data Exploration:** overlapping histograms per feature and a full scatter matrix
- **Tab 4 — Raw Data:** filterable table with CSV download option

---

## Deployment

The dashboard is deployed on **Streamlit Cloud**:  
🔗 https://iris-species-7qkosz3zr5cxpy7qut6bif.streamlit.app/

---

## Key Findings

- The Random Forest model achieves **93.33% accuracy** on the test set with perfect separation of *Iris-setosa* from the other two species.
- Petal length and petal width are the most discriminative features; sepal width shows the most overlap between *versicolor* and *virginica*.
- No missing values were found in the dataset; no imputation was required.
