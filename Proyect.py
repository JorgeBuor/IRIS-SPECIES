import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)

# ── Configuración de página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Iris Species Classification",
    page_icon="🌸",
    layout="wide"
)

st.title("🌸 Iris Species Classification Dashboard")
st.caption("Data Mining — Universidad de la Costa | Authors: Jorge Estiiven Burgos Ortega · Diego Navarro Gómez")

# ── Carga del dataset ────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("Iris.csv")
    return df

# ── Entrenamiento del modelo ─────────────────────────────────────────────────
@st.cache_resource
def train_model(df):
    feature_cols = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
    X = df[feature_cols].values
    y = df["Species"].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    metrics = {
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted"),
        "Recall":    recall_score(y_test, y_pred, average="weighted"),
        "F1 Score":  f1_score(y_test, y_pred, average="weighted"),
    }

    cm = confusion_matrix(y_test, y_pred)

    return clf, scaler, le, metrics, cm

df = load_data()
model, scaler, le, metrics, cm = train_model(df)

feature_cols  = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
species_names = le.classes_          # ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

# ── Filtros interactivos ─────────────────────────────────────────────────────
st.sidebar.header("🔎 Filters")

species_filter = st.sidebar.multiselect(
    "Species (exploration view)",
    options=list(species_names),
    default=list(species_names)
)

show_grid = st.sidebar.checkbox("Show grid on charts", value=True)
palette   = st.sidebar.selectbox("Color palette", ["muted", "Set2", "pastel", "dark"])

import matplotlib as mpl  # noqa: E402
colors = plt.get_cmap(
    {"muted": "tab10", "Set2": "Set2", "pastel": "Pastel1", "dark": "Dark2"}[palette]
).colors

# ── Filtrado del dataframe ───────────────────────────────────────────────────
df_f = df[df["Species"].isin(species_filter)].copy()

if df_f.empty:
    st.warning("No data available for the selected filters.")
    st.stop()

# ── KPI Cards ────────────────────────────────────────────────────────────────
st.subheader("📊 Model Performance Metrics")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Accuracy",       f"{metrics['Accuracy']:.2%}")
col2.metric("Precision",      f"{metrics['Precision']:.2%}")
col3.metric("Recall",         f"{metrics['Recall']:.2%}")
col4.metric("F1 Score",       f"{metrics['F1 Score']:.2%}")
col5.metric("Training Samples", f"{int(len(df) * 0.8)}")

st.divider()

# ── Visualizaciones ──────────────────────────────────────────────────────────
st.subheader("📈 Analysis")

tab1, tab2, tab3, tab4 = st.tabs([
    "🤖 Model Performance",
    "🌺 Predict Species",
    "📉 Data Exploration",
    "📋 Raw Data"
])

# ────────────────────────────────────────────────────────────────────────────
# TAB 1: Model Performance — Confusion Matrix + Metrics bar chart
# ────────────────────────────────────────────────────────────────────────────
with tab1:
    short_names = [s.replace("Iris-", "") for s in species_names]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Confusion Matrix ─────────────────────────────────────────────────────
    im = axes[0].imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=axes[0])
    axes[0].set_xticks(np.arange(len(short_names)))
    axes[0].set_yticks(np.arange(len(short_names)))
    axes[0].set_xticklabels(short_names, fontsize=10)
    axes[0].set_yticklabels(short_names, fontsize=10)
    axes[0].set_title("Confusion Matrix", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Predicted Label", fontsize=11)
    axes[0].set_ylabel("True Label", fontsize=11)
    for i in range(len(short_names)):
        for j in range(len(short_names)):
            axes[0].text(
                j, i, str(cm[i, j]),
                ha="center", va="center", fontsize=14, fontweight="bold",
                color="white" if cm[i, j] > cm.max() / 2 else "black"
            )

    # ── Metrics bar chart ────────────────────────────────────────────────────
    metric_names  = list(metrics.keys())
    metric_values = [v * 100 for v in metrics.values()]
    bar_colors    = [colors[i % len(colors)] for i in range(len(metric_names))]

    bars = axes[1].bar(
        metric_names, metric_values,
        color=bar_colors, edgecolor="white", width=0.5, alpha=0.88
    )
    for bar in bars:
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{bar.get_height():.2f}%",
            ha="center", va="bottom", fontsize=10, fontweight="bold"
        )
    axes[1].set_title("Classification Metrics (Random Forest)", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Score (%)", fontsize=11)
    axes[1].set_ylim(0, 112)
    axes[1].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    axes[1].grid(show_grid, alpha=0.4)

    plt.tight_layout()
    st.pyplot(fig)

# ────────────────────────────────────────────────────────────────────────────
# TAB 2: Predicción interactiva + 3D Scatter Plot
# ────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### Enter flower measurements to predict its species")

    c1, c2, c3, c4 = st.columns(4)
    sepal_length = c1.slider("Sepal Length (cm)", 4.0, 8.0, 5.8, 0.1)
    sepal_width  = c2.slider("Sepal Width (cm)",  2.0, 4.5, 3.0, 0.1)
    petal_length = c3.slider("Petal Length (cm)", 1.0, 7.0, 4.0, 0.1)
    petal_width  = c4.slider("Petal Width (cm)",  0.1, 2.5, 1.3, 0.1)

    sample        = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    sample_scaled = scaler.transform(sample)
    pred_enc      = model.predict(sample_scaled)[0]
    pred_proba    = model.predict_proba(sample_scaled)[0]
    pred_species  = le.inverse_transform([pred_enc])[0]
    pred_label    = pred_species.replace("Iris-", "")

    st.markdown(f"### 🌺 Predicted Species: **{pred_label}**")

    prob_cols = st.columns(len(species_names))
    for i, (sp, prob) in enumerate(zip(species_names, pred_proba)):
        prob_cols[i].metric(sp.replace("Iris-", ""), f"{prob:.1%}")

    # ── 3D Scatter Plot ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(10, 7))
    ax  = fig.add_subplot(111, projection="3d")

    for i, sp in enumerate(species_names):
        mask = df["Species"] == sp
        sub  = df[mask]
        ax.scatter(
            sub["SepalLengthCm"], sub["SepalWidthCm"], sub["PetalLengthCm"],
            c=[colors[i % len(colors)]], label=sp.replace("Iris-", ""),
            alpha=0.55, s=40, edgecolors="white", linewidth=0.3
        )

    # Nuevo punto
    ax.scatter(
        [sepal_length], [sepal_width], [petal_length],
        c="red", s=220, marker="*", label="New Sample",
        edgecolors="darkred", linewidth=1.2, zorder=5
    )

    ax.set_xlabel("Sepal Length (cm)", fontsize=9, labelpad=8)
    ax.set_ylabel("Sepal Width (cm)",  fontsize=9, labelpad=8)
    ax.set_zlabel("Petal Length (cm)", fontsize=9, labelpad=8)
    ax.set_title("3D Scatter — New Sample vs Dataset",
                 fontsize=13, fontweight="bold", pad=15)
    ax.legend(title="Species", loc="upper left", fontsize=9)

    st.pyplot(fig)

# ────────────────────────────────────────────────────────────────────────────
# TAB 3: Data Exploration — Histogramas + Scatter Matrix
# ────────────────────────────────────────────────────────────────────────────
with tab3:
    # ── Histogramas de distribución por feature ──────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    feat_labels = ["Sepal Length (cm)", "Sepal Width (cm)",
                   "Petal Length (cm)", "Petal Width (cm)"]

    for idx, (col, label) in enumerate(zip(feature_cols, feat_labels)):
        for i, sp in enumerate(species_filter):
            sub = df_f[df_f["Species"] == sp]
            axes[idx].hist(
                sub[col], bins=15, alpha=0.65,
                color=colors[i % len(colors)],
                label=sp.replace("Iris-", ""),
                edgecolor="white"
            )
        axes[idx].set_title(f"Distribution: {label}", fontsize=11, fontweight="bold")
        axes[idx].set_xlabel(label, fontsize=10)
        axes[idx].set_ylabel("Frequency", fontsize=10)
        axes[idx].legend(title="Species", fontsize=8)
        axes[idx].grid(show_grid, alpha=0.4)

    plt.suptitle("Feature Distributions by Species", fontsize=14,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    st.pyplot(fig)

    st.divider()

    # ── Scatter Matrix ───────────────────────────────────────────────────────
    st.markdown("#### 🔵 Feature Scatter Matrix")

    n = len(feature_cols)
    short_feat = ["Sep.L", "Sep.W", "Pet.L", "Pet.W"]

    fig2, axes2 = plt.subplots(n, n, figsize=(13, 11))

    for r in range(n):
        for c_idx in range(n):
            ax = axes2[r][c_idx]
            if r == c_idx:
                for i, sp in enumerate(species_filter):
                    sub = df_f[df_f["Species"] == sp]
                    ax.hist(sub[feature_cols[r]], bins=12, alpha=0.65,
                            color=colors[i % len(colors)], edgecolor="white")
            else:
                for i, sp in enumerate(species_filter):
                    sub = df_f[df_f["Species"] == sp]
                    ax.scatter(sub[feature_cols[c_idx]], sub[feature_cols[r]],
                               color=colors[i % len(colors)], alpha=0.5, s=12)
            ax.tick_params(labelsize=6)
            if r == n - 1:
                ax.set_xlabel(short_feat[c_idx], fontsize=8)
            if c_idx == 0:
                ax.set_ylabel(short_feat[r], fontsize=8)

    plt.suptitle("Scatter Matrix — Iris Features", fontsize=13,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    st.pyplot(fig2)

# ────────────────────────────────────────────────────────────────────────────
# TAB 4: Raw data
# ────────────────────────────────────────────────────────────────────────────
with tab4:
    st.write(f"Showing **{len(df_f)}** records matching the current filters.")
    st.dataframe(df_f.reset_index(drop=True), use_container_width=True)
    st.download_button(
        label="⬇️ Download filtered data as CSV",
        data=df_f.to_csv(index=False).encode("utf-8"),
        file_name="iris_filtered_data.csv",
        mime="text/csv"
    )

st.divider()
st.caption("Universidad de la Costa — Data Mining | Final Project: Iris Species Classification")
