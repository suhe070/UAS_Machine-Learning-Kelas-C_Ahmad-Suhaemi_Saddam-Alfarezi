"""
PREDIKSI EMPLOYEE ATTRITION MENGGUNAKAN MACHINE LEARNING
=========================================================
Perbandingan Logistic Regression, Random Forest, XGBoost, dan LightGBM
dengan Evaluasi Cost-Sensitive dan Interpretasi SHAP

Dataset : IBM HR Analytics Employee Attrition & Performance (1470 baris, 35 kolom)
Sumber  : Kaggle - pavansubhasht/ibm-hr-analytics-attrition-dataset (dataset publik, dibuat IBM)

Cara pakai:
    python3 attrition_pipeline.py
Semua output (metrik, plot) akan tersimpan di folder ./outputs/
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import shap

RANDOM_STATE = 42
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------------
df = pd.read_csv("hr_attrition.csv")
print("Ukuran dataset:", df.shape)

# ------------------------------------------------------------------
# 2. EDA RINGKAS
# ------------------------------------------------------------------
eda_report = {}
eda_report["shape"] = df.shape
eda_report["missing_values_total"] = int(df.isnull().sum().sum())
eda_report["target_distribution"] = df["Attrition"].value_counts().to_dict()
eda_report["attrition_rate_percent"] = round(
    (df["Attrition"] == "Yes").mean() * 100, 2
)

# Kolom konstan / tidak informatif (verifikasi manual dari data dictionary IBM)
constant_cols = [c for c in df.columns if df[c].nunique() == 1]
eda_report["constant_columns_dropped"] = constant_cols

# Distribusi target
plt.figure(figsize=(5, 4))
sns.countplot(data=df, x="Attrition", palette=["#4C72B0", "#DD8452"])
plt.title("Distribusi Target: Attrition")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/01_distribusi_target.png", dpi=150)
plt.close()

# Korelasi fitur numerik
plt.figure(figsize=(12, 10))
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()
sns.heatmap(corr, cmap="coolwarm", center=0, linewidths=0.3)
plt.title("Correlation Heatmap - Fitur Numerik")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/02_correlation_heatmap.png", dpi=150)
plt.close()

# ------------------------------------------------------------------
# 3. PREPROCESSING
# ------------------------------------------------------------------
data = df.drop(columns=constant_cols + ["EmployeeNumber"], errors="ignore")

target = "Attrition"
y = data[target].map({"Yes": 1, "No": 0})
X = data.drop(columns=[target])

cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(exclude="object").columns.tolist()

# One-hot encoding untuk kategorikal
X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)

# Split train-test (stratified, karena target imbalance ~16% churn)
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Scaling untuk model berbasis jarak/linear (Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

eda_report["train_size"] = X_train.shape[0]
eda_report["test_size"] = X_test.shape[0]
eda_report["n_features_after_encoding"] = X_encoded.shape[1]

# ------------------------------------------------------------------
# 4. MODELING - 4 ALGORITMA PEMBANDING
# ------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        eval_metric="logloss", random_state=RANDOM_STATE
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        class_weight="balanced", random_state=RANDOM_STATE, verbose=-1
    ),
}

results = []
roc_curves = {}
trained_models = {}

for name, model in models.items():
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

    trained_models[name] = model

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    # Cost-sensitive framing:
    # Asumsi biaya kehilangan karyawan (False Negative) = 5x biaya intervensi
    # retensi yang salah sasaran (False Positive). Rasio ini bisa disesuaikan
    # HR pengguna sesuai data biaya rekrutmen riil perusahaan.
    tn, fp, fn, tp = cm.ravel()
    cost_fn_unit = 5
    cost_fp_unit = 1
    total_cost = fn * cost_fn_unit + fp * cost_fp_unit

    results.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4),
        "ROC-AUC": round(auc, 4),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "Estimated_Cost_Units": int(total_cost),
    })

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_curves[name] = (fpr, tpr, auc)

results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False)
results_df.to_csv(f"{OUT_DIR}/model_comparison_results.csv", index=False)
print("\n=== HASIL PERBANDINGAN MODEL ===")
print(results_df.to_string(index=False))

# ------------------------------------------------------------------
# 5. VISUALISASI: ROC CURVE GABUNGAN
# ------------------------------------------------------------------
plt.figure(figsize=(7, 6))
for name, (fpr, tpr, auc) in roc_curves.items():
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Perbandingan ROC Curve - 4 Algoritma")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/03_roc_curve_comparison.png", dpi=150)
plt.close()

# ------------------------------------------------------------------
# 6. CONFUSION MATRIX - MODEL TERBAIK (berdasarkan ROC-AUC)
# ------------------------------------------------------------------
best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]

if best_model_name == "Logistic Regression":
    y_pred_best = best_model.predict(X_test_scaled)
else:
    y_pred_best = best_model.predict(X_test)

cm_best = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(5, 4))
sns.heatmap(cm_best, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Attrition", "Attrition"],
            yticklabels=["No Attrition", "Attrition"])
plt.title(f"Confusion Matrix - {best_model_name} (Model Terbaik)")
plt.ylabel("Aktual")
plt.xlabel("Prediksi")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/04_confusion_matrix_best_model.png", dpi=150)
plt.close()

# ------------------------------------------------------------------
# 7. INTERPRETABILITY - SHAP (untuk model terbaik, non-linear)
# ------------------------------------------------------------------
shap_model_name = best_model_name if best_model_name != "Logistic Regression" else "XGBoost"
shap_model = trained_models[shap_model_name]

explainer = shap.TreeExplainer(shap_model)
shap_values = explainer.shap_values(X_test)

plt.figure()
shap.summary_plot(shap_values, X_test, show=False, max_display=15)
plt.title(f"SHAP Summary Plot - {shap_model_name}")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/05_shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()

# Feature importance dari SHAP (mean |SHAP value|)
shap_importance = pd.DataFrame({
    "feature": X_test.columns,
    "mean_abs_shap": np.abs(shap_values).mean(axis=0)
}).sort_values("mean_abs_shap", ascending=False)
shap_importance.to_csv(f"{OUT_DIR}/shap_feature_importance.csv", index=False)

plt.figure(figsize=(8, 6))
top15 = shap_importance.head(15)
plt.barh(top15["feature"][::-1], top15["mean_abs_shap"][::-1], color="#4C72B0")
plt.xlabel("Mean |SHAP value|")
plt.title(f"Top 15 Fitur Paling Berpengaruh - {shap_model_name}")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/06_shap_top_features_bar.png", dpi=150)
plt.close()

# ------------------------------------------------------------------
# 8. SIMPAN RINGKASAN LENGKAP UNTUK PENULISAN PAPER
# ------------------------------------------------------------------
summary = {
    "eda": eda_report,
    "best_model": best_model_name,
    "shap_model_used": shap_model_name,
    "top_5_features_shap": shap_importance.head(5).to_dict(orient="records"),
    "model_results": results_df.to_dict(orient="records"),
}
with open(f"{OUT_DIR}/summary_for_paper.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n=== SELESAI. Semua output tersimpan di folder outputs/ ===")
print(f"Model terbaik berdasarkan ROC-AUC: {best_model_name}")
print(f"Top 5 fitur paling berpengaruh (SHAP): {[d['feature'] for d in shap_importance.head(5).to_dict(orient='records')]}")
