import os, json, joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

DATA = "FINAL_READY_DATASET.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Keep the same feature set used by the original project/UI.
FEATURES = [
    "Water_Score", "Temp_Score", "Soil_Score",
    "Avg_Rainfall_mm", "Production_MT",
    "District_Temp_Min", "District_Temp_Max",
    "Water_Min", "Water_Max"
]

df = pd.read_csv(DATA)

def make_target(score):
    if score < 0.40:
        return "Low"
    elif score < 0.70:
        return "Moderate"
    return "Good"

df["Target"] = df["Suitability_Score"].apply(make_target)

def make_models(seed=42):
    rf = RandomForestClassifier(
        n_estimators=150, max_depth=8, class_weight="balanced",
        random_state=seed, n_jobs=1
    )
    xgb = XGBClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85,
        eval_metric="mlogloss",
        random_state=seed, n_jobs=1
    )
    return rf, xgb

# Global models, used only when a district is too small to validate safely.
Xg = df[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)
enc_global = LabelEncoder()
yg = enc_global.fit_transform(df["Target"])
rf_global, xgb_global = make_models()
rf_global.fit(Xg, yg)
xgb_global.fit(Xg, yg)

min_global = int(pd.Series(yg).value_counts().min())
if min_global >= 2:
    folds = min(5, min_global)
    cvg = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    rf_global_acc = float(cross_val_score(make_models()[0], Xg, yg, cv=cvg, scoring="accuracy", n_jobs=1).mean())
    xgb_global_acc = float(cross_val_score(make_models()[1], Xg, yg, cv=cvg, scoring="accuracy", n_jobs=1).mean())
else:
    rf_global_acc = float(accuracy_score(yg, rf_global.predict(Xg)))
    xgb_global_acc = float(accuracy_score(yg, xgb_global.predict(Xg)))

registry = {
    "DEFAULT_MODEL": "XGBoost",
    "features": FEATURES,
    "global_random_forest_accuracy": round(rf_global_acc, 4),
    "global_xgboost_accuracy": round(xgb_global_acc, 4),
    "districts": {}
}
results = []

for district, g in df.groupby("District"):
    g = g.copy()
    X = g[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)
    enc = LabelEncoder()
    y = enc.fit_transform(g["Target"])
    classes = np.unique(y)
    counts = pd.Series(y).value_counts()
    min_class = int(counts.min())
    rf, xgb = make_models()

    if len(classes) >= 2 and min_class >= 2 and len(g) >= 10:
        folds = min(5, min_class)
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        rf_acc = float(cross_val_score(rf, X, y, cv=cv, scoring="accuracy", n_jobs=1).mean())
        xgb_acc = float(cross_val_score(xgb, X, y, cv=cv, scoring="accuracy", n_jobs=1).mean())
        method = f"{folds}-fold stratified CV"
        rf.fit(X, y)
        xgb.fit(X, y)
        rf_bundle = {"model": rf, "encoder": enc, "features": FEATURES}
        xgb_bundle = {"model": xgb, "encoder": enc, "features": FEATURES}
    elif len(classes) >= 2 and min_class >= 2 and len(g) >= 6:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
        rf.fit(Xtr, ytr)
        xgb.fit(Xtr, ytr)
        rf_acc = float(accuracy_score(yte, rf.predict(Xte)))
        xgb_acc = float(accuracy_score(yte, xgb.predict(Xte)))
        method = "stratified 70/30 holdout"
        rf.fit(X, y)
        xgb.fit(X, y)
        rf_bundle = {"model": rf, "encoder": enc, "features": FEATURES}
        xgb_bundle = {"model": xgb, "encoder": enc, "features": FEATURES}
    else:
        rf_acc = rf_global_acc
        xgb_acc = xgb_global_acc
        method = "global model comparison (district has insufficient class diversity/data)"
        rf_bundle = {"model": rf_global, "encoder": enc_global, "features": FEATURES}
        xgb_bundle = {"model": xgb_global, "encoder": enc_global, "features": FEATURES}

    winner = "Random Forest" if rf_acc >= xgb_acc else "XGBoost"
    safe = "".join(c if c.isalnum() or c in " _-" else "_" for c in str(district)).strip()
    rf_file = f"{safe}_random_forest.pkl"
    xgb_file = f"{safe}_xgboost.pkl"
    joblib.dump(rf_bundle, os.path.join(MODEL_DIR, rf_file))
    joblib.dump(xgb_bundle, os.path.join(MODEL_DIR, xgb_file))

    registry["districts"][str(district)] = {
        "model": winner,
        "random_forest_accuracy": round(rf_acc, 4),
        "xgboost_accuracy": round(xgb_acc, 4),
        "evaluation_method": method,
        "random_forest_file": f"models/{rf_file}",
        "xgboost_file": f"models/{xgb_file}",
        "classes": [str(c) for c in enc.classes_]
    }
    results.append({
        "District": district,
        "Random Forest Accuracy": round(rf_acc, 4),
        "XGBoost Accuracy": round(xgb_acc, 4),
        "Selected Model": winner,
        "Evaluation Method": method,
        "Classes": len(classes),
        "Samples": len(g)
    })

out = pd.DataFrame(results).sort_values("District")
out.to_csv("model_results.csv", index=False)
with open("district_best_models.json", "w", encoding="utf-8") as f:
    json.dump(registry, f, indent=2)

print(out.to_string(index=False))
print("\nFeatures used:", FEATURES)
print(f"Global RF: {rf_global_acc:.4f}")
print(f"Global XGBoost: {xgb_global_acc:.4f}")
print("\nDone. Both models use the original 9 project features and the higher validation accuracy is selected per district.")
