"""Train the student success model and save it to model.joblib."""
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold, cross_val_score, train_test_split

df = pd.read_csv("augmented_balanced_dataset_m.csv")
FEATURES = [c for c in df.columns if c != "rating"]
X = df[FEATURES]
y = (df["rating"] == "High").astype(int)  # High = Pass (1), Low = Fail (0)

# The dataset is augmented and contains duplicate rows. Keep identical rows
# together in the same split so the test score is not inflated by leakage.
groups = X.astype(str).agg("|".join, axis=1).factorize()[0]

model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)

cv = cross_val_score(model, X, y, cv=GroupKFold(5), groups=groups)
print(f"Grouped 5-fold CV accuracy: {cv.mean():.3f} (+/- {cv.std():.3f})")

# Standard stratified 80/20 split (fixed seed). Because the data contains
# duplicate rows, this score is optimistic; the grouped CV above is stricter.
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
model.fit(Xtr, ytr)
pred = model.predict(Xte)
acc = accuracy_score(yte, pred)
print(f"Standard 80/20 test accuracy: {acc:.3f}")
print(confusion_matrix(yte, pred))
print(classification_report(yte, pred, target_names=["Fail (Low)", "Pass (High)"]))

# Final model on all data
model.fit(X, y)
importances = dict(zip(FEATURES, map(float, model.feature_importances_)))
medians = X.median().to_dict()
joblib.dump(model, "model.joblib")
json.dump(
    {"features": FEATURES, "cv_accuracy": float(cv.mean()), "holdout_accuracy": float(acc),
     "importances": importances, "medians": medians,
     "max": X.quantile(0.99).to_dict()},
    open("meta.json", "w"), indent=2)
print("Saved model.joblib and meta.json")
