"""
Brake Failure Prediction - Improved ML Pipeline
================================================
Improvements over original:
  - Feature scaling with StandardScaler
  - Class imbalance handled via class_weight='balanced'
  - Cross-validation for reliable accuracy
  - Multiple models compared (Logistic Regression, Random Forest, XGBoost)
  - Confusion matrix visualization
  - Cleaner prediction interface with input validation
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────

df = pd.read_csv('/content/brake_failure_dataset_custom.csv')

print("=" * 55)
print("  BRAKE FAILURE PREDICTION — ML PIPELINE")
print("=" * 55)
print(f"\nDataset shape : {df.shape[0]} rows × {df.shape[1]} columns")
print("\nFirst 5 rows:")
print(df.head())

print("\nClass distribution:")
counts = df['Brake_Failure'].value_counts()
for label, count in counts.items():
    name = "Brake Fail" if label == 1 else "Good Condition"
    pct  = count / len(df) * 100
    print(f"  {name:>15} ({label}): {count} samples ({pct:.1f}%)")

# ── 2. PREPROCESSING ──────────────────────────────────────────────────────────

le = LabelEncoder()
df['Road_Condition'] = le.fit_transform(df['Road_Condition'])   # Dry=0, Wet=1

FEATURES = [
    'Vehicle_Speed',
    'Brake_Temperature',
    'Brake_Usage_Frequency',
    'Vehicle_Load',
    'Road_Condition',
    'Brake_Pad_Thickness',
]

X = df[FEATURES]
y = df['Brake_Failure']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y   # stratify preserves class ratio
)

print(f"\nTrain set : {len(X_train)} samples")
print(f"Test  set : {len(X_test)} samples")

# ── 3. MODELS ─────────────────────────────────────────────────────────────────
# Each model wrapped in a Pipeline so scaling is applied correctly
# and cannot leak from train → test.

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)),
    ]),
    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)),
    ]),
}

# Optional XGBoost — only used if the library is installed
try:
    from xgboost import XGBClassifier
    models["XGBoost"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    XGBClassifier(scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
                                  use_label_encoder=False, eval_metric='logloss',
                                  random_state=42)),
    ])
except ImportError:
    print("\nXGBoost not installed — skipping. Run: pip install xgboost")

# ── 4. TRAIN & EVALUATE ALL MODELS ───────────────────────────────────────────

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_model_name = None
best_f1         = 0.0
results         = {}

print("\n" + "=" * 55)
print("  MODEL COMPARISON")
print("=" * 55)

for name, pipeline in models.items():
    # Cross-validated F1 (weighted) on training set
    cv_f1 = cross_val_score(pipeline, X_train, y_train,
                             cv=cv, scoring='f1_weighted').mean()

    # Fit on full training set, evaluate on held-out test set
    pipeline.fit(X_train, y_train)
    y_pred     = pipeline.predict(X_test)
    test_acc   = accuracy_score(y_test, y_pred)
    report     = classification_report(y_test, y_pred,
                                        target_names=["Good Condition", "Brake Fail"],
                                        output_dict=True)
    test_f1    = report['weighted avg']['f1-score']

    results[name] = {
        "pipeline" : pipeline,
        "cv_f1"    : cv_f1,
        "test_acc" : test_acc,
        "test_f1"  : test_f1,
        "y_pred"   : y_pred,
        "report"   : report,
    }

    print(f"\n▶ {name}")
    print(f"  CV F1 (weighted) : {cv_f1:.3f}")
    print(f"  Test Accuracy    : {test_acc:.3f}")
    print(f"  Test F1          : {test_f1:.3f}")
    print()
    print(classification_report(y_test, y_pred,
                                  target_names=["Good Condition", "Brake Fail"]))

    cm = confusion_matrix(y_test, y_pred)
    print("  Confusion Matrix:")
    print(f"    TP={cm[1,1]}  FP={cm[0,1]}")
    print(f"    FN={cm[1,0]}  TN={cm[0,0]}")

    if test_f1 > best_f1:
        best_f1         = test_f1
        best_model_name = name

# ── 5. BEST MODEL SUMMARY ─────────────────────────────────────────────────────

print("\n" + "=" * 55)
print(f"  BEST MODEL → {best_model_name}  (Test F1: {best_f1:.3f})")
print("=" * 55)

best_pipeline = results[best_model_name]["pipeline"]

# Feature importances (only for tree-based models)
if hasattr(best_pipeline.named_steps['clf'], 'feature_importances_'):
    importances = best_pipeline.named_steps['clf'].feature_importances_
    print("\nFeature Importances:")
    for feat, imp in sorted(zip(FEATURES, importances), key=lambda x: -x[1]):
        bar = "█" * int(imp * 40)
        print(f"  {feat:<28} {imp:.3f}  {bar}")

# ── 6. PREDICT NEW VEHICLE ────────────────────────────────────────────────────

def predict_brake_status(speed, temperature, usage_freq, load,
                          road_condition, pad_thickness,
                          pipeline=best_pipeline):
    """
    Predict brake failure risk for a single vehicle reading.

    Parameters
    ----------
    speed          : Vehicle speed (km/h)
    temperature    : Brake temperature (°C)
    usage_freq     : Brake usage frequency (counts)
    load           : Vehicle load (categorical: 0=empty, 1=half, 2=full)
    road_condition : 0 = Dry, 1 = Wet
    pad_thickness  : Brake pad thickness (mm)

    Returns
    -------
    dict with 'prediction' (int), 'label' (str), 'probability' (float)
    """
    # --- Input validation ---
    if road_condition not in (0, 1):
        raise ValueError("road_condition must be 0 (Dry) or 1 (Wet)")
    if load not in (0, 1, 2):
        raise ValueError("load must be 0 (empty), 1 (half), or 2 (full)")
    if pad_thickness <= 0:
        raise ValueError("pad_thickness must be positive")

    new_data = pd.DataFrame([{
        'Vehicle_Speed'         : speed,
        'Brake_Temperature'     : temperature,
        'Brake_Usage_Frequency' : usage_freq,
        'Vehicle_Load'          : load,
        'Road_Condition'        : road_condition,
        'Brake_Pad_Thickness'   : pad_thickness,
    }])

    prediction  = pipeline.predict(new_data)[0]
    probability = pipeline.predict_proba(new_data)[0][1]  # P(Brake Fail)

    return {
        "prediction"  : int(prediction),
        "label"       : "⚠️  BRAKE FAILURE RISK" if prediction == 1 else "✅ Good Condition",
        "probability" : round(float(probability), 3),
    }


# ── 7. EXAMPLE PREDICTIONS ────────────────────────────────────────────────────

print("\n" + "=" * 55)
print("  EXAMPLE PREDICTIONS")
print("=" * 55)

test_cases = [
    dict(speed=120, temperature=133, usage_freq=11, load=2,
         road_condition=1, pad_thickness=18,
         label="High-speed wet road, heavy load"),
    dict(speed=40,  temperature=60,  usage_freq=5,  load=0,
         road_condition=0, pad_thickness=12,
         label="City driving, dry road, light load"),
    dict(speed=80,  temperature=200, usage_freq=25, load=2,
         road_condition=1, pad_thickness=3,
         label="Highway, overheated brakes, thin pads"),
]

for tc in test_cases:
    label = tc.pop("label")
    result = predict_brake_status(**tc)
    print(f"\nScenario : {label}")
    print(f"  Result  : {result['label']}")
    print(f"  P(Fail) : {result['probability']:.1%}")
