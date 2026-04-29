"""
Brake Failure Prediction — Streamlit App
"""

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Brake Failure Predictor",
    page_icon="🚗",
    layout="wide",
)

st.title("🚗 Brake Failure Prediction")
st.markdown("ML-powered brake safety analysis using vehicle sensor data.")

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    df = pd.read_csv('brake_failure_dataset_custom.csv', sep='\t')
    df.columns = df.columns.str.strip()
    return df

df = load_data()

with st.expander("📊 Dataset Overview", expanded=False):
    st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    st.dataframe(df.head())

    counts = df['Brake_Failure'].value_counts()
    dist_df = pd.DataFrame({
        'Class': ['Good Condition', 'Brake Failure'],
        'Count': [counts.get(0, 0), counts.get(1, 0)],
        'Percentage': [
            f"{counts.get(0,0)/len(df)*100:.1f}%",
            f"{counts.get(1,0)/len(df)*100:.1f}%",
        ]
    })
    st.table(dist_df)

# ── 2. PREPROCESSING ──────────────────────────────────────────────────────────

@st.cache_resource
def train_models(df):
    le = LabelEncoder()
    df = df.copy()
    df['Road_Condition'] = le.fit_transform(df['Road_Condition'])

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
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)),
        ]),
    }

    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
                use_label_encoder=False, eval_metric='logloss', random_state=42
            )),
        ])
    except ImportError:
        pass

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    best_model_name = None
    best_f1 = 0.0

    for name, pipeline in models.items():
        cv_f1 = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1_weighted').mean()
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)
        report = classification_report(
            y_test, y_pred,
            target_names=["Good Condition", "Brake Fail"],
            output_dict=True
        )
        test_f1 = report['weighted avg']['f1-score']
        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
            "pipeline": pipeline,
            "cv_f1": cv_f1,
            "test_acc": test_acc,
            "test_f1": test_f1,
            "y_pred": y_pred,
            "report": report,
            "cm": cm,
        }

        if test_f1 > best_f1:
            best_f1 = test_f1
            best_model_name = name

    return results, best_model_name, FEATURES, X_test, y_test

with st.spinner("Training models..."):
    results, best_model_name, FEATURES, X_test, y_test = train_models(df)

best_pipeline = results[best_model_name]["pipeline"]

# ── 3. MODEL COMPARISON ───────────────────────────────────────────────────────

st.subheader("📈 Model Comparison")

comparison_data = []
for name, r in results.items():
    comparison_data.append({
        "Model": name,
        "CV F1 (weighted)": f"{r['cv_f1']:.3f}",
        "Test Accuracy": f"{r['test_acc']:.3f}",
        "Test F1": f"{r['test_f1']:.3f}",
        "Best": "⭐" if name == best_model_name else "",
    })
st.table(pd.DataFrame(comparison_data))

st.success(f"**Best Model:** {best_model_name}  |  Test F1: {results[best_model_name]['test_f1']:.3f}")

# ── 4. BEST MODEL DETAILS ─────────────────────────────────────────────────────

with st.expander(f"🔍 {best_model_name} — Detailed Report", expanded=False):
    report = results[best_model_name]["report"]
    cm = results[best_model_name]["cm"]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Classification Report**")
        report_df = pd.DataFrame(report).transpose().round(3)
        st.dataframe(report_df)

    with col2:
        st.markdown("**Confusion Matrix**")
        cm_df = pd.DataFrame(
            cm,
            index=["Actual: Good", "Actual: Failure"],
            columns=["Predicted: Good", "Predicted: Failure"]
        )
        st.dataframe(cm_df)

    # Feature importances
    clf = best_pipeline.named_steps['clf']
    if hasattr(clf, 'feature_importances_'):
        st.markdown("**Feature Importances**")
        importances = clf.feature_importances_
        imp_df = pd.DataFrame({
            'Feature': FEATURES,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        st.bar_chart(imp_df.set_index('Feature'))

# ── 5. PREDICTION INTERFACE ───────────────────────────────────────────────────

st.subheader("🔮 Predict Brake Failure Risk")
st.markdown("Enter vehicle parameters to get a real-time brake failure prediction.")

col1, col2, col3 = st.columns(3)

with col1:
    speed = st.slider("Vehicle Speed (km/h)", min_value=0, max_value=200, value=80)
    temperature = st.slider("Brake Temperature (°C)", min_value=0, max_value=400, value=100)

with col2:
    usage_freq = st.slider("Brake Usage Frequency", min_value=0, max_value=50, value=10)
    pad_thickness = st.slider("Brake Pad Thickness (mm)", min_value=1, max_value=30, value=10)

with col3:
    load_label = st.selectbox("Vehicle Load", options=["Empty (0)", "Half (1)", "Full (2)"])
    load = int(load_label.split("(")[1].replace(")", ""))
    road_label = st.selectbox("Road Condition", options=["Dry (0)", "Wet (1)"])
    road_condition = int(road_label.split("(")[1].replace(")", ""))

if st.button("🔍 Predict", use_container_width=True):
    new_data = pd.DataFrame([{
        'Vehicle_Speed': speed,
        'Brake_Temperature': temperature,
        'Brake_Usage_Frequency': usage_freq,
        'Vehicle_Load': load,
        'Road_Condition': road_condition,
        'Brake_Pad_Thickness': pad_thickness,
    }])

    prediction = best_pipeline.predict(new_data)[0]
    probability = best_pipeline.predict_proba(new_data)[0][1]

    st.divider()
    res_col1, res_col2 = st.columns(2)

    with res_col1:
        if prediction == 1:
            st.error(f"### ⚠️ BRAKE FAILURE RISK DETECTED")
        else:
            st.success(f"### ✅ Good Condition")

    with res_col2:
        st.metric("Failure Probability", f"{probability:.1%}")
        st.progress(float(probability))

# ── 6. EXAMPLE SCENARIOS ──────────────────────────────────────────────────────

with st.expander("🧪 Run Example Scenarios", expanded=False):
    test_cases = [
        dict(speed=120, temperature=133, usage_freq=11, load=2, road_condition=1, pad_thickness=18,
             label="High-speed wet road, heavy load"),
        dict(speed=40,  temperature=60,  usage_freq=5,  load=0, road_condition=0, pad_thickness=12,
             label="City driving, dry road, light load"),
        dict(speed=80,  temperature=200, usage_freq=25, load=2, road_condition=1, pad_thickness=3,
             label="Highway, overheated brakes, thin pads"),
    ]

    for tc in test_cases:
        label = tc.pop("label")
        new_data = pd.DataFrame([{
            'Vehicle_Speed': tc['speed'],
            'Brake_Temperature': tc['temperature'],
            'Brake_Usage_Frequency': tc['usage_freq'],
            'Vehicle_Load': tc['load'],
            'Road_Condition': tc['road_condition'],
            'Brake_Pad_Thickness': tc['pad_thickness'],
        }])
        pred = best_pipeline.predict(new_data)[0]
        prob = best_pipeline.predict_proba(new_data)[0][1]
        status = "⚠️ BRAKE FAILURE RISK" if pred == 1 else "✅ Good Condition"

        st.markdown(f"**{label}**")
        cols = st.columns([3, 1])
        cols[0].write(status)
        cols[1].write(f"P(Fail): {prob:.1%}")
        st.divider()
