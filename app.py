"""
Streamlit App – Cricket Match Winner Prediction
Supports all 6 classification models with evaluation metrics & confusion matrix.
Models are trained in-memory on the dataset (no .pkl dependency).
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cricket Match Winner Prediction",
    page_icon="🏏",
    layout="wide",
)

st.title("🏏 Cricket Match Winner Prediction")
st.markdown("Predict T20 match outcomes using **6 Machine Learning models** trained on the Match dataset.")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(BASE_DIR, "Match_dataset.csv")

MODEL_NAMES = [
    "Logistic Regression",
    "Decision Tree",
    "KNN",
    "Naive Bayes",
    "Random Forest (Ensemble)",
    "XGBoost (Ensemble)",
]


# ── Helper: Feature Engineering ──────────────────────────────────────────────
def engineer_features(df: pd.DataFrame):
    """Apply feature engineering and return X (scaled), y, and feature column names."""
    df = df.copy()

    df["Ranking_Diff"] = df["Team_A_Ranking"] - df["Team_B_Ranking"]
    df["Form_Diff"] = df["Team_A_Form"] - df["Team_B_Form"]
    df["Tech_Diff"] = df["Team_A_Tech_Index"] - df["Team_B_Tech_Index"]
    df["H2H_Diff"] = df["HeadToHead_A_Wins"] - df["HeadToHead_B_Wins"]
    df["Team_A_Won_Toss"] = (df["Toss_Winner"] == "Team_A").astype(int)
    df["Toss_Bat"] = (df["Toss_Decision"] == "Bat").astype(int)

    le_pitch = LabelEncoder()
    df["Pitch_Type_Enc"] = le_pitch.fit_transform(df["Pitch_Type"])
    le_stage = LabelEncoder()
    df["Stage_Enc"] = le_stage.fit_transform(df["Stage"])

    feature_cols = [
        "Team_A_Ranking", "Team_B_Ranking", "Team_A_Form", "Team_B_Form",
        "HeadToHead_A_Wins", "HeadToHead_B_Wins", "Venue_HomeAdvantage_A",
        "Venue_HomeAdvantage_B", "Avg_T20_Score_Venue", "Team_A_Tech_Index",
        "Team_B_Tech_Index", "Match_Total", "Ranking_Diff", "Form_Diff",
        "Tech_Diff", "H2H_Diff", "Team_A_Won_Toss", "Toss_Bat",
        "Pitch_Type_Enc", "Stage_Enc",
    ]

    X = df[feature_cols].values

    y = None
    if "Winner" in df.columns:
        le_target = LabelEncoder()
        y = le_target.fit_transform(df["Winner"])

    return X, y


def _build_model(name: str):
    """Return an untrained sklearn/xgb estimator for the given model name."""
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=1000, random_state=42)
    elif name == "Decision Tree":
        return DecisionTreeClassifier(max_depth=10, random_state=42)
    elif name == "KNN":
        return KNeighborsClassifier(n_neighbors=7)
    elif name == "Naive Bayes":
        return GaussianNB()
    elif name == "Random Forest (Ensemble)":
        return RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
    elif name == "XGBoost (Ensemble)":
        return XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            use_label_encoder=False, eval_metric="logloss", random_state=42,
        )


@st.cache_data
def train_and_evaluate(_X_train, _X_test, _y_train, _y_test, model_names):
    """Train each model, evaluate on test set, return results dict."""
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(_X_train)
    X_te = scaler.transform(_X_test)

    results = {}
    for name in model_names:
        model = _build_model(name)
        model.fit(X_tr, _y_train)
        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)[:, 1]

        results[name] = {
            "Accuracy": round(accuracy_score(_y_test, y_pred), 4),
            "AUC": round(roc_auc_score(_y_test, y_prob), 4),
            "Precision": round(precision_score(_y_test, y_pred), 4),
            "Recall": round(recall_score(_y_test, y_pred), 4),
            "F1": round(f1_score(_y_test, y_pred), 4),
            "MCC": round(matthews_corrcoef(_y_test, y_pred), 4),
            "y_pred": y_pred.tolist(),
            "y_prob": y_prob.tolist(),
        }
    return results


# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")

# a. Dataset upload (CSV)  [1 mark]
uploaded_file = st.sidebar.file_uploader(
    "📂 Upload CSV (same schema as Match_dataset.csv)",
    type=["csv"],
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success(f"Uploaded: {uploaded_file.name} ({len(df)} rows)")
else:
    df = pd.read_csv(DEFAULT_DATA)
    st.sidebar.info("Using default Match_dataset.csv")

# b. Model selection dropdown (multiple models)  [1 mark]
selected_models = st.sidebar.multiselect(
    "🤖 Select model(s) to evaluate",
    MODEL_NAMES,
    default=MODEL_NAMES,
)

st.sidebar.markdown("---")
st.sidebar.markdown("Built for the ML Classification Assignment")

# ── Data Preview ─────────────────────────────────────────────────────────────
with st.expander("📊 Dataset Preview", expanded=False):
    st.dataframe(df.head(20), use_container_width=True)
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    if "Winner" in df.columns:
        col3.metric("Classes", df["Winner"].nunique())

# ── Feature Engineering ──────────────────────────────────────────────────────
X, y = engineer_features(df)

if y is None:
    st.warning("The uploaded CSV does not contain a 'Winner' column. Metrics cannot be computed.")
    st.stop()

# ── Train-Test Split & Evaluate ──────────────────────────────────────────────
if not selected_models:
    st.info("Please select at least one model from the sidebar.")
    st.stop()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

with st.spinner("Training models..."):
    results = train_and_evaluate(X_train, X_test, y_train, y_test, tuple(selected_models))

# Convert lists back to arrays for metric computation below
for m in results.values():
    m["y_pred"] = np.array(m["y_pred"])
    m["y_prob"] = np.array(m["y_prob"])

# ── c. Display Evaluation Metrics  [1 mark] ─────────────────────────────────
st.header("📈 Evaluation Metrics")

comparison_rows = []
for name, m in results.items():
    comparison_rows.append({
        "Model": name,
        "Accuracy": m["Accuracy"],
        "AUC": m["AUC"],
        "Precision": m["Precision"],
        "Recall": m["Recall"],
        "F1": m["F1"],
        "MCC": m["MCC"],
    })

comparison_df = pd.DataFrame(comparison_rows).set_index("Model")
st.dataframe(comparison_df.style.highlight_max(axis=0, color="#c6efce"), use_container_width=True)

# Bar chart comparison
fig_bar, ax_bar = plt.subplots(figsize=(10, 4))
comparison_df.plot(kind="bar", ax=ax_bar)
ax_bar.set_ylabel("Score")
ax_bar.set_title("Model Comparison")
ax_bar.set_ylim(0, 1.05)
ax_bar.legend(loc="lower right", fontsize=8)
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
st.pyplot(fig_bar)

# ── d. Confusion Matrix / Classification Report  [1 mark] ───────────────────
st.header("🔍 Confusion Matrix & Classification Report")

tabs = st.tabs(list(results.keys()))
target_names = ["Team_A", "Team_B"]

for tab, (model_name, m) in zip(tabs, results.items()):
    with tab:
        col_cm, col_cr = st.columns(2)

        with col_cm:
            cm = confusion_matrix(y_test, m["y_pred"])
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=target_names, yticklabels=target_names, ax=ax,
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(f"Confusion Matrix – {model_name}")
            plt.tight_layout()
            st.pyplot(fig)

        with col_cr:
            report = classification_report(
                y_test, m["y_pred"], target_names=target_names, output_dict=True
            )
            st.subheader("Classification Report")
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)

st.markdown("---")
st.caption("Cricket Match Winner Prediction – ML Classification Assignment")
