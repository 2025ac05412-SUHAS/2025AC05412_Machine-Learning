# Streamlit app - loads saved pkl models, no retraining

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from train_models import (
    MODEL_FILES,
    TARGET_COLUMN,
    evaluate_predictions,
    positive_class_probability,
)

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "model"
RESULTS_DIR = PROJECT_ROOT / "results"
METADATA_PATH = MODEL_DIR / "metadata.pkl"
COMPARISON_PATH = RESULTS_DIR / "model_comparison.csv"


@st.cache_resource
def load_metadata():
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            "Saved model metadata was not found. Run `python train_models.py` first."
        )
    with open(METADATA_PATH, "rb") as file:
        return pickle.load(file)


@st.cache_resource
def load_models():
    models = {}
    missing = []

    for model_name, filename in MODEL_FILES.items():
        model_path = MODEL_DIR / filename
        if not model_path.exists():
            missing.append(str(model_path))
            continue
        with open(model_path, "rb") as file:
            models[model_name] = pickle.load(file)

    if missing:
        raise FileNotFoundError(
            "Missing saved model files. Run `python train_models.py` first.\n"
            + "\n".join(missing)
        )

    return models


def load_training_comparison():
    if not COMPARISON_PATH.exists():
        return None
    return pd.read_csv(COMPARISON_PATH)


def validate_uploaded_csv(df, feature_columns):
    missing_features = [column for column in feature_columns if column not in df.columns]
    if missing_features:
        return (
            "The uploaded CSV is missing required feature columns:\n\n"
            + ", ".join(missing_features)
        )

    if TARGET_COLUMN not in df.columns:
        return (
            f"The uploaded CSV must include the target column '{TARGET_COLUMN}' "
            "so evaluation metrics can be calculated."
        )

    if df.empty:
        return "The uploaded CSV does not contain any rows."

    feature_data = df[feature_columns]
    if feature_data.isnull().any().any():
        return "The uploaded CSV contains missing values in the required feature columns."

    try:
        feature_data.apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError):
        return "All required feature columns must be numeric."

    if df[TARGET_COLUMN].isnull().any():
        return f"The target column '{TARGET_COLUMN}' contains missing values."

    return None


def round_metrics(df):
    metric_columns = ["Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"]
    display_df = df.copy()
    for column in metric_columns:
        if column in display_df.columns:
            display_df[column] = display_df[column].map(lambda value: f"{value:.4f}")
    return display_df


def main():
    st.set_page_config(
        page_title="Phishing Website Detection",
        layout="centered",
    )

    st.title("Phishing Website Detection")
    st.write(
        "Upload the test csv and pick a model. The models are already trained "
        "and saved in the model folder."
    )

    try:
        metadata = load_metadata()
        models = load_models()
    except FileNotFoundError as error:
        st.error(str(error))
        return

    feature_columns = metadata["feature_columns"]
    training_comparison = load_training_comparison()

    uploaded_file = st.file_uploader("Upload test CSV", type=["csv"])
    selected_model_name = st.selectbox("Select model", list(MODEL_FILES.keys()))
    run_evaluation = st.button("Evaluate")

    if training_comparison is not None:
        st.subheader("Saved training comparison")
        st.dataframe(
            round_metrics(training_comparison),
            use_container_width=True,
            hide_index=True,
        )

    if not run_evaluation:
        return

    if uploaded_file is None:
        st.error("Please upload a CSV file before evaluation.")
        return

    try:
        uploaded_df = pd.read_csv(uploaded_file)
    except Exception:
        st.error("The uploaded file could not be read as a CSV.")
        return

    validation_error = validate_uploaded_csv(uploaded_df, feature_columns)
    if validation_error:
        st.error(validation_error)
        return

    X = uploaded_df[feature_columns].apply(pd.to_numeric, errors="raise")
    y_true = uploaded_df[TARGET_COLUMN]

    selected_model = models[selected_model_name]
    y_pred = selected_model.predict(X)
    y_score = positive_class_probability(selected_model, X)
    metrics = evaluate_predictions(y_true, y_pred, y_score)

    st.subheader(f"Results: {selected_model_name}")

    metric_columns = st.columns(6)
    for column, (metric_name, value) in zip(metric_columns, metrics.items()):
        column.metric(metric_name, f"{value:.4f}")

    st.subheader("Confusion matrix")
    labels = list(selected_model.classes_)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    class_names = [metadata["class_labels"].get(label, str(label)) for label in labels]
    confusion_df = pd.DataFrame(
        matrix,
        index=[f"Actual {name}" for name in class_names],
        columns=[f"Predicted {name}" for name in class_names],
    )
    st.dataframe(confusion_df, use_container_width=True)

    st.subheader("Classification report")
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df, use_container_width=True)

    st.subheader("Comparison on uploaded data")
    uploaded_comparison_rows = []
    for model_name, model in models.items():
        predictions = model.predict(X)
        scores = positive_class_probability(model, X)
        row_metrics = evaluate_predictions(y_true, predictions, scores)
        uploaded_comparison_rows.append({"Model": model_name, **row_metrics})

    uploaded_comparison = pd.DataFrame(uploaded_comparison_rows)
    st.dataframe(
        round_metrics(uploaded_comparison),
        use_container_width=True,
        hide_index=True,
    )


if __name__ == "__main__":
    main()
