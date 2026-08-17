# Train the 5 models, dump pickle files, and write test_data.csv

import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "phishing-websites-data" / "Phishing_Websites_Data.csv"
MODEL_DIR = PROJECT_ROOT / "model"
RESULTS_DIR = PROJECT_ROOT / "results"
TEST_DATA_PATH = PROJECT_ROOT / "test_data.csv"

TARGET_COLUMN = "Result"
RANDOM_STATE = 42
TEST_SIZE = 0.20
POSITIVE_LABEL = 1

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree Classifier": "decision_tree.pkl",
    "K-Nearest Neighbors": "knn.pkl",
    "Gaussian Naive Bayes": "gaussian_nb.pkl",
    "Random Forest": "random_forest.pkl",
}


def load_and_clean_data(csv_path):
    df = pd.read_csv(csv_path)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' was not found in the dataset.")

    feature_columns = [column for column in df.columns if column != TARGET_COLUMN]
    original_rows = len(df)

    exact_duplicates = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)

    # same features, different labels -> drop these so they dont leak in split
    conflicting_mask = df.duplicated(subset=feature_columns, keep=False)
    conflicting_rows = int(conflicting_mask.sum())
    df = df.loc[~conflicting_mask].reset_index(drop=True)

    print(f"Loaded rows              : {original_rows}")
    print(f"Exact duplicate rows     : {exact_duplicates}")
    print(f"Conflicting-label rows   : {conflicting_rows}")
    print(f"Rows after cleaning      : {len(df)}")
    print(f"Features                 : {len(feature_columns)}")

    return df, feature_columns


def build_models():
    return {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Decision Tree Classifier": DecisionTreeClassifier(
            random_state=RANDOM_STATE,
        ),
        "K-Nearest Neighbors": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    KNeighborsClassifier(n_neighbors=5),
                ),
            ]
        ),
        "Gaussian Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def positive_class_probability(model, X):
    probabilities = model.predict_proba(X)
    class_index = list(model.classes_).index(POSITIVE_LABEL)
    return probabilities[:, class_index]


def evaluate_predictions(y_true, y_pred, y_score):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_score),
        "Precision": precision_score(
            y_true,
            y_pred,
            pos_label=POSITIVE_LABEL,
            zero_division=0,
        ),
        "Recall": recall_score(
            y_true,
            y_pred,
            pos_label=POSITIVE_LABEL,
            zero_division=0,
        ),
        "F1 Score": f1_score(
            y_true,
            y_pred,
            pos_label=POSITIVE_LABEL,
            zero_division=0,
        ),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def main():
    print("=" * 70)
    print("PHISHING WEBSITE MODEL TRAINING")
    print("=" * 70)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df, feature_columns = load_and_clean_data(DATA_PATH)

    X = df[feature_columns]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Training samples         : {len(X_train)}")
    print(f"Test samples             : {len(X_test)}")
    print(f"Train class counts       : {y_train.value_counts().to_dict()}")
    print(f"Test class counts        : {y_test.value_counts().to_dict()}")

    test_data = X_test.copy()
    test_data[TARGET_COLUMN] = y_test.values
    test_data.to_csv(TEST_DATA_PATH, index=False)
    print(f"\nSaved held-out test data : {TEST_DATA_PATH}")

    models = build_models()
    comparison_rows = []

    for model_name, model in models.items():
        print("\n" + "-" * 70)
        print(f"Training: {model_name}")

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        scores = positive_class_probability(model, X_test)
        metrics = evaluate_predictions(y_test, predictions, scores)

        model_path = MODEL_DIR / MODEL_FILES[model_name]
        with open(model_path, "wb") as file:
            pickle.dump(model, file)

        row = {"Model": model_name, **metrics}
        comparison_rows.append(row)

        print(f"Saved model              : {model_path}")
        for metric_name, value in metrics.items():
            print(f"{metric_name:<12}: {value:.4f}")

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_path = RESULTS_DIR / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    metadata = {
        "feature_columns": feature_columns,
        "target_column": TARGET_COLUMN,
        "positive_label": POSITIVE_LABEL,
        "model_files": MODEL_FILES,
        "class_labels": {
            -1: "Phishing",
            1: "Legitimate",
        },
    }
    with open(MODEL_DIR / "metadata.pkl", "wb") as file:
        pickle.dump(metadata, file)

    print("\n" + "=" * 70)
    print("FINAL MODEL COMPARISON")
    print("=" * 70)
    print(
        comparison_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )
    print(f"\nComparison saved to      : {comparison_path}")
    print("Training complete.")


if __name__ == "__main__":
    main()
