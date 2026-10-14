
from pathlib import Path
import json

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)



def find_dataset():
    """Load the main HLD classification dataset."""
    path = DATA_DIR / "sample_sections.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    df = pd.read_csv(path)

    if not {"text", "label"}.issubset(df.columns):
        raise ValueError(
            "Dataset must contain 'text' and 'label' columns."
        )

    df = df[["text", "label"]].dropna().copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()

    df = df[
        (df["text"] != "") &
        (df["label"] != "")
    ]

    print(f"Dataset loaded: {path}")
    print(f"Rows: {len(df)}")
    print("Categories:")
    print(df["label"].value_counts().to_string())

    return df



   
def main():
    df = find_dataset()

    print("\nDataset size:", len(df))
    print("\nClass distribution:")
    print(df["label"].value_counts().to_string())

    # A stratified split needs enough samples in every class.
    counts = df["label"].value_counts()
    if counts.min() < 2:
        raise ValueError(
            "Every class needs at least 2 samples for a stratified split. "
            "Add more labeled examples before training."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.25,
        random_state=42,
        stratify=df["label"],
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced"
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=42, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
        ),
    }

    results = []
    reports = {}

    for name, classifier in models.items():
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                max_features=10000,
            )),
            ("classifier", classifier),
        ])

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        )

        accuracy = accuracy_score(y_test, predictions)

        results.append({
            "model": name,
            "accuracy": accuracy,
            "weighted_precision": precision,
            "weighted_recall": recall,
            "weighted_f1": f1,
        })

        reports[name] = {
            "classification_report": classification_report(
                y_test, predictions, zero_division=0
            ),
            "y_test": y_test.tolist(),
            "predictions": predictions.tolist(),
        }

        print(f"\n{name}")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Weighted precision: {precision:.3f}")
        print(f"Weighted recall: {recall:.3f}")
        print(f"Weighted F1-score: {f1:.3f}")
        print(classification_report(
            y_test, predictions, zero_division=0
        ))

        # Save each trained model for later integration.
        filename = name.lower().replace(" ", "_")
        joblib.dump(pipeline, MODEL_DIR / f"{filename}.joblib")

    results_df = pd.DataFrame(results)
    results_df.to_csv(REPORT_DIR / "model_comparison.csv", index=False)

    best_name = results_df.sort_values(
        ["weighted_f1", "accuracy"], ascending=False
    ).iloc[0]["model"]

    best = models[best_name]
    best_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2), max_features=10000
        )),
        ("classifier", best),
    ])
    best_pipeline.fit(X_train, y_train)
    joblib.dump(best_pipeline, MODEL_DIR / "section_classifier.joblib")

    best_predictions = best_pipeline.predict(X_test)
    report = classification_report(
        y_test, best_predictions, zero_division=0
    )
    (REPORT_DIR / "classification_report.txt").write_text(
        f"Selected model: {best_name}\n\n{report}",
        encoding="utf-8",
    )

    labels = sorted(df["label"].unique())
    cm = confusion_matrix(y_test, best_predictions, labels=labels)
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=labels
    )
    display.plot(xticks_rotation=45, values_format="d")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    summary = {
        "selected_model": best_name,
        "dataset_size": len(df),
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "results": results,
        "warning": (
            "Results depend on dataset size, class balance, and label quality. "
            "A small test set is not sufficient for reliable generalization claims."
        ),
    }
    (REPORT_DIR / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nModel comparison:")
    print(results_df.to_string(index=False))
    print(f"\nSelected model: {best_name}")
    print(f"Reports saved in: {REPORT_DIR}")
    print(f"Models saved in: {MODEL_DIR}")


if __name__ == "__main__":
    main()
