import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
)
import sys
import os

# Adjust the Python path to include the parent directory of 'src'
current_script_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_script_path)
# Navigate up to 'src' directory: decision_tree -> comparison -> src
src_dir = os.path.dirname(os.path.dirname(current_dir))
# Navigate up to project root: src -> project_root
project_root = os.path.dirname(src_dir)
sys.path.insert(0, project_root)

from src.utils.pipeline import LabelPipeline  # type: ignore

if __name__ == "__main__":
    SEED = 42
    np.random.seed(SEED)

    base_path = project_root

    train_df_path = os.path.join(base_path, "data/traindata.csv")
    unlabeled_preds_path = os.path.join(base_path, "data/unlabeled_predictions.csv")

    print(f"Loading training data from: {train_df_path}")
    try:
        train_df = pd.read_csv(train_df_path)
    except FileNotFoundError:
        print(f"Error: Training data file not found at {train_df_path}. Exiting.")
        sys.exit(1)

    print(f"Loading pseudo-labels from: {unlabeled_preds_path}")
    try:
        temp_df = pd.read_csv(unlabeled_preds_path, header=None)
    except FileNotFoundError:
        print(
            f"Error: Pseudo-labels file not found at {unlabeled_preds_path}. Exiting."
        )
        sys.exit(1)

    labels_for_pseudo = None
    if temp_df.shape[0] > 0 and temp_df.shape[1] > 0:
        labels_for_pseudo = temp_df.iloc[:, 0]
        print(
            f"Info: Loaded {len(labels_for_pseudo)} pseudo-labels from {unlabeled_preds_path}."
        )
    else:
        print(
            f"Warning: {unlabeled_preds_path} is empty or has no columns. No pseudo-labels loaded."
        )
        sys.exit(
            "Exiting: Pseudo-labels are required for Decision Tree baseline based on pseudo-labeled data."
        )

    lp = LabelPipeline(train_df)
    labeled_signals, targets = lp.add_labels(labels_for_pseudo, cutedge=(500, 1000))

    print("Original pseudo-labels distribution:")
    print(targets.value_counts(normalize=True))

    if not isinstance(labeled_signals, pd.DataFrame):
        labeled_signals = pd.DataFrame(labeled_signals)
    if not isinstance(targets, pd.Series):
        targets = pd.Series(targets)

    print(
        f"Total labeled signals: {labeled_signals.shape[0]}, Total targets: {targets.shape[0]}"
    )
    if labeled_signals.empty or targets.empty:
        raise ValueError(
            "Data loading or processing resulted in empty signals or targets."
        )

    X_train, X_val, y_train_pseudo, y_val_pseudo = train_test_split(
        labeled_signals,
        targets,
        test_size=0.2,
        random_state=SEED,
        stratify=targets,
        shuffle=True,
    )

    print(f"Decision Tree training set size: {X_train.shape[0]}")
    print(f"Decision Tree validation set size: {X_val.shape[0]}")

    # Decision Tree Model
    dt_model = DecisionTreeClassifier(random_state=SEED)

    print("\nTraining Decision Tree on the training portion of pseudo-labeled data...")
    dt_model.fit(X_train, y_train_pseudo)

    # Evaluate on Training Set (Pseudo-labels)
    print(
        "\nEvaluating Decision Tree on the training portion of pseudo-labeled data (for reference):"
    )
    y_train_pred_dt = dt_model.predict(X_train)
    print(
        f"Training F1 Score: {f1_score(y_train_pseudo, y_train_pred_dt, average='binary', zero_division=0):.4f}"
    )
    print(
        f"Training Precision Score: {precision_score(y_train_pseudo, y_train_pred_dt, average='binary', zero_division=0):.4f}"
    )
    print(
        f"Training Recall Score: {recall_score(y_train_pseudo, y_train_pred_dt, average='binary', zero_division=0):.4f}"
    )
    print(f"Training Accuracy: {accuracy_score(y_train_pseudo, y_train_pred_dt):.4f}")
    print("Training Classification Report:")
    print(
        classification_report(
            y_train_pseudo,
            y_train_pred_dt,
            target_names=["Class 0", "Class 1"],
            zero_division=0,
        )
    )

    # Evaluate on Validation Set (Pseudo-labels)
    print(
        "\nEvaluating Decision Tree on the validation portion of pseudo-labeled data:"
    )
    y_val_pred_dt = dt_model.predict(X_val)
    val_f1 = f1_score(y_val_pseudo, y_val_pred_dt, average="binary", zero_division=0)
    val_precision = precision_score(
        y_val_pseudo, y_val_pred_dt, average="binary", zero_division=0
    )
    val_recall = recall_score(
        y_val_pseudo, y_val_pred_dt, average="binary", zero_division=0
    )
    val_accuracy = accuracy_score(y_val_pseudo, y_val_pred_dt)
    print(f"Validation F1 Score: {val_f1:.4f}")
    print(f"Validation Precision Score: {val_precision:.4f}")
    print(f"Validation Recall Score: {val_recall:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print("Validation Classification Report:")
    print(
        classification_report(
            y_val_pseudo,
            y_val_pred_dt,
            target_names=["Class 0", "Class 1"],
            zero_division=0,
        )
    )

    # Save validation metrics
    validation_metrics_data = {
        "Metric": ["F1 Score", "Precision", "Recall", "Accuracy"],
        "Score": [val_f1, val_precision, val_recall, val_accuracy],
    }
    validation_metrics_df = pd.DataFrame(validation_metrics_data)
    # Save in the current script's directory (src/comparison/decision_tree/)
    validation_metrics_save_path = os.path.join(
        current_dir, "decision_tree_validation_evaluation_metrics.csv"
    )
    validation_metrics_df.to_csv(validation_metrics_save_path, index=False)
    print(
        f"Decision Tree validation evaluation metrics saved to {validation_metrics_save_path}"
    )

    print("\nDecision Tree baseline script finished.")
