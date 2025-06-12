import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
)
import sys
import os

# Adjust the Python path to include the parent directory of 'src'
# This allows importing modules from 'utils' and 'model'
# Get the absolute path of the current script
current_script_path = os.path.abspath(__file__)
# Get the directory containing the current script (e.g., /home/zivmax/bmedesign-cpsc/src/comparison/kmeans)
current_dir = os.path.dirname(current_script_path)
# Get the 'src' directory
src_dir = os.path.dirname(os.path.dirname(current_dir))
# Get the project root directory (parent of 'src')
project_root = os.path.dirname(src_dir)
# Add the project root to the Python path
sys.path.insert(0, project_root)

from src.utils.pipeline import LabelPipeline  # type: ignore


def map_clusters_to_labels(kmeans_labels, true_labels):
    """
    Maps KMeans cluster labels to true class labels.
    Assumes binary classification.
    """
    # Create a mapping for cluster 0
    cluster0_mask = kmeans_labels == 0
    if np.sum(cluster0_mask) == 0:  # No samples in cluster 0
        # If cluster 0 is empty, assign the most frequent label of cluster 1 to it,
        # and the other label to cluster 1. This is a fallback.
        # Or, more simply, if cluster 0 is empty, all samples are in cluster 1.
        # We can assume cluster 1 maps to the most frequent true label among those samples.
        # And cluster 0 would map to the other label.
        # However, a simpler approach if one cluster is empty is to assume
        # the non-empty cluster represents the majority true label of its members.
        # This scenario should be rare with k=2 and sufficient data.
        # For now, let's assume cluster 0 maps to 0 and cluster 1 maps to 1 if cluster 0 is empty.
        # This will likely lead to poor results but avoids errors.
        # A better handling would be to check true_labels for the samples in cluster 1.
        if np.sum(kmeans_labels == 1) > 0:
            label_for_cluster1 = pd.Series(true_labels[kmeans_labels == 1]).mode()[0]
            label_for_cluster0 = 1 - label_for_cluster1
        else:  # both clusters are empty - highly unlikely
            label_for_cluster0 = 0
            label_for_cluster1 = 1

    else:
        label_for_cluster0 = pd.Series(true_labels[cluster0_mask]).mode()[0]

    # Create a mapping for cluster 1
    cluster1_mask = kmeans_labels == 1
    if np.sum(cluster1_mask) == 0:  # No samples in cluster 1
        # Similar to above, if cluster 1 is empty.
        # Assume cluster 1 maps to the label not taken by cluster 0.
        label_for_cluster1 = 1 - label_for_cluster0
    else:
        # Check if the mode for cluster 1 is the same as for cluster 0
        # If so, assign the other label to cluster 1
        mode_cluster1 = pd.Series(true_labels[cluster1_mask]).mode()[0]
        if mode_cluster1 == label_for_cluster0:
            label_for_cluster1 = 1 - label_for_cluster0
        else:
            label_for_cluster1 = mode_cluster1

    # Ensure labels are different if possible and both clusters have samples
    if (
        label_for_cluster0 == label_for_cluster1
        and np.sum(cluster0_mask) > 0
        and np.sum(cluster1_mask) > 0
    ):
        # This can happen if one cluster is much larger or cleaner, and the other is mixed.
        # Or if the data is not well separable by kmeans into the true classes.
        # Fallback: assign 0 to cluster 0 and 1 to cluster 1, or vice-versa,
        # based on which assignment yields better overall accuracy on the training labels.

        # Option 1: cluster 0 -> 0, cluster 1 -> 1
        mapping1 = {0: 0, 1: 1}
        preds1 = np.array([mapping1[l] for l in kmeans_labels])
        acc1 = accuracy_score(true_labels, preds1)

        # Option 2: cluster 0 -> 1, cluster 1 -> 0
        mapping2 = {0: 1, 1: 0}
        preds2 = np.array([mapping2[l] for l in kmeans_labels])
        acc2 = accuracy_score(true_labels, preds2)

        if acc1 >= acc2:
            label_for_cluster0 = 0
            label_for_cluster1 = 1
        else:
            label_for_cluster0 = 1
            label_for_cluster1 = 0

    return {0: label_for_cluster0, 1: label_for_cluster1}


if __name__ == "__main__":
    SEED = 42
    np.random.seed(SEED)

    # Define paths relative to the project root
    # The script is in /home/zivmax/bmedesign-cpsc/src/comparison/kmeans/main.py
    # Project root is /home/zivmax/bmedesign-cpsc/
    base_path = project_root  # Defined earlier using os.path

    train_df_path = os.path.join(base_path, "data/traindata.csv")
    test_df_path = os.path.join(base_path, "data/testdata.csv")
    unlabeled_preds_path = os.path.join(base_path, "data/unlabeled_predictions.csv")

    print(f"Loading training data from: {train_df_path}")
    train_df = pd.read_csv(train_df_path)

    print(f"Loading pseudo-labels from: {unlabeled_preds_path}")
    temp_df = pd.read_csv(unlabeled_preds_path, header=None)
    labels_for_pseudo = None

    if (
        temp_df.shape[0] > 0 and temp_df.shape[1] > 0
    ):  # Check if df is not empty and has at least one column
        labels_for_pseudo = temp_df.iloc[:, 0]
        print(
            f"Info: Loaded {len(labels_for_pseudo)} pseudo-labels from {unlabeled_preds_path}."
        )
    else:
        print(
            f"Warning: {unlabeled_preds_path} is empty or has no columns. No pseudo-labels loaded."
        )
        # Exiting because pseudo-labels are essential for this script's purpose
        sys.exit(
            "Exiting: Pseudo-labels are required for K-Means baseline based on pseudo-labeled data."
        )

    lp = LabelPipeline(train_df)
    # Using the exact pseudo-labeled data means we use labels_for_pseudo directly
    # The cutedge parameter was (500, 1000) in main.py, let's keep it consistent
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

    # Split data into training and validation (final_val in main.py)
    # We use the 'targets' which are the pseudo-labels for both training K-Means and evaluating it.
    X_train, X_val, y_train_pseudo, y_val_pseudo = train_test_split(
        labeled_signals,
        targets,
        test_size=0.2,
        random_state=SEED,
        stratify=targets,
        shuffle=True,
    )

    print(f"KMeans training set size: {X_train.shape[0]}")
    print(f"KMeans validation set size: {X_val.shape[0]}")

    # K-Means Model
    kmeans = KMeans(n_clusters=2, random_state=SEED, n_init="auto")

    print("\nTraining K-Means on the training portion of pseudo-labeled data...")
    # Fit K-Means on the features of the training set
    kmeans.fit(X_train)

    # Get cluster assignments for the training data
    train_cluster_labels = kmeans.labels_

    # Map cluster labels (0 or 1 from KMeans) to actual pseudo-labels (0 or 1 from targets)
    # We use y_train_pseudo to find the best mapping
    print("Mapping K-Means clusters to pseudo-labels using training data...")
    cluster_to_class_mapping = map_clusters_to_labels(
        train_cluster_labels, y_train_pseudo.to_numpy()
    )
    print(f"K-Means cluster to class mapping: {cluster_to_class_mapping}")

    # Predict classes on the training set using the mapping
    y_train_pred_kmeans = np.array(
        [cluster_to_class_mapping[label] for label in train_cluster_labels]
    )

    print(
        "\nEvaluating K-Means on the training portion of pseudo-labeled data (for reference):"
    )
    print(
        f"Training F1 Score: {f1_score(y_train_pseudo, y_train_pred_kmeans, average='binary'):.4f}"
    )  # Specify average for binary
    print(
        f"Training Precision Score: {precision_score(y_train_pseudo, y_train_pred_kmeans, average='binary'):.4f}"
    )
    print(
        f"Training Recall Score: {recall_score(y_train_pseudo, y_train_pred_kmeans, average='binary'):.4f}"
    )
    print(
        f"Training Accuracy: {accuracy_score(y_train_pseudo, y_train_pred_kmeans):.4f}"
    )
    print("Training Classification Report:")
    print(
        classification_report(
            y_train_pseudo, y_train_pred_kmeans, target_names=["Class 0", "Class 1"]
        )
    )

    # Predict clusters on the validation set
    print("\nPredicting K-Means clusters on the validation set...")
    val_cluster_labels = kmeans.predict(X_val)

    # Convert cluster labels to class predictions using the mapping derived from the training set
    y_val_pred_kmeans = np.array(
        [cluster_to_class_mapping[label] for label in val_cluster_labels]
    )

    print("\nEvaluating K-Means on the validation portion of pseudo-labeled data:")
    val_f1 = f1_score(
        y_val_pseudo, y_val_pred_kmeans, average="binary"
    )  # Specify average for binary
    val_precision = precision_score(y_val_pseudo, y_val_pred_kmeans, average="binary")
    val_recall = recall_score(y_val_pseudo, y_val_pred_kmeans, average="binary")
    val_accuracy = accuracy_score(y_val_pseudo, y_val_pred_kmeans)
    print(f"Validation F1 Score: {val_f1:.4f}")
    print(f"Validation Precision Score: {val_precision:.4f}")
    print(f"Validation Recall Score: {val_recall:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print("Validation Classification Report:")
    print(
        classification_report(
            y_val_pseudo, y_val_pred_kmeans, target_names=["Class 0", "Class 1"]
        )
    )

    # Save validation metrics
    validation_metrics_data = {
        "Metric": ["F1 Score", "Precision", "Recall", "Accuracy"],
        "Score": [val_f1, val_precision, val_recall, val_accuracy],
    }
    validation_metrics_df = pd.DataFrame(validation_metrics_data)
    validation_metrics_save_path = os.path.join(
        current_dir, "kmeans_validation_evaluation_metrics.csv"
    )
    validation_metrics_df.to_csv(validation_metrics_save_path, index=False)
    print(
        f"K-Means validation evaluation metrics saved to {validation_metrics_save_path}"
    )

    print("\nK-Means baseline script finished.")
