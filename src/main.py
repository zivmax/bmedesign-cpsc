from model import HybridModel
from pipeline import LabelPipeline, InteractionPipeline
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Load data
train_df = pd.read_csv(r'data\traindata.csv')
test_df = pd.read_csv(r'data\testdata.csv')

# Get labeled data
lp = LabelPipeline(train_df)
labeled_signals, targets = lp.get_labeled_data()


X_train_val, X_test, y_train_val, y_test = train_test_split(
    labeled_signals, targets, test_size=0.2, 
    random_state=42, stratify=targets
)


cnn_params = {
    'input_length': 4000,
    'embedding_dim': 800,
    'kernel_sizes': [1],
    'num_filters': 2,
    'drop_out': 0.9,
}
classifier_params = {}


model = HybridModel(cnn_params, classifier_params)
cv_results = model.train(
    X_train_val, y_train_val,
    num_epochs=2,
    batch_size=12,
    model_save_path='best_model.pth'
)


test_results = model.evaluate(X_test, y_test)
print(f"Test F1 Score: {test_results['f1']:.4f}")