from model import HybridModel
from pipeline import LabelPipeline, InteractionPipeline
import pandas as pd
import numpy as np


train_df = pd.read_csv(r'data\traindata.csv')
test_df = pd.read_csv(r'data\traindata.csv')

cnn_params = {
    'input_length':4000,
    'embedding_dim': 64,
    'kernel_sizes': [3, 5, 7],
    'num_filters': 128,
    'drop_out': 0.5,
}

classifier_params = {}
lp = LabelPipeline(train_df)

labeled_train_signals, train_targets = lp.get_labeled_data()
train_interaction_df = InteractionPipeline.get_interaction_features(labeled_train_signals)

print(train_interaction_df.head())
model = HybridModel(cnn_params, classifier_params)
