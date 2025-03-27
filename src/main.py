from model import HybridModel
from pipeline import LabelPipeline, InteractionPipeline
from plot import Plots
import pandas as pd
import numpy as np


train_df = pd.read_csv(r'data\traindata.csv')
test_df = pd.read_csv(r'data\traindata.csv')

def generate_labeled_plots(labeled_signal_df, interaction_df):
    plots = Plots(labeled_signal_df, interaction_df)
    plots.random_single_sample_plots()
    plots.interaction_df_plots()
    plots.labeled_signal_df_plots()

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


model = HybridModel(cnn_params, classifier_params)
model.train(labeled_train_signals, train_targets)
