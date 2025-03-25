from model import HybridModel
from pipeline import FEPipeline
import pandas as pd
import numpy as np


train_df = pd.read_csv(r'bmedesign-cpsc\data\traindata.csv')
test_df = pd.read_csv(r'bmedesign-cpsc\data\testdata.csv')

cnn_params = {
    'input_length':4000,
    'embedding_dim': 64,
    'kernel_sizes': [3, 5, 7],
    'num_filters': 128,
    'drop_out': 0.5,
}

classifier_params = {}
model = HybridModel(cnn_params, classifier_params)
