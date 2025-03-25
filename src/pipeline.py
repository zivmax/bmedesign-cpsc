from utils import SignalOps, SignalFeatures
from typing import Tuple
import pandas as pd
import numpy as np

NPERSEG = 1024
class FEPipeline:
    def __init__(self, df:pd.DataFrame, cutedge:Tuple=None):
        self.main_df = df.copy()
        self.cut_edge = cutedge
        self.interaction_df = pd.DataFrame()
        self.target_df = None
        self._prepare_interaction_features()
        self._prepare_target()

    def _prepare_interaction_features(self):
        self.interaction_df['mean'] = self.main_df.mean(axis=1)
        self.interaction_df['std'] = self.main_df.std(axis=1)
        self.interaction_df['fft_std'] = self.main_df.apply(
            lambda x: SignalOps.fft_components(x.values)[1].std(), 
            axis=1)
        self.interaction_df['ps_density_mean'] = self.main_df.apply(lambda x: SignalFeatures.ps_density(x, nperseg=NPERSEG)[1].mean(), axis=1)
        self.interaction_df['ps_density_std'] = self.main_df.apply(lambda x: SignalFeatures.ps_density(x, nperseg=NPERSEG)[1].std(), axis=1)
    def _prepare_target(self):
        if self.cut_edge:
            self.main_df['target'] = -1
            self.main_df.iloc[0:self.cut_edge[0], -1] = 1
            self.main_df.iloc[self.cut_edge[0]:self.cut_edge[1], 
                              -1] = 0
            self.target_df = self.main_df['target']
            self.main_df = self.main_df.drop('target', axis=1)

    def get_labeled_data(self):
        return self.main_df[:self.cut_edge[1]], self.interaction_df[:self.cut_edge[1]], self.target_df[:self.cut_edge[1]] if self.cut_edge else None
    
    def get_total_interaction_features(self):
        return self.interaction_df
    
    def get_total_target(self):
        return self.target_df