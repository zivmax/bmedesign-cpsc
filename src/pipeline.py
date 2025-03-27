from utils import SignalOps, SignalFeatures
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple
import pandas as pd
import numpy as np

NPERSEG = 1024
class LabelPipeline:
    def __init__(self, df:pd.DataFrame, cutedge:Tuple=(500, 1000)):
        self.main_df = df.copy()
        self.cut_edge = cutedge
        self.target_df = None
        self._prepare_target()

    def _prepare_target(self):
        if self.cut_edge:
            self.main_df['target'] = -1
            self.main_df.iloc[0:self.cut_edge[0], -1] = 1
            self.main_df.iloc[self.cut_edge[0]:self.cut_edge[1], 
                              -1] = 0
            # Fix: distribution leak
            self.labeled_df = self.main_df[:self.cut_edge[1]].sample(frac=1, random_state=42).reset_index(drop=True)
            self.labeled_target_df = self.labeled_df['target']
            self.target_df = self.main_df['target']
            self.main_df = self.main_df.drop('target', axis=1)
            self.labeled_df = self.labeled_df.drop('target', axis=1)

    def get_labeled_data(self):
        return self.labeled_df, self.labeled_target_df if self.cut_edge else None
    
    def get_total_target(self):
        return self.target_df
    

class InteractionPipeline:
    
    @staticmethod    
    def get_interaction_features(raw_df:pd.DataFrame, scaler=MinMaxScaler()):
        interaction_df = pd.DataFrame()
        main_df = raw_df.copy()
        interaction_df['mean'] = main_df.mean(axis=1)
        interaction_df['std'] = main_df.std(axis=1)
        interaction_df['fft_std'] = main_df.apply(
            lambda x: SignalOps.fft_components(x.values)[1].std(), axis=1)
        interaction_df['ps_density_mean'] = main_df.apply(
            lambda x: SignalFeatures.ps_density(x, nperseg=NPERSEG)[1].mean(), axis=1)
        interaction_df['ps_density_std'] = main_df.apply(
            lambda x: SignalFeatures.ps_density(x, nperseg=NPERSEG)[1].std(), axis=1)

        interaction_df = pd.DataFrame(scaler.fit_transform(interaction_df), 
                                           columns=interaction_df.columns)
        return interaction_df
        