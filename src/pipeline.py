from utils import SignalOps, SignalFeatures
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple
import pandas as pd
import numpy as np

NPERSEG = 1024
SEED = 42
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
            # shuffle only the labeled part of the data
            self.labeled_df = self.main_df[:self.cut_edge[1]].sample(frac=1, random_state=SEED).reset_index(drop=True)
            self.labeled_target_df = self.labeled_df['target']

            self.target_df = self.main_df['target']
            self.main_df = self.main_df.drop('target', axis=1)
            self.labeled_df = self.labeled_df.drop('target', axis=1)

    def get_labeled_data(self):
        # Fix: return none if cut_edge is not set
        return (self.labeled_df, self.labeled_target_df) if self.cut_edge else None
    
    def get_raw_data(self):
        return self.target_df if self.cut_edge else None, self.main_df


class SignalAugmentationPipeline:
    def __init__(self, labeled_df:pd.DataFrame,
                 labeled_target_df:pd.DataFrame,
                 window_length=40, lag=20, diff=True):
        self.labeled_df = labeled_df
        self.labeled_target_df = labeled_target_df
        self.final_labeled_df = pd.DataFrame()
        self.final_labeled_target_df = pd.DataFrame()
        self._apply_moving_avg(window_length)
        self._apply_time_shift(lag, diff)
    
    def _apply_moving_avg(self, window_length):
        moving_avg_df = self.labeled_df.apply(
            lambda x: SignalOps.moving_average(x, window_length), axis=0)
        moving_avg_target_df = self.labeled_target_df.copy()
        self.final_labeled_df = pd.concat([self.final_labeled_df, moving_avg_df], axis=0)

        self.final_labeled_target_df = pd.concat([self.final_labeled_target_df, 
                                          moving_avg_target_df], axis=0)

    def _apply_time_shift(self, lag, diff):
        time_shift_df = self.labeled_df.apply(
            lambda x: SignalOps.time_shift(x, lag, diff), axis=0)

        time_shift_target_df = self.labeled_target_df.copy()
        self.final_labeled_df = pd.concat([self.final_labeled_df, time_shift_df], axis=0)
        self.final_labeled_target_df = pd.concat([self.final_labeled_target_df,
                                            time_shift_target_df], axis=0)

    def get_total_labeled_data(self):
        return pd.concat([self.final_labeled_df,
                          self.labeled_df]), pd.concat([self.final_labeled_target_df,
                                                        self.labeled_target_df])
    def get_processed_data(self):
        return self.final_labeled_df, self.final_labeled_target_df

# NOTE: sklearn pipeline structure
class InteractionPipeline:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.is_fitted = False
        
    def fit(self, raw_df:pd.DataFrame):
        interaction_df = self._extract_features(raw_df)
        self.scaler.fit(interaction_df)
        self.is_fitted = True
        return self
        
    def transform(self, raw_df:pd.DataFrame):
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")
        interaction_df = self._extract_features(raw_df)
        return pd.DataFrame(self.scaler.transform(interaction_df), 
                           columns=interaction_df.columns)
    
    def fit_transform(self, raw_df:pd.DataFrame):
        self.fit(raw_df)
        return self.transform(raw_df)
    
    def _extract_features(self, raw_df:pd.DataFrame):
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
        return interaction_df
        