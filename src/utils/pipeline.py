from .signal_toolkit import SignalOps, SignalFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from typing import Tuple
import pandas as pd
import numpy as np

NPERSEG = 1024
SEED = 42


class LabelPipeline:
    def __init__(self, df: pd.DataFrame, cutedge: Tuple = (500, 1000)):
        self.main_df = df.copy()
        self.cut_edge = cutedge
        self.target_df = None
        self._prepare_target()

    def _prepare_target(self):
        if self.cut_edge:
            self.main_df["target"] = -1
            self.main_df.iloc[0 : self.cut_edge[0], -1] = 1
            self.main_df.iloc[self.cut_edge[0] : self.cut_edge[1], -1] = 0

            self.labeled_df = self.main_df[: self.cut_edge[1]]
            self.labeled_df = shuffle(self.labeled_df, random_state=SEED)
            self.labeled_target_df = self.labeled_df["target"]

            self.target_df = self.main_df["target"]
            self.main_df = self.main_df.drop("target", axis=1)
            self.labeled_df = self.labeled_df.drop("target", axis=1)

    def get_labeled_data(self):
        # Fix: return none if cut_edge is not set
        return (self.labeled_df, self.labeled_target_df) if self.cut_edge else None

    def get_raw_data(self):
        return self.target_df if self.cut_edge else None, self.main_df

    def add_labels(self, labels, cutedge: Tuple = (500, 1000)):
        self.main_df["target"] = -1  # Initialize all to -1 (unlabeled)
        self.main_df.iloc[0 : cutedge[0], -1] = 1  # Known state 1
        self.main_df.iloc[cutedge[0] : cutedge[1], -1] = 0  # Known state 0

        # Determine the slice for pseudo-labels
        pseudo_label_slice = self.main_df.iloc[cutedge[1] :]

        if labels is not None:
            labels_array = np.array(labels).flatten()
            if len(labels_array) == len(pseudo_label_slice):
                self.main_df.iloc[cutedge[1] :, -1] = labels_array
                print(f"Successfully applied {len(labels_array)} pseudo-labels.")
            else:
                print(
                    f"Warning: Length mismatch for pseudo-labels. Expected {len(pseudo_label_slice)}, got {len(labels_array)}. Pseudo-labels will not be applied to the tail."
                )
                # Option: Fill with a default value like -1 if lengths don't match, or leave as is.
                # For now, if lengths don't match, the tail remains -1 (or whatever it was before this check).
                # If the intention is to ensure the tail is explicitly marked, ensure it's -1.
                # self.main_df.iloc[cutedge[1]:, -1] = -1 # Or some other default if not applying partial pseudo labels
        else:
            print(
                "Info: No pseudo-labels provided or loaded. The tail end of the data will remain as initially set (e.g., -1)."
            )
            # The tail is already -1 from the initial `self.main_df['target'] = -1`

        self.target_df = self.main_df["target"]
        self.main_df = self.main_df.drop("target", axis=1)

        return self.main_df, self.target_df


class SignalAugmentationPipeline:
    def __init__(
        self,
        labeled_df: pd.DataFrame,
        labeled_target_df: pd.DataFrame,
        window_length=40,
        lag=20,
        noise_level=0.1,
        diff=True,
    ):
        self.labeled_df = labeled_df
        self.labeled_target_df = labeled_target_df
        self.final_labeled_df = pd.DataFrame()
        self.final_labeled_target_df = pd.DataFrame()
        self.moving_avg_df = self._apply_moving_avg(window_length)
        self.time_shift_df = self._apply_time_shift(lag, diff)
        self.noise_df = self._apply_noise(noise_level)

    def _apply_moving_avg(self, window_length):
        moving_avg_df = self.labeled_df.apply(
            lambda x: SignalOps.moving_average(x, window_length), axis=0
        )
        moving_avg_target_df = self.labeled_target_df.copy()
        self.final_labeled_df = pd.concat(
            [self.final_labeled_df, moving_avg_df], axis=0
        )

        self.final_labeled_target_df = pd.concat(
            [self.final_labeled_target_df, moving_avg_target_df], axis=0
        )
        return moving_avg_df

    def _apply_time_shift(self, lag, diff):
        time_shift_df = self.labeled_df.apply(
            lambda x: SignalOps.time_shift(x, lag, diff), axis=0
        )

        time_shift_target_df = self.labeled_target_df.copy()
        self.final_labeled_df = pd.concat(
            [self.final_labeled_df, time_shift_df], axis=0
        )
        self.final_labeled_target_df = pd.concat(
            [self.final_labeled_target_df, time_shift_target_df], axis=0
        )
        return time_shift_df

    def _apply_noise(self, noise_level):
        noise_df = self.labeled_df.apply(
            lambda x: SignalOps.add_noise(x, noise_level), axis=0
        )
        noise_target_df = self.labeled_target_df.copy()
        self.final_labeled_df = pd.concat([self.final_labeled_df, noise_df], axis=0)
        self.final_labeled_target_df = pd.concat(
            [self.final_labeled_target_df, noise_target_df], axis=0
        )
        return noise_df

    def get_total_labeled_data(self):
        return pd.concat([self.final_labeled_df, self.labeled_df]), pd.concat(
            [self.final_labeled_target_df, self.labeled_target_df]
        )

    def get_augmentation_data(self):
        return self.moving_avg_df, self.time_shift_df, self.noise_df

    def get_processed_data(self):
        return self.final_labeled_df.reset_index(
            drop=True
        ), self.final_labeled_target_df.reset_index(drop=True)


# NOTE: sklearn pipeline structure
class InteractionPipeline:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.is_fitted = False

    def fit(self, raw_df: pd.DataFrame):
        interaction_df = self._extract_features(raw_df)
        self.scaler.fit(interaction_df)
        self.is_fitted = True
        return self

    def transform(self, raw_df: pd.DataFrame):
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")
        interaction_df = self._extract_features(raw_df)
        return pd.DataFrame(
            self.scaler.transform(interaction_df), columns=interaction_df.columns
        )

    def fit_transform(self, raw_df: pd.DataFrame):
        self.fit(raw_df)
        return self.transform(raw_df)

    def _extract_features(self, raw_df: pd.DataFrame):
        interaction_df = pd.DataFrame()
        main_df = raw_df.copy()
        interaction_df["mean"] = main_df.mean(axis=1)
        interaction_df["std"] = main_df.std(axis=1)
        interaction_df["fft_std"] = main_df.apply(
            lambda x: SignalOps.fft_components(x.values)[1].std(), axis=1
        )
        interaction_df["ps_density_mean"] = main_df.apply(
            lambda x: SignalFeatures.ps_density(x, nperseg=NPERSEG)[1].mean(), axis=1
        )
        interaction_df["ps_density_std"] = main_df.apply(
            lambda x: SignalFeatures.ps_density(x, nperseg=NPERSEG)[1].std(), axis=1
        )
        return interaction_df
