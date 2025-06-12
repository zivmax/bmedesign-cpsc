import pandas as pd
from typing import Tuple
from utils import SignalPlot, SignalOps
import random
import os


class Plots:
    def __init__(self, labeled_df: pd.DataFrame, 
                 interaction_df: pd.DataFrame,
                 cutedge: Tuple[int, int] = (500, 1000)):

        self.labeled_df = labeled_df
        self.interaction_df = interaction_df
        self.cut_edge = cutedge

    def random_single_sample_plots(self, fs=400, duration=10, moving_avg=100, lag=400, 
                                   base_path = r"src/imgs/signal_overview"):

        if not os.path.exists(base_path):
            os.makedirs(base_path)
        ill_idx = random.choice(range(0, self.cut_edge[0]))
        healthy_idx = random.choice(range(self.cut_edge[0], self.cut_edge[1]))

        ill_signal = self.labeled_df.iloc[ill_idx]
        healthy_signal = self.labeled_df.iloc[healthy_idx]
        healthy_moving_avg = SignalOps.moving_average(healthy_signal, 
                                                      window_length=moving_avg)
        ill_moving_avg = SignalOps.moving_average(ill_signal, 
                                                  window_length=moving_avg)
        healty_avg_tuple = (healthy_signal, healthy_moving_avg)
        ill_avg_tuple = (ill_signal, ill_moving_avg)


        SignalPlot.fft_plot(ill_signal, fs=fs, duration=duration, 
                            path=base_path + '/fft_ill.png')
        SignalPlot.fft_plot(healthy_signal, fs=fs, duration=duration, 
                            path=base_path + '/fft_healthy.png')

        SignalPlot.time_shift_diff_plot(healthy_signal, ill_signal, lag=lag, 
                                        path=base_path + '/time_shift_diff.png')

        SignalPlot.binary_class_differential_plot(healty_avg_tuple, ill_avg_tuple,
                                                path=base_path + '/moving_avg_diff.png')
    
    def labeled_signal_df_plots(self, cutedge=500, base_path = r"src/imgs/signal_overview"):
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        SignalPlot.statistics_plot(self.labeled_df, cutedge=cutedge,
                                path=base_path + '/signal_statistics.png')
        
    def interaction_df_plots(self, cutedge=500, path='src/imgs/interaction_features/'):
        if not os.path.exists(path):
            os.makedirs(path)
        SignalPlot.signal_feature_plot(self.interaction_df, cutedge=cutedge,
                                path=path)