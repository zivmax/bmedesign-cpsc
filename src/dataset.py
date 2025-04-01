from torch.utils.data import Dataset
import numpy as np


class SignalDataset(Dataset):
    def __init__(self, dataframe, target=None):
        self.signals = dataframe.values.astype(np.float32)
        self.signals = self.signals.reshape(self.signals.shape[0], 1, 
                                            self.signals.shape[1])
        
        self.targets = target.values.astype(np.int64) if target is not None else None

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        sample = self.signals[idx]
        target = self.targets[idx] if self.targets is not None else None

        return sample, target


