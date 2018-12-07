import pandas as pd

import torch as t
from torch.utils.data import Dataset


class SignalProteins(Dataset):
    def __init__(
            self,
            data: pd.DataFrame,
            metadata: pd.DataFrame,
    ):
        self.data = data
        self.metadata = metadata
        if not len(self.data) == len(self.metadata):
            raise ValueError(
                'data doesn\'t have same length as metadata '
                f'({len(self.data)} vs. ({len(self.metadata)}))')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return dict(
            input=t.tensor(self.data.iloc[idx].values).float(),
            label=t.tensor(self.metadata.signal[idx]).long(),
        )
