from typing import Optional

import numpy as np

import pandas as pd

import torch as t
from torch.utils.data import DataLoader, Dataset


class WordBags(Dataset):
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
            entry=self.metadata.index[idx],
            input=t.as_tensor(self.data.iloc[idx].values).float(),
            label=t.as_tensor(self.metadata.signal[idx]).long(),
        )


class Sequences(Dataset):
    _amino_acids = list('ACDEFGHIKLMNPQRSTVWY')
    _onehot = pd.DataFrame(np.eye(
        len(_amino_acids))).set_index([_amino_acids])

    def __init__(
            self,
            data: pd.DataFrame,
            truncate: Optional[int] = None,
    ):
        self.data = data
        self._truncate = (
            (lambda x: x[:truncate])
            if truncate else
            (lambda x: x)
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        def _encode(x):
            try:
                return t.as_tensor(self._onehot.loc[x].values, dtype=t.float)
            except KeyError:
                return t.zeros((self._onehot.shape[1],), dtype=t.float)
        return dict(
            entry=self.data.index[idx],
            input=t.cat([
                _encode(c)[None, ...]
                for c in self._truncate(self.data['sequence'].iloc[idx])
            ]),
            label=t.as_tensor(self.data.signal[idx], dtype=t.long),
        )


def make_sequence_loader(dataset: Sequences, *args, **kwargs):
    def _collate_fn(xs):
        data = [x['input'] for x in xs]
        label = t.as_tensor([x['label'] for x in xs])
        length = t.as_tensor([x.shape[0] for x in data], dtype=t.long)
        return dict(
            input=dict(
                data=t.cat([
                    t.nn.functional.pad(
                        x, (0, 0, 0, max(length) - x.shape[0]))[None, ...]
                    for x in data
                ]).transpose(0, 1),
                length=length,
            ),
            label=label,
            entry=[x['entry'] for x in xs],
        )

    return DataLoader(dataset, *args, collate_fn=_collate_fn, **kwargs)
