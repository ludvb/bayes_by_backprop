#!/usr/bin/env python3

import argparse as ap

import numpy as np

import pandas as pd


def split(data, proportion):
    n = data.shape[0]
    a = np.random.choice(n, int(n * proportion), replace=False)
    b = np.setdiff1d(range(n), a)
    return data.iloc[a], data.iloc[b]


def main():
    args = ap.ArgumentParser()
    args.add_argument('data', type=str)
    args.add_argument('--training-size', type=float, default=0.9)
    opts = args.parse_args()

    data = pd.read_csv(opts.data, sep='\t', header=0)

    for name, data in zip(('training_set', 'test_set'),
                          split(data, opts.training_size)):
        data.to_csv(
            f'{name}.tsv.gz',
            compression='gzip',
            sep='\t',
            header=True,
            index=False,
        )


if __name__ == '__main__':
    main()
