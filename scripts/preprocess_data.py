#!/usr/bin/env python3

import argparse as ap

import logging
from logging import DEBUG, INFO, log

import numpy as np

import pandas as pd


logging.basicConfig(level=DEBUG)


def split(entries, proportion):
    n = len(entries)
    a = np.random.choice(n, int(n * proportion), replace=False)
    b = np.setdiff1d(range(n), a)
    return entries[a], entries[b]


def run(data_file, training_size, min_seq_length, min_word_count):
    log(INFO, 'reading data...')
    data = pd.read_csv(
        data_file,
        sep='\t',
        header=0,
        index_col=0,
    )

    log(INFO, 'data contains %d observations', len(data))

    short_seqs = data['sequence'].apply(lambda x: len(x) < min_seq_length)
    data = data.drop(data.index[short_seqs])
    log(
        INFO, 'dropped %d sequences with length < %d...',
        sum(short_seqs),
        min_seq_length,
    )

    word_bags = data.iloc[:, 3:]
    metadata = data.iloc[:, :3]

    log(INFO, 'computing train/test split...')
    training_ids, test_ids = split(data.index, training_size)
    log(
        INFO,
        'training (test) set contains %d (%d) observations',
        *map(len, (training_ids, test_ids)),
    )

    avg_count = (
        word_bags
        .loc[training_ids]
        .mean(axis=0)
    )
    too_few = word_bags.columns[avg_count < min_word_count]
    word_bags = word_bags.drop(too_few, axis=1)
    log(
        INFO,
        'dropped %d words avg count less than %f',
        len(too_few),
        min_word_count,
    )

    log(INFO, 'centering word bags...')
    word_bags = (
        (word_bags - word_bags.loc[training_ids].mean(axis=0))
        / word_bags.loc[training_ids].std(axis=0)
    )

    # drop words that had zero sd (up to fp precision)
    has_nans = (
        word_bags
        .loc[training_ids]
        .columns[np.isnan(word_bags).any(axis=0)]
    )
    word_bags = word_bags.drop(has_nans, axis=1)
    log(
        INFO,
        'dropped %d words with zero st.dev in training set',
        len(has_nans),
    )

    def _save(data, filename, **kwargs):
        log(INFO, 'saving %s...', filename)
        data.to_csv(
            filename,
            **{
                k: kwargs[k] if k in kwargs else v
                for k, v in
                dict(
                    compression='gzip',
                    sep='\t',
                    header=True,
                    index=True,
                ).items()
            },
        )

    for name, ids in zip(
            ('training_set', 'test_set'),
            (training_ids, test_ids),
    ):
        _save(metadata.loc[ids], f'metadata-{name}.tsv.gz')
        _save(word_bags.loc[ids].round(4), f'wordbags-{name}.tsv.gz')


def main():
    args = ap.ArgumentParser()
    args.add_argument('data_file', metavar='data', type=str)
    args.add_argument('--training-size', type=float, default=0.9)
    args.add_argument('--min-seq-length', type=int, default=100)
    args.add_argument('--min-word-count', type=float, default=0.2)
    run(**vars(args.parse_args()))


if __name__ == '__main__':
    main()
