#!/usr/bin/env python3

import argparse as ap

from datetime import datetime as dt

import os
import os.path as osp

import sys

from tempfile import gettempdir

import numpy as np

import pandas as pd

import torch as t
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

from . import __version__
from .apply import apply
from .dataset import WordBags, Sequences, make_sequence_loader
from .logging import DEBUG, ERROR, INFO, log, set_level
from .network import LSTM, MLP
from .train import train
from .utils import get_device, restore_state


def _train(
        data_file,
        wordbags_file,
        mle,
        learning_rate,
        state,
        num_hidden,
        hidden_size,
        learn_prior,
        **kwargs,
):
    if mle:
        from .network import Deterministic, Uninformative
        prior_distribution = Uninformative
        prior_kwargs = None
        variational_distribution = Deterministic
        variational_kwargs = None
        update_samples = 1
    else:
        from .network import Normal, NormalMixture
        prior_distribution = NormalMixture
        prior_kwargs = None
        variational_distribution = Normal
        variational_kwargs = None
        update_samples = 1

    log(DEBUG, 'reading data...')
    data = pd.read_csv(data_file, sep='\t', header=0, index_col=0)

    if wordbags_file:
        log(DEBUG, 'reading word bags...')
        wordbags = pd.read_csv(
            wordbags_file,
            sep='\t',
            header=0,
            index_col=0,
        )
        dataset = WordBags(wordbags, data.drop('sequence', axis=1))
        _loader_factory = DataLoader
        batch_size = 2048
    else:
        wordbags = None
        dataset = Sequences(data=data)
        _loader_factory = make_sequence_loader
        batch_size = 16

    log(INFO, 'batch size is set to %d', batch_size)

    log(INFO, 'performing validation split...')
    training_set_idxs = np.random.choice(
        len(dataset),
        int(0.9 * len(dataset)),
        replace=False,
    )
    validation_set_idxs = np.setdiff1d(range(len(dataset)), training_set_idxs)
    training_set = _loader_factory(
        Subset(dataset, training_set_idxs),
        shuffle=False,
        batch_size=batch_size,
        num_workers=len(os.sched_getaffinity(0)),
    )
    validation_set = _loader_factory(
        Subset(dataset, validation_set_idxs),
        shuffle=False,
        batch_size=batch_size,
        num_workers=len(os.sched_getaffinity(0)),
    )
    log(INFO, 'training (validation) set is of size %d (%d)',
        len(training_set), len(validation_set))

    if state:
        log(INFO, 'restoring state from %s', state)
        state = restore_state(state)
        network = state['network']
        optimizer = state['optimizer']
    else:
        log(INFO, 'initializing network')
        signal_frac = np.sum(data.signal) / len(data)
        if wordbags:
            network = MLP(
                input_size=wordbags.shape[1],
                output_size=2,
                num_hidden=num_hidden,
                hidden_size=hidden_size,
                prior_distribution=prior_distribution,
                prior_kwargs=prior_kwargs,
                variational_distribution=variational_distribution,
                variational_kwargs=variational_kwargs,
                dataset_size=len(training_set_idxs),
                class_weights=[0.5 / (1 - signal_frac), 0.5 / signal_frac],
                learn_prior=learn_prior,
            )
        else:
            network = LSTM(
                output_size=2,
                input_size=len(Sequences._amino_acids),
                hidden_size=25,
                prior_distribution=prior_distribution,
                prior_kwargs=prior_kwargs,
                variational_distribution=variational_distribution,
                variational_kwargs=variational_kwargs,
                dataset_size=len(training_set_idxs),
                class_weights=[0.5 / (1 - signal_frac), 0.5 / signal_frac],
                learn_prior=learn_prior,
            )
        network = network.to(get_device())
        optimizer = t.optim.Adam(
            network.parameters(), lr=learning_rate)

    train(
        network,
        optimizer,
        training_set,
        validation_set,
        update_samples=update_samples,
        **kwargs,
    )


def _apply(
        data_file,
        wordbags_file,
        state,
        **kwargs,
):
    log(INFO, 'restoring state from %s', state)
    network = restore_state(state)['network']

    log(DEBUG, 'reading data...')
    data = pd.read_csv(data_file, sep='\t', header=0, index_col=0)

    if isinstance(network, MLP):
        if wordbags_file is None:
            raise ValueError(
                'network is MLP but no wordbags file was provided')
        log(DEBUG, 'reading word bags...')
        wordbags = pd.read_csv(
            wordbags_file,
            sep='\t',
            header=0,
            index_col=0,
        )
        dataset = WordBags(wordbags, data.drop('sequence', axis=1))
        _loader_factory = DataLoader
        batch_size = 2048
    else:
        dataset = Sequences(data=data)
        _loader_factory = make_sequence_loader
        batch_size = 16

    data_loader = _loader_factory(
        dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=len(os.sched_getaffinity(0)),
    )

    apply(network, data_loader, **kwargs)


def main():
    args = ap.ArgumentParser(
        description=f'Version: {__version__:s}',
        formatter_class=ap.RawTextHelpFormatter,
    )
    args.add_argument('-v', '--version', action='version', version=__version__)
    args.add_argument('--verbose', action='store_true')

    sp = args.add_subparsers()
    train_parser = sp.add_parser('train')
    apply_parser = sp.add_parser('apply')

    train_parser.set_defaults(run=_train)
    apply_parser.set_defaults(run=_apply)

    train_parser.add_argument('data_file', metavar='data', type=str)
    train_parser.add_argument(
        '--wordbags',
        dest='wordbags_file',
        type=str,
        help='. '.join([
            'file with wordbags.',
            'if specified, will train bag of words model.',
            'otherwise, train LSTM model.',
        ])
    )
    train_parser.add_argument('--state', type=t.load)
    train_parser.add_argument(
        '--num-hidden',
        type=int,
        default=3,
    )
    train_parser.add_argument(
        '--hidden-size',
        type=int,
        default=100,
    )
    train_parser.add_argument(
        '--mle',
        action='store_true',
        help='optimize by maximum likelihood',
    )
    train_parser.add_argument(
        '--learn-prior',
        action='store_true',
        help='optimize prior parameters',
    )
    train_parser.add_argument(
        '--epochs',
        type=int,
        dest='nepochs',
    )
    train_parser.add_argument(
        '--learning-rate',
        type=float,
        default=5e-3,
    )
    train_parser.add_argument(
        '--output-prefix',
        type=str,
        default=osp.join(
            osp.abspath(osp.curdir),
            f'protonet-{dt.now().isoformat()}',
        ),
        help='where to store the output files',
    )

    apply_parser.add_argument('data_file', metavar='data', type=str)
    apply_parser.add_argument(
        '--wordbags',
        dest='wordbags_file',
        type=str,
        help='file with wordbags.',
    )
    apply_parser.add_argument('--state', required=True, type=t.load)
    apply_parser.add_argument(
        '--output-prefix',
        type=str,
        default=osp.join(
            gettempdir(),
            f'protonet-{dt.now().isoformat()}',
        ),
        help='where to store the output files',
    )

    opts = vars(args.parse_args())
    try:
        run = opts.pop('run')
    except KeyError:
        args.print_help()
        sys.exit(0)

    if opts.pop('verbose'):
        set_level(DEBUG)
    else:
        set_level(INFO)

    log(INFO, 'running version %s with args %s', __version__,
        ', '.join([f'{k}={v}' for k, v in opts.items()]))

    log(INFO, 'using device: "%s"', str(get_device().type))

    try:
        run(**opts)
    except Exception as err:  # pylint: disable=broad-except
        from traceback import format_exc
        from .logging import LOGGER
        trace = err.__traceback__
        while trace.tb_next is not None:
            trace = trace.tb_next
        frame = trace.tb_frame
        LOGGER.findCaller = (
            lambda self, stack_info=None, f=frame:
            (f.f_code.co_filename, f.f_lineno, f.f_code.co_name, None)
        )
        log(ERROR, str(err))
        log(DEBUG, format_exc().strip())
        sys.exit(1)


if __name__ == '__main__':
    main()
