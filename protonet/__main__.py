#!/usr/bin/env python3

import argparse as ap

from datetime import datetime as dt

import os.path as osp

import sys

import numpy as np

import pandas as pd

import torch as t

from . import __version__
from .apply import apply
from .dataset import SignalProteins
from .logging import DEBUG, ERROR, INFO, log, set_level
from .network import MLP
from .train import train
from .utils import get_device, restore_state


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

    train_parser.set_defaults(run=train)
    apply_parser.set_defaults(
        run=lambda netwok, _optimizer, *args, **kwargs:
        apply(network, *args, **kwargs),
    )

    train_parser.add_argument('data_file', metavar='data', type=str)
    train_parser.add_argument('metadata_file', metavar='metadata', type=str)
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

    apply_parser.add_argument('data', type=str)
    apply_parser.add_argument('--state', required=True, type=t.load)

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

    log(DEBUG, 'reading data...')
    data = SignalProteins(
        pd.read_csv(
            opts.pop('data_file'),
            sep='\t',
            header=0,
            index_col=0,
        ),
        pd.read_csv(
            opts.pop('metadata_file'),
            sep='\t',
            header=0,
            index_col=0,
        ).drop('sequence', axis=1),
    )

    state = opts.pop('state')
    if state:
        log(INFO, 'restoring state from %s', state)
        state = restore_state(state)
        network = state['network']
        optimizer = state['optimizer']
        del opts['num_hidden']
        del opts['hidden_size']
    else:
        log(INFO, 'initializing network')
        signal_frac = np.sum(data.metadata.signal) / len(data.metadata)
        if opts.pop('mle'):
            from .network import Uninformative, Deterministic
            prior_distribution = Uninformative
            prior_kwargs = None
            variational_distribution = Deterministic
            variational_kwargs = None
        else:
            from .network import Normal, NormalMixture
            from .network import Uninformative, Deterministic
            prior_distribution = NormalMixture
            prior_kwargs = None
            variational_distribution = Normal
            variational_kwargs = None
        network = MLP(
            input_size=len(data[0]['input']),
            output_size=2,
            num_hidden=opts.pop('num_hidden'),
            hidden_size=opts.pop('hidden_size'),
            prior_distribution=prior_distribution,
            prior_kwargs=prior_kwargs,
            variational_distribution=variational_distribution,
            variational_kwargs=variational_kwargs,
            class_weights=[0.5 / (1 - signal_frac), 0.5 / signal_frac],
            learn_prior=opts.pop('learn_prior'),
        ).to(get_device())
        optimizer = t.optim.Adam(
            network.parameters(), lr=opts.pop('learning_rate'))

    try:
        run(
            network,
            optimizer,
            data,
            **opts,
        )
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
