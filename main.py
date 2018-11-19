#!/usr/bin/env python3

import argparse as ap

from functools import partial, reduce, wraps

import itertools as it

import logging
from logging import log, DEBUG, INFO

import os

import operator as op

import subprocess as sp

import sys

from typing import Any, Callable, List, Optional, Tuple

import numpy as np

import pandas as pd

import torch as t
from torch.nn import init
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import Subset

from tqdm import tqdm

VERSION: str = (sp.run(
    'git describe --dirty --always',
    capture_output=True,
    shell=True,
).stdout.decode().strip())

DEVICE: t.device = t.device('cuda' if t.cuda.is_available() else 'cpu')


def with_interrupt_handler(handler):
    import signal

    def _decorator(func):
        @wraps(func)
        def _wrapper(*args, **kwargs):
            signal.signal(signal.SIGINT, handler)
            func(*args, **kwargs)
            signal.signal(signal.SIGINT, signal.SIG_IGN)

        return _wrapper

    return _decorator


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
                f'({len(self.data)} vs. ({len(self.metadata)}))',
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return dict(
            input=t.tensor(self.data.iloc[idx].values).float(),
            label=t.tensor(self.metadata.signal[idx]).long(),
        )


class Linear(t.nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weights = t.Tensor(in_features, out_features)
        self.bias = t.Tensor(out_features)

    def forward(self, x) -> t.Tensor:
        return (x @ self.weights) + self.bias


class MLELinear(Linear):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        self.weights = t.nn.Parameter(self.weights)
        self.bias = t.nn.Parameter(self.bias)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weights, a=np.sqrt(5))
        self.bias.data[...] = 0.


class BNNLinear(Linear):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        self.w_mu = t.nn.Parameter(t.Tensor(*self.weights.shape))
        self.w_sd = t.nn.Parameter(t.Tensor(*self.weights.shape))

        self.b_mu = t.nn.Parameter(t.Tensor(*self.bias.shape))
        self.b_sd = t.nn.Parameter(t.Tensor(*self.bias.shape))

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.w_mu, a=np.sqrt(5))
        self.b_mu.data[...] = 0.

        self.w_sd.data[...] = -100.
        self.b_sd.data[...] = -100.

    def forward(self, x: t.Tensor) -> t.Tensor:
        self.weights = (
            self.w_mu + t.log(1 + t.exp(self.w_sd)) * t.randn_like(self.w_sd))
        self.bias = (
            self.b_mu + t.log(1 + t.exp(self.b_sd)) * t.randn_like(self.b_sd))
        return super().forward(x)


def mle_loss(network, outputs, labels, class_weights):
    return -(
        t.sum(
            class_weights[labels]
            * t.log(outputs[range(len(outputs)), labels]),
        )
        /
        t.sum(class_weights[labels])
    )


def bnn_loss(network, outputs, labels, class_weights):
    return (
        mle_loss(network, outputs, labels, class_weights) +
        reduce(
            op.add,
            ((
                t.sum(
                    t.distributions.kl_divergence(
                        t.distributions.Normal(layer.w_mu, layer.w_sd),
                        t.distributions.Normal(0., 1.),
                    ))
                +
                t.sum(
                    t.distributions.kl_divergence(
                        t.distributions.Normal(layer.b_mu, layer.b_sd),
                        t.distributions.Normal(0., 1.),
                    ))
            ) for layer in network._forward if isinstance(layer, BNNLinear))
        ))


class MLP(t.nn.Module):
    def __init__(
            self,
            input_size: int,
            output_size: int,
            num_hidden: int,
            hidden_size: int,
            bnn: bool = False,
            class_weights: Optional[List[float]] = None,
            activation: Optional[t.nn.Module] = None,
    ):
        super().__init__()

        self.class_weights = t.tensor(
            class_weights if class_weights
            else np.repeat(1., output_size)
        )

        if bnn:
            layer_type = BNNLinear
            self._loss = partial(
                bnn_loss,
                self,
                class_weights=self.class_weights,
            )
        else:
            layer_type = MLELinear
            self._loss = partial(
                mle_loss,
                self,
                class_weights=self.class_weights,
            )

        if activation is None:
            activation = t.nn.LeakyReLU(inplace=True)

        self._forward = t.nn.Sequential(
            layer_type(input_size, hidden_size),
            activation,
            *reduce(op.add, [[
                layer_type(hidden_size, hidden_size),
                activation,
            ] for _ in range(num_hidden)]),
            layer_type(hidden_size, output_size),
            t.nn.Softmax(dim=1),
        )

    def to(self, *args, **kwargs):
        self.class_weights.data = self.class_weights.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def forward(self, x):
        inputs = x['input']
        labels = x['label']

        y = self._forward(inputs)

        return dict(
            prediction=y,
            accuracy=t.mean((t.argmax(y, dim=1) == labels).float()).item(),
            weighted_accuracy=(
                t.sum(
                    self.class_weights[labels]
                    * (t.argmax(y, dim=1) == labels).float(),
                ).item()
                /
                t.sum(self.class_weights[labels]).item()
            ),
            loss=self._loss(y, labels),
        )


def store_state(
        network: t.nn.Module,
        optimizer: Optimizer,
        path: str,
        prefix: Optional[str] = None,
) -> None:
    import datetime as dt
    import os.path as osp

    if not prefix:
        prefix = 'checkpoint'

    name = f'{prefix}-{dt.datetime.now().isoformat()}'

    filename = osp.join(path, f'{name}.pkl')
    log(INFO, 'saving state to %s...', filename)

    t.save(
        dict(
            network=network,
            optimizer=optimizer,
        ),
        filename,
    )


def restore_state(state_dict: dict) -> dict:
    state_dict['network'].to(DEVICE)
    for weight_params in state_dict['optimizer'].state.values():
        param: t.Tensor
        for param in filter(t.is_tensor, weight_params.values()):
            param.data = param.to(DEVICE)
    return state_dict


def run_train(
        network: t.nn.Module,
        optimizer: Optimizer,
        data: DataLoader,
        checkpoint_path: str,
        nepochs: Optional[int] = None,
        k_fold_split: int = 10,
):
    def _interrupt_handler(*_):
        from multiprocessing import current_process
        if current_process().name == 'MainProcess':
            raise Exception("interrupted")  # u.Interrupted()

    def checkpoint(name=None):
        store_state(
            network=network,
            optimizer=optimizer,
            path=checkpoint_path,
            prefix=name,
        )

    def _step_function(output_hook) -> Callable[[Any], None]:
        def _wrapper(data_generator) -> Tuple[List[float], List[float]]:
            data_tracker = tqdm(data_generator, dynamic_ncols=True)
            loss: List[float] = []
            accuracy: List[float] = []
            weighted_accuracy: List[float] = []
            for x in data_tracker:
                optimizer.zero_grad()
                y = network({
                    k: v.to(DEVICE) if isinstance(v, t.Tensor) else v
                    for k, v in x.items()
                })
                if np.isnan(y['loss'].cpu().detach().numpy()):
                    import ipdb; ipdb.set_trace()
                output_hook(y)
                loss += [y['loss'].item()]
                accuracy += [y['accuracy']]
                weighted_accuracy += [y['weighted_accuracy']]
                data_tracker.set_description(
                    ' / '.join([
                        f'loss: {np.mean(loss):.4e}',
                        f'acc: {np.mean(accuracy) * 100:.1f} %',
                        f'wt.acc: {np.mean(weighted_accuracy) * 100:.1f} %',
                    ]),
                )
            log(INFO, 'average loss=%.4e', np.mean(loss))
            log(INFO, 'average accuracy=%2.1f', np.mean(accuracy))
            log(INFO, 'average accuracy=%2.1f', np.mean(weighted_accuracy))
            return loss, accuracy, weighted_accuracy

        return _wrapper

    @_step_function
    def _train_step(y):
        y['loss'].backward()
        optimizer.step()

    @_step_function
    def _valid_step(_y):
        pass

    @with_interrupt_handler(_interrupt_handler)
    def _run_training_loop():
        best_loss: float = float('inf')

        for epoch in it.takewhile(
                lambda x: nepochs is None or x <= nepochs,
                it.count(1),
        ):
            log(INFO, 'starting epoch %d of %s...', epoch,
                str(nepochs) if nepochs else 'inf')

            log(INFO, 'performing %d-fold split...', k_fold_split)
            train_set_idxs = np.random.choice(
                len(data),
                int((1 - 1 / k_fold_split) * len(data)),
                replace=False,
            )
            valid_set_idxs = np.setdiff1d(range(len(data)), train_set_idxs)
            train_set = DataLoader(
                Subset(data, train_set_idxs),
                shuffle=True,
                batch_size=2048,
                num_workers=len(os.sched_getaffinity(0)),
            )
            valid_set = DataLoader(
                Subset(data, valid_set_idxs),
                shuffle=False,
                batch_size=2048,
                num_workers=len(os.sched_getaffinity(0)),
            )

            log(INFO, 'performing training steps...')
            network.train()
            t.enable_grad()
            _, _, _ = _train_step(train_set)

            log(INFO, 'performing validation steps...')
            network.eval()
            t.no_grad()
            loss, _, _ = _valid_step(valid_set)

            avg_loss = np.mean(loss)
            if best_loss > avg_loss:
                best_loss = avg_loss
                checkpoint()

    _run_training_loop()
    log(INFO, 'all epochs done')
    checkpoint('final')


def run_apply(
        network: t.nn.Module,
        data: DataLoader,
):
    pass


def main():
    from tempfile import gettempdir

    args = ap.ArgumentParser(
        description=f'Version: {VERSION:s}',
        formatter_class=ap.RawTextHelpFormatter,
    )
    args.add_argument('-v', '--version', action='version', version=VERSION)
    args.add_argument('--verbose', action='store_true')

    sp = args.add_subparsers()
    train_parser = sp.add_parser('train')
    apply_parser = sp.add_parser('apply')

    train_parser.set_defaults(run=run_train)
    apply_parser.set_defaults(
        run=lambda netwok, _optimizer, *args, **kwargs:
        run_apply(network, *args, **kwargs),
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
        '--bnn',
        action='store_true',
        help='use variational weights',
    )
    train_parser.add_argument(
        '--epochs',
        type=int,
        dest='nepochs',
    )
    train_parser.add_argument(
        '--checkpoint-path',
        type=str,
        default=gettempdir(),
        help='checkpoint path',
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
        logging.getLogger().setLevel(DEBUG)
    else:
        logging.getLogger().setLevel(INFO)

    log(INFO, 'running version %s with args %s', VERSION,
        ', '.join([f'{k}={v}' for k, v in opts.items()]))

    log(INFO, 'using device: "%s"', str(DEVICE.type))

    log(INFO, 'reading data...')
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
        network = state['network']
        optimizer = state['optimizer']
    else:
        log(INFO, 'initializing network')
        signal_frac = np.sum(data.metadata.signal) / len(data.metadata)
        network = MLP(
            input_size=len(data[0]['input']),
            output_size=2,
            num_hidden=opts.pop('num_hidden'),
            hidden_size=opts.pop('hidden_size'),
            bnn=opts.pop('bnn'),
            class_weights=[1 / (1 - signal_frac), 1 / signal_frac],
        ).to(DEVICE)
        optimizer = t.optim.Adam(network.parameters())  # , lr=1e-5)

    run(
        network,
        optimizer,
        data,
        **opts,
    )


if __name__ == '__main__':
    main()
