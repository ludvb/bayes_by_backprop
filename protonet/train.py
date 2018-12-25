from collections import OrderedDict

import itertools as it

import os
import os.path as osp

from typing import Any, Callable, List, Optional, Tuple

import numpy as np

import torch as t
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from tqdm import tqdm

from .utils import (
    get_device,
    make_csv_writer,
    store_state,
    with_interrupt_handler,
    zip_dicts,
)

from .logging import INFO, log


class Interrupt(Exception):
    pass


def train(
        network: t.nn.Module,
        optimizer: Optimizer,
        training_set: DataLoader,
        validation_set: DataLoader,
        output_prefix: str,
        nepochs: Optional[int] = None,
        update_samples: int = 1,
):
    try:
        os.makedirs(output_prefix)
        os.makedirs(osp.join(output_prefix, 'checkpoints'))
    except OSError as e:
        raise RuntimeError(f'failed to create output directory ({e})')

    def _interrupt_handler(*_):
        from multiprocessing import current_process
        if current_process().name == 'MainProcess':
            raise Interrupt()

    def checkpoint(name=None):
        store_state(
            network=network,
            optimizer=optimizer,
            path=osp.join(output_prefix, 'checkpoints'),
            prefix=name,
        )

    def _collect_items(d):
        d_ = {}
        for k, v in d.items():
            try:
                d_[k] = v.item()
            except (ValueError, AttributeError):
                pass
        return d_

    device = get_device()

    def _step_function(step_func) -> Callable[[Any], None]:
        def _wrapper(data_generator) -> Tuple[List[float], List[float]]:
            results: List[dict] = []

            def _fmt(k, v):
                return (
                    f'{k}: {v:.5f}'
                    if abs(v) < 10 or abs(v) < 0.10 else
                    f'{k}: {v:.2e}'
                )

            data_tracker = tqdm(data_generator, dynamic_ncols=True)
            for x in data_tracker:
                y = step_func({
                    k: v.to(device) if isinstance(v, t.Tensor) else v
                    for k, v in x.items()
                })
                results.append(_collect_items(y))
                data_tracker.set_description(
                    ' / '.join([_fmt(k, v) for k, v in results[-1].items()]))

            results_ = zip_dicts(results)
            log(INFO, 'means: %s', ', '.join([
                _fmt(k, m)
                for k, v in results_.items() for m in (np.mean(v),)
            ]))

            return results_
        return _wrapper

    @_step_function
    def _train_step(x):
        optimizer.zero_grad()
        results: List[dict] = []
        for _ in range(update_samples):
            y = network(x)
            results.append(_collect_items(y))
            y['loss'].backward()
        optimizer.step()
        return {k: np.mean(v) for k, v in zip_dicts(results).items()}

    @_step_function
    def _valid_step(x):
        return network(x)

    write_data = make_csv_writer(
        osp.join(output_prefix, 'training_data.csv'),
        ['epoch', 'iteration', 'validation', 'type', 'value'],
    )

    @with_interrupt_handler(_interrupt_handler)
    def _run_training_loop():
        best_loss: float = float('inf')

        for epoch in it.takewhile(
                lambda x: nepochs is None or x <= nepochs,
                it.count(1),
        ):
            log(INFO, 'starting epoch %d of %s...', epoch,
                str(nepochs) if nepochs else 'inf')

            log(INFO, 'performing training steps...')
            network.train()
            t.enable_grad()
            train_data = _train_step(training_set)

            log(INFO, 'performing validation steps...')
            network.eval()
            t.no_grad()
            valid_data = _valid_step(validation_set)

            avg_loss = np.mean(valid_data['loss'])
            if best_loss > avg_loss:
                best_loss = avg_loss
                checkpoint(f'epoch-{epoch:03d}')

            log(INFO, 'writing training data')
            for validation, data in ((0, train_data), (1, valid_data)):
                for k, vs in data.items():
                    for i, v in enumerate(vs):
                        write_data(OrderedDict([
                            ('epoch', epoch),
                            ('iteration', i),
                            ('validation', validation),
                            ('type', k),
                            ('value', v),
                        ]))

    try:
        _run_training_loop()
        log(INFO, 'all epochs finished, stopping')
    except Interrupt:
        log(INFO, 'interrupted, stopping')

    checkpoint('final')
