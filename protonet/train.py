from collections import OrderedDict

import itertools as it

import os
import os.path as osp

from typing import List, Optional

import numpy as np

import torch as t
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from .utils import (
    Interrupt,
    collect_items,
    get_device,
    make_csv_writer,
    step_function,
    store_state,
    with_interrupt_handler,
    zip_dicts,
)

from .logging import INFO, log


def train(
        network: t.nn.Module,
        optimizer: Optimizer,
        training_set: DataLoader,
        validation_set: DataLoader,
        output_prefix: str,
        nepochs: Optional[int] = None,
        update_samples: int = 1,
        start_epoch: int = 1,
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

    def checkpoint(epoch, name=None):
        store_state(
            network=network,
            optimizer=optimizer,
            epoch=epoch,
            path=osp.join(output_prefix, 'checkpoints'),
            prefix=name,
        )

    device = get_device()

    @step_function(device)
    def _train_step(x):
        optimizer.zero_grad()
        results: List[dict] = []
        for _ in range(update_samples):
            y = network.forward_with_loss(x)
            results.append(collect_items(y))
            y['loss'].backward()
        optimizer.step()
        return {k: np.mean(v) for k, v in zip_dicts(results).items()}

    @step_function(device)
    def _valid_step(x):
        return network.forward_with_loss(x)

    write_data = make_csv_writer(
        osp.join(output_prefix, 'training_data.csv'),
        ['epoch', 'iteration', 'validation', 'type', 'value'],
    )

    epoch: int

    @with_interrupt_handler(_interrupt_handler)
    def _run_training_loop():
        nonlocal epoch

        best_loss: float = float('inf')

        for epoch in it.takewhile(
                lambda x: nepochs is None or x <= nepochs,
                it.count(start_epoch),
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
                checkpoint(epoch, f'epoch-{epoch:03d}')

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

    checkpoint(epoch, 'final')
