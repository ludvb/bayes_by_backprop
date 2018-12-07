import itertools as it

import os

from typing import Any, Callable, List, Optional, Tuple

import numpy as np

import torch as t
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

from tqdm import tqdm

from .utils import get_device, store_state, with_interrupt_handler
from .logging import INFO, log


class Interrupt(Exception):
    pass


def train(
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
            raise Interrupt()

    def checkpoint(name=None):
        store_state(
            network=network,
            optimizer=optimizer,
            path=checkpoint_path,
            prefix=name,
        )

    device = get_device()

    def _step_function(step_func) -> Callable[[Any], None]:
        def _wrapper(data_generator) -> Tuple[List[float], List[float]]:
            data_tracker = tqdm(data_generator, dynamic_ncols=True)
            loss: List[float] = []
            accuracy: List[float] = []
            weighted_accuracy: List[float] = []
            for x in data_tracker:
                y = step_func({
                    k: v.to(device) if isinstance(v, t.Tensor) else v
                    for k, v in x.items()
                })
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
            log(INFO, 'average accuracy=%2.1f', 100 * np.mean(accuracy))
            log(INFO, 'average wt. accuracy=%2.1f',
                100 * np.mean(weighted_accuracy))
            return loss, accuracy, weighted_accuracy

        return _wrapper

    @_step_function
    def _train_step(x):
        optimizer.zero_grad()
        y = network(x)
        y['loss'].backward()
        optimizer.step()
        return y

    @_step_function
    def _valid_step(x):
        return network(x)

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
                shuffle=False,
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

    try:
        _run_training_loop()
        log(INFO, 'all epochs finished, stopping')
    except Interrupt:
        log(INFO, 'interrupted, stopping')

    checkpoint('final')
