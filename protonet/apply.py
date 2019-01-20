from collections import OrderedDict

import os
import os.path as osp

from typing import List

import numpy as np

import torch as t
from torch.utils.data import DataLoader

from .utils import (
    Interrupt,
    first_non_existant,
    get_device,
    make_csv_writer,
    step_function,
    with_interrupt_handler,
)

from .logging import INFO, WARNING, log

from .network import LSTM, MLP


def apply(
        network: t.nn.Module,
        data: DataLoader,
        output_prefix: str,
        track_outputs: bool = True,
        samples: int = 100,
):
    if track_outputs and isinstance(network, MLP):
        log(WARNING, 'can\'t track output of MLP, ignoring.')

    try:
        os.makedirs(output_prefix, exist_ok=True)
    except OSError as e:
        raise RuntimeError(f'failed to create output directory ({e})')

    def _interrupt_handler(*_):
        from multiprocessing import current_process
        if current_process().name == 'MainProcess':
            raise Interrupt()

    write_data = make_csv_writer(
        first_non_existant(osp.join(output_prefix, 'data.csv')),
        ['entry', 'position', 'sample', 'prediction0', 'prediction1'],
    )

    kwargs = (
        dict(track_outputs=track_outputs)
        if isinstance(network, LSTM) else
        {}
    )

    @with_interrupt_handler(_interrupt_handler)
    @step_function(get_device())
    def _run(x):
        predictions: List[t.Tensor] = []
        for i in range(samples):
            ys = [
                x.detach().cpu().numpy()
                for x in network(x['input'], **kwargs)
            ]
            if track_outputs:
                for e, j, z in (
                        (e, j, z)
                        for e, zs in zip(x['entry'], ys)
                        for j, z in enumerate(zs)
                ):
                    p1, p2 = np.exp(z)
                    write_data(OrderedDict((
                        ('entry', e),
                        ('position', j),
                        ('sample', i),
                        ('prediction0', p1),
                        ('prediction1', p2),
                    )))
                _predictions = np.stack([y[-1] for y in ys])
            else:
                for p1, p2 in t.exp(ys):
                    write_data(OrderedDict((
                        ('entry', e),
                        ('position', -1),
                        ('sample', i),
                        ('prediction0', p1),
                        ('prediction1', p2),
                    )))
                _predictions = ys
            predictions.append(_predictions)
        return dict(
            accuracy=np.mean(
                np.argmax(np.mean(np.stack(predictions), axis=0), axis=1)
                ==
                x['label'].detach().cpu().numpy()
            ),
        )

    try:
        _run(data)
    except Interrupt:
        log(INFO, 'interrupted, stopping')
