from collections import OrderedDict

from typing import Any, Callable, List, Optional, Tuple

import numpy as np

import torch as t
from torch.optim.optimizer import Optimizer

from tqdm import tqdm

from .logging import INFO, log


class Interrupt(Exception):
    pass


def collect_items(d):
    d_ = {}
    for k, v in d.items():
        try:
            d_[k] = v.item()
        except (ValueError, AttributeError):
            pass
    return d_


def step_function(device) -> Callable[
        [t.device],
        Callable[[Any], Tuple[List[float], List[float]]],
]:
    def _decorator(step_func):
        def _wrapper(data_generator):
            results: List[dict] = []

            def _fmt(k, v):
                return (
                    f'{k}: {v:.5f}'
                    if abs(v) < 10 or abs(v) < 0.10 else
                    f'{k}: {v:.2e}'
                )

            data_tracker = tqdm(data_generator, dynamic_ncols=True)
            for x in data_tracker:
                y = step_func(to_device(x, device))
                results.append(collect_items(y))
                data_tracker.set_description(
                    ' / '.join([_fmt(k, v) for k, v in results[-1].items()]))

            results_ = zip_dicts(results)
            log(INFO, 'means: %s', ', '.join([
                _fmt(k, m)
                for k, v in results_.items() for m in (np.mean(v),)
            ]))

            return results_
        return _wrapper
    return _decorator


def get_device() -> t.device:
    return t.device('cuda' if t.cuda.is_available() else 'cpu')


def to_device(obj, device):
    if isinstance(obj, t.Tensor):
        obj.data = obj.to(device)
    elif isinstance(obj, list) or isinstance(obj, tuple):
        obj = type(obj)([to_device(x, device) for x in obj])
    elif isinstance(obj, dict):
        obj = {k: to_device(v, device) for k, v in obj.items()}
    return obj


def with_interrupt_handler(handler):
    from functools import wraps
    import signal

    def _decorator(func):
        @wraps(func)
        def _wrapper(*args, **kwargs):
            signal.signal(signal.SIGINT, handler)
            func(*args, **kwargs)
            signal.signal(signal.SIGINT, signal.SIG_IGN)

        return _wrapper

    return _decorator


def store_state(
        network: t.nn.Module,
        optimizer: Optimizer,
        path: str,
        prefix: Optional[str] = None,
) -> None:
    import os.path as osp
    from .logging import INFO, log

    if not prefix:
        prefix = 'checkpoint'

    filename = osp.join(path, f'{prefix}.pkl')
    log(INFO, 'saving state to %s...', filename)

    t.save(
        dict(
            network=network,
            optimizer=optimizer,
        ),
        filename,
    )


def restore_state(state_dict: dict) -> dict:
    dev = get_device()
    state_dict['network'] = state_dict['network'].to(dev)
    for weight_params in state_dict['optimizer'].state.values():
        param: t.Tensor
        for param in filter(t.is_tensor, weight_params.values()):
            param.data = param.to(dev)
    return state_dict


def make_csv_writer(
        path: str,
        headers: List[str],
):
    with open(path, 'w') as f:
        f.write(','.join(headers) + '\n')

    def _write_data(data: OrderedDict):
        def _checkvalue(x):
            k, (k_, v) = x
            if k != k_:
                raise ValueError(
                    'failed to write csv data '
                    f'(column name mismatch: "{k}" != "{k_}")'
                )
            return str(v)

        with open(path, 'a') as f:
            f.write(
                ','.join(map(_checkvalue, zip(headers, data.items())))
                +
                '\n'
            )

    return _write_data


def zip_dicts(ds: List[dict]) -> dict:
    d = {k: [] for k in ds[0].keys()}
    for d_ in ds:
        for k, v in d_.items():
            try:
                d[k].append(v)
            except AttributeError:
                raise ValueError('dict keys are inconsistent')
    return d
