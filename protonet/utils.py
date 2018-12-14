from typing import Optional

import torch as t
from torch.optim.optimizer import Optimizer


def get_device() -> t.device:
    return t.device('cuda' if t.cuda.is_available() else 'cpu')


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
    import datetime as dt
    import os.path as osp
    from .logging import INFO, log

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
    dev = get_device()
    state_dict['network'] = state_dict['network'].to(dev)
    for weight_params in state_dict['optimizer'].state.values():
        param: t.Tensor
        for param in filter(t.is_tensor, weight_params.values()):
            param.data = param.to(dev)
    return state_dict
