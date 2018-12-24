from abc import abstractmethod

from collections import OrderedDict

from functools import reduce

import operator as op

from typing import List, Optional

import numpy as np

import torch as t
from torch.nn import init

from .logging import log, DEBUG


FLOAT_MAX = 1e10


class SamplingError(Exception):
    pass


class Distribution(t.nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    @abstractmethod
    def sample(self):
        pass

    @abstractmethod
    def p(self, x):
        pass


class Normal(Distribution):
    def __init__(self, shape, loc=None, scale=None):
        loc = t.as_tensor(loc or 0.)
        scale = t.as_tensor(scale or 0.)
        super().__init__(shape)
        [self.loc, self.scale_] = [
            t.nn.Parameter(t.ones(self.shape) * x.detach())
            for x in (loc, scale)
        ]

    @property
    def scale(self):
        return t.nn.functional.softplus(self.scale_)

    def sample(self):
        return self.loc + self.scale * t.randn_like(self.scale)

    def p(self, x):
        return t.distributions.Normal(self.loc, self.scale).log_prob(x)


class NormalMixture(Distribution):
    def __init__(self, shape, loc1=None, scale1=None, loc2=None, scale2=None):
        loc1 = t.as_tensor(loc1 or 0.)
        scale1 = t.as_tensor(scale1 or -2.)
        loc2 = t.as_tensor(loc2 or 0.)
        scale2 = t.as_tensor(scale2 or 10.)
        super().__init__(shape)
        self.d1 = Normal(shape, loc1, scale1)
        self.d2 = Normal(shape, loc2, scale2)

    def sample(self):
        if np.random.uniform() > 0.5:
            return self.d1.sample()
        return self.d2.sample()

    def p(self, x):
        return (
            np.log(0.5)
            +
            t.logsumexp(
                t.cat((self.d1.p(x)[None, ...], self.d2.p(x)[None, ...])),
                dim=0,
            )
        )


class Uninformative(Distribution):
    def sample(self):
        raise SamplingError(
            'attempted to sample from uninformative distribution!')

    def p(self, x):
        return FLOAT_MAX * t.ones(self.shape, requires_grad=False).to(x)


class Deterministic(Distribution):
    def __init__(self, shape, x=None):
        super().__init__(shape)
        if x is None:
            x = t.tensor(0.)
        self.x = t.nn.Parameter(
            t.as_tensor(x).clone().detach() * t.ones(shape))

    def sample(self):
        return self.x

    def p(self, x):
        return (FLOAT_MAX *
                (1 - 2 * t.ones(self.shape, requires_grad=False).to(x) *
                 (x != self.x).float()))


class Linear(t.nn.Module):
    def __init__(
            self,
            prior_w: Distribution,
            prior_b: Distribution,
            variational_w: Distribution,
            variational_b: Distribution,
            learn_prior: bool = False,
    ):
        super().__init__()
        self.prior_w = prior_w
        self.prior_b = prior_b
        self.variational_w = variational_w
        self.variational_b = variational_b
        self.w, self.b = None, None
        if not learn_prior:
            for d in (self.prior_w, self.prior_b):
                for p in d.parameters():
                    p.requires_grad = False

    def forward(self, x):
        self.w = self.variational_w.sample()
        self.b = self.variational_b.sample()
        return x @ self.w + self.b


class MLP(t.nn.Module):
    def __init__(
            self,
            input_size: int,
            output_size: int,
            num_hidden: int,
            hidden_size: int,
            class_weights: Optional[List[float]] = None,
            activation: Optional[t.nn.Module] = None,
            prior_distribution: type = Uninformative,
            prior_kwargs: Optional[dict] = None,
            variational_distribution: type = Deterministic,
            variational_kwargs: Optional[dict] = None,
            **kwargs,
    ):
        if class_weights is None:
            class_weights = np.repeat(1., output_size)

        if activation is None:
            activation = t.nn.LeakyReLU(inplace=True)

        if prior_kwargs is None:
            prior_kwargs = {}

        if variational_kwargs is None:
            variational_kwargs = {}

        super().__init__()

        self.class_weights = t.nn.Parameter(
            t.as_tensor(class_weights).float(), requires_grad=False)
        log(DEBUG, 'class weights: %s',
            ' | '.join([f'{i:4d}' for i in range(len(self.class_weights))]))
        log(DEBUG, '               %s',
            ' | '.join([f'{w:1.2f}' for w in self.class_weights]))

        log(DEBUG, 'prior distribution=%s(%s)', prior_distribution.__name__,
            ','.join([f'{a}={b}' for a, b in prior_kwargs]))
        log(DEBUG, 'variational distribution=%s(%s)',
            variational_distribution.__name__, ','.join(
                [f'{a}={b}' for a, b in variational_kwargs.items()]))

        self.layers = []

        w_prior = prior_distribution((1, ), **variational_kwargs)
        b_prior = prior_distribution((1, ), **variational_kwargs)

        def _make_layer(in_size, out_size):
            layer = Linear(
                w_prior, b_prior,
                variational_distribution((in_size, out_size), **prior_kwargs),
                variational_distribution((out_size, ), **prior_kwargs),
                **kwargs,
            )

            # if the variational distribution is deterministic, we need to
            # break symmetry by applying some jitter to the weights
            if variational_distribution == Deterministic:
                init.kaiming_uniform_(layer.variational_w.x, a=np.sqrt(5))

            self.layers.append(layer)
            return layer

        self._forward = t.nn.Sequential(
            _make_layer(input_size, hidden_size),
            activation,
            *reduce(op.add, [[
                _make_layer(hidden_size, hidden_size),
                activation,
            ] for _ in range(num_hidden)]),
            _make_layer(hidden_size, output_size),
            t.nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        inputs = x['input']
        labels = x['label']

        y = self._forward(inputs)

        likelihood_loss = t.nn.functional.nll_loss(
            y, labels, self.class_weights)
        prior_loss = sum([
            t.sum(l.variational_w.p(l.w) - l.prior_w.p(l.w))
            +
            t.sum(l.variational_b.p(l.b) - l.prior_b.p(l.b))
            for l in self.layers
        ])

        return OrderedDict(
            prediction=y,
            loss=likelihood_loss + prior_loss,
            acc=t.mean((t.argmax(y, dim=1) == labels).float()),
            wt_acc=t.sum(
                (t.argmax(y, dim=1) == labels).float()
                *
                self.class_weights[labels]
            ) / t.sum(self.class_weights[labels]),
            px=-likelihood_loss,
            dqp=prior_loss,
        )
