from numbers import Real
from typing import Any

import torch
from pyro.distributions import Bernoulli, constraints


class _ReinMaxMixin:
    has_rsample = True
    arg_constraints = {"temperature": constraints.greater_than(0)}

    def __init__(self, temperature: Real | torch.Tensor, *args, **kwargs):
        self.temperature = torch.as_tensor(temperature)
        super().__init__(*args, **kwargs)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(self.__class__, _instance)
        new.temperature = self.temperature
        return super().expand(batch_shape, new)


class ReinMaxBernoulli(_ReinMaxMixin, Bernoulli):
    """ReinMax version of the Bernoulli distribution.

    This class implements a Bernoulli distribution with ReinMax gradient estimation
    based on https://arxiv.org/pdf/2304.08612
    """

    arg_constraints = _ReinMaxMixin.arg_constraints | Bernoulli.arg_constraints

    def rsample(self, sample_shape: torch.Size = torch.Size()):
        sample = torch.atleast_1d(super().sample(sample_shape))
        return _ReinMaxGrad.apply(
            torch.stack((self.probs, 1 - self.probs), dim=-1),
            torch.stack((self.logits, -self.logits), dim=-1),
            self.temperature,
            torch.stack((sample, 1 - sample), dim=-1),
        )[..., 0]


class _ReinMaxGrad(torch.autograd.Function):
    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[torch.Tensor], output: torch.Tensor) -> None:
        ctx.save_for_backward(*inputs)

    @staticmethod
    def forward(
        probs: torch.Tensor, logits: torch.Tensor, temperature: torch.Tensor, sample: torch.Tensor
    ) -> torch.Tensor:
        return sample[:]  # we need the slice here, otherwise this doesn't play well with
        # pyro.infer.inspect.get_dependencies and becomes an empty tensor

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor | None]:
        probs, logits, temperature, sample = ctx.saved_tensors

        pid2 = 0.5 * (torch.softmax(logits / temperature, dim=-1) + sample)
        grad1 = 2 * grad_output * pid2
        grad1 = grad1 - grad1.sum(-1, keepdim=True) * pid2

        retgrad1, retgrad2 = None, None
        if ctx.needs_input_grad[0]:
            retgrad1 = (
                torch.autograd.grad(logits, probs, grad1, materialize_grads=True, retain_graph=True)[0]
                - 0.5 * grad_output
            )
        if ctx.needs_input_grad[1]:
            retgrad2 = (
                grad1
                - 0.5 * torch.autograd.grad(probs, logits, grad_output, materialize_grads=True, retain_graph=True)[0]
            )

        return (retgrad1, retgrad2, None, None)
