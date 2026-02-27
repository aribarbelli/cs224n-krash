from typing import Callable, Iterable, Tuple
import math

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # State should be stored in this dictionary.
                state = self.state[p]

                # Access hyperparameters from the `group` dictionary.
                alpha = group["lr"]
                b_1, b_2 = group["betas"]

                ### TODO: Complete the implementation of AdamW here, reading and saving
                ###       your state in the `state` dictionary above.
                ###       The hyperparameters can be read from the `group` dictionary
                ###       (they are lr, betas, eps, weight_decay, as saved in the constructor).
                ###
                ###       To complete this implementation:
                ###       1. Update the first and second moments of the gradients.
                ###       2. Apply bias correction
                ###          (using the "efficient version" given in https://arxiv.org/abs/1412.6980;
                ###          also given in the pseudo-code in the project description).
                ###       3. Update parameters (p.data).
                ###       4. Apply weight decay after the main gradient-based updates.
                ###
                ###       Refer to the default project handout for more details.
                ### YOUR CODE HERE

                if len(state) == 0:
                    state["t"] = 0
                    state["m_t"] = torch.zeros_like(p.data)
                    state["v_t"] = torch.zeros_like(p.data)
                # t ← t + 1
                state["t"] += 1
                # gt ← ∇ft(θt−1) (Get gradients w.r.t. stochastic objective function at timestep t)
                # mt ← β1 · mt−1 + (1 − β1) · gt (Update biased first moment estimate)
                state["m_t"] = (b_1 * state["m_t"]) + ((1 - b_1) * grad)
                # vt ← β2 · vt−1 + (1 − β2) · gt^2
                state["v_t"] = (b_2 * state["v_t"]) + ((1 - b_2) * (grad**2))

                # (Update biased second raw moment estimate)
                if group["correct_bias"]:
                    # mˆt ← mt/(1 − β1^t) (Compute bias-corrected first moment estimate)
                    m_hat = state["m_t"] / (1 - (b_1**state["t"]))
                    # vˆt ← vt/(1 − β2^t) (Compute bias-corrected second raw moment estimate)
                    v_hat = state["v_t"] / (1 - (b_2**state["t"]))
                    denom = torch.sqrt(v_hat) + group["eps"]
                    update = m_hat / denom
                else:
                    denom = torch.sqrt(state["v_t"]) + group["eps"]
                    update = state["m_t"] / denom

                # θt ← θt−1 − α · mˆt/(sqrt(vˆt) + ϵ)
                p.data = p.data - alpha * update
                if group["weight_decay"] != 0.0:
                    p.data = p.data - alpha * group["weight_decay"] * p.data

        return loss
