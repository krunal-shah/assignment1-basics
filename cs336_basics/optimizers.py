import torch
import math
from collections.abc import Callable


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1, eps=1e-8):
        defaults = {"lr": lr, "beta1": betas[0], "beta2": betas[1], "weight_decay": weight_decay, "eps": eps}
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 1)
                lr_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                momentum = beta1 * state.get("momentum", 0) + (1 - beta1) * p.grad.data
                velocity = beta2 * state.get("velocity", 0) + (1 - beta2) * p.grad.data * p.grad.data

                p.data -= lr * weight_decay * p.data
                p.data -= lr_t * momentum / (eps + torch.sqrt(velocity))

                state["momentum"] = momentum
                state["velocity"] = velocity
                state["t"] = t + 1
        return loss

    def update_lr(self, lr):
        for group in self.param_groups:
            group["lr"] = lr


def cosine_lr_scheduler(t, lr_max, lr_min, warmup_itr, annealing_itr):
    if t < warmup_itr:
        return t * lr_max / warmup_itr
    elif t <= annealing_itr:
        return (
            lr_min + (1 + math.cos(math.pi * (t - warmup_itr) / (annealing_itr - warmup_itr))) * (lr_max - lr_min) / 2
        )
    return lr_min


def gradient_clipping(parameters, max_norm, eps=1e-6):
    squared_norm = 0
    parameters_list = []
    for parameter in parameters:
        if parameter.grad is not None:
            parameters_list.append(parameter)
            squared_norm += torch.sum(torch.square(parameter.grad.data))
    norm = math.sqrt(squared_norm)

    if norm > max_norm:
        for parameter in parameters_list:
            if parameter.grad is not None:
                parameter.grad = parameter.grad * max_norm / (norm + eps)

    return norm
