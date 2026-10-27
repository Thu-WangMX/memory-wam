"""Collect logging scalars without synchronizing the training hot path."""

import torch


def gather_training_metrics(accelerator, loss, loss_dict, grad_norm):
    """Gather one optimizer step's existing metrics, only on logging steps.

    All ranks must call this together. Loss dictionaries have the same keys on
    every rank, as required by the former per-key collectives. Scalars remain
    detached on the device until a single gather and a single host transfer.
    This deliberately retains the last accumulation microbatch's values rather
    than introducing averaging over microbatches or over the logging interval.
    """

    def scalar(value):
        if torch.is_tensor(value):
            value = value.detach().to(device=loss.device, dtype=torch.float32)
        else:
            value = torch.tensor(float(value), device=loss.device, dtype=torch.float32)
        if value.numel() != 1:
            raise ValueError('Training log metrics must be scalars')
        return value.reshape(())

    keys = list(loss_dict)
    local = torch.stack([scalar(loss), *(scalar(loss_dict[key]) for key in keys), scalar(grad_norm)])
    gathered = accelerator.gather(local.unsqueeze(0)).reshape(-1, local.numel())
    # Reduce each contiguous rank vector exactly as the former scalar gather
    # did; a matrix mean(dim=0) may choose a different floating-point reduction.
    means = torch.stack([column.contiguous().mean() for column in gathered.unbind(dim=1)])
    values = means.cpu().tolist()
    return values[0], dict(zip(keys, values[1:-1])), values[-1]
