import torch.nn as nn
import torch


def bce_loss_pytorch(predictions: torch.Tensor, targets: torch.Tensor, logits: bool=False) -> torch.Tensor:
    """
    Computes the binary cross entropy loss, either with logits or without.

    Parameters
    ----------
    predictions: Predictions of shape (N, dim), where dim is the dimensionality of the feature space.
    targets: Labels of shape (N, dim),where dim is the dimensionality of the feature space, or (N).
    logits: Whether to use torch's bce with logits function or torch's bce function.

    Returns
    -------
    loss: Loss value.
    """

    if predictions.size()[1] == 1:
        predictions = predictions.view(-1)

    if logits:
        loss_fun = nn.BCEWithLogitsLoss()
    else:
        loss_fun = nn.BCELoss()
    loss = loss_fun(predictions, targets)
    return loss
