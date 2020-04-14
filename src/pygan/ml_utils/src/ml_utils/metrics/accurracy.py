import numpy as np
import torch


def accuracy_pytorch(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Computes the binary cross entropy loss, either with logits or without.

    Parameters
    ----------
    predictions: Predictions of shape (N, dim), where dim is the dimensionality of the feature space.
    targets: Labels of shape (N, dim),where dim is the dimensionality of the feature space, or (N).

    Returns
    -------
    loss: Loss value.
    """

    if predictions.size()[1] == 1:
        predictions = predictions.view(-1).detach().cpu().numpy()
        predictions = predictions[predictions>0.5].astype(np.uint8)
        targets = targets.detach().cpu().numpy().astype(np.uint8)
        accuracy = np.mean(np.equal(predictions, targets).astype(np.uint8), axis=0).item()
    else:
        print('Prediction size not supported')
        accuracy = 0.0
    return accuracy



