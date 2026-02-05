import torch.nn as nn


class BaseLoss(nn.Module):
    """Base class for loss functions with common reduction options.

    Parameters
    ----------
    reduction : str, default="mean"
        Reduction to apply to elementwise losses. Supported values are
        ``"mean"``, ``"sum"``, and ``"none"``.
    **kwargs : Any
        Ignored for forward compatibility.
    """

    def __init__(self, reduction: str = "mean", **kwargs):
        super().__init__()
        self.reduction = reduction

    def _reduce(self, loss):
        """Reduce an elementwise loss tensor.

        Parameters
        ----------
        loss : torch.Tensor
            Elementwise loss values.

        Returns
        -------
        torch.Tensor
            Reduced loss tensor.
        """
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
