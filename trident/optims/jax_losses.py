"""JAX losses. Classification and image losses use channels last by default."""
from trident.optims.losses import PUBLIC_LOSSES
from trident.optims.loss_kernels import (
    ClassificationLoss as _ClassificationLoss,
    ActiveContourLoss, AdaptiveWingLoss, BCELoss, BCEWithLogitsLoss, CenterLoss,
    CosineSimilarityLoss, CrossEntropyLoss, DiceLoss, EdgeLoss, ExponentialLoss,
    F1ScoreLoss, FocalLoss, GPLoss, GeneralizedDiceLoss, HardTripletLoss,
    HuberLoss, IoULoss, ItakuraSaitoLoss, KLDivergenceLoss, L1Loss,
    L2Loss, LovaszSoftmax, MSELoss, MS_SSIMLoss, NLLLoss,
    PerceptionLoss, SmoothL1Loss, SoftIoULoss, StyleLoss, TransformInvariantLoss,
    TripletLoss, TripletMarginLoss, TverskyLoss, WingLoss,
)

__all__ = ['get_loss', '_ClassificationLoss', *PUBLIC_LOSSES]


def get_loss(loss_name):
    if loss_name is None:
        return None
    if loss_name not in __all__ or loss_name == 'get_loss':
        raise ValueError(f"Unknown JAX loss {loss_name!r}")
    return globals()[loss_name]
