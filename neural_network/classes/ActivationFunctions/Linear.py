from .ActivationFunction import ActivationFunction
import numpy as np


__all__ = ["Linear"]


class Linear(ActivationFunction):
    def __init__(self) -> None:
        super().__init__(
            lambda v: v,
            lambda v: np.ones(shape=v.shape)
        )