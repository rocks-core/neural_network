from .ActivationFunction import ActivationFunction


__all__ = ["Linear"]


class Linear(ActivationFunction):
    def __init__(self) -> None:
        super().__init__(
            lambda x: x,
            lambda _: 1
        )