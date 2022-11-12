from .LossFunction import LossFunction

__all__ = ["MSE"]


class MSE(LossFunction):
    def __init__(self) -> None:
        super().__init__(
            lambda o, d: 0.5 * (o - d) ** 2,
            lambda o, d: -(o - d)
        )
