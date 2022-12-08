import pickle
from typing import Union

from neural_network.classes.Results.ValidationResult import ValidationResult
from neural_network.classes.Results.TestResult import TestResult
from neural_network.classes.Results import Result


class ResultCollection:
    """
    Collection of multiple result of different models
    """
    def __init__(self):
        self.list = []

    def add_result(self, result: Union[Result, ValidationResult, TestResult]):
        self.list.append(result)

    def sort(self, metric, reverse):
        """
        Sort the collection according to the metric specified and reversing the order if reverse is True
        """
        self.list.sort(key=lambda r: r.metrics[metric], reverse=reverse)


    def __iter__(self):
        return self.list.__iter__()

    def plot_one(self, i, *args, **kwargs):
        """
        Plot a single result of the collection
        Args:
            i (int): the index of the element of the collection to be plotted
        """
        self.list[i].plot(*args, **kwargs)

    def plot(self, *args, **kwargs):
        """
        Plot all results in the collection
        """
        for r in self.list:
            r.plot(*args, **kwargs)

    def dump(self, path):
        with open(path, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(path):
        with open(path, "rb") as file:
            return pickle.load(file)
