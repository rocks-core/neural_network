import pickle

from neural_network.classes.Results import *


class ResultCollection:
    def __init__(self):
        self.list = []

    def add_result(self, result):
        self.list.append(result)

    def sort(self, metric, reversed):
        self.list.sort(key=lambda r: r.metrics[metric], reverse=reversed)


    def __iter__(self):
        return self.list.__iter__()

    def plot(self, i, *args, **kwargs):
        self.list[i].plot(*args, **kwargs)

    def plot_all(self, *args, **kwargs):
        for r in self.list:
            r.plot(*args, **kwargs)

    def dump(self, path):
        with open(path, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(path):
        with open(path, "rb") as file:
            return pickle.load(file)
