import pickle

from neural_network.classes.Results import *


class ResultCollection:
    def __init__(self):
        self.dictionary = {}

    def add_result(self, result, key=None):
        if not key:
            key = len(self.dictionary)
        self.dictionary[key] = result

    def __iter__(self):
        return self.dictionary.items()

    def plot(self, key, *args, **kwargs):
        self.dictionary[key].plot(*args, **kwargs)

    def plot_all(self, *args, **kwargs):
        for r in self.dictionary.values():
            r.plot(*args, **kwargs)

    def dump(self, path):
        with open(path, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(path):
        with open(path, "rb") as file:
            return pickle.load(file)
