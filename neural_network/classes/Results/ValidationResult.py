import numpy as np
import pickle
import os


class ValidationResult:
    """
    Class that hold the result of one or more models with the same hyperparameters
    """
    def __init__(self, hp_config, results, name="", comments=""):
        """
        Args:
            hp_config (dict): configuration of hyperparameters
            results (list): list of result on validation set of the models used
            name (str, optional): optional name to recognize the results
            comments (str, optional): optional comments of the results
        """
        self.hp_config = hp_config
        self.results = results
        for r in self.results:
            r.hp_config = hp_config
        self.metrics = {}

        # do the average of all the metrics of the result provided
        for m in results[0].metrics.keys():
            self.metrics[m] = np.array([r.metrics[m] for r in results]).mean()
        self.name = name
        self.comments = comments

    def plot(self, *args, **kwargs):
        """
        Plot the curves of the metrics specified in *args for all the results.

        Keyword Args:
            title (str): title for the plot
            save_path (str): if there are multiple models the path provided is used as a directory where all the plots are saved
            show (bool): if true display the plots
        """
        if "title" in kwargs:
            title = kwargs["title"]
        if "save_path" in kwargs:
            save = kwargs["save_path"]
        for i, r in enumerate(self.results):
            if "title" in kwargs and len(self.results) > 1:
                kwargs["title"] = title + "/fold" + str(i)

            if "save_path" in kwargs and len(self.results) > 1:
                kwargs["save_path"] = save + "/fold" + str(i)

            r.plot(*args, **kwargs)

    def dump(self, path):
        folder = os.path.dirname(path)
        os.makedirs(folder, exist_ok=True)
        with open(path, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(path):
        with open(path, "rb") as file:
            return pickle.load(file)