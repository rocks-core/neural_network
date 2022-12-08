import numpy as np


class FoldResult:
    """
    Class that hold the result of multiple model with the same hyperparameters used in K-fold cross validation
    """
    def __init__(self, hp_config, results, name="", comments=""):
        """
        Args:
            hp_config (dict): configuration of hyperparameters
            results (list): list of result of the models used
            name (str, optional): optional name to recognize the results
            comments (str, optional): optional comments of the results
        """
        self.hp_config = hp_config
        self.results = results
        self.metrics = {}
        for m in results[0].metrics.keys():
            self.metrics[m] = np.array([r.metrics[m] for r in results]).mean()
        self.name = name
        self.comments = comments

    def plot(self, *args, **kwargs):
        """
        Plot the curves of the metrics specified in *args for all the results.

        Keyword Args:
            title (str): title for the plot
            save_path (str): if provided the path is used as a directory where all the plots are saved
            show (bool): if true display the plot
        """
        if "title" in kwargs:
            title = kwargs["title"]
        if "save_path" in kwargs:
            save = kwargs["save_path"]
        for i, r in enumerate(self.results):
            if "title" in kwargs:
                kwargs["title"] = title + "/fold" + str(i)

            if "save_path" in kwargs:
                kwargs["save_path"] = save + "/fold" + str(i)

            r.plot(*args, **kwargs)
