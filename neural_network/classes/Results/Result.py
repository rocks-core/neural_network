import matplotlib.pyplot as plt
import pickle


class Result:
    """
    Class that hold the result of a single model
    """
    def __init__(self, metrics: dict, history: dict, hp_config: dict = None, name="", comments=""):
        """
        Args:
            metrics (dict): dict of final score on metrics after training
            history (dict): history of score on metrics during the training (used to plot training curves)
            hp_config (dict, optional): dictionary of hyperparameter configuration used (None if Result is part of a FoldResult object)
            name (str, optional): optional name to recognize the result
            comments (str, optional): optional comments of the result
        """
        self.metrics = metrics
        self.history = history
        self.hp_config = hp_config
        self.name = name
        self.comments = comments

    def plot(self, *args, **kwargs):
        """
        Plot the curves of the metric specified in *args in a unique plot

        Keyword Args:
            title (str): title for the plot
            save_path (str): if provided the plot will be saved in path
            show (bool): if true display the plot
        """
        fig, ax = plt.subplots()
        for s in args:
            y = self.history[s]
            x = range(len(y))
            ax.plot(x, y, label=s)
            ax.set_xlabel(str(s))
            ax.legend()

        if "title" in kwargs:
            ax.set_title(kwargs["title"])

        if "save_path" in kwargs:
            fig.savefig(kwargs["save_path"])

        if "show" not in kwargs or kwargs["show"]:
            plt.show()
        plt.close()

    def dump(self, path):
        with open(path, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(path):
        with open(path, "rb") as file:
            return pickle.load(file)
