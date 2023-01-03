import matplotlib.pyplot as plt
import pickle
import os

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
        fig.set_figwidth(11)
        fig.set_figheight(11)
        for s in args:
            y = self.history[s]
            x = range(len(y))
            if "val_" in s:
                linestyle = "dashed"
            else:
                linestyle = "solid"
            ax.tick_params(axis='both', labelsize=16)
            ax.plot(x, y, label=s, linestyle=linestyle)
            ax.legend()

        if "title" in kwargs:
            ax.set_title(kwargs["title"])
        else:
            ax.set_title(str(self.hp_config))

        if "save_path" in kwargs:
            if "title" in kwargs:
                title = kwargs["title"]
            else:
                title = "plot"
            os.makedirs(kwargs["save_path"], exist_ok=True)
            fig.savefig(kwargs["save_path"] + "/" +  title)

        if "show" not in kwargs or kwargs["show"]:
            plt.show()
        plt.close()

    def dump(self, path):
        folder = os.path.dirname(path)
        os.makedirs(folder, exist_ok=True)
        with open(path, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(path):
        with open(path, "rb") as file:
            return pickle.load(file)
