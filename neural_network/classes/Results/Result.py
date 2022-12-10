import matplotlib.pyplot as plt


class Result:
    def __init__(self, metrics, history, hp_config=None, name="", comments=""):
        self.metrics = metrics
        self.history = history
        self.hp_config = hp_config
        self.name = name
        self.comments = comments

    def plot(self, *args, **kwargs):
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
