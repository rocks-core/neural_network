import matplotlib.pyplot as plt


class Result:
    def __init__(self, metrics, result, hp_config=None, name="", comments=""):
        self.metrics = metrics
        self.result = result
        self.hp_config = None
        self.name = name
        self.comments = comments

    def plot(self, *args, **kwargs):
        fig, ax = plt.subplots()
        for s in args:
            y = self.result[s]
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
