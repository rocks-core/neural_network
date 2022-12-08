import numpy as np


class FoldResult:
    def __init__(self, hp_config, results, name="", comments=""):
        self.hp_config = hp_config
        self.results = results
        self.metrics = {}
        for m in results[0].metrics.keys():
            self.metrics[m] = np.array([r.metrics[m] for r in results]).mean()
        self.name = name
        self.comments = comments

    def plot(self, *args, **kwargs):
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
