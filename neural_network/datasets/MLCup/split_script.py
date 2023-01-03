import pandas as pd

from neural_network.utils import split_samples

dataset = pd.read_csv("../ML-CUP22-TR.csv", skiprows=7, index_col=0)

train, test, _ = split_samples(dataset, 0.75, 0.25, 0., shuffle=True)

train.to_csv("./train.csv", header=False)
test.to_csv("./test.csv", header=False)
