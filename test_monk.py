import numpy as np
import pandas as pd
from neural_network import datasets
from neural_network import Model
from neural_network.classes import ActivationFunctions
from neural_network.classes.Layers import InputLayer, OutputLayer, HiddenLayer
from neural_network.classes.Optimizers import SGD
from neural_network.classes.LossFunctions import MSE
from neural_network.classes.Initializer import Uniform
from neural_network.utils import split_samples

dataset_attribute_columns = ["a1", "a2", "a3", "a4", "a5", "a6"]
dataset_class_column = "class"

df, dft = datasets.read_monk1()

df = pd.get_dummies(df, columns=dataset_attribute_columns)
dft = pd.get_dummies(dft, columns=dataset_attribute_columns)


y = df.pop("class").to_numpy()
X = df.to_numpy(dtype=np.float32)

perm = np.random.permutation(X.shape[0])
X = X[perm]
y = y[perm]

train_x, val_x, _ = split_samples(X, 0.75, 0.35, 0.)
train_y, val_y, _ = split_samples(y, 0.75, 0.35, 0.)

test_y = dft.pop("class").to_numpy()
test_x = dft.to_numpy(dtype=np.float32)

layers = [
        InputLayer((None, X.shape[-1]), 20, ActivationFunctions.Sigmoid(), initializer=Uniform(-1, 1)),
        HiddenLayer(10, ActivationFunctions.Sigmoid(), initializer=Uniform(-1, 1)),
        OutputLayer(1, ActivationFunctions.Sigmoid(), initializer=Uniform(-1, 1))
    ]

model = Model(
    layers=layers,
    loss=MSE(),
    metrics=["mse", "binary_accuracy"],
    optimizer=SGD(learning_rate=0.5, momentum=0.7, regularization=1e-5),
    batch_size=20,
    n_epochs=500,
    verbose=True
)

h = model.fit(train_x, train_y, validation_data=[val_x, val_y])
h.plot("mse", "val_mse")
h.plot("binary_accuracy", "val_binary_accuracy")
print(model.evaluate(test_x, test_y))
a = 0