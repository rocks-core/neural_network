import pandas as pd
from neural_network import datasets
from neural_network.classes import MLClassifier
from neural_network.classes.Layer import *
from neural_network.classes import ActivationFunctions
from neural_network.classes.Optimizers import *
from neural_network.classes.LossFunctions import MSE
from neural_network.classes.Initializer import *
from neural_network.classes.Callbacks import EarlyStopping, WandbLogger
import time

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

test_y = dft.pop("class").to_numpy()
test_x = dft.to_numpy(dtype=np.float32)

layers = [
        InputLayer((None, X.shape[-1]), 15, ActivationFunctions.Sigmoid(), initializer=Gaussian(0., 0.1)),
        HiddenLayer(10, ActivationFunctions.Sigmoid(), initializer=Uniform(-1, 1)),
        OutputLayer(1, ActivationFunctions.Sigmoid(), initializer=Gaussian(0., 0.1))
    ]

model = MLClassifier(
    layers=layers,
    loss=MSE(),
    optimizer=SGD(learning_rate=0.12, momentum=0., regularization=0.),
    metrics=["mse", "mae"],
    batch_size=100,
    verbose=True
)

t = time.time_ns()
h = model.fit(X, y, validation_data=[test_x, test_y], epochs=500,
              callbacks=[EarlyStopping("val_mse", patience=50, mode="min", min_delta=1e-3, restore_best_weight=True),
                         # WandbLogger("all")
                         ])
t = time.time_ns() - t
print(f"training took {t*1e-9} seconds")
h.plot("mse", "val_mse")
