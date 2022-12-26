import pandas as pd
from neural_network import datasets
from neural_network.classes import Model
from neural_network.classes.Layers import *
from neural_network.classes import ActivationFunctions
from neural_network.classes.Optimizers import *
from neural_network.classes.LossFunctions import MSE, MeanEuclideanDistance
from neural_network.classes.Initializer import *
from neural_network.classes.Callbacks import EarlyStopping, WandbLogger
import time
from neural_network.utils import split_samples
# import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# dataset_attribute_columns = ["a1", "a2", "a3", "a4", "a5", "a6"]
# dataset_class_column = "class"

# (X, y), (test_x, test_y) = tf.keras.datasets.boston_housing.load_data(
#     path="boston_housing.npz", test_split=0.2, seed=113
# )
#
# X = minmax_scale(X)
# test_x = minmax_scale(test_x)
# df, dft = datasets.read_monk1()
#
# df = pd.get_dummies(df, columns=dataset_attribute_columns)
# dft = pd.get_dummies(dft, columns=dataset_attribute_columns)
#
#
# y = df.pop("class").to_numpy()
# X = df.to_numpy(dtype=np.float32)
#
# perm = np.random.permutation(X.shape[0])
# X = X[perm]
# y = y[perm]
#
# test_y = dft.pop("class").to_numpy()
# test_x = dft.to_numpy(dtype=np.float32)


dataset_attribute_columns = ["a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9"]
dataset_class_column = ["target_x", "target_y"]
dataset = pd.read_csv("neural_network/datasets/ML-CUP22-TR.csv", skiprows=7, index_col=0, names= dataset_attribute_columns + dataset_class_column)

train, val, _ = split_samples(dataset, 0.3, 0.7, 0., shuffle=True)

train_y = train[dataset_class_column].to_numpy()
train_x = train[dataset_attribute_columns].to_numpy()

val_y = val[dataset_class_column].to_numpy()
val_x = val[dataset_attribute_columns].to_numpy()

# scaler = MinMaxScaler()
# train_x = scaler.fit_transform(train_x)
# val_x = scaler.transform(val_x)
# train_y = scaler.fit_transform(train_y)
# val_y = scaler.transform(val_y)

layers = [
        InputLayer((None, train_x.shape[-1]), 15, ActivationFunctions.TanH(), initializer=Gaussian(0., 0.1)),
        # HiddenLayer(35, ActivationFunctions.TanH(), initializer=Gaussian(0., 0.1)),
        # HiddenLayer(50, ActivationFunctions.TanH(), initializer=Gaussian(0., 0.1)),
        # HiddenLayer(50, ActivationFunctions.TanH(), initializer=Gaussian(0., 0.1)),
        HiddenLayer(22, ActivationFunctions.TanH(), initializer=Gaussian(0., 0.1)),
        OutputLayer(2, ActivationFunctions.Linear(), initializer=Gaussian(0., 0.1))
    ]

model = Model(
    layers=layers,
    loss=MeanEuclideanDistance(),
    optimizer=NesterovSGD(learning_rate=0.05, momentum=0.9, regularization=0.),
    metrics=["mse", "mean_euclidean_distance"],
    verbose=True
)

t = time.time_ns()
h = model.fit(train_x, train_y, validation_data=[val_x, val_y], epochs=200, batch_size=100,
              callbacks=[# EarlyStopping("val_mse", patience=50, mode="min", min_delta=1e-3, restore_best_weight=True),
                         # WandbLogger("all")
                         ])
t = time.time_ns() - t
print(f"training took {t*1e-9} seconds")
h.plot("mse", "val_mse")
h.plot("mean_euclidean_distance", "val_mean_euclidean_distance")
