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
dataset = pd.read_csv("neural_network/datasets/MLCup/train.csv", skiprows=7, index_col=0, names= dataset_attribute_columns + dataset_class_column)

train, val, _ = split_samples(dataset, 0.75, 0.25, 0., shuffle=True)

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
        InputLayer((None, train_x.shape[-1]), 10, ActivationFunctions.Sigmoid(), initializer=Uniform(-1, 1)),
        # HiddenLayer(10, ActivationFunctions.TanH(), initializer=Uniform(0., 0.1)),
        HiddenLayer(20, ActivationFunctions.Sigmoid(), initializer=Uniform(-1, 1)),
        OutputLayer(2, ActivationFunctions.Linear(), initializer=Uniform(-1, 1))
    ]

model = Model(
    layers=layers,
    loss=MeanEuclideanDistance(),
    optimizer=SGD(learning_rate=0.05, momentum=0.5, regularization=0.0),
    metrics=["mse", "mean_euclidean_distance"],
    verbose=True
)

# learning rate 0.05 - 0.0001
# momentum 0. - 0.5
# regularization 0., 0.0001 - 0.0000001
# unit in layer 5, 10, 20

t = time.time_ns()
h = model.fit(train_x, train_y, validation_data=[val_x, val_y], epochs=1000, batch_size=200,
              callbacks=[EarlyStopping("val_mean_euclidean_distance", patience=50, mode="min", min_delta=1e-2, restore_best_weight=False),
                         # WandbLogger("all")
                         ])
t = time.time_ns() - t
print(f"training took {t*1e-9} seconds")
h.plot("mse", "val_mse")
h.plot("mean_euclidean_distance", "val_mean_euclidean_distance")
