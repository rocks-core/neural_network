import numpy as np
import pandas as pd
from neural_network.classes import ADModel
import neural_network.classes.autodiff as ad

from neural_network import datasets
from neural_network.classes import Model
from neural_network.classes.Layers import *
from neural_network.classes import ActivationFunctions
from neural_network.classes.Optimizers import *
from neural_network.classes.LossFunctions import MSE
from neural_network.classes.Initializer import *
from neural_network.classes.Callbacks import EarlyStopping, WandbLogger
import time

df, dft = datasets.read_monk1()

dataset_attribute_columns = ["a1", "a2", "a3", "a4", "a5", "a6"]
dataset_class_column = "class"
df = pd.get_dummies(df, columns=dataset_attribute_columns)
dft = pd.get_dummies(dft, columns=dataset_attribute_columns)

y = df.pop("class").to_numpy()
X = df.to_numpy(dtype=np.float32)

perm = np.random.permutation(X.shape[0])
X = X[perm]
y = y[perm]
y = np.reshape(y, (-1, 1))

test_y = dft.pop("class").to_numpy()
test_x = dft.to_numpy(dtype=np.float32)

ad_layers = [
	ad.Layers.InputLayer((None, X.shape[-1]), 5, ad.ActivationFunctions.Sigmoid(), initializer=Uniform(-1., 1.)),
	# ad.Layers.DenseLayer(5, ad.ActivationFunctions.Sigmoid(), initializer=Uniform(-1., 1.)),
	ad.Layers.DenseLayer(1, ad.ActivationFunctions.Sigmoid(), initializer=Uniform(-1., 1.))
]

ad_model = ADModel(
	layers=ad_layers,
	loss=ad.LossFunction.MSE(),
	optimizer=SGD(learning_rate=0.1, momentum=0.05, regularization=0.),
	metrics=["mse", "mae", "binary_accuracy"],
	verbose=True
)

layers = [
	InputLayer((None, X.shape[-1]), 5, ActivationFunctions.Sigmoid(), initializer=Uniform(-1., 1.)),
	# HiddenLayer(5, ActivationFunctions.Sigmoid(), initializer=Uniform(-1., 1.)),
	OutputLayer(1, ActivationFunctions.Sigmoid(), initializer=Uniform(-1., 1.))
]

bp_model = Model(layers=layers,
                 loss=MSE(),
                 optimizer=SGD(learning_rate=0.1, momentum=0.05, regularization=0.),
                 metrics=["mse", "mae", "binary_accuracy"],
                 verbose=True)

bp_model.set_weights(ad_model.get_weights())

# keras_layer = [keras.layers.Dense(5, activation=keras.activations.sigmoid),
#                # keras.layers.Dense(5, activation=keras.activations.sigmoid),
#                keras.layers.Dense(1, activation=keras.activations.sigmoid)]
#
# keras_model = keras.Sequential(keras_layer)
# keras_model.compile(optimizer=keras.optimizers.SGD(0.15),
#                     loss=keras.losses.MSE,
#                     metrics=["accuracy"])
# keras_model.build(input_shape=(None, X.shape[-1]))
# weights = ad_model.get_weights()
# keras_weights = []
# for w in weights:
# 	keras_weights.append(w[1:, :])
# 	keras_weights.append(w[1, :])
# keras_model.set_weights(keras_weights)
#
# with tf.GradientTape() as tape:
# 	inputs = X
# 	for l in keras_model.layers:
# 		inputs = l(inputs)
# 	output = inputs
# 	loss = tf.losses.MSE(y, output)
#
# keras_deltas = tape.gradient(loss, keras_model.trainable_weights)[-2].numpy()
#
# ad_deltas = -ad_model.fit_pattern(X, y)[-1][1:]
#
# ratio = keras_deltas / ad_deltas
#
# bp_deltas = -bp_model.fit_pattern(X, y)[-1][1:]
# div_delta = bp_deltas / 124
#
# deltas = []
# for x_s, y_s in zip(np.split(X, 124), np.split(y, 124)):
# 	deltas.append(bp_model.fit_pattern(x_s, y_s)[-1])
#
# d = np.sum(deltas, axis=0)
# a = 0
#
# a = ad_deltas[0] / bp_deltas[0]

t = time.time_ns()
h = ad_model.fit(X, y, validation_data=[test_x, test_y], epochs=500, batch_size=124,
                 callbacks=[EarlyStopping("val_mse", patience=50, mode="min", min_delta=1e-3, restore_best_weight=True),
                            # WandbLogger("all")
                            ])
t = time.time_ns() - t
print(f"training took {t * 1e-9} seconds")
h.plot("mse", "val_mse")
h.plot("binary_accuracy", "val_binary_accuracy")

t = time.time_ns()
h = bp_model.fit(X, y, validation_data=[test_x, test_y], epochs=500, batch_size=124,
                 callbacks=[EarlyStopping("val_mse", patience=50, mode="min", min_delta=1e-3, restore_best_weight=True),
                            # WandbLogger("all")
                            ])
t = time.time_ns() - t
print(f"training took {t * 1e-9} seconds")
h.plot("mse", "val_mse")
h.plot("binary_accuracy", "val_binary_accuracy")
