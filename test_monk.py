from neural_network import datasets
from neural_network import Model
from neural_network.classes import ActivationFunctions
from neural_network.classes.Layers import InputLayer, OutputLayer, HiddenLayer
from neural_network.classes.LossFunctions import MSE
from neural_network.classes.Initializer import Uniform
from model_builder_monk import model_builder
from neural_network.classes.Callbacks import EarlyStopping
from neural_network.classes.LossFunctions import MeanEuclideanDistance
from neural_network.classes.Optimizers import *
from neural_network.classes.Validation import *
import pandas as pd
from neural_network.classes.ActivationFunctions import Sigmoid, TanH
from neural_network.classes.Validation import TunerCV

dataset_attribute_columns = ["a1", "a2", "a3", "a4", "a5", "a6"]
dataset_class_column = "class"

paths = ["dumps/monk_results/monk1/", "dumps/monk_results/monk2/", "dumps/monk_results/monk3/"]
for i, p in enumerate(paths):

    df, dft = datasets.read_monk(i+1)

    df = pd.get_dummies(df, columns=dataset_attribute_columns)
    dft = pd.get_dummies(dft, columns=dataset_attribute_columns)


    y = df.pop("class").to_numpy()
    X = df.to_numpy(dtype=np.float32)

    perm = np.random.permutation(X.shape[0])
    X = X[perm]
    y = y[perm]

    test_y = dft.pop("class").to_numpy()
    test_x = dft.to_numpy(dtype=np.float32)


    hp = {"num_hidden_layers": Hyperparameter(
        generator_logic="all_from_list",
        generator_space=[2],
        unfold=True),
        "neurons_in_layer_1": Hyperparameter(
        generator_logic="all_from_list",
        generator_space=[5, 10, 20],
        unfold=True),
        "neurons_in_layer_2": Hyperparameter(
        generator_logic="all_from_list",
        generator_space=[5, 10, 20],
        unfold=True),
        "loss_function": Hyperparameter(
        generator_logic="all_from_list",
        generator_space=[MSE],
        unfold=True),
        "optimizer": Hyperparameter(
        generator_logic="all_from_list",
        generator_space=[SGD],
        unfold=True),
        "learning_rate": Hyperparameter(
        generator_logic="all_from_list",
        generator_space=[0.5, 0.3, 0.1, 0.05],
        unfold=True),
        "momentum": Hyperparameter(
        generator_logic="all_from_list",
        generator_space=[0.7, 0.4, 0.3],
        unfold=True),
        "regularization": Hyperparameter(
        generator_logic="all_from_list",
        generator_space=[0., 1e-7, 1e-6, 1e-5],
        unfold=True),
        "activation_function": Hyperparameter(
        generator_logic="all_from_list",
        generator_space=[TanH],
        unfold=True),
        "batch_size": Hyperparameter(
        generator_logic="all_from_list",
        generator_space=[50],
        unfold=True),
        "num_epochs": Hyperparameter(
        generator_logic="all_from_list",
        generator_space=[500],
        unfold=True),
        "callbacks": Hyperparameter(
        generator_logic="all_from_list",
        generator_space=[[EarlyStopping(monitor="val_mse", patience=50, mode="min", min_delta=1e-4, restore_best_weight=False)]],
        unfold=True)
    }

    tuner = TunerCV(ConfigurationGenerator(hp, mode="grid"), model_builder, n_fold=4, verbose=True,
                    default_metric="val_binary_accuracy", default_reverse=True)

    val_res = tuner.fit(X, y)
    best_model = tuner.best_model()
    refit_results = best_model.fit(X, y)
    best_model.dump_weights(p + "best_weights.pickle")
    test_result = best_model.evaluate_result(test_x, test_y)
    test_result.validation_results = val_res
    test_result.refit_results = refit_results
    test_result.dump(p + "test_results.pickle")










# layers = [
#         InputLayer((None, X.shape[-1]), 20, ActivationFunctions.Sigmoid(), initializer=Uniform(-1, 1)),
#         HiddenLayer(10, ActivationFunctions.Sigmoid(), initializer=Uniform(-1, 1)),
#         OutputLayer(1, ActivationFunctions.Sigmoid(), initializer=Uniform(-1, 1))
#     ]
#
# model = Model(
#     layers=layers,
#     loss=MSE(),
#     metrics=["mse", "binary_accuracy"],
#     optimizer=SGD(learning_rate=0.5, momentum=0.7, regularization=1e-5),
#     batch_size=20,
#     n_epochs=500,
#     verbose=True
# )
#
# h = model.fit(train_x, train_y, validation_data=[val_x, val_y])
# h.plot("mse", "val_mse")
# h.plot("binary_accuracy", "val_binary_accuracy")
# print(model.evaluate(test_x, test_y))
# a = 0