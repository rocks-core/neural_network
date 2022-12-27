from neural_network.classes.Callbacks import EarlyStopping, WandbLogger
from neural_network.classes.LossFunctions import MeanEuclideanDistance
from neural_network import datasets
from neural_network.classes.Optimizers import *
from neural_network.classes.Validation import *
from neural_network.utils import split_samples
from neural_network import utils
import pandas as pd
import wandb
from neural_network.classes.Validation.model_builder import model_builder

dataset_attribute_columns = ["a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9"]
dataset_class_column = ["target_x", "target_y"]
dataset = pd.read_csv("neural_network/datasets/ML-CUP22-TR.csv", skiprows=7, index_col=0, names= dataset_attribute_columns + dataset_class_column)

# train, val, _ = split_samples(dataset, 0.3, 0.7, 0., shuffle=True)

dataset_y = dataset[dataset_class_column].to_numpy()
dataset_x = dataset[dataset_attribute_columns].to_numpy()


# def model_builder(hp):
#     layers = [
#         InputLayer((None, dataset_x.shape[-1]), hp["units1"], ActivationFunctions.Sigmoid(), initializer=Uniform(-0.1, 0.1)),
#         HiddenLayer(hp["units2"], ActivationFunctions.Sigmoid(), initializer=Uniform(-0.1, 0.1)),
#         OutputLayer(2, ActivationFunctions.Linear(), initializer=Uniform(-0.1, 0.1))
#     ]
#
#     model = Model(
#         layers=layers,
#         loss=MeanEuclideanDistance(),
#         n_epochs=200,
#         batch_size=50,
#         optimizer=SGD(learning_rate=hp["learning_rate"], momentum=hp["momentum"], regularization=hp["regularization"]),
#         metrics=["mse", "mean_euclidean_distance"],
#         shuffle=True,
#         verbose=False
#     )
#     return model

# def model_builder(hp):
#     return None

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
    generator_space=[MeanEuclideanDistance],
    unfold=True),
    "optimizer": Hyperparameter(
    generator_logic="all_from_list",
    generator_space=[SGD],
    unfold=True),
    "learning_rate": Hyperparameter(
    generator_logic="all_from_list",
    generator_space=[0.05, 0.03, 0.01, 0.005, 0.001],
    unfold=True),
    "momentum": Hyperparameter(
    generator_logic="all_from_list",
    generator_space=[0.5, 0.1, 0.01, 0.],
    unfold=True),
    "regularization": Hyperparameter(
    generator_logic="all_from_list",
    generator_space=[0., 0.000001, 0.00001, 0.0001],
    unfold=True),
    "batch_size": Hyperparameter(
    generator_logic="all_from_list",
    generator_space=[100],
    unfold=True),
    "num_epochs": Hyperparameter(
    generator_logic="all_from_list",
    generator_space=[700],
    unfold=True),
    "callbacks": Hyperparameter(
    generator_logic="all_from_list",
    generator_space=[[EarlyStopping(monitor="val_mean_euclidean_distance", patience=50, mode="min", min_delta=1e-4, restore_best_weight=True)]],
    unfold=True)
}
# tuner = TunerHO(ConfigurationGenerator(hp, mode="grid"), model_builder, validation_size=0.3, verbose=True)
tuner = TunerCV(ConfigurationGenerator(hp, mode="random", num_trials=8), model_builder, n_fold=4, verbose=True,
                default_metric="val_mean_euclidean_distance", default_reverse=False)
tester = TesterCV(tuner, n_fold=4, verbose=True)

r = tester.fit(dataset_x, dataset_y)
r.dump("./dumps/test1.pickle")

r = TestResult.load("./dumps/test1.pickle")

r.validation_results[0].plot_one(0, "mse", "val_mse")
r.validation_results[0].plot_one(0, "mean_euclidean_distance", "val_mean_euclidean_distance")
