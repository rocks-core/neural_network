from sklearn.preprocessing import MinMaxScaler
from model_builder import model_builder
from neural_network.classes.Callbacks import EarlyStopping
from neural_network.classes.LossFunctions import MeanEuclideanDistance
from neural_network.classes.Optimizers import *
from neural_network.classes.Validation import *
import pandas as pd
from neural_network.classes.ActivationFunctions import Sigmoid
from neural_network.classes.Validation import TunerCV


dataset_attribute_columns = ["a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9"]
dataset_class_column = ["target_x", "target_y"]
dataset = pd.read_csv("neural_network/datasets/MLCup/train.csv", skiprows=7, index_col=0, names= dataset_attribute_columns + dataset_class_column)

dataset_y = dataset[dataset_class_column].to_numpy()
dataset_x = dataset[dataset_attribute_columns].to_numpy()

hp = {"num_hidden_layers": Hyperparameter(
    generator_logic="all_from_list",
    generator_space=[3],
    unfold=True),
    "neurons_in_layer_1": Hyperparameter(
    generator_logic="all_from_list",
    generator_space=[10, 15, 20],
    unfold=True),
    "neurons_in_layer_2": Hyperparameter(
    generator_logic="all_from_list",
    generator_space=[10, 15, 20],
    unfold=True),
    "neurons_in_layer_3": Hyperparameter(
    generator_logic="all_from_list",
    generator_space=[10, 15, 20],
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
    generator_space=[0.05, 0.03, 0.01, 0.005],
    unfold=True),
    "momentum": Hyperparameter(
    generator_logic="all_from_list",
    generator_space=[0.6, 0.5, 0.4],
    unfold=True),
    "regularization": Hyperparameter(
    generator_logic="all_from_list",
    generator_space=[0., 1e-8, 1e-7, 1e-6],
    unfold=True),
    "activation_function": Hyperparameter(
    generator_logic="all_from_list",
    generator_space=[Sigmoid],
    unfold=True),
    "batch_size": Hyperparameter(
    generator_logic="all_from_list",
    generator_space=[200],
    unfold=True),
    "num_epochs": Hyperparameter(
    generator_logic="all_from_list",
    generator_space=[1000],
    unfold=True),
    "callbacks": Hyperparameter(
    generator_logic="all_from_list",
    generator_space=[[EarlyStopping(monitor="val_mean_euclidean_distance", patience=50, mode="min", min_delta=1e-2, restore_best_weight=False)]],
    unfold=True)
}

tuner = TunerCV(ConfigurationGenerator(hp, mode="grid"), model_builder, n_fold=4, verbose=True,
                default_metric="val_mean_euclidean_distance", default_reverse=False)

res = tuner.fit(dataset_x, dataset_y)
res.dump("dumps/fine_search_scaled.pickle")


