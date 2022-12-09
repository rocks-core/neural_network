from neural_network import ActivationFunctions
from neural_network import LossFunctions
from neural_network import MLClassifier
from neural_network import datasets
from neural_network.classes.Layer import HiddenLayer, OutputLayer, InputLayer
from neural_network.classes.Optimizers import SGD, NesterovSGD
from neural_network.classes.Initializer import Uniform
from neural_network.classes.Validation import ConfigurationGenerator, Hyperparameter, model_builder
import neural_network.utils
import numpy as np


def main():
    dataset_attribute_columns = ["a1", "a2", "a3", "a4", "a5", "a6"]
    dataset_class_column = "class"
    number_inputs = len(dataset_attribute_columns)
    loss_function = LossFunctions.MSE()

    tr_df, vl_df, _ = neural_network.utils.split_samples(
        df=datasets.read_monk1()[0],
        tr_size=0.7,
        vl_size=0.3,
        ts_size=0.0
    )

    tr_inputs = tr_df[dataset_attribute_columns].to_numpy(dtype=np.float32)
    tr_outputs = tr_df[dataset_class_column].to_numpy()
    vl_inputs = vl_df[dataset_attribute_columns].to_numpy(dtype=np.float32)
    vl_outputs = vl_df[dataset_class_column].to_numpy()

    grid = test_grid_search()

    for config in grid:
        model = model_builder(
            config=config,
            input_shape=tr_inputs.shape[-1],
            output_shape=1,
            verbose=True
        )

        model.fit(tr_inputs, tr_outputs, validation_data=[vl_inputs, vl_outputs])


def test_grid_search():
    return ConfigurationGenerator(
        mode="grid",
        num_trials=1,
        folded_hyper_space={

            "loss_function": Hyperparameter(
                generator_logic="all_from_list",
                generator_space=[LossFunctions.MSE]
            ),
            "optimizer": Hyperparameter(
                generator_logic="all_from_list",
                generator_space=[NesterovSGD]
            ),
            "optimizer_learning_rate": Hyperparameter(
                generator_logic="all_from_list",
                generator_space=[0.1]
            ),
            "optimizer_momentum": Hyperparameter(
                generator_logic="all_from_list",
                generator_space=[0.9]
            ),
            "optimizer_regularization": Hyperparameter(
                generator_logic="all_from_list",
                generator_space=[0.]
            ),
            "batch_size": Hyperparameter(
                generator_logic="all_from_list",
                generator_space=[200]
            ),
            "num_epochs": Hyperparameter(
                generator_logic="all_from_list",
                generator_space=[1000]
            ),
            "num_hidden_layers": Hyperparameter(
                generator_logic="all_from_list",
                generator_space=[3]
            ),
            "neurons_in_layer_1": Hyperparameter(
                generator_logic="all_from_list",
                generator_space=[20, 30]
            ),
            "neurons_in_layer_2": Hyperparameter(
                generator_logic="all_from_list",
                generator_space=[20, 30]
            ),
            "neurons_in_layer_3": Hyperparameter(
                generator_logic="all_from_list",
                generator_space=[20, 30]
            )
        }
    )


if __name__ == "__main__":
    main()
