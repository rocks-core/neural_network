from neural_network import ActivationFunctions
from neural_network import LossFunctions
from neural_network import Model
from neural_network import datasets
from neural_network.classes.Layers import HiddenLayer, OutputLayer, InputLayer
from neural_network.classes.Optimizers import SGD, NesterovSGD
from neural_network.classes.Initializer import Uniform
from neural_network.classes.Validation import ConfigurationGenerator, Hyperparameter, model_builder
import neural_network.utils
import numpy as np
import pandas as pd
from neural_network.classes.Validation import TunerCV, TesterCV
from neural_network.classes.Callbacks.EarlyStopping import EarlyStopping


def main():
    dataset_attribute_columns = ["a1", "a2", "a3", "a4", "a5", "a6"]
    dataset_class_column = "class"

    grid = ConfigurationGenerator(
        mode="grid",
        folded_hyper_space={
            "loss_function": Hyperparameter(
                generator_logic="all_from_list",
                generator_space=[LossFunctions.MSE],
                unfold=True
            ),
            "optimizer": Hyperparameter(
                generator_logic="all_from_list",
                generator_space=[SGD],
                unfold=True
            ),
            "learning_rate": Hyperparameter(
                generator_logic="all_from_list",
                generator_space=[0.1, 0.05, 0.01, 0.005],
                unfold=True
            ),
            "momentum": Hyperparameter(
                generator_logic="all_from_list",
                generator_space=[0., 0.5, 0.9],
                unfold=True
            ),
            "regularization": Hyperparameter(
                generator_logic="all_from_list",
                generator_space=[0.0001, 0.00001, 0.000001],
                unfold=True
            ),
            "batch_size": Hyperparameter(
                generator_logic="all_from_list",
                generator_space=[200],
                unfold=True
            ),
            "num_epochs": Hyperparameter(
                generator_logic="all_from_list",
                generator_space=[1000],
                unfold=True
            ),
            "num_hidden_layers": Hyperparameter(
                generator_logic="all_from_list",
                generator_space=[2],
                unfold=True
            ),
            "neurons_in_layer_1": Hyperparameter(
                generator_logic="all_from_list",
                generator_space=[5, 10, 20],
                unfold=True
            ),
            "neurons_in_layer_2": Hyperparameter(
                generator_logic="all_from_list",
                generator_space=[5, 10, 20],
                unfold=True
            ),
            "callbacks": Hyperparameter(
                generator_logic="all_from_list",
                generator_space=[
                    [EarlyStopping("val_mse", 50, "min", 0.01, False)]
                ],
                unfold=True
            )
        }
    )

    internal_folds = 4
    external_folds = 4

    results = ["monk1_results.pickle", "monk2_results.pickle", "monk3_results.pickle"]
    dfs = [datasets.read_monk1(), datasets.read_monk2(), datasets.read_monk3()]

    for result_name, (tr, ts) in zip(results, dfs):
        #result_name = "monk1_results.pickle"
        #tr, ts = datasets.read_monk1()
        merged_df = pd.concat([tr, ts])

        merged_inputs = merged_df[dataset_attribute_columns].to_numpy(dtype=np.float32)
        merged_outputs = merged_df[dataset_class_column].to_numpy()

        selection_tuner = TunerCV(
            configurations=grid,
            model_builder=model_builder,
            n_fold=internal_folds,
            verbose=False,
            default_metric="binary_accuracy",
            default_reverse=True
        )
        tuner = TesterCV(
            tuner=selection_tuner,
            n_fold=external_folds,
            verbose=True
        )
        test_result = tuner.fit(
            inputs=merged_inputs,
            outputs=merged_outputs
        )
        test_result.dump(f"/home/bendico765/Scrivania/statuto/{result_name}")

if __name__ == "__main__":
    main()
