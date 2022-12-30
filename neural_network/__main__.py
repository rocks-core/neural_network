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
                generator_space=[0.5, 0.3, 0.1, 0.05],
                unfold=True
            ),
            "momentum": Hyperparameter(
                generator_logic="all_from_list",
                generator_space=[0.1, 0.5, 0.7],
                unfold=True
            ),
            "regularization": Hyperparameter(
                generator_logic="all_from_list",
                generator_space=[0., 0.00001, 0.000001, 0.0000001],
                unfold=True
            ),
            "batch_size": Hyperparameter(
                generator_logic="all_from_list",
                generator_space=[20],
                unfold=True
            ),
            "num_epochs": Hyperparameter(
                generator_logic="all_from_list",
                generator_space=[500],
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
                    [EarlyStopping("val_mse", 50, "min", 0.0001, False)]
                ],
                unfold=True
            )
        }
    )

    internal_folds = 4
    #external_folds = 4

    results = ["monk1_results.pickle", "monk2_results.pickle", "monk3_results.pickle"]
    #results = ["monk1_results.pickle"]
    dfs = [datasets.read_monk1(), datasets.read_monk2(), datasets.read_monk3()]
    #dfs = [datasets.read_monk1()]
    """
    # TEST WITH DOUBLE K FOLD
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
    """

    for result_name, (train_df, test_df) in zip(results, dfs):
        #result_name = "monk1.pickle"
        #train_df, test_df = datasets.read_monk1()

        trainval_inputs = train_df[dataset_attribute_columns].to_numpy(dtype=np.float32)
        trainval_outputs = train_df[dataset_class_column].to_numpy()
        test_inputs = test_df[dataset_attribute_columns].to_numpy(dtype=np.float32)
        test_outputs = test_df[dataset_class_column].to_numpy()

        selection_tuner = TunerCV(
            configurations=grid,
            model_builder=model_builder,
            n_fold=internal_folds,
            verbose=False,
            default_metric="binary_accuracy",
            default_reverse=True
        )
        validation_result = selection_tuner.fit(trainval_inputs, trainval_outputs)
        model = selection_tuner.best_model()

        refit_result = model.fit(trainval_inputs, trainval_outputs)
        test_results = model.evaluate_result(test_inputs, test_outputs)

        refit_result.dump(f"/home/bendico765/Scrivania/statuto/refit_{result_name}")
        validation_result.dump(f"/home/bendico765/Scrivania/statuto/validation_{result_name}")
        test_results.dump(f"/home/bendico765/Scrivania/statuto/test_{result_name}")

if __name__ == "__main__":
    main()
