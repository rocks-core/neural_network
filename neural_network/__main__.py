import numpy as np

from neural_network import ActivationFunctions
from neural_network.classes.Layer import HiddenLayer, OutputLayer
from neural_network import MLClassifier
from neural_network import datasets


if __name__ == "__main__":
    number_inputs = 2
    layer_sizes = (2, 2, 1)
    activation_functions = (
        ActivationFunctions.Linear(),
        ActivationFunctions.Linear(),
        ActivationFunctions.Sigmoid()
    )
    tr_df, vl_df, _ = datasets.circle()
    n_trials = 5

    tr_inputs = tr_df[["x", "y"]].to_numpy()
    tr_outputs = tr_df[["class"]].to_numpy()
    vl_inputs = vl_df[["x", "y"]].to_numpy()
    vl_outputs = vl_df[["class"]].to_numpy()

    trials = []

    layers = [
        (  # 2 units hidden layer with linear act. fun
            HiddenLayer(2, ActivationFunctions.Linear()),
            np.random.rand(2, 3) * 0.5 - 0.2
        ),
        (  # 2 units hidden layer with linear act. fun
            HiddenLayer(2, ActivationFunctions.Linear()),
            np.random.rand(2, 3) * 0.5 - 0.2
        ),
        (  # 1 unit output layer with sigmoid function
            OutputLayer(1, ActivationFunctions.Sigmoid()),
            np.random.rand(1, 3) * 0.5 - 0.2
        ),
    ]
    """
    layers = {
        0: {  # 2 units hidden layer with linear act. fun
            "layer": HiddenLayer(2, ActivationFunctions.Linear()),
            "weights": np.random.rand(2, 3) * 0.5 - 0.2
        },
        1: {  # 2 units hidden layer with linear act. fun
            "layer": HiddenLayer(2, ActivationFunctions.Linear()),
            "weights": np.random.rand(2, 3) * 0.5 - 0.2
        },
        2: {  # 1 unit output layer with sigmoid function
            "layer": OutputLayer(1, ActivationFunctions.Sigmoid()),
            "weights": np.random.rand(1, 3) * 0.5 - 0.2
        }
    }
    """
    for _ in range(n_trials):
        classifier = MLClassifier(
            layers=layers,
            batch_size=1,
            learning_rate=0.001,
            n_epochs=200,
            verbose=True
        )
        # training model
        classifier.fit(tr_inputs, tr_outputs)
        print("Done training")

        # validating result
        correct_predictions = 0
        for (input, expected_output) in zip(vl_inputs, vl_outputs):
            real_output = classifier.predict(input)[0]
            if round(real_output) == expected_output:
                correct_predictions += 1

        trials.append(100 * (correct_predictions / len(vl_df)))

    print(f"min: {min(trials)}")
    print(f"max: {max(trials)}")
    avg = lambda l: sum(l)/len(l) if len(l) != 0 else 0
    print(f"avg: {avg(trials)}")
