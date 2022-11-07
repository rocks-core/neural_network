from neural_network import ActivationFunctions
from neural_network import MLClassifier
from neural_network import datasets
import neural_network.utils


if __name__ == "__main__":
    dataset_attribute_columns = ["a1", "a2", "a3", "a4", "a5", "a6"]
    dataset_class_column = "class"
    number_inputs = len(dataset_attribute_columns)
    layer_sizes = (2, 2, 1)

    tr_df, vl_df, _ = neural_network.utils.split_samples(
        df=datasets.read_monk1()[0],
        tr_size=0.7,
        vl_size=0.3,
        ts_size=0.0
    )
    n_trials = 5

    tr_inputs = tr_df[dataset_attribute_columns].to_numpy()
    tr_outputs = tr_df[dataset_class_column].to_numpy()
    vl_inputs = vl_df[dataset_attribute_columns].to_numpy()
    vl_outputs = vl_df[dataset_class_column].to_numpy()

    trials = []
    for _ in range(n_trials):
        classifier = MLClassifier(
            number_inputs=number_inputs,
            layer_sizes=layer_sizes,
            activation_functions=(
                ActivationFunctions.Linear(),
                ActivationFunctions.Linear(),
                ActivationFunctions.Sigmoid()
            ),
            #regularization_term=0.01,
            batch_size=10,
            learning_rate=0.001,
            n_epochs=200,
            verbose=False
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

